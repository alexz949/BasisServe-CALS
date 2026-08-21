"""Compiled static-ragged NCCL AllGather and decode runtime.

The runtime owns a dedicated NCCL communicator.  Every source width is fixed
for the lifetime of a layer plan, but widths may differ between TP ranks.  A
single compiled call gathers exact (unpadded) source payloads, packs rank-major
receives into token-major order when necessary, and optionally launches the
replicated decoder GEMM.

The C++/CUDA extension is compiled lazily.  Importing this module never starts
a compiler and never initializes a communicator.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F


_EXTENSION_NAME = "basisserve_ragged_allgather_v3"
_EXTENSION: Any | None = None
_EXTENSION_LOCK = threading.Lock()
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_GATHER_ALGORITHMS = ("direct", "pairwise", "ring", "biring_grouped")


def _source_directory() -> Path:
    return Path(__file__).resolve().parent / "csrc"


def _nccl_paths() -> tuple[Path, Path]:
    site_packages = Path(torch.__file__).resolve().parent.parent
    root = site_packages / "nvidia" / "nccl"
    include = root / "include"
    library = root / "lib"
    if not (include / "nccl.h").is_file():
        raise FileNotFoundError(f"NCCL header is unavailable under {include}")
    if not (library / "libnccl.so.2").is_file():
        raise FileNotFoundError(f"NCCL shared library is unavailable under {library}")
    return include, library


def _cuda_target_include() -> Path | None:
    candidate = Path(sys.prefix) / "targets" / "x86_64-linux" / "include"
    return candidate if (candidate / "cuda_runtime_api.h").is_file() else None


def _prefer_environment_cuda_toolkit() -> None:
    """Point the JIT builder at a conda CUDA toolkit when one is complete."""

    prefix = Path(sys.prefix)
    nvcc = prefix / "bin" / "nvcc"
    runtime_header = prefix / "targets" / "x86_64-linux" / "include" / "cuda_runtime.h"
    if nvcc.is_file() and runtime_header.is_file():
        os.environ.setdefault("CUDA_HOME", str(prefix))


def _match_loaded_cxx_runtime() -> None:
    """Avoid emitting symbols newer than the libstdc++ already loaded by Torch."""

    requested = os.environ.get("BASISSERVE_RAGGED_AG_CXX")
    if requested:
        os.environ["CXX"] = requested
        return
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return
    loaded = next(
        (line for line in maps.read_text(encoding="utf-8").splitlines() if "libstdc++.so" in line),
        "",
    )
    system_cxx = Path("/usr/bin/g++")
    if "/usr/lib/" in loaded and system_cxx.is_file():
        os.environ["CXX"] = str(system_cxx)
        # ``cpp_extension`` forwards CC to NVCC as ``-ccbin``.  Pointing it
        # at gcc can fail on cluster nodes where only the g++ driver knows how
        # to locate cc1plus, so use the same C++ driver for both build paths.
        os.environ["CC"] = str(system_cxx)


def load_ragged_allgather_extension(*, verbose: bool | None = None) -> Any:
    """Build or load the C++/CUDA extension exactly once in this process."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        _prefer_environment_cuda_toolkit()
        _match_loaded_cxx_runtime()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6+PTX")
        from torch.utils.cpp_extension import load

        source = _source_directory()
        nccl_include, nccl_library = _nccl_paths()
        include_paths = [source, nccl_include]
        cuda_target_include = _cuda_target_include()
        if cuda_target_include is not None:
            include_paths.append(cuda_target_include)
        requested_verbose = (
            os.environ.get("BASISSERVE_RAGGED_AG_BUILD_VERBOSE", "0") == "1"
            if verbose is None
            else bool(verbose)
        )
        _EXTENSION = load(
            name=_EXTENSION_NAME,
            sources=[
                str(source / "ragged_allgather.cpp"),
                str(source / "ragged_allgather_pack.cu"),
            ],
            extra_include_paths=[str(path) for path in include_paths],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--threads", "4"],
            extra_ldflags=[
                f"-L{nccl_library}",
                "-l:libnccl.so.2",
                f"-Wl,-rpath,{nccl_library}",
            ],
            with_cuda=True,
            verbose=requested_verbose,
        )
        return _EXTENSION


@dataclass(frozen=True)
class StaticRaggedPlan:
    """One layer's process-rank-ordered, compile-time communication geometry."""

    source_widths: tuple[int, ...]
    offsets: tuple[int, ...]
    total_width: int
    padded_total_width: int

    @classmethod
    def from_source_widths(cls, source_widths: Sequence[int]) -> "StaticRaggedPlan":
        widths = tuple(int(width) for width in source_widths)
        if not widths:
            raise ValueError("static ragged plan requires at least one source")
        if len(widths) > 32:
            raise ValueError("static ragged plan supports at most 32 sources")
        if any(width <= 0 for width in widths):
            raise ValueError(f"source widths must be positive, got {widths}")
        offsets: list[int] = []
        prefix = 0
        for width in widths:
            offsets.append(prefix)
            prefix += width
        return cls(
            source_widths=widths,
            offsets=tuple(offsets),
            total_width=prefix,
            padded_total_width=len(widths) * max(widths),
        )

    @classmethod
    def from_head_ranks(
        cls,
        source_head_ranks: Sequence[int],
        *,
        heads_per_source: int,
    ) -> "StaticRaggedPlan":
        if heads_per_source <= 0:
            raise ValueError("heads_per_source must be positive")
        return cls.from_source_widths(
            tuple(heads_per_source * int(rank) for rank in source_head_ranks)
        )

    @property
    def padding_overhead(self) -> float:
        return self.padded_total_width / self.total_width - 1.0

    def rank_major_offsets(self, batch: int) -> tuple[int, ...]:
        if batch <= 0:
            raise ValueError("batch must be positive")
        return tuple(batch * offset for offset in self.offsets)


class RaggedNcclCommunicator:
    """Python lifetime wrapper around the compiled, dedicated NCCL communicator."""

    def __init__(self, implementation: Any) -> None:
        self._implementation = implementation
        self._registered_workspaces: dict[
            tuple[torch.dtype, int, tuple[int, ...], int], Tensor
        ] = {}

    @classmethod
    def from_process_group(
        cls,
        process_group: dist.ProcessGroup | None = None,
        *,
        device: torch.device | int | None = None,
    ) -> "RaggedNcclCommunicator":
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        if str(dist.get_backend(process_group)).lower() != "nccl":
            raise ValueError("compiled ragged AllGather requires an NCCL process group")
        if device is None:
            device_index = torch.cuda.current_device()
        elif isinstance(device, torch.device):
            if device.type != "cuda":
                raise ValueError(f"ragged communicator requires CUDA, got {device}")
            device_index = torch.cuda.current_device() if device.index is None else device.index
        else:
            device_index = int(device)
        resolved_device = torch.device("cuda", device_index)
        torch.cuda.set_device(resolved_device)
        group_rank = dist.get_rank(process_group)
        world_size = dist.get_world_size(process_group)
        extension = load_ragged_allgather_extension()

        if group_rank == 0:
            unique_id = extension.nccl_unique_id()
            unique_id_cpu = torch.tensor(
                list(unique_id),
                dtype=torch.uint8,
            )
        else:
            # NCCL unique IDs are currently 128 bytes.  Querying on rank zero
            # and broadcasting the length keeps this wrapper independent of
            # that implementation constant.
            unique_id_cpu = torch.empty(128, dtype=torch.uint8)
        length = torch.tensor(
            [int(unique_id_cpu.numel()) if group_rank == 0 else 0],
            dtype=torch.int64,
            device=resolved_device,
        )
        source_global_rank = (
            0
            if process_group is None
            else dist.get_global_rank(process_group, 0)
        )
        dist.broadcast(length, src=source_global_rank, group=process_group)
        expected_length = int(length.cpu().item())
        if group_rank != 0:
            unique_id_cpu = torch.empty(expected_length, dtype=torch.uint8)
        unique_id_cuda = unique_id_cpu.to(device=resolved_device)
        dist.broadcast(unique_id_cuda, src=source_global_rank, group=process_group)
        unique_id = bytes(unique_id_cuda.cpu().tolist())
        implementation = extension.RaggedNcclCommunicator(
            unique_id,
            group_rank,
            world_size,
            resolved_device.index,
        )
        return cls(implementation)

    @property
    def rank(self) -> int:
        return int(self._implementation.rank)

    @property
    def world_size(self) -> int:
        return int(self._implementation.world_size)

    @property
    def device_index(self) -> int:
        return int(self._implementation.device_index)

    @property
    def is_closed(self) -> bool:
        return bool(self._implementation.is_closed)

    def all_gather(
        self,
        local_latent: Tensor,
        plan: StaticRaggedPlan,
        *,
        algorithm: str = "direct",
        registered: bool = False,
    ) -> Tensor:
        """Gather exact-width sources.

        A registered call reuses a communicator-owned output arena.  Its return
        value remains valid until another registered call with the same plan,
        batch, dtype, and CUDA stream is enqueued.
        """

        self._validate_plan(plan)
        self._validate_algorithm(algorithm)
        workspace = (
            self._registered_workspace(local_latent, plan)
            if registered
            else None
        )
        return self._implementation.all_gather(
            local_latent,
            list(plan.source_widths),
            algorithm,
            workspace,
        )

    def all_gather_rank_major(
        self,
        local_latent: Tensor,
        plan: StaticRaggedPlan,
        *,
        algorithm: str = "direct",
        registered: bool = False,
    ) -> Tensor:
        """Gather exact-width sources without the token-major layout pack.

        The returned flat buffer concatenates each source's contiguous
        ``[batch, source_width]`` block in process-rank order.  This boundary
        is useful for communication profiling and future source-wise decode.
        """

        self._validate_plan(plan)
        self._validate_algorithm(algorithm)
        workspace = (
            self._registered_workspace(local_latent, plan)
            if registered
            else None
        )
        return self._implementation.all_gather_rank_major(
            local_latent,
            list(plan.source_widths),
            algorithm,
            workspace,
        )

    def all_gather_decode(
        self,
        local_latent: Tensor,
        decoder: Tensor,
        plan: StaticRaggedPlan,
        bias: Tensor | None = None,
        *,
        algorithm: str = "direct",
        registered: bool = False,
    ) -> Tensor:
        self._validate_plan(plan)
        self._validate_algorithm(algorithm)
        workspace = (
            self._registered_workspace(local_latent, plan)
            if registered
            else None
        )
        return self._implementation.all_gather_decode(
            local_latent,
            decoder,
            list(plan.source_widths),
            bias,
            algorithm,
            workspace,
        )

    def close(self) -> None:
        if not self.is_closed:
            with torch.cuda.device(self.device_index):
                torch.cuda.synchronize(self.device_index)
            self._implementation.close()
            self._registered_workspaces.clear()

    def _validate_plan(self, plan: StaticRaggedPlan) -> None:
        if len(plan.source_widths) != self.world_size:
            raise ValueError(
                f"plan has {len(plan.source_widths)} sources for TP{self.world_size}"
            )

    @staticmethod
    def _validate_algorithm(algorithm: str) -> None:
        if algorithm not in _GATHER_ALGORITHMS:
            raise ValueError(
                f"unknown ragged algorithm {algorithm!r}; "
                f"expected one of {_GATHER_ALGORITHMS}"
            )

    def _registered_workspace(
        self,
        local_latent: Tensor,
        plan: StaticRaggedPlan,
    ) -> Tensor:
        if local_latent.ndim != 2:
            raise ValueError("registered ragged latent must be a matrix")
        if local_latent.device != torch.device("cuda", self.device_index):
            raise ValueError("registered ragged latent is on the wrong device")
        stream = int(torch.cuda.current_stream(self.device_index).cuda_stream)
        key = (
            local_latent.dtype,
            int(local_latent.shape[0]),
            plan.source_widths,
            stream,
        )
        workspace = self._registered_workspaces.get(key)
        if workspace is None:
            workspace = self._implementation.create_registered_workspace(
                local_latent,
                int(local_latent.shape[0]) * plan.total_width,
            )
            self._registered_workspaces[key] = workspace
        return workspace


class StaticRaggedPrivateAllGatherOutput(nn.Module):
    """Inference-only local encoder plus compiled ragged AllGather/decode."""

    def __init__(
        self,
        local_input_factor: Tensor,
        decoder: Tensor,
        *,
        plan: StaticRaggedPlan,
        communicator: RaggedNcclCommunicator,
        bias: Tensor | None = None,
        algorithm: str = "direct",
        registered: bool = False,
    ) -> None:
        super().__init__()
        if local_input_factor.ndim != 2:
            raise ValueError("local_input_factor must be [local_input, local_wire]")
        if decoder.ndim != 2:
            raise ValueError("decoder must be [total_wire, hidden]")
        if len(plan.source_widths) != communicator.world_size:
            raise ValueError("ragged plan and communicator world sizes differ")
        local_width = plan.source_widths[communicator.rank]
        if int(local_input_factor.shape[1]) != local_width:
            raise ValueError(
                "local input factor width differs from the rank-local wire: "
                f"{local_input_factor.shape[1]} != {local_width}"
            )
        if int(decoder.shape[0]) != plan.total_width:
            raise ValueError(
                f"decoder width {decoder.shape[0]} != plan width {plan.total_width}"
            )
        if local_input_factor.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"unsupported ragged factor dtype {local_input_factor.dtype}")
        if decoder.dtype != local_input_factor.dtype:
            raise TypeError("local encoder and decoder dtypes must match")
        if decoder.device != local_input_factor.device:
            raise ValueError("local encoder and decoder devices must match")
        if decoder.device != torch.device("cuda", communicator.device_index):
            raise ValueError("ragged factors are not on the communicator device")
        if bias is not None:
            if bias.shape != (decoder.shape[1],):
                raise ValueError("ragged decoder bias has the wrong shape")
            if bias.dtype != decoder.dtype or bias.device != decoder.device:
                raise ValueError("ragged decoder bias must match decoder dtype/device")
        communicator._validate_algorithm(algorithm)

        self.plan = plan
        self.communicator = communicator
        self.algorithm = algorithm
        self.registered = bool(registered)
        self.local_projection_weight = nn.Parameter(
            local_input_factor.detach().T.contiguous(),
            requires_grad=False,
        )
        self.decoder = nn.Parameter(decoder.detach().contiguous(), requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().contiguous(), requires_grad=False)

    @property
    def local_in_features(self) -> int:
        return int(self.local_projection_weight.shape[1])

    @property
    def local_wire_width(self) -> int:
        return int(self.local_projection_weight.shape[0])

    @property
    def out_features(self) -> int:
        return int(self.decoder.shape[1])

    def forward(self, local_hidden_states: Tensor) -> Tensor:
        if torch.is_grad_enabled() and local_hidden_states.requires_grad:
            raise RuntimeError("ragged private AllGather is inference-only")
        if int(local_hidden_states.shape[-1]) != self.local_in_features:
            raise ValueError(
                f"expected local input width {self.local_in_features}, "
                f"got {local_hidden_states.shape[-1]}"
            )
        leading_shape = tuple(local_hidden_states.shape[:-1])
        latent = F.linear(local_hidden_states, self.local_projection_weight)
        flat_latent = latent.reshape(-1, self.local_wire_width).contiguous()
        output = self.communicator.all_gather_decode(
            flat_latent,
            self.decoder,
            self.plan,
            self.bias,
            algorithm=self.algorithm,
            registered=self.registered,
        )
        return output.reshape(*leading_shape, self.out_features)


def pack_rank_major_for_test(
    rank_major: Tensor,
    plan: StaticRaggedPlan,
    *,
    batch: int,
) -> Tensor:
    """Expose only the CUDA layout kernel for focused correctness tests."""

    extension = load_ragged_allgather_extension()
    return extension.pack_rank_major(
        rank_major,
        list(plan.source_widths),
        int(batch),
    )


__all__ = [
    "RaggedNcclCommunicator",
    "StaticRaggedPlan",
    "StaticRaggedPrivateAllGatherOutput",
    "load_ragged_allgather_extension",
    "pack_rank_major_for_test",
]
