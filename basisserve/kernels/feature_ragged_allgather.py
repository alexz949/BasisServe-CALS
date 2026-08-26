"""Feature-major ragged transport for C1 / source-private TP latents.

The receive arena is laid out as ``[sum(source_widths), tokens]``.  Source ``s``
occupies the contiguous row interval ``offset[s]:offset[s+1]`` and sends
``local.T``.  Consequently the arena already equals ``X.T`` for
``X = cat([X_0, ..., X_{P-1}], dim=-1)``; no global rank-major -> token-major
packing kernel is needed.  Decoding is one GEMM::

    Y = arena.T @ decoder

Two transports share the same layout:

* ``feature_direct``: in-place NCCL AllGather for uniform widths and an
  exact-width NCCL ring for ragged widths.
* ``feature_rma``: NCCL 2.29+ ``PutSignal`` / ``WaitSignal`` one-sided push.

The RMA path is deliberately strict: the workspace is prepared outside the hot
path, has a fixed token count/dtype, and is bound to one CUDA stream.  These
constraints make signal reuse and arena lifetime explicit rather than hiding an
unsafe cache behind the Python API.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.distributed as dist

from basisserve.kernels.ragged_allgather import (
    StaticRaggedPlan,
    _match_loaded_cxx_runtime,
    _nccl_paths,
)


_EXTENSION_NAME = "basisserve_feature_ragged_allgather_v8"
_DTYPE_TO_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}
_DIRECT_DTYPES = frozenset((*_DTYPE_TO_CODE, torch.uint8))


def _cuda_build_paths() -> tuple[list[Path], Path, str, Path]:
    """Validate the complete CUDA toolkit selected by the cluster module."""

    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not configured:
        raise FileNotFoundError(
            "CUDA_HOME is unset; load nvidia/cuda12/cuda/12.4.1 before running"
        )
    toolkit = Path(configured).expanduser().resolve()
    nvcc = toolkit / "bin" / "nvcc"
    if not nvcc.is_file():
        raise FileNotFoundError(f"CUDA compiler is unavailable: {nvcc}")
    includes = [toolkit / "include"]
    for header in ("cuda_runtime_api.h", "crt/host_defines.h", "nv/target"):
        if not any((path / header).exists() for path in includes):
            raise FileNotFoundError(f"CUDA build header {header!r} is unavailable")
    runtime = toolkit / "lib64"
    if not runtime.is_dir():
        runtime = toolkit / "lib"
    unversioned = runtime / "libcudart.so"
    if unversioned.is_file():
        runtime_name = unversioned.name
    else:
        versioned = sorted(runtime.glob("libcudart.so.*"), reverse=True)
        if not versioned:
            raise FileNotFoundError("CUDA runtime library is unavailable")
        runtime_name = versioned[0].name
    return includes, runtime, runtime_name, toolkit


@lru_cache(maxsize=1)
def _load_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("feature ragged collectives require CUDA")

    _match_loaded_cxx_runtime()
    source_root = Path(__file__).resolve().parent / "csrc"
    sources = (
        source_root / "feature_ragged_allgather.cpp",
        source_root / "feature_ragged_allgather_pack.cu",
    )
    if any(not source.exists() for source in sources):
        raise FileNotFoundError("feature-major C++/CUDA extension sources are incomplete")

    nccl_include, nccl_library = _nccl_paths()
    cuda_includes, cuda_library, cuda_runtime_name, cuda_home = _cuda_build_paths()
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.9+PTX")
    from torch.utils import cpp_extension

    cpp_extension.CUDA_HOME = str(cuda_home)
    include_paths = [str(nccl_include), *(str(path) for path in cuda_includes)]
    verbose = os.environ.get("BASISSERVE_VERBOSE_BUILD", "0") == "1"
    build_directory = os.environ.get("BASISSERVE_EXT_BUILD_DIR")

    extra_ldflags = [
        f"-L{nccl_library}",
        "-l:libnccl.so.2",
        f"-Wl,-rpath,{nccl_library}",
        f"-L{cuda_library}",
        f"-l:{cuda_runtime_name}",
        f"-Wl,-rpath,{cuda_library}",
        "-lc10_cuda",
        "-ltorch_cuda",
    ]

    return cpp_extension.load(
        name=_EXTENSION_NAME,
        sources=[str(source) for source in sources],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--threads", "4"],
        extra_include_paths=include_paths,
        extra_ldflags=extra_ldflags,
        with_cuda=True,
        build_directory=build_directory,
        verbose=verbose,
    )


def decode_feature_major(
    arena: torch.Tensor,
    decoder: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode a feature-major receive arena with exactly one GEMM.

    Args:
        arena: Contiguous ``[K, tokens]`` source-concatenated receive arena.
        decoder: Source-major decoder ``[K, hidden]`` using the same source and
            within-source coordinate ordering as ``StaticRaggedPlan.source_widths``.
        bias: Optional ``[hidden]`` bias. ``torch.addmm`` folds it into the GEMM.

    Returns:
        Token-major output ``[tokens, hidden]``.
    """

    if arena.ndim != 2 or decoder.ndim != 2:
        raise ValueError(
            f"arena and decoder must be matrices, got {arena.shape=} and {decoder.shape=}"
        )
    if arena.shape[0] != decoder.shape[0]:
        raise ValueError(
            f"decoder K={decoder.shape[0]} does not match arena K={arena.shape[0]}"
        )
    if arena.device != decoder.device or arena.dtype != decoder.dtype:
        raise ValueError("arena and decoder must share device and dtype")
    if not arena.is_contiguous() or not decoder.is_contiguous():
        raise ValueError("arena and decoder must be contiguous")
    if torch.is_grad_enabled() and (arena.requires_grad or decoder.requires_grad):
        raise RuntimeError("feature ragged decode is inference-only")

    token_major_view = arena.transpose(0, 1)
    if bias is None:
        return torch.mm(token_major_view, decoder)
    if bias.ndim != 1 or bias.shape[0] != decoder.shape[1]:
        raise ValueError(
            f"bias must have shape [{decoder.shape[1]}], got {tuple(bias.shape)}"
        )
    if bias.device != decoder.device or bias.dtype != decoder.dtype:
        raise ValueError("bias must match decoder device and dtype")
    if torch.is_grad_enabled() and bias.requires_grad:
        raise RuntimeError("feature ragged decode is inference-only")
    return torch.addmm(bias, token_major_view, decoder)


class FeatureRaggedCommunicator:
    """A TP communicator shared by multiple layer-specific ragged plans.

    Create one object per TP process group, then pass different
    ``StaticRaggedPlan`` objects on successive layer calls.  The RMA arena may be
    prepared for the largest layer K and reused by all smaller plans.
    """

    def __init__(self, impl) -> None:
        self._impl = impl
        self._closed = False
        self._direct_workspaces: dict[
            tuple[torch.dtype, int, tuple[int, ...], int], torch.Tensor
        ] = {}
        self._shared_direct_workspace: torch.Tensor | None = None
        self._shared_direct_shape: tuple[int, int, torch.dtype, int] | None = None

    @classmethod
    def from_distributed(
        cls,
        group: Optional[dist.ProcessGroup] = None,
        *,
        device: torch.device | int | None = None,
    ) -> "FeatureRaggedCommunicator":
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        if str(dist.get_backend(group)).lower() != "nccl":
            raise RuntimeError("feature ragged communicator requires an NCCL process group")

        if device is None:
            device_index = torch.cuda.current_device()
        elif isinstance(device, torch.device):
            if device.type != "cuda":
                raise ValueError(f"feature ragged communicator requires CUDA, got {device}")
            device_index = (
                torch.cuda.current_device() if device.index is None else device.index
            )
        else:
            device_index = int(device)
        resolved_device = torch.device("cuda", device_index)
        torch.cuda.set_device(resolved_device)

        extension = _load_extension()
        group_rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        if group is None or group is dist.group.WORLD:
            src_global_rank = 0
        elif hasattr(dist, "get_global_rank"):
            src_global_rank = dist.get_global_rank(group, 0)
        else:
            raise RuntimeError(
                "subgroups require torch.distributed.get_global_rank; use the world group"
            )

        unique_id_size = int(extension.get_unique_id_size())
        if group_rank == 0:
            unique_id = extension.get_unique_id()
            unique_id_cpu = torch.tensor(list(unique_id), dtype=torch.uint8)
            if unique_id_cpu.numel() != unique_id_size:
                raise RuntimeError("NCCL returned an unexpected unique-id size")
        else:
            unique_id_cpu = torch.empty(unique_id_size, dtype=torch.uint8)
        unique_id_cuda = unique_id_cpu.to(device=resolved_device)
        dist.broadcast(unique_id_cuda, src=src_global_rank, group=group)
        unique_id = bytes(unique_id_cuda.cpu().tolist())
        impl = extension.FeatureRaggedCommunicator(
            unique_id,
            group_rank,
            world_size,
            device_index,
        )
        return cls(impl)

    @property
    def rank(self) -> int:
        return int(self._impl.rank)

    @property
    def world_size(self) -> int:
        return int(self._impl.world_size)

    @property
    def device(self) -> int:
        return int(self._impl.device)

    @property
    def nccl_version(self) -> int:
        return int(self._impl.nccl_version())

    @property
    def rma_available(self) -> bool:
        return bool(self._impl.rma_compiled() and self._impl.rma_runtime_version_ok())

    def _check_plan(self, plan: StaticRaggedPlan) -> None:
        if len(plan.source_widths) != self.world_size:
            raise ValueError(
                f"plan has {len(plan.source_widths)} sources, communicator has {self.world_size}"
            )

    def configure_direct_workspace(
        self,
        *,
        tokens: int,
        max_total_width: int,
        dtype: torch.dtype,
    ) -> None:
        """Allocate one stream-bound arena shared by every sequential layer."""

        selected_tokens = int(tokens)
        selected_width = int(max_total_width)
        if selected_tokens <= 0 or selected_width <= 0:
            raise ValueError("packed direct workspace dimensions must be positive")
        if dtype not in _DIRECT_DTYPES:
            raise TypeError(f"packed direct workspace does not support {dtype}")
        stream = int(torch.cuda.current_stream(self.device).cuda_stream)
        shape = (selected_tokens, selected_width, dtype, stream)
        if shape == self._shared_direct_shape:
            return
        self._direct_workspaces.clear()
        self._shared_direct_workspace = torch.empty(
            selected_tokens * selected_width,
            dtype=dtype,
            device=torch.device("cuda", self.device),
        )
        self._shared_direct_shape = shape

    def prepare_rma(
        self,
        *,
        tokens: int,
        max_total_width: int,
        dtype: torch.dtype,
    ) -> None:
        """Collectively create/register the one-sided symmetric arena.

        Call once on every TP rank, outside the measured hot path and on the same
        CUDA stream that will later invoke ``feature_rma``.  Reconfiguration is
        also collective.
        """

        if dtype not in _DTYPE_TO_CODE:
            raise TypeError(f"RMA supports fp16/bf16/fp32, got {dtype}")
        self._impl.prepare_rma(int(tokens), int(max_total_width), _DTYPE_TO_CODE[dtype])

    def local_feature_major_view(self, plan: StaticRaggedPlan) -> torch.Tensor:
        """Return the next ``[local_width, tokens]`` registered RMA slot.

        A fused/modified attention kernel can write its compact output directly
        here.  Passing the returned tensor to ``gather(..., local_is_feature_major=True)``
        avoids even the source-local transpose/copy. Acquire this view immediately
        before every gather; successful RMA gathers alternate between two slots.
        """

        self._check_plan(plan)
        return self._impl.rma_local_feature_view(list(plan.source_widths))

    def direct_local_feature_major_view(
        self,
        plan: StaticRaggedPlan,
        *,
        tokens: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return this rank's packed slot in the shared direct arena."""

        self._check_plan(plan)
        arena = self._direct_workspace_for_shape(
            tokens=int(tokens),
            dtype=dtype,
            plan=plan,
        )
        return arena.narrow(
            0,
            plan.offsets[self.rank],
            plan.source_widths[self.rank],
        )

    def _direct_workspace_for_shape(
        self,
        *,
        tokens: int,
        dtype: torch.dtype,
        plan: StaticRaggedPlan,
    ) -> torch.Tensor:
        selected_tokens = int(tokens)
        if selected_tokens <= 0:
            raise ValueError("packed direct token count must be positive")
        stream = int(torch.cuda.current_stream(self.device).cuda_stream)
        if self._shared_direct_workspace is not None:
            assert self._shared_direct_shape is not None
            maximum_tokens, maximum_width, configured_dtype, configured_stream = (
                self._shared_direct_shape
            )
            if dtype != configured_dtype:
                raise TypeError("packed direct workspace dtype differs from local coordinates")
            if stream != configured_stream:
                raise RuntimeError("packed direct workspace belongs to another CUDA stream")
            if selected_tokens > maximum_tokens or plan.total_width > maximum_width:
                raise RuntimeError(
                    "packed direct workspace is smaller than the current layer: "
                    f"tokens={selected_tokens}/{maximum_tokens}, "
                    f"width={plan.total_width}/{maximum_width}"
                )
            elements = selected_tokens * plan.total_width
            return self._shared_direct_workspace[:elements].view(
                plan.total_width,
                selected_tokens,
            )
        key = (dtype, selected_tokens, plan.source_widths, stream)
        workspace = self._direct_workspaces.get(key)
        if workspace is None:
            workspace = torch.empty(
                plan.total_width,
                selected_tokens,
                dtype=dtype,
                device=torch.device("cuda", self.device),
            )
            self._direct_workspaces[key] = workspace
        return workspace

    def _direct_workspace(
        self,
        local: torch.Tensor,
        plan: StaticRaggedPlan,
        *,
        local_is_feature_major: bool,
    ) -> torch.Tensor:
        if local.ndim != 2:
            raise ValueError(f"local coordinates must be a matrix, got {tuple(local.shape)}")
        tokens = int(local.shape[1] if local_is_feature_major else local.shape[0])
        return self._direct_workspace_for_shape(
            tokens=tokens,
            dtype=local.dtype,
            plan=plan,
        )

    def gather(
        self,
        local: torch.Tensor,
        plan: StaticRaggedPlan,
        *,
        backend: str = "feature_direct",
        local_is_feature_major: bool = False,
    ) -> torch.Tensor:
        """Return the exact feature-major ``[sum(widths), tokens]`` arena.

        The direct backend reuses a stream-private workspace; its returned
        contents remain valid until the next direct call with the same plan,
        token count, dtype, and CUDA stream is enqueued.
        """

        self._check_plan(plan)
        if backend == "feature_direct":
            workspace = self._direct_workspace(
                local,
                plan,
                local_is_feature_major=local_is_feature_major,
            )
            return self._impl.gather_feature_direct(
                local,
                list(plan.source_widths),
                bool(local_is_feature_major),
                workspace,
            )
        if backend == "feature_rma":
            return self._impl.gather_feature_rma(
                local, list(plan.source_widths), bool(local_is_feature_major)
            )
        raise ValueError(
            f"unknown backend {backend!r}; expected 'feature_direct' or 'feature_rma'"
        )

    def all_gather_decode(
        self,
        local: torch.Tensor,
        plan: StaticRaggedPlan,
        decoder: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        backend: str = "feature_direct",
        local_is_feature_major: bool = False,
    ) -> torch.Tensor:
        arena = self.gather(
            local,
            plan,
            backend=backend,
            local_is_feature_major=local_is_feature_major,
        )
        return decode_feature_major(arena, decoder, bias)

    def close(self) -> None:
        """Collective clean shutdown; invoke on every rank in the same order."""

        if not self._closed:
            torch.cuda.synchronize(self.device)
            self._impl.close()
            self._direct_workspaces.clear()
            self._shared_direct_workspace = None
            self._shared_direct_shape = None
            self._closed = True


def pure_torch_feature_major_reference(parts: Iterable[torch.Tensor]) -> torch.Tensor:
    """Reference arena used by CPU tests and debugging."""

    materialized = list(parts)
    if not materialized:
        raise ValueError("parts must not be empty")
    batch = materialized[0].shape[0]
    if any(part.ndim != 2 or part.shape[0] != batch for part in materialized):
        raise ValueError("all parts must have shape [tokens, source_width]")
    return torch.cat([part.transpose(0, 1).contiguous() for part in materialized], dim=0)


__all__ = [
    "FeatureRaggedCommunicator",
    "decode_feature_major",
    "pure_torch_feature_major_reference",
]
