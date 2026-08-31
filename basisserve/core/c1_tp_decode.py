"""C1 factor loading and the post-attention TP decode boundary.

The offline ragged C1 checkpoint stores one Value encoder per physical KV
head and one decoder per query head, both padded to the largest source rank in
that layer.  This module turns that artifact into the exact tensors consumed
by a TP run where process rank ``s`` owns KV head ``s`` and its contiguous
query-head group.

The runtime boundary deliberately starts after dense attention.  Its local
input is ``[..., local_query_heads, head_dim]``; compact Value-cache attention
can later produce the same coordinates directly without changing the ragged
collective or decoder layout defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch import Tensor, nn

from basisserve.kernels.ragged_allgather import (
    RaggedNcclCommunicator,
    StaticRaggedPlan,
)


RAGGED_C1_FACTOR_FORMAT = "basisserve.qwen3_32b.gqa_c1.ragged_schedule_als.v1"
_FACTOR_TENSORS = {
    "value_coordinate_encoders",
    "head_output_decoders",
    "source_ranks",
}


def file_sha256(path: Path) -> str:
    """Hash a file without materializing it in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class C1TPGeometry:
    """Model geometry for one-KV-group-per-process GQA tensor parallelism."""

    hidden_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    tp_size: int

    def __post_init__(self) -> None:
        values = (
            self.hidden_size,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
            self.num_layers,
            self.tp_size,
        )
        if any(value <= 0 for value in values):
            raise ValueError(f"C1 TP geometry must be positive, got {values}")
        if self.num_kv_heads != self.tp_size:
            raise ValueError(
                "C1 TP ownership requires exactly one physical KV head per rank: "
                f"num_kv_heads={self.num_kv_heads}, tp_size={self.tp_size}"
            )
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("query heads must divide evenly across physical KV heads")

    @property
    def query_heads_per_rank(self) -> int:
        return self.num_query_heads // self.tp_size

    @classmethod
    def from_model_config(
        cls,
        path: Path,
        *,
        tp_size: int,
    ) -> "C1TPGeometry":
        config = json.loads(path.read_text(encoding="utf-8"))
        required = (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_hidden_layers",
        )
        missing = tuple(name for name in required if name not in config)
        if missing:
            raise ValueError(f"model config is missing C1 geometry fields: {missing}")
        return cls(
            hidden_size=int(config["hidden_size"]),
            num_query_heads=int(config["num_attention_heads"]),
            num_kv_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["head_dim"]),
            num_layers=int(config["num_hidden_layers"]),
            tp_size=int(tp_size),
        )

    def ownership(self, process_rank: int) -> "C1TPHeadOwnership":
        rank = int(process_rank)
        if not 0 <= rank < self.tp_size:
            raise ValueError(f"process rank {rank} is outside TP{self.tp_size}")
        start = rank * self.query_heads_per_rank
        return C1TPHeadOwnership(
            process_rank=rank,
            kv_head=rank,
            query_head_start=start,
            query_head_stop=start + self.query_heads_per_rank,
        )


@dataclass(frozen=True)
class C1TPHeadOwnership:
    """The physical KV group and query heads owned by one process rank."""

    process_rank: int
    kv_head: int
    query_head_start: int
    query_head_stop: int

    @property
    def query_head_count(self) -> int:
        return self.query_head_stop - self.query_head_start

    @property
    def query_head_slice(self) -> slice:
        return slice(self.query_head_start, self.query_head_stop)


def headwise_c1_encode(local_attention: Tensor, encoder: Tensor) -> Tensor:
    """Apply one shared physical-head encoder as a single dense GEMM.

    ``local_attention`` ends in ``[local_query_heads, head_dim]`` and
    ``encoder`` is ``[head_dim, source_rank]``.  All leading dimensions and
    query heads are folded into the GEMM row dimension, then the per-head
    coordinates are flattened in head-major order.  No block-diagonal tensor
    is constructed.
    """

    if local_attention.ndim < 2:
        raise ValueError("local attention must have query-head and head dimensions")
    if encoder.ndim != 2:
        raise ValueError("C1 encoder must be [head_dim, source_rank]")
    if int(local_attention.shape[-1]) != int(encoder.shape[0]):
        raise ValueError(
            "local attention head dimension differs from the C1 encoder: "
            f"{local_attention.shape[-1]} != {encoder.shape[0]}"
        )
    if (
        local_attention.device != encoder.device
        or local_attention.dtype != encoder.dtype
    ):
        raise ValueError("local attention must match the C1 encoder dtype/device")
    leading_shape = tuple(local_attention.shape[:-2])
    query_heads = int(local_attention.shape[-2])
    head_dim = int(encoder.shape[0])
    source_rank = int(encoder.shape[1])
    flat_heads = local_attention.reshape(-1, head_dim)
    flat_coordinates = torch.mm(flat_heads, encoder)
    return flat_coordinates.reshape(*leading_shape, query_heads * source_rank)


@dataclass(frozen=True)
class PackedC1TPLayer:
    """One layer's compact TP factors on their requested runtime device."""

    layer_index: int
    process_rank: int
    ownership: C1TPHeadOwnership
    source_ranks: tuple[int, ...]
    plan: StaticRaggedPlan
    local_encoder: Tensor
    local_decoder: Tensor
    global_decoder: Tensor
    manifest_sha256: str
    artifact_sha256: str

    @property
    def local_rank(self) -> int:
        return self.source_ranks[self.process_rank]

    @property
    def local_wire_width(self) -> int:
        return self.plan.source_widths[self.process_rank]

    @property
    def hidden_size(self) -> int:
        return int(self.global_decoder.shape[1])

    def encode_local_attention(self, local_attention: Tensor) -> Tensor:
        """Encode each owned query head with the shared physical-head ``A_s``."""

        expected_tail = (self.ownership.query_head_count, self.local_encoder.shape[0])
        if local_attention.ndim < 2 or tuple(local_attention.shape[-2:]) != expected_tail:
            raise ValueError(
                "local attention must end in [owned_query_heads, head_dim]: "
                f"expected {expected_tail}, got {tuple(local_attention.shape)}"
            )
        return headwise_c1_encode(local_attention, self.local_encoder)

    def decode_local_attention(self, local_attention: Tensor) -> Tensor:
        """Reference local C1 contribution before the output AllReduce."""

        return self.encode_local_attention(local_attention) @ self.local_decoder


class C1TPFactorLoader:
    """Validate a ragged C1 checkpoint and pack layers for static TP ownership."""

    def __init__(
        self,
        factor_dir: str | Path,
        *,
        model_config: str | Path,
        tp_size: int = 8,
        expected_result_sha256: str | None = None,
    ) -> None:
        self.factor_dir = Path(factor_dir).expanduser().resolve()
        result_path = self.factor_dir / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        self.result_path = result_path
        self.result_sha256 = file_sha256(result_path)
        if (
            expected_result_sha256 is not None
            and self.result_sha256 != expected_result_sha256
        ):
            raise ValueError(
                "ragged C1 result hash mismatch: "
                f"expected {expected_result_sha256}, observed {self.result_sha256}"
            )
        model_config_path = Path(model_config).expanduser().resolve()
        if model_config_path.is_dir():
            model_config_path = model_config_path / "config.json"
        if not model_config_path.is_file():
            raise FileNotFoundError(model_config_path)
        self.model_config_path = model_config_path
        self.geometry = C1TPGeometry.from_model_config(
            model_config_path,
            tp_size=tp_size,
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("ragged C1 result must be a JSON object")
        self.result = result
        self._validate_result()

    @property
    def schedule(self) -> tuple[tuple[int, ...], ...]:
        selected = self.result["selection"]["selected_schedule"]
        return tuple(tuple(map(int, layer)) for layer in selected)

    @property
    def schedule_sha256(self) -> str:
        return _canonical_sha256(self.schedule)

    @property
    def distributed_identity_sha256(self) -> str:
        geometry = (
            self.geometry.hidden_size,
            self.geometry.num_query_heads,
            self.geometry.num_kv_heads,
            self.geometry.head_dim,
            self.geometry.num_layers,
            self.geometry.tp_size,
        )
        return _canonical_sha256(
            {
                "result_sha256": self.result_sha256,
                "schedule_sha256": self.schedule_sha256,
                "geometry": geometry,
            }
        )

    def _validate_result(self) -> None:
        result = self.result
        if (
            result.get("format") != RAGGED_C1_FACTOR_FORMAT
            or result.get("status") != "complete"
        ):
            raise ValueError("ragged C1 factor result is incomplete or incompatible")
        fit_config = result.get("fit_config")
        selection = result.get("selection")
        artifacts = result.get("artifacts")
        if not isinstance(fit_config, dict) or not isinstance(selection, dict):
            raise ValueError("ragged C1 result is missing fit_config or selection")
        if not isinstance(artifacts, dict):
            raise ValueError("ragged C1 result is missing layer artifacts")
        observed_model_hash = file_sha256(self.model_config_path)
        if fit_config.get("model_config_sha256") != observed_model_hash:
            raise ValueError("ragged C1 factors belong to another model config")
        expected_layers = tuple(range(self.geometry.num_layers))
        if tuple(map(int, result.get("layers", ()))) != expected_layers:
            raise ValueError("ragged C1 result does not cover every model layer")
        try:
            artifact_layers = {int(layer) for layer in artifacts}
        except (TypeError, ValueError) as error:
            raise ValueError("ragged C1 artifact keys must be integer layer indices") from error
        if artifact_layers != set(expected_layers):
            raise ValueError("ragged C1 result does not contain every layer artifact")
        schedule = selection.get("selected_schedule")
        if not isinstance(schedule, list) or len(schedule) != self.geometry.num_layers:
            raise ValueError("ragged C1 result has an incompatible rank schedule")
        if any(
            not isinstance(layer, list) or len(layer) != self.geometry.tp_size
            for layer in schedule
        ):
            raise ValueError(
                f"ragged C1 rank schedule does not define TP{self.geometry.tp_size}"
            )
        try:
            flat_ranks = [int(rank) for layer in schedule for rank in layer]
        except (TypeError, ValueError) as error:
            raise ValueError("ragged C1 schedule contains a non-integer rank") from error
        if any(not 0 < rank <= self.geometry.head_dim for rank in flat_ranks):
            raise ValueError("ragged C1 schedule contains an invalid source rank")
        source_rank_sum = sum(flat_ranks)
        if source_rank_sum != int(selection.get("source_rank_sum", -1)):
            raise ValueError("ragged C1 schedule violates its recorded rank budget")
        if fit_config.get("rank_schedule") != schedule:
            raise ValueError("ragged C1 fit config and selected schedule disagree")
        if int(fit_config.get("source_rank_sum", -1)) != source_rank_sum:
            raise ValueError("ragged C1 fit config records another source rank budget")
        for process_rank in range(self.geometry.tp_size):
            ownership = self.geometry.ownership(process_rank)
            if ownership.kv_head != process_rank:
                raise AssertionError("TP ownership is not process-rank ordered")
            if ownership.query_head_count != self.geometry.query_heads_per_rank:
                raise AssertionError("TP query-head ownership is not uniform")

    def _artifact_path(self, artifact: Mapping[str, Any], layer_index: int) -> Path:
        file_name = artifact.get("file")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"layer {layer_index} artifact has no file name")
        relative = Path(file_name)
        if relative.is_absolute():
            raise ValueError(f"layer {layer_index} artifact path must be relative")
        path = (self.factor_dir / relative).resolve()
        try:
            path.relative_to(self.factor_dir)
        except ValueError as error:
            raise ValueError(
                f"layer {layer_index} artifact escapes the factor directory"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _validate_tensor_metadata(
        self,
        artifact: Mapping[str, Any],
        payload: Mapping[str, Tensor],
        layer_index: int,
    ) -> None:
        metadata = artifact.get("tensors")
        if not isinstance(metadata, dict) or set(metadata) != _FACTOR_TENSORS:
            raise ValueError(f"layer {layer_index} has incomplete tensor metadata")
        for name, tensor in payload.items():
            expected = metadata[name]
            if not isinstance(expected, dict):
                raise ValueError(f"layer {layer_index} has invalid metadata for {name}")
            if expected.get("shape") != list(tensor.shape):
                raise ValueError(f"layer {layer_index} metadata shape mismatch for {name}")
            if expected.get("dtype") != str(tensor.dtype):
                raise ValueError(f"layer {layer_index} metadata dtype mismatch for {name}")

    def _load_layer_cpu(
        self,
        layer_index: int,
    ) -> tuple[tuple[int, ...], StaticRaggedPlan, Tensor, Tensor, str]:
        layer = int(layer_index)
        if not 0 <= layer < self.geometry.num_layers:
            raise ValueError(f"layer {layer} is outside the model")
        artifact = self.result["artifacts"][str(layer)]
        if not isinstance(artifact, dict):
            raise ValueError(f"layer {layer} artifact record must be an object")
        path = self._artifact_path(artifact, layer)
        observed_artifact_sha256 = file_sha256(path)
        if artifact.get("sha256") != observed_artifact_sha256:
            raise ValueError(f"ragged C1 artifact hash mismatch at layer {layer}")
        payload = load_file(str(path), device="cpu")
        if set(payload) != _FACTOR_TENSORS:
            raise ValueError(f"unexpected ragged C1 tensors at layer {layer}")
        self._validate_tensor_metadata(artifact, payload, layer)

        ranks_tensor = payload["source_ranks"]
        if ranks_tensor.dtype != torch.int32 or tuple(ranks_tensor.shape) != (
            self.geometry.tp_size,
        ):
            raise ValueError(f"unexpected source-rank tensor at layer {layer}")
        source_ranks = tuple(map(int, ranks_tensor.tolist()))
        if source_ranks != self.schedule[layer]:
            raise ValueError(f"ragged C1 rank mismatch at layer {layer}")
        maximum_rank = max(source_ranks)
        encoders = payload["value_coordinate_encoders"]
        decoders = payload["head_output_decoders"]
        expected_encoder_shape = (
            self.geometry.num_kv_heads,
            self.geometry.head_dim,
            maximum_rank,
        )
        expected_decoder_shape = (
            self.geometry.num_query_heads,
            maximum_rank,
            self.geometry.hidden_size,
        )
        if tuple(encoders.shape) != expected_encoder_shape:
            raise ValueError(f"unexpected ragged encoder shape at layer {layer}")
        if tuple(decoders.shape) != expected_decoder_shape:
            raise ValueError(f"unexpected ragged decoder shape at layer {layer}")
        if encoders.dtype != decoders.dtype or not encoders.is_floating_point():
            raise TypeError(f"ragged C1 factors have incompatible dtypes at layer {layer}")
        if artifact.get("encoder_sha256") != _tensor_sha256(encoders):
            raise ValueError(f"ragged C1 encoder hash mismatch at layer {layer}")
        if artifact.get("decoder_sha256") != _tensor_sha256(decoders):
            raise ValueError(f"ragged C1 decoder hash mismatch at layer {layer}")

        plan = StaticRaggedPlan.from_head_ranks(
            source_ranks,
            heads_per_source=self.geometry.query_heads_per_rank,
        )
        decoder_blocks: list[Tensor] = []
        for source, source_rank in enumerate(source_ranks):
            source_ownership = self.geometry.ownership(source)
            block = decoders[
                source_ownership.query_head_slice,
                :source_rank,
                :,
            ].reshape(
                source_ownership.query_head_count * source_rank,
                self.geometry.hidden_size,
            )
            decoder_blocks.append(block)
        compact_global_decoder = torch.cat(decoder_blocks, dim=0).contiguous()
        return (
            source_ranks,
            plan,
            encoders,
            compact_global_decoder,
            observed_artifact_sha256,
        )

    def _pack_process_ranks(
        self,
        *,
        layer_index: int,
        process_ranks: Sequence[int],
        device: torch.device | str,
        dtype: torch.dtype | None,
    ) -> tuple[PackedC1TPLayer, ...]:
        (
            source_ranks,
            plan,
            encoders,
            compact_global_decoder,
            artifact_sha256,
        ) = self._load_layer_cpu(layer_index)
        ownerships = tuple(
            self.geometry.ownership(process_rank) for process_rank in process_ranks
        )

        target_device = torch.device(device)
        target_dtype = encoders.dtype if dtype is None else dtype
        if not torch.empty((), dtype=target_dtype).is_floating_point():
            raise TypeError(f"C1 runtime dtype must be floating point, got {target_dtype}")
        compact_global_decoder = compact_global_decoder.to(
            device=target_device,
            dtype=target_dtype,
        ).contiguous()
        packed: list[PackedC1TPLayer] = []
        for ownership in ownerships:
            local_encoder = encoders[
                ownership.kv_head,
                :,
                : source_ranks[ownership.kv_head],
            ].contiguous().to(
                device=target_device,
                dtype=target_dtype,
            )
            local_start = plan.offsets[ownership.process_rank]
            local_stop = local_start + plan.source_widths[ownership.process_rank]
            local_decoder = compact_global_decoder[local_start:local_stop]
            packed.append(
                PackedC1TPLayer(
                    layer_index=int(layer_index),
                    process_rank=ownership.process_rank,
                    ownership=ownership,
                    source_ranks=source_ranks,
                    plan=plan,
                    local_encoder=local_encoder,
                    local_decoder=local_decoder,
                    global_decoder=compact_global_decoder,
                    manifest_sha256=self.result_sha256,
                    artifact_sha256=artifact_sha256,
                )
            )
        return tuple(packed)

    def load_layer(
        self,
        layer_index: int,
        *,
        process_rank: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> PackedC1TPLayer:
        """Load and pack one layer in process-rank/source order."""

        return self._pack_process_ranks(
            layer_index=layer_index,
            process_ranks=(process_rank,),
            device=device,
            dtype=dtype,
        )[0]

    def load_virtual_tp_layer(
        self,
        layer_index: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> tuple[PackedC1TPLayer, ...]:
        """Load all virtual ranks once for a single-process TP simulation."""

        return self._pack_process_ranks(
            layer_index=layer_index,
            process_ranks=tuple(range(self.geometry.tp_size)),
            device=device,
            dtype=dtype,
        )


def assert_distributed_loader_consensus(
    loader: C1TPFactorLoader,
    process_group: dist.ProcessGroup | None = None,
) -> None:
    """Fail before custom NCCL setup if ranks loaded different manifests/plans."""

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    world_size = dist.get_world_size(process_group)
    if world_size != loader.geometry.tp_size:
        raise ValueError(
            f"distributed world is TP{world_size}, checkpoint expects TP{loader.geometry.tp_size}"
        )
    backend = str(dist.get_backend(process_group)).lower()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend == "nccl"
        else torch.device("cpu")
    )
    local_identity = torch.tensor(
        list(bytes.fromhex(loader.distributed_identity_sha256)),
        dtype=torch.uint8,
        device=device,
    )
    gathered = torch.empty(
        world_size * local_identity.numel(),
        dtype=torch.uint8,
        device=device,
    )
    dist.all_gather_into_tensor(gathered, local_identity, group=process_group)
    identities = tuple(
        bytes(row).hex()
        for row in gathered.cpu().reshape(world_size, -1).tolist()
    )
    if any(identity != identities[0] for identity in identities):
        raise ValueError(f"C1 manifest or rank schedule differs across ranks: {identities}")


class C1RaggedOutputTPDecode(nn.Module):
    """Exact ragged AllGather/decode for rank-local C1 coordinates.

    Compact-V attention produces this module's input directly, with final
    width ``owned_query_heads * r_s``.  No post-attention encoder GEMM is
    present on this serving path.
    """

    def __init__(
        self,
        packed: PackedC1TPLayer,
        *,
        communicator: RaggedNcclCommunicator,
        algorithm: str = "direct",
        registered: bool = False,
    ) -> None:
        super().__init__()
        if communicator.rank != packed.process_rank:
            raise ValueError("packed factors belong to another process rank")
        if communicator.world_size != len(packed.plan.source_widths):
            raise ValueError("packed plan and communicator world sizes differ")
        communicator._validate_algorithm(algorithm)
        expected_device = torch.device("cuda", communicator.device_index)
        if packed.local_encoder.device != expected_device:
            raise ValueError("packed factors are not on the communicator device")
        if packed.global_decoder.device != expected_device:
            raise ValueError("packed decoder is not on the communicator device")
        if packed.local_encoder.dtype != packed.global_decoder.dtype:
            raise TypeError("packed encoder and decoder dtypes differ")
        self.layer_index = packed.layer_index
        self.process_rank = packed.process_rank
        self.ownership = packed.ownership
        self.plan = packed.plan
        self.communicator = communicator
        self.algorithm = algorithm
        self.registered = bool(registered)
        self.register_buffer("global_decoder", packed.global_decoder)

    @property
    def local_wire_width(self) -> int:
        return self.plan.source_widths[self.process_rank]

    @property
    def out_features(self) -> int:
        return int(self.global_decoder.shape[1])

    def forward(self, local_coordinates: Tensor) -> Tensor:
        if torch.is_grad_enabled() and local_coordinates.requires_grad:
            raise RuntimeError("C1 TP decode is inference-only")
        if local_coordinates.ndim < 1 or int(local_coordinates.shape[-1]) != self.local_wire_width:
            raise ValueError(
                "local C1 coordinates must end in this rank's ragged wire width: "
                f"expected {self.local_wire_width}, got {tuple(local_coordinates.shape)}"
            )
        if (
            local_coordinates.device != self.global_decoder.device
            or local_coordinates.dtype != self.global_decoder.dtype
        ):
            raise ValueError("local C1 coordinates must match decoder dtype/device")
        leading_shape = tuple(local_coordinates.shape[:-1])
        flat_latent = local_coordinates.reshape(-1, self.local_wire_width)
        output = self.communicator.all_gather_decode(
            flat_latent,
            self.global_decoder,
            self.plan,
            algorithm=self.algorithm,
            registered=self.registered,
        )
        return output.reshape(*leading_shape, self.out_features)


class C1PostAttentionTPDecode(nn.Module):
    """Dense-attention oracle: headwise encode, then ragged output decode."""

    def __init__(
        self,
        packed: PackedC1TPLayer,
        *,
        communicator: RaggedNcclCommunicator,
        algorithm: str = "direct",
        registered: bool = False,
    ) -> None:
        super().__init__()
        self.layer_index = packed.layer_index
        self.process_rank = packed.process_rank
        self.ownership = packed.ownership
        self.register_buffer("local_encoder", packed.local_encoder)
        self.output_decode = C1RaggedOutputTPDecode(
            packed,
            communicator=communicator,
            algorithm=algorithm,
            registered=registered,
        )

    @property
    def local_wire_width(self) -> int:
        return self.output_decode.local_wire_width

    @property
    def out_features(self) -> int:
        return self.output_decode.out_features

    def forward(self, local_attention: Tensor) -> Tensor:
        if torch.is_grad_enabled() and local_attention.requires_grad:
            raise RuntimeError("C1 TP decode is inference-only")
        expected_tail = (self.ownership.query_head_count, self.local_encoder.shape[0])
        if local_attention.ndim < 2 or tuple(local_attention.shape[-2:]) != expected_tail:
            raise ValueError(
                "local attention must end in [owned_query_heads, head_dim]: "
                f"expected {expected_tail}, got {tuple(local_attention.shape)}"
            )
        latent = headwise_c1_encode(local_attention, self.local_encoder)
        return self.output_decode(latent)


__all__ = [
    "C1PostAttentionTPDecode",
    "C1RaggedOutputTPDecode",
    "C1TPFactorLoader",
    "C1TPGeometry",
    "C1TPHeadOwnership",
    "PackedC1TPLayer",
    "RAGGED_C1_FACTOR_FORMAT",
    "assert_distributed_loader_consensus",
    "file_sha256",
    "headwise_c1_encode",
]
