"""Actual TP4 runtime for Qwen3-8B Wo-only collective compression.

Every arm keeps Q/K/V attention and the KV cache dense.  Only the row-parallel
``o_proj`` boundary changes:

* dense: local 1024->4096 projection followed by full-output AllReduce;
* Wo-C1: local 1024->512 encoder, packed private AllGather, joint decoder;
* Wo-LR wire: local 1024->1024 encoder, latent AllReduce, shared decoder;
* Wo-LR capacity: local 1024->2048 encoder, latent AllReduce, shared decoder;
* Wo-C1 local-AR: the C1 encoder/decoder blocks followed by hidden AllReduce.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F

from basisserve.core.qwen3_8b_tp4_decode import (
    HIDDEN_SIZE,
    QUERY_HEADS_PER_PROCESS,
    Qwen3TP4DenseDecodeAttention,
    TP_SIZE,
    file_sha256,
)
from basisserve.core.tp_output import LowRankAllReduceOutput
from basisserve.kernels.feature_ragged_allgather import (
    FeatureRaggedCommunicator,
    PreparedUniformAllGather,
    decode_feature_major,
)
from basisserve.kernels.ragged_allgather import StaticRaggedPlan


PHASE1_FORMAT = "basisserve.qwen3_8b.wo_c1_lr_ar_phase1.v2"
PHASE1_LAYER_FORMAT = "basisserve.qwen3_8b.wo_c1_lr_ar_phase1.layer.v2"
ARMS = (
    "dense",
    "wo_lr_ar_wire",
    "wo_lr_ar_capacity",
    "wo_c1_ag",
    "wo_c1_local_ar",
)
LOCAL_INPUT_WIDTH = HIDDEN_SIZE // TP_SIZE
C1_LOCAL_RANK = 512
LR_SHARED_RANK = 1024
LR_CAPACITY_RANK = TP_SIZE * C1_LOCAL_RANK
EXPECTED_FACTOR_KEYS = {
    "c1_source_encoders",
    "c1_source_decoders",
    "lr_wire_input_factor",
    "lr_wire_shared_decoder",
    "lr_wire_singular_values",
    "lr_capacity_input_factor",
    "lr_capacity_shared_decoder",
    "lr_capacity_singular_values",
}


@dataclass(frozen=True)
class Qwen3TP4WOLayerFactors:
    """Process-local factors and replicated decoder for one Wo-only arm."""

    layer_index: int
    arm: str
    local_input_factor: Tensor
    global_decoder: Tensor
    artifact_path: Path
    artifact_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in ARMS[1:]:
            raise ValueError(f"unsupported Wo factor arm {self.arm}")
        if self.arm in ("wo_c1_ag", "wo_c1_local_ar"):
            expected_rank = C1_LOCAL_RANK
            expected_decoder_rank = TP_SIZE * C1_LOCAL_RANK
        elif self.arm == "wo_lr_ar_wire":
            expected_rank = LR_SHARED_RANK
            expected_decoder_rank = LR_SHARED_RANK
        else:
            expected_rank = LR_CAPACITY_RANK
            expected_decoder_rank = LR_CAPACITY_RANK
        if tuple(self.local_input_factor.shape) != (
            LOCAL_INPUT_WIDTH,
            expected_rank,
        ):
            raise ValueError("Wo local input factor has incompatible geometry")
        if tuple(self.global_decoder.shape) != (
            expected_decoder_rank,
            HIDDEN_SIZE,
        ):
            raise ValueError("Wo global decoder has incompatible geometry")
        if self.local_input_factor.dtype != self.global_decoder.dtype:
            raise TypeError("Wo encoder and decoder dtypes differ")
        if not bool(torch.isfinite(self.local_input_factor).all()) or not bool(
            torch.isfinite(self.global_decoder).all()
        ):
            raise ValueError("Wo factors contain non-finite values")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _broadcast_validation_error(error: str, *, device: torch.device) -> None:
    encoded = error.encode("utf-8")
    length = torch.tensor(len(encoded), dtype=torch.int64, device=device)
    dist.broadcast(length, src=0)
    size = int(length.item())
    payload = torch.zeros(max(size, 1), dtype=torch.uint8, device=device)
    if dist.get_rank() == 0 and size:
        payload[:size] = torch.tensor(list(encoded), dtype=torch.uint8, device=device)
    dist.broadcast(payload, src=0)
    if size:
        message = bytes(payload[:size].cpu().tolist()).decode("utf-8")
        raise RuntimeError(f"Phase-1 factor validation failed: {message}")


def load_qwen3_8b_wo_phase1(
    phase1_dir: str | Path,
    *,
    model_path: str | Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Validate the complete Phase-1 bank, hashing artifacts once on TP rank 0."""

    root = Path(phase1_dir).expanduser().resolve()
    selected_model = Path(model_path).expanduser().resolve()
    results_path = root / "results.json"
    result = _load_json(results_path)
    records = {int(row["layer"]): row for row in result["layers"]}
    error = ""
    if dist.get_rank() == 0:
        try:
            if result.get("format") != PHASE1_FORMAT or result.get("status") != "complete":
                raise ValueError("incompatible or incomplete results.json")
            if result["model"]["config_sha256"] != file_sha256(
                selected_model / "config.json"
            ):
                raise ValueError("model config differs from Phase-1")
            if result["method"]["scope"] != (
                "post-attention W_o only; dense V and dense KV cache"
            ):
                raise ValueError("factor bank is not Wo-only")
            if tuple(sorted(records)) != tuple(range(36)):
                raise ValueError("factor bank does not cover all 36 layers")
            for layer, record in records.items():
                sidecar = _load_json(root / f"layer_{layer:03d}.json")
                if sidecar != record or sidecar.get("format") != PHASE1_LAYER_FORMAT:
                    raise ValueError(f"layer {layer} sidecar differs from results")
                if sidecar.get("status") != "passed":
                    raise ValueError(f"layer {layer} did not pass Phase-1 gates")
                path = root / sidecar["artifact"]["file"]
                if file_sha256(path) != sidecar["artifact"]["sha256"]:
                    raise ValueError(f"layer {layer} artifact hash mismatch")
        except Exception as exc:  # broadcast the exact rank-0 validation failure
            error = f"{type(exc).__name__}: {exc}"
    _broadcast_validation_error(error, device=device)
    return result, records


def factors_from_payload(
    payload: Mapping[str, Tensor],
    *,
    layer_index: int,
    arm: str,
    process_rank: int,
    artifact_path: Path,
    artifact_sha256: str,
) -> Qwen3TP4WOLayerFactors:
    """Select one TP source from an already verified Phase-1 payload."""

    if not 0 <= process_rank < TP_SIZE:
        raise ValueError("process rank is outside TP4")
    if arm in ("wo_c1_ag", "wo_c1_local_ar"):
        encoders = payload["c1_source_encoders"]
        decoders = payload["c1_source_decoders"]
        if tuple(encoders.shape) != (TP_SIZE, LOCAL_INPUT_WIDTH, C1_LOCAL_RANK):
            raise ValueError("C1 source encoders have incompatible geometry")
        if tuple(decoders.shape) != (
            TP_SIZE,
            C1_LOCAL_RANK,
            HIDDEN_SIZE,
        ):
            raise ValueError("C1 source decoders have incompatible geometry")
        local_input = encoders[process_rank].contiguous()
        global_decoder = decoders.reshape(
            TP_SIZE * C1_LOCAL_RANK,
            HIDDEN_SIZE,
        ).contiguous()
    elif arm in ("wo_lr_ar_wire", "wo_lr_ar_capacity"):
        prefix = "lr_wire" if arm == "wo_lr_ar_wire" else "lr_capacity"
        selected_rank = (
            LR_SHARED_RANK if arm == "wo_lr_ar_wire" else LR_CAPACITY_RANK
        )
        input_factor = payload[f"{prefix}_input_factor"]
        decoder = payload[f"{prefix}_shared_decoder"]
        if tuple(input_factor.shape) != (HIDDEN_SIZE, selected_rank):
            raise ValueError("wire LR input factor has incompatible geometry")
        if tuple(decoder.shape) != (selected_rank, HIDDEN_SIZE):
            raise ValueError("wire LR decoder has incompatible geometry")
        start = process_rank * LOCAL_INPUT_WIDTH
        local_input = input_factor[start : start + LOCAL_INPUT_WIDTH].contiguous()
        global_decoder = decoder.contiguous()
    else:
        raise ValueError(f"unsupported Wo factor arm {arm}")
    return Qwen3TP4WOLayerFactors(
        layer_index=int(layer_index),
        arm=arm,
        local_input_factor=local_input,
        global_decoder=global_decoder,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
    )


def load_qwen3_tp4_wo_layer_factors(
    phase1_dir: Path,
    record: Mapping[str, Any],
    *,
    layer_index: int,
    arm: str,
    process_rank: int,
) -> Qwen3TP4WOLayerFactors:
    artifact_path = phase1_dir / str(record["artifact"]["file"])
    if arm in ("wo_c1_ag", "wo_c1_local_ar"):
        required = ("c1_source_encoders", "c1_source_decoders")
    elif arm == "wo_lr_ar_wire":
        required = ("lr_wire_input_factor", "lr_wire_shared_decoder")
    elif arm == "wo_lr_ar_capacity":
        required = ("lr_capacity_input_factor", "lr_capacity_shared_decoder")
    else:
        raise ValueError(f"unsupported Wo factor arm {arm}")
    with safe_open(artifact_path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != EXPECTED_FACTOR_KEYS:
            raise ValueError(f"Phase-1 tensor keys differ at layer {layer_index}")
        payload = {key: handle.get_tensor(key) for key in required}
    return factors_from_payload(
        payload,
        layer_index=layer_index,
        arm=arm,
        process_rank=process_rank,
        artifact_path=artifact_path,
        artifact_sha256=str(record["artifact"]["sha256"]),
    )


class PackedPrivateAllGatherOutput(nn.Module):
    """BF16 local encoder, packed NCCL AllGather, and replicated decoder."""

    def __init__(
        self,
        local_input_factor: Tensor,
        global_decoder: Tensor,
        *,
        communicator: FeatureRaggedCommunicator,
    ) -> None:
        super().__init__()
        if tuple(local_input_factor.shape) != (
            LOCAL_INPUT_WIDTH,
            C1_LOCAL_RANK,
        ):
            raise ValueError("packed C1 local encoder has incompatible geometry")
        if tuple(global_decoder.shape) != (
            TP_SIZE * C1_LOCAL_RANK,
            HIDDEN_SIZE,
        ):
            raise ValueError("packed C1 decoder has incompatible geometry")
        self.local_projection_weight = nn.Parameter(
            local_input_factor.transpose(0, 1).contiguous(),
            requires_grad=False,
        )
        self.global_decoder = nn.Parameter(
            global_decoder.contiguous(),
            requires_grad=False,
        )
        self.communicator = communicator
        self.plan = StaticRaggedPlan.from_source_widths(
            (C1_LOCAL_RANK,) * TP_SIZE
        )
        self._prepared: dict[int, PreparedUniformAllGather] = {}

    @property
    def local_rank(self) -> int:
        return C1_LOCAL_RANK

    @property
    def gathered_rank(self) -> int:
        return TP_SIZE * C1_LOCAL_RANK

    def clear_prepared(self) -> None:
        self._prepared.clear()

    def prepare(self, rows: int) -> None:
        selected_rows = int(rows)
        if selected_rows <= 0:
            raise ValueError("packed C1 row count must be positive")
        self._prepared[selected_rows] = self.communicator.prepare_uniform(
            self.plan,
            tokens=selected_rows,
            dtype=self.local_projection_weight.dtype,
            backend="uniform_nccl",
        )

    def forward(self, local_hidden_states: Tensor) -> Tensor:
        if int(local_hidden_states.shape[-1]) != LOCAL_INPUT_WIDTH:
            raise ValueError("packed C1 input width differs from one TP source")
        leading = tuple(local_hidden_states.shape[:-1])
        flat = local_hidden_states.reshape(-1, LOCAL_INPUT_WIDTH)
        rows = int(flat.shape[0])
        prepared = self._prepared.get(rows)
        if prepared is None:
            raise RuntimeError(
                f"packed C1 AllGather for {rows} rows was not prepared"
            )
        local = F.linear(flat, self.local_projection_weight)
        arena = prepared.gather(local, local_is_feature_major=False)
        output = decode_feature_major(arena, self.global_decoder)
        return output.reshape(*leading, HIDDEN_SIZE)


class PrivateLocalDecodeAllReduceOutput(nn.Module):
    """The exact C1 approximate function with a hidden-width AllReduce."""

    def __init__(
        self,
        local_input_factor: Tensor,
        global_decoder: Tensor,
        *,
        process_rank: int,
    ) -> None:
        super().__init__()
        if tuple(local_input_factor.shape) != (
            LOCAL_INPUT_WIDTH,
            C1_LOCAL_RANK,
        ):
            raise ValueError("local-AR C1 encoder has incompatible geometry")
        if tuple(global_decoder.shape) != (
            TP_SIZE * C1_LOCAL_RANK,
            HIDDEN_SIZE,
        ):
            raise ValueError("local-AR C1 decoder has incompatible geometry")
        selected_process = int(process_rank)
        if not 0 <= selected_process < TP_SIZE:
            raise ValueError("process rank is outside TP4")
        start = selected_process * C1_LOCAL_RANK
        self.local_projection_weight = nn.Parameter(
            local_input_factor.transpose(0, 1).contiguous(),
            requires_grad=False,
        )
        self.local_decoder_weight = nn.Parameter(
            global_decoder[start : start + C1_LOCAL_RANK]
            .transpose(0, 1)
            .contiguous(),
            requires_grad=False,
        )

    @property
    def local_rank(self) -> int:
        return C1_LOCAL_RANK

    def forward(self, local_hidden_states: Tensor) -> Tensor:
        if int(local_hidden_states.shape[-1]) != LOCAL_INPUT_WIDTH:
            raise ValueError("local-AR C1 input width differs from one TP source")
        latent = F.linear(local_hidden_states, self.local_projection_weight)
        output = F.linear(latent, self.local_decoder_weight)
        dist.all_reduce(output)
        return output


class Qwen3TP4WOC1DecodeAttention(Qwen3TP4DenseDecodeAttention):
    """Dense attention/KV cache followed by Wo-C1 private AllGather."""

    def __init__(
        self,
        base_attention: nn.Module,
        factors: Qwen3TP4WOLayerFactors,
        communicator: FeatureRaggedCommunicator,
    ) -> None:
        super().__init__(base_attention)
        if factors.arm != "wo_c1_ag" or factors.layer_index != self.layer_idx:
            raise ValueError("Wo-C1 factors belong to another arm or layer")
        self.o_proj = PackedPrivateAllGatherOutput(
            factors.local_input_factor.to(
                device=base_attention.q_proj.weight.device,
                dtype=base_attention.q_proj.weight.dtype,
            ),
            factors.global_decoder.to(
                device=base_attention.q_proj.weight.device,
                dtype=base_attention.q_proj.weight.dtype,
            ),
            communicator=communicator,
        )
        self.factor_path = str(factors.artifact_path)
        self.factor_sha256 = factors.artifact_sha256

    def clear_cache(self) -> None:
        super().clear_cache()
        self.o_proj.clear_prepared()

    def configure_cache(
        self,
        *,
        batch_size: int,
        capacity: int,
        max_forward_tokens: int = 1,
    ) -> None:
        super().configure_cache(
            batch_size=batch_size,
            capacity=capacity,
            max_forward_tokens=max_forward_tokens,
        )
        rows = int(batch_size) * int(max_forward_tokens)
        self.o_proj.communicator.configure_direct_workspace(
            tokens=rows,
            max_total_width=self.o_proj.gathered_rank,
            dtype=self.o_proj.local_projection_weight.dtype,
        )
        self.o_proj.prepare(rows)
        if int(max_forward_tokens) != 1:
            self.o_proj.prepare(int(batch_size))

    def _project_output(self, local_output: Tensor) -> Tensor:
        batch = int(local_output.shape[0])
        tokens = int(local_output.shape[2])
        token_major = local_output.transpose(1, 2).reshape(
            batch,
            tokens,
            QUERY_HEADS_PER_PROCESS * self.value_head_dim,
        )
        return self.o_proj(token_major)


class Qwen3TP4WOC1LocalAllReduceDecodeAttention(Qwen3TP4DenseDecodeAttention):
    """Dense attention followed by the C1 local decoder and hidden AllReduce."""

    def __init__(
        self,
        base_attention: nn.Module,
        factors: Qwen3TP4WOLayerFactors,
    ) -> None:
        super().__init__(base_attention)
        if factors.arm != "wo_c1_local_ar" or factors.layer_index != self.layer_idx:
            raise ValueError("Wo-C1 local-AR factors belong to another arm or layer")
        device = base_attention.q_proj.weight.device
        dtype = base_attention.q_proj.weight.dtype
        self.o_proj = PrivateLocalDecodeAllReduceOutput(
            factors.local_input_factor.to(device=device, dtype=dtype),
            factors.global_decoder.to(device=device, dtype=dtype),
            process_rank=dist.get_rank(),
        )
        self.factor_path = str(factors.artifact_path)
        self.factor_sha256 = factors.artifact_sha256

    def _project_output(self, local_output: Tensor) -> Tensor:
        batch = int(local_output.shape[0])
        tokens = int(local_output.shape[2])
        token_major = local_output.transpose(1, 2).reshape(
            batch,
            tokens,
            QUERY_HEADS_PER_PROCESS * self.value_head_dim,
        )
        return self.o_proj(token_major)


class Qwen3TP4WOLRAllReduceDecodeAttention(Qwen3TP4DenseDecodeAttention):
    """Dense attention/KV cache followed by wire-matched LR-AllReduce."""

    def __init__(
        self,
        base_attention: nn.Module,
        factors: Qwen3TP4WOLayerFactors,
    ) -> None:
        super().__init__(base_attention)
        if factors.arm not in (
            "wo_lr_ar_wire",
            "wo_lr_ar_capacity",
        ) or factors.layer_index != self.layer_idx:
            raise ValueError("Wo-LR factors belong to another arm or layer")
        device = base_attention.q_proj.weight.device
        dtype = base_attention.q_proj.weight.dtype
        self.o_proj = LowRankAllReduceOutput(
            factors.local_input_factor.to(device=device, dtype=dtype),
            factors.global_decoder.transpose(0, 1).to(device=device, dtype=dtype),
            process_group=None,
        )
        self.factor_path = str(factors.artifact_path)
        self.factor_sha256 = factors.artifact_sha256

    def _project_output(self, local_output: Tensor) -> Tensor:
        batch = int(local_output.shape[0])
        tokens = int(local_output.shape[2])
        token_major = local_output.transpose(1, 2).reshape(
            batch,
            tokens,
            QUERY_HEADS_PER_PROCESS * self.value_head_dim,
        )
        return self.o_proj(token_major)


def install_qwen3_8b_wo_tp4_attention(
    model: nn.Module,
    *,
    arm: str,
    model_path: str | Path,
    phase1_dir: str | Path,
) -> tuple[
    tuple[
        Qwen3TP4DenseDecodeAttention
        | Qwen3TP4WOC1DecodeAttention
        | Qwen3TP4WOC1LocalAllReduceDecodeAttention
        | Qwen3TP4WOLRAllReduceDecodeAttention,
        ...,
    ],
    dict[str, Any],
]:
    if arm not in ARMS:
        raise ValueError(f"unsupported Wo TP4 arm {arm}")
    if not dist.is_initialized() or dist.get_world_size() != TP_SIZE:
        raise RuntimeError("Wo runtime requires an initialized TP4 process group")
    device = model.model.embed_tokens.weight.device
    phase1, records = load_qwen3_8b_wo_phase1(
        phase1_dir,
        model_path=model_path,
        device=device,
    )
    observed = (
        int(model.config.hidden_size),
        int(model.config.num_hidden_layers),
        int(model.config.num_attention_heads),
        int(model.config.num_key_value_heads),
    )
    if observed != (HIDDEN_SIZE, 36, 32, 8):
        raise ValueError(f"loaded model has incompatible Qwen3-8B geometry {observed}")

    root = Path(phase1_dir).expanduser().resolve()
    process_rank = dist.get_rank()
    communicator = (
        FeatureRaggedCommunicator.from_distributed(device=device)
        if arm == "wo_c1_ag"
        else None
    )
    installed = []
    for layer_index, layer in enumerate(model.model.layers):
        base_attention = layer.self_attn
        if arm == "dense":
            replacement = Qwen3TP4DenseDecodeAttention(base_attention)
        else:
            factors = load_qwen3_tp4_wo_layer_factors(
                root,
                records[layer_index],
                layer_index=layer_index,
                arm=arm,
                process_rank=process_rank,
            )
            if arm == "wo_c1_ag":
                assert communicator is not None
                replacement = Qwen3TP4WOC1DecodeAttention(
                    base_attention,
                    factors,
                    communicator,
                )
            elif arm == "wo_c1_local_ar":
                replacement = Qwen3TP4WOC1LocalAllReduceDecodeAttention(
                    base_attention,
                    factors,
                )
            else:
                replacement = Qwen3TP4WOLRAllReduceDecodeAttention(
                    base_attention,
                    factors,
                )
        replacement.eval()
        layer.self_attn = replacement
        installed.append(replacement)
    return tuple(installed), phase1


def close_qwen3_8b_wo_tp4(
    modules: Sequence[
        Qwen3TP4DenseDecodeAttention
        | Qwen3TP4WOC1DecodeAttention
        | Qwen3TP4WOC1LocalAllReduceDecodeAttention
        | Qwen3TP4WOLRAllReduceDecodeAttention
    ],
) -> None:
    c1_modules = tuple(
        module for module in modules if isinstance(module, Qwen3TP4WOC1DecodeAttention)
    )
    communicators = {
        id(module.o_proj.communicator): module.o_proj.communicator
        for module in c1_modules
    }
    for module in c1_modules:
        module.o_proj.clear_prepared()
    for communicator in communicators.values():
        communicator.close()


__all__ = [
    "ARMS",
    "C1_LOCAL_RANK",
    "LR_CAPACITY_RANK",
    "LR_SHARED_RANK",
    "LOCAL_INPUT_WIDTH",
    "PackedPrivateAllGatherOutput",
    "PrivateLocalDecodeAllReduceOutput",
    "Qwen3TP4WOC1DecodeAttention",
    "Qwen3TP4WOC1LocalAllReduceDecodeAttention",
    "Qwen3TP4WOLRAllReduceDecodeAttention",
    "Qwen3TP4WOLayerFactors",
    "close_qwen3_8b_wo_tp4",
    "factors_from_payload",
    "install_qwen3_8b_wo_tp4_attention",
    "load_qwen3_8b_wo_phase1",
    "load_qwen3_tp4_wo_layer_factors",
]
