"""DeepSeek-V2-Lite TP4 C1 attention-output communication runtime.

The fitted checkpoint contains eight logical attention-output sources, each
covering two consecutive MLA value heads (``2 * 128 = 256`` features).  On a
physical TP4 runtime, every process owns two consecutive logical sources.  C1
encodes those two post-attention source blocks locally, performs one packed
AllGather across the four processes, and applies the complete source-major
decoder on every process.  MLA latent KV and the KV cache remain unchanged.
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

from basisserve.kernels.feature_ragged_allgather import (
    FeatureRaggedCommunicator,
    decode_feature_major,
)
from basisserve.kernels.ragged_allgather import StaticRaggedPlan


TP_SIZE = 4
LOGICAL_SOURCES = 8
SOURCES_PER_PROCESS = LOGICAL_SOURCES // TP_SIZE
NUM_ATTENTION_HEADS = 16
VALUE_HEAD_DIM = 128
SOURCE_WIDTH = 2 * VALUE_HEAD_DIM
LOCAL_ATTENTION_WIDTH = SOURCES_PER_PROCESS * SOURCE_WIDTH
HIDDEN_SIZE = 2048
NUM_LAYERS = 27
FACTOR_FORMAT = "basisserve.deepseek_v2_lite.tp8_source_wo_c1.layer_global_kl.v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DeepseekV2LiteC1FactorLayer:
    layer_index: int
    source_rank: int
    encoders: Tensor
    decoders: Tensor
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        rank = int(self.source_rank)
        if not 0 < rank <= SOURCE_WIDTH:
            raise ValueError(f"invalid DeepSeek C1 source rank {rank}")
        if tuple(self.encoders.shape) != (LOGICAL_SOURCES, SOURCE_WIDTH, rank):
            raise ValueError(
                "DeepSeek C1 encoders must have shape "
                f"{(LOGICAL_SOURCES, SOURCE_WIDTH, rank)}, got "
                f"{tuple(self.encoders.shape)}"
            )
        if tuple(self.decoders.shape) != (LOGICAL_SOURCES, rank, HIDDEN_SIZE):
            raise ValueError(
                "DeepSeek C1 decoders must have shape "
                f"{(LOGICAL_SOURCES, rank, HIDDEN_SIZE)}, got "
                f"{tuple(self.decoders.shape)}"
            )
        if self.encoders.dtype != self.decoders.dtype:
            raise TypeError("DeepSeek C1 encoders and decoders must share a dtype")


def _load_result(factor_dir: Path) -> Mapping[str, Any]:
    path = factor_dir / "result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("format") != FACTOR_FORMAT or result.get("status") != "complete":
        raise ValueError(f"{path} is not a completed DeepSeek layer Global-KL result")
    selection = result.get("selection", {})
    if selection.get("selected_candidate") != "mean_dp":
        raise ValueError("DeepSeek runtime requires the selected mean-DP schedule")
    schedule = tuple(map(int, selection.get("selected_schedule", ())))
    if len(schedule) != NUM_LAYERS:
        raise ValueError("DeepSeek mean-DP schedule does not cover all decoder layers")
    return result


def load_deepseek_v2_lite_c1_factors(
    factor_dir: str | Path,
) -> tuple[DeepseekV2LiteC1FactorLayer, ...]:
    root = Path(factor_dir).expanduser().resolve()
    result = _load_result(root)
    selection = result["selection"]
    schedule = tuple(map(int, selection["selected_schedule"]))
    artifacts = result.get("selected_artifacts", {})
    layers: list[DeepseekV2LiteC1FactorLayer] = []
    for layer_index, rank in enumerate(schedule):
        record = artifacts.get(str(layer_index))
        if not isinstance(record, dict):
            raise ValueError(f"missing selected artifact for layer {layer_index}")
        if int(record.get("source_rank", -1)) != rank:
            raise ValueError(f"artifact rank differs from schedule at layer {layer_index}")
        path = root / str(record["file"])
        expected_hash = str(record["sha256"])
        observed_hash = file_sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(f"factor hash mismatch at layer {layer_index}")
        payload = load_file(str(path), device="cpu")
        if set(payload) != {"source_encoders", "source_decoders"}:
            raise ValueError(f"unexpected tensors in {path}")
        layers.append(
            DeepseekV2LiteC1FactorLayer(
                layer_index=layer_index,
                source_rank=rank,
                encoders=payload["source_encoders"].contiguous(),
                decoders=payload["source_decoders"].contiguous(),
                path=path,
                sha256=observed_hash,
            )
        )
    return tuple(layers)


class DeepseekV2LiteTP4C1OutputProjection(nn.Module):
    """Replace one TP4 row-parallel ``o_proj`` with C1 private AllGather."""

    def __init__(
        self,
        factors: DeepseekV2LiteC1FactorLayer,
        communicator: FeatureRaggedCommunicator,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if not dist.is_initialized() or dist.get_world_size() != TP_SIZE:
            raise RuntimeError("DeepSeek-V2-Lite runtime requires physical TP4")
        process_rank = dist.get_rank()
        first_source = process_rank * SOURCES_PER_PROCESS
        last_source = first_source + SOURCES_PER_PROCESS
        rank = int(factors.source_rank)
        self.layer_index = int(factors.layer_index)
        self.source_rank = rank
        self.local_source_indices = tuple(range(first_source, last_source))
        self.local_wire_width = SOURCES_PER_PROCESS * rank
        self.factor_path = str(factors.path)
        self.factor_sha256 = factors.sha256
        self.communicator = communicator
        self.plan = StaticRaggedPlan.from_source_widths(
            (self.local_wire_width,) * TP_SIZE
        )
        local_encoders = factors.encoders[first_source:last_source].to(
            device=device,
            dtype=dtype,
        )
        packed_encoder = torch.zeros(
            LOCAL_ATTENTION_WIDTH,
            self.local_wire_width,
            device=device,
            dtype=dtype,
        )
        for local_source in range(SOURCES_PER_PROCESS):
            input_start = local_source * SOURCE_WIDTH
            output_start = local_source * rank
            packed_encoder[
                input_start : input_start + SOURCE_WIDTH,
                output_start : output_start + rank,
            ].copy_(local_encoders[local_source])
        self.register_buffer("local_encoder", packed_encoder.contiguous())
        self.register_buffer(
            "global_decoder",
            factors.decoders.reshape(LOGICAL_SOURCES * rank, HIDDEN_SIZE).to(
                device=device,
                dtype=dtype,
            ),
        )

    def forward(self, local_attention_output: Tensor) -> Tensor:
        if int(local_attention_output.shape[-1]) != LOCAL_ATTENTION_WIDTH:
            raise ValueError(
                "DeepSeek TP4 local attention width must be "
                f"{LOCAL_ATTENTION_WIDTH}, got {tuple(local_attention_output.shape)}"
            )
        leading_shape = tuple(local_attention_output.shape[:-1])
        rows = local_attention_output.numel() // LOCAL_ATTENTION_WIDTH
        local_coordinates = torch.mm(
            local_attention_output.reshape(rows, LOCAL_ATTENTION_WIDTH),
            self.local_encoder,
        )
        arena = self.communicator.gather(
            local_coordinates,
            self.plan,
            backend="feature_direct",
        )
        decoded = decode_feature_major(arena, self.global_decoder)
        return decoded.reshape(*leading_shape, HIDDEN_SIZE)


def install_deepseek_v2_lite_tp4_c1_output_projections(
    model: nn.Module,
    *,
    factor_dir: str | Path,
) -> tuple[DeepseekV2LiteTP4C1OutputProjection, ...]:
    """Install all 27 post-attention C1 collectives after HF TP sharding."""

    if not dist.is_initialized() or dist.get_world_size() != TP_SIZE:
        raise RuntimeError("DeepSeek-V2-Lite C1 installation requires TP4")
    layers: Sequence[nn.Module] = model.model.layers
    if len(layers) != NUM_LAYERS:
        raise ValueError(f"expected {NUM_LAYERS} DeepSeek layers, got {len(layers)}")
    factors = load_deepseek_v2_lite_c1_factors(factor_dir)
    reference = model.model.embed_tokens.weight
    communicator = FeatureRaggedCommunicator.from_distributed(device=reference.device)
    installed: list[DeepseekV2LiteTP4C1OutputProjection] = []
    for layer_index, (layer, factor) in enumerate(zip(layers, factors)):
        base = layer.self_attn.o_proj
        weight = getattr(base, "weight", None)
        if weight is None or tuple(weight.shape) != (HIDDEN_SIZE, LOCAL_ATTENTION_WIDTH):
            raise ValueError(
                f"layer {layer_index} o_proj is not a TP4 row shard: "
                f"{None if weight is None else tuple(weight.shape)}"
            )
        if getattr(base, "bias", None) is not None:
            raise ValueError("DeepSeek-V2-Lite C1 runtime does not support o_proj bias")
        replacement = DeepseekV2LiteTP4C1OutputProjection(
            factor,
            communicator,
            device=weight.device,
            dtype=weight.dtype,
        ).eval()
        layer.self_attn.o_proj = replacement
        installed.append(replacement)
    return tuple(installed)


def configure_deepseek_v2_lite_tp4_c1_workspace(
    modules: Sequence[DeepseekV2LiteTP4C1OutputProjection],
    *,
    max_tokens: int,
) -> None:
    if not modules:
        return
    communicators = {id(module.communicator): module.communicator for module in modules}
    if len(communicators) != 1:
        raise RuntimeError("DeepSeek C1 layers must share one communicator")
    communicator = next(iter(communicators.values()))
    communicator.configure_direct_workspace(
        tokens=int(max_tokens),
        max_total_width=max(module.plan.total_width for module in modules),
        dtype=modules[0].global_decoder.dtype,
    )


def close_deepseek_v2_lite_tp4_c1_communicator(
    modules: Sequence[DeepseekV2LiteTP4C1OutputProjection],
) -> None:
    communicators = {id(module.communicator): module.communicator for module in modules}
    for communicator in communicators.values():
        communicator.close()


__all__ = [
    "DeepseekV2LiteC1FactorLayer",
    "DeepseekV2LiteTP4C1OutputProjection",
    "HIDDEN_SIZE",
    "LOCAL_ATTENTION_WIDTH",
    "LOGICAL_SOURCES",
    "NUM_LAYERS",
    "SOURCE_WIDTH",
    "SOURCES_PER_PROCESS",
    "TP_SIZE",
    "close_deepseek_v2_lite_tp4_c1_communicator",
    "configure_deepseek_v2_lite_tp4_c1_workspace",
    "file_sha256",
    "install_deepseek_v2_lite_tp4_c1_output_projections",
    "load_deepseek_v2_lite_c1_factors",
]
