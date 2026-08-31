"""Serving-ready C1 factors for the Qwen3-32B vLLM integration.

The local encoders are folded into the Value projection during checkpoint
loading.  Compact attention therefore emits the process-rank-ordered wire
block directly; this module packs that block's replicated global decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from basisserve.core.qwen3_32b_tp4_decode import (
    HEAD_DIM,
    HIDDEN_SIZE,
    KV_HEADS_PER_PROCESS,
    NUM_KV_HEADS,
    QUERY_HEADS_PER_KV_HEAD,
    QUERY_HEADS_PER_PROCESS,
    TP_SIZE,
    Qwen3_32BTP4C1FactorLayer,
)


@dataclass(frozen=True)
class Qwen3_32BTP4UniformC1OutputFactors:
    """Serving-ready uniform C1 factors for one TP4 process and layer."""

    layer_index: int
    process_rank: int
    source_rank: int
    local_encoders: Tensor
    decoder_weight: Tensor
    manifest_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not 0 <= self.process_rank < TP_SIZE:
            raise ValueError(f"process rank is outside TP{TP_SIZE}")
        expected_encoders = (
            KV_HEADS_PER_PROCESS,
            HEAD_DIM,
            self.source_rank,
        )
        expected_decoder = (
            HIDDEN_SIZE,
            NUM_KV_HEADS * QUERY_HEADS_PER_KV_HEAD * self.source_rank,
        )
        if tuple(self.local_encoders.shape) != expected_encoders:
            raise ValueError(
                f"local encoders must have shape {expected_encoders}, got "
                f"{tuple(self.local_encoders.shape)}"
            )
        if tuple(self.decoder_weight.shape) != expected_decoder:
            raise ValueError(
                f"decoder weight must have shape {expected_decoder}, got "
                f"{tuple(self.decoder_weight.shape)}"
            )
        if self.local_encoders.dtype != self.decoder_weight.dtype:
            raise TypeError("C1 encoder and decoder dtypes differ")
        if self.local_encoders.device != self.decoder_weight.device:
            raise ValueError("C1 encoder and decoder devices differ")

    @property
    def local_wire_width(self) -> int:
        return QUERY_HEADS_PER_PROCESS * self.source_rank

    @property
    def global_wire_width(self) -> int:
        return TP_SIZE * self.local_wire_width


def pack_qwen3_32b_tp4_uniform_c1_output_factors(
    factors: Qwen3_32BTP4C1FactorLayer,
    *,
    process_rank: int,
    manifest_sha256: str,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Qwen3_32BTP4UniformC1OutputFactors:
    """Pack one uniform layer in vLLM TP-rank wire order."""

    rank = int(process_rank)
    if not 0 <= rank < TP_SIZE:
        raise ValueError(f"process rank is outside TP{TP_SIZE}: {rank}")
    if len(set(factors.source_ranks)) != 1:
        raise ValueError(
            "the initial vLLM C1 serving path requires one uniform source rank"
        )
    source_rank = factors.source_ranks[0]
    source_start = rank * KV_HEADS_PER_PROCESS
    source_stop = source_start + KV_HEADS_PER_PROCESS
    local_encoders = factors.encoders[
        source_start:source_stop,
        :,
        :source_rank,
    ].to(device=device, dtype=dtype).contiguous()

    decoder_blocks: list[Tensor] = []
    for source in range(NUM_KV_HEADS):
        head_start = source * QUERY_HEADS_PER_KV_HEAD
        head_stop = head_start + QUERY_HEADS_PER_KV_HEAD
        decoder_blocks.append(
            factors.decoders[head_start:head_stop, :source_rank]
            .reshape(QUERY_HEADS_PER_KV_HEAD * source_rank, HIDDEN_SIZE)
        )
    global_decoder = torch.cat(decoder_blocks, dim=0).to(
        device=device,
        dtype=dtype,
    )
    decoder_weight = global_decoder.T.contiguous()
    return Qwen3_32BTP4UniformC1OutputFactors(
        layer_index=factors.layer_index,
        process_rank=rank,
        source_rank=source_rank,
        local_encoders=local_encoders,
        decoder_weight=decoder_weight,
        manifest_sha256=str(manifest_sha256),
        artifact_sha256=factors.sha256,
    )


def decode_qwen3_32b_c1_coordinates(
    global_coordinates: Tensor,
    decoder_weight: Tensor,
) -> Tensor:
    """Decode rank-major C1 coordinates into the replicated hidden state."""

    if global_coordinates.ndim != 2 or decoder_weight.ndim != 2:
        raise ValueError("C1 coordinates and decoder weight must both be matrices")
    if int(global_coordinates.shape[1]) != int(decoder_weight.shape[1]):
        raise ValueError(
            "global coordinate width differs from the C1 decoder input width"
        )
    if (
        global_coordinates.device != decoder_weight.device
        or global_coordinates.dtype != decoder_weight.dtype
    ):
        raise ValueError("C1 coordinates and decoder must match dtype/device")
    return F.linear(global_coordinates, decoder_weight)


__all__ = [
    "Qwen3_32BTP4UniformC1OutputFactors",
    "decode_qwen3_32b_c1_coordinates",
    "pack_qwen3_32b_tp4_uniform_c1_output_factors",
]
