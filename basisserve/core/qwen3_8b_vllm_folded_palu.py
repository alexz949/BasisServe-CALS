"""Authenticated Qwen3-8B PaLU M/G2/G4 folding for standard vLLM."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import load_file
import torch
from torch import Tensor

from basisserve.core.qwen3_8b_vllm_folded_c1 import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_QUERY_HEADS,
    file_sha256,
)


CHECKPOINT_FORMAT = "basisserve.qwen3_8b.iclr_v_factors.v1"


@dataclass(frozen=True)
class Qwen3_8BFoldedPaLUCheckpoint:
    """Authenticated PaLU artifact and its per-layer rank schedule."""

    checkpoint_dir: Path
    manifest_sha256: str
    model_config_sha256: str
    factor_path: Path
    factor_sha256: str
    head_group_size: int
    layer_ranks: tuple[tuple[int, ...], ...]


def load_qwen3_8b_folded_palu_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_manifest_sha256: str,
) -> Qwen3_8BFoldedPaLUCheckpoint:
    """Authenticate a Qwen3-8B PaLU M/G2/G4 32K-Fisher R80 checkpoint."""

    resolved = checkpoint_dir.expanduser().resolve()
    manifest_path = resolved / "manifest.json"
    assert manifest_path.is_file()
    observed_manifest_sha256 = file_sha256(manifest_path)
    assert observed_manifest_sha256 == expected_manifest_sha256
    manifest: Mapping[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    compression = manifest.get("compression", {})
    assert manifest.get("format") == CHECKPOINT_FORMAT
    assert manifest.get("status") == "complete"
    assert compression.get("method") == "palu-fisher"
    assert compression.get("projection") == "v"
    head_group_size = int(compression.get("head_group_size", 0))
    assert head_group_size in (1, 2, 4)
    assert (
        manifest.get("run_id")
        == {
            1: "Q3-8B-PALUM-R80",
            2: "Q3-8B-PALUG2-R80",
            4: "Q3-8B-PALUG4-R80",
        }[head_group_size]
    )
    assert int(compression.get("equivalent_rank_target", 0)) == 80
    layer_ranks = tuple(
        tuple(map(int, ranks)) for ranks in compression.get("layer_ranks", ())
    )
    assert len(layer_ranks) == NUM_LAYERS
    num_groups = NUM_KV_HEADS // head_group_size
    assert all(len(ranks) == num_groups for ranks in layer_ranks)
    assert all(
        len(set(ranks)) == 1
        and all(0 < rank <= HEAD_DIM * head_group_size for rank in ranks)
        for ranks in layer_ranks
    )
    assert compression.get("allocation") == "official_palu_fisher_uniform"
    calibration = manifest.get("calibration", {})
    assert int(calibration.get("samples", 0)) == 32
    assert int(calibration.get("sequence_length", 0)) == 32768
    assert sum(map(sum, layer_ranks)) == int(
        compression.get("rank_sum_across_layers", -1)
    )
    artifact = manifest.get("artifact", {})
    factor_path = (resolved / str(artifact.get("file", ""))).resolve()
    assert factor_path.is_file()
    factor_sha256 = str(artifact.get("sha256"))
    assert file_sha256(factor_path) == factor_sha256
    assert int(artifact.get("tensor_count", 0)) == 2 * NUM_LAYERS
    return Qwen3_8BFoldedPaLUCheckpoint(
        checkpoint_dir=resolved,
        manifest_sha256=observed_manifest_sha256,
        model_config_sha256=str(manifest.get("model", {}).get("config_sha256")),
        factor_path=factor_path,
        factor_sha256=factor_sha256,
        head_group_size=head_group_size,
        layer_ranks=layer_ranks,
    )


@torch.no_grad()
def reconstructed_palu_value_projection(
    writer: Tensor,
    decoder: Tensor,
    ranks: tuple[int, ...],
) -> Tensor:
    """Return the dense V projection represented by PaLU group factors."""

    num_groups = len(ranks)
    group_width = int(decoder.shape[1])
    hidden_size = int(writer.shape[1])
    maximum_rank = max(ranks)
    assert num_groups > 0
    assert tuple(writer.shape) == (sum(ranks), hidden_size)
    assert tuple(decoder.shape) == (num_groups, group_width, maximum_rank)
    work_writer = writer.float()
    work_decoder = decoder.to(device=writer.device, dtype=torch.float32)
    dense_value = torch.zeros(
        num_groups * group_width,
        hidden_size,
        device=writer.device,
        dtype=torch.float32,
    )
    writer_offset = 0
    for group, rank in enumerate(ranks):
        dense_rows = slice(group * group_width, (group + 1) * group_width)
        dense_value[dense_rows].copy_(
            work_decoder[group, :, :rank]
            @ work_writer[writer_offset : writer_offset + rank]
        )
        writer_offset += rank
    return dense_value


@torch.no_grad()
def fold_qwen3_8b_palu_layer_into_vllm_weights(
    qkv_weight: Tensor,
    writer: Tensor,
    decoder: Tensor,
    ranks: tuple[int, ...],
) -> None:
    """Fold one PaLU group factorization into TP1 vLLM's V projection."""

    query_width = NUM_QUERY_HEADS * HEAD_DIM
    key_width = NUM_KV_HEADS * HEAD_DIM
    value_offset = query_width + key_width
    assert tuple(qkv_weight.shape) == (
        query_width + 2 * key_width,
        HIDDEN_SIZE,
    )
    folded_value = reconstructed_palu_value_projection(
        writer.to(device=qkv_weight.device),
        decoder.to(device=qkv_weight.device),
        ranks,
    )
    qkv_weight[value_offset:].copy_(folded_value.to(dtype=qkv_weight.dtype))


def load_qwen3_8b_palu_factors(
    checkpoint: Qwen3_8BFoldedPaLUCheckpoint,
) -> dict[str, Tensor]:
    """Load and authenticate the single PaLU factor artifact."""

    assert file_sha256(checkpoint.factor_path) == checkpoint.factor_sha256
    factors = load_file(str(checkpoint.factor_path), device="cpu")
    assert set(factors) == {
        name
        for layer in range(NUM_LAYERS)
        for name in (
            f"layers.{layer}.v_writer.weight",
            f"layers.{layer}.v_decoder.weight",
        )
    }
    return factors


__all__ = [
    "CHECKPOINT_FORMAT",
    "Qwen3_8BFoldedPaLUCheckpoint",
    "fold_qwen3_8b_palu_layer_into_vllm_weights",
    "load_qwen3_8b_folded_palu_checkpoint",
    "load_qwen3_8b_palu_factors",
    "reconstructed_palu_value_projection",
]
