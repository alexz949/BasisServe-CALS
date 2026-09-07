"""Authenticated Qwen3-8B C1 folding for a standard vLLM KV cache."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import load_file
import torch
from torch import Tensor


CHECKPOINT_FORMAT = "basisserve.qwen3_8b.iclr_v_factors.v1"
NUM_LAYERS = 36
HIDDEN_SIZE = 4096
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Qwen3_8BFoldedC1Layer:
    """One authenticated layer factor reference and its physical-source ranks."""

    layer_index: int
    ranks: tuple[int, ...]
    factor_path: Path
    factor_sha256: str


@dataclass(frozen=True)
class Qwen3_8BFoldedC1Checkpoint:
    """Small manifest index used by the vLLM model worker."""

    checkpoint_dir: Path
    manifest_sha256: str
    method: str
    model_config_sha256: str
    layers: tuple[Qwen3_8BFoldedC1Layer, ...]


def load_qwen3_8b_folded_c1_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_manifest_sha256: str,
) -> Qwen3_8BFoldedC1Checkpoint:
    """Authenticate a uniform or two-sided-KL ICLR C1 checkpoint manifest."""

    resolved = checkpoint_dir.expanduser().resolve()
    manifest_path = resolved / "manifest.json"
    assert manifest_path.is_file()
    observed_manifest_sha256 = file_sha256(manifest_path)
    assert observed_manifest_sha256 == expected_manifest_sha256
    manifest: Mapping[str, Any] = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    assert manifest.get("format") == CHECKPOINT_FORMAT
    assert manifest.get("status") == "complete"
    compression = manifest.get("compression", {})
    method = str(compression.get("method"))
    assert method in {"c1-two-sided-kl", "c1-uniform"}
    assert compression.get("projection") == "joint_v_o"
    equivalent_rank_target = int(
        compression.get("equivalent_rank_target", 0)
    )
    assert 0 < equivalent_rank_target <= HEAD_DIM
    expected_rank_sum = NUM_LAYERS * NUM_KV_HEADS * equivalent_rank_target
    assert int(compression.get("rank_sum_across_layers", 0)) == expected_rank_sum
    rows = manifest.get("layers", ())
    assert len(rows) == NUM_LAYERS
    layers = []
    for layer_index, row in enumerate(rows):
        ranks = tuple(map(int, row.get("ranks", ())))
        factor_path = (resolved / str(row.get("file", ""))).resolve()
        assert int(row.get("layer", -1)) == layer_index
        assert len(ranks) == NUM_KV_HEADS
        assert all(0 < rank <= HEAD_DIM for rank in ranks)
        assert factor_path.is_file()
        factor_sha256 = str(row.get("sha256"))
        assert file_sha256(factor_path) == factor_sha256
        layers.append(
            Qwen3_8BFoldedC1Layer(
                layer_index=layer_index,
                ranks=ranks,
                factor_path=factor_path,
                factor_sha256=factor_sha256,
            )
        )
    assert sum(sum(layer.ranks) for layer in layers) == expected_rank_sum
    return Qwen3_8BFoldedC1Checkpoint(
        checkpoint_dir=resolved,
        manifest_sha256=observed_manifest_sha256,
        method=method,
        model_config_sha256=str(manifest.get("model", {}).get("config_sha256")),
        layers=tuple(layers),
    )


def load_qwen3_8b_folded_c1_layer(
    record: Qwen3_8BFoldedC1Layer,
) -> tuple[Tensor, Tensor]:
    """Load one layer and normalize allocated/uniform tensor payloads."""

    assert file_sha256(record.factor_path) == record.factor_sha256
    payload = load_file(str(record.factor_path), device="cpu")
    expected = {"value_coordinate_encoders", "head_output_decoders"}
    if "source_ranks" in payload:
        assert tuple(map(int, payload["source_ranks"].tolist())) == record.ranks
        expected.add("source_ranks")
    assert set(payload) == expected
    encoders = payload["value_coordinate_encoders"].contiguous()
    decoders = payload["head_output_decoders"].contiguous()
    maximum_rank = max(record.ranks)
    assert tuple(encoders.shape) == (NUM_KV_HEADS, HEAD_DIM, maximum_rank)
    assert tuple(decoders.shape) == (
        NUM_QUERY_HEADS,
        maximum_rank,
        HIDDEN_SIZE,
    )
    return encoders, decoders


@torch.no_grad()
def folded_c1_projection_weights(
    dense_value: Tensor,
    encoders: Tensor,
    decoders: Tensor,
    ranks: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    """Return dense-slot V/O weights for arbitrary valid GQA C1 geometry."""

    num_sources = len(ranks)
    head_dim = int(encoders.shape[1])
    num_query_heads = int(decoders.shape[0])
    hidden_size = int(dense_value.shape[1])
    maximum_rank = max(ranks)
    assert num_sources > 0 and num_query_heads % num_sources == 0
    assert tuple(dense_value.shape) == (num_sources * head_dim, hidden_size)
    assert tuple(encoders.shape) == (num_sources, head_dim, maximum_rank)
    assert tuple(decoders.shape) == (
        num_query_heads,
        maximum_rank,
        hidden_size,
    )
    device = dense_value.device
    work_value = dense_value.detach().float()
    work_encoders = encoders.to(device=device, dtype=torch.float32)
    work_decoders = decoders.to(device=device, dtype=torch.float32)
    folded_value = torch.zeros_like(work_value)
    folded_output = torch.zeros(
        hidden_size,
        num_query_heads * head_dim,
        device=device,
        dtype=torch.float32,
    )
    query_heads_per_source = num_query_heads // num_sources
    for source, rank in enumerate(ranks):
        dense_rows = slice(source * head_dim, (source + 1) * head_dim)
        latent_rows = slice(source * head_dim, source * head_dim + rank)
        folded_value[latent_rows].copy_(
            work_encoders[source, :, :rank].T @ work_value[dense_rows]
        )
        first_query_head = source * query_heads_per_source
        for head in range(
            first_query_head,
            first_query_head + query_heads_per_source,
        ):
            latent_columns = slice(head * head_dim, head * head_dim + rank)
            folded_output[:, latent_columns].copy_(
                work_decoders[head, :rank].T
            )
    return folded_value, folded_output


@torch.no_grad()
def fold_qwen3_8b_c1_layer_into_vllm_weights(
    qkv_weight: Tensor,
    output_weight: Tensor,
    encoders: Tensor,
    decoders: Tensor,
    ranks: tuple[int, ...],
) -> None:
    """Fold C1 into TP1 vLLM fused-QKV and dense output-projection slots."""

    query_width = NUM_QUERY_HEADS * HEAD_DIM
    key_width = NUM_KV_HEADS * HEAD_DIM
    value_offset = query_width + key_width
    assert tuple(qkv_weight.shape) == (
        query_width + 2 * key_width,
        HIDDEN_SIZE,
    )
    assert tuple(output_weight.shape) == (HIDDEN_SIZE, query_width)
    assert len(ranks) == NUM_KV_HEADS
    maximum_rank = max(ranks)
    assert tuple(encoders.shape) == (NUM_KV_HEADS, HEAD_DIM, maximum_rank)
    assert tuple(decoders.shape) == (
        NUM_QUERY_HEADS,
        maximum_rank,
        HIDDEN_SIZE,
    )
    folded_value, folded_output = folded_c1_projection_weights(
        qkv_weight[value_offset:],
        encoders,
        decoders,
        ranks,
    )
    qkv_weight[value_offset:].copy_(folded_value.to(dtype=qkv_weight.dtype))
    output_weight.copy_(folded_output.to(dtype=output_weight.dtype))


__all__ = [
    "CHECKPOINT_FORMAT",
    "HEAD_DIM",
    "HIDDEN_SIZE",
    "NUM_KV_HEADS",
    "NUM_LAYERS",
    "NUM_QUERY_HEADS",
    "Qwen3_8BFoldedC1Checkpoint",
    "Qwen3_8BFoldedC1Layer",
    "file_sha256",
    "folded_c1_projection_weights",
    "fold_qwen3_8b_c1_layer_into_vllm_weights",
    "load_qwen3_8b_folded_c1_checkpoint",
    "load_qwen3_8b_folded_c1_layer",
]
