#!/usr/bin/env python3
"""Build the algebraically exact pair-rank256 Qwen3 Key runtime control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor
from transformers import AutoConfig


FORMAT = "basisserve.qwen3_8b.pairwise_kq_svd.v1"
NUM_LAYERS = 36
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAIR_RANK = 2 * HEAD_DIM


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)


def identity_pair_factors(
    *,
    num_layers: int,
    num_kv_heads: int,
    num_query_heads: int,
    head_dim: int,
) -> tuple[Tensor, Tensor]:
    """Return pair encoders/readouts that exactly concatenate adjacent Keys."""

    if min(num_layers, num_kv_heads, num_query_heads, head_dim) <= 0:
        raise ValueError("identity Pairwise-K geometry must be positive")
    if num_layers % 2:
        raise ValueError("identity Pairwise-K requires an even layer count")
    pair_rank = 2 * head_dim
    pair_key = (
        torch.eye(pair_rank, dtype=torch.float32)
        .view(1, 1, pair_rank, pair_rank)
        .expand(num_layers // 2, num_kv_heads, pair_rank, pair_rank)
        .clone()
    )
    selectors = torch.zeros(2, head_dim, pair_rank, dtype=torch.float32)
    selectors[0, :, :head_dim] = torch.eye(head_dim, dtype=torch.float32)
    selectors[1, :, head_dim:] = torch.eye(head_dim, dtype=torch.float32)
    pair_query = (
        selectors[torch.arange(num_layers) % 2]
        .view(num_layers, 1, head_dim, pair_rank)
        .expand(num_layers, num_query_heads, head_dim, pair_rank)
        .clone()
    )
    return pair_key.contiguous(), pair_query.contiguous()


def _validate_model(model_path: Path) -> None:
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    observed = (
        str(config.model_type),
        int(config.num_hidden_layers),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(config.head_dim),
    )
    expected = ("qwen3", NUM_LAYERS, NUM_QUERY_HEADS, NUM_KV_HEADS, HEAD_DIM)
    if observed != expected:
        raise ValueError(f"expected Qwen3-8B geometry {expected}, found {observed}")


def build(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    model_path = args.model.expanduser().resolve()
    source_dir = args.source_factor_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _validate_model(model_path)

    source_result_path = source_dir / "result.json"
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if source_result.get("format") != FORMAT or source_result.get("status") != "complete":
        raise ValueError("source Pairwise-K checkpoint is incomplete or incompatible")
    if source_result["model"]["config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("source Pairwise-K checkpoint belongs to another model")
    source_artifact = source_result["artifacts"]["factors"]
    source_factor_path = source_dir / source_artifact["file"]
    if _sha256(source_factor_path) != source_artifact["sha256"]:
        raise ValueError("source Pairwise-K factor artifact hash mismatch")
    source_factors = load_file(str(source_factor_path), device="cpu")
    independent_key = source_factors["independent_key_projector"].float().contiguous()
    independent_query = source_factors["independent_query_projector"].float().contiguous()
    if tuple(independent_key.shape) != (NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, 64):
        raise ValueError("source independent Key control does not have rank 64")
    if tuple(independent_query.shape) != (NUM_LAYERS, NUM_QUERY_HEADS, HEAD_DIM, 64):
        raise ValueError("source independent Query control does not have rank 64")
    pair_key, pair_query = identity_pair_factors(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        num_query_heads=NUM_QUERY_HEADS,
        head_dim=HEAD_DIM,
    )
    factors = {
        "independent_key_projector": independent_key,
        "independent_query_projector": independent_query,
        "pair_key_projector": pair_key,
        "pair_query_projector": pair_query,
    }

    output_dir.mkdir(parents=True)
    factor_path = output_dir / "factors.safetensors"
    _atomic_safetensors(factor_path, factors)
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "geometry": {
            "layers": NUM_LAYERS,
            "adjacent_layer_pairs": NUM_LAYERS // 2,
            "query_heads": NUM_QUERY_HEADS,
            "physical_kv_heads": NUM_KV_HEADS,
            "query_heads_per_kv_head": NUM_QUERY_HEADS // NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "independent_rank_per_layer": 64,
            "pair_rank_per_layer_pair": PAIR_RANK,
            "average_pair_key_rank_per_layer": PAIR_RANK // 2,
            "pair_key_scalars_per_pair_token": PAIR_RANK,
            "logical_key_cache_ratio_vs_bf16_dense": 1.0,
            "logical_dense_v_total_kv_ratio_vs_bf16_dense": 1.0,
        },
        "calibration": {
            "used": False,
            "reason": "algebraic full-rank identity correctness control",
            "source_independent_control": str(source_result_path),
            "source_independent_control_sha256": _sha256(source_result_path),
        },
        "method": {
            "coordinate": "post-RoPE, after Qwen3 q_norm/k_norm",
            "pair_code": "exact concatenation [K_even, K_odd]",
            "even_layer_query": "exact zero-padded readout [Q_even, 0]",
            "odd_layer_query": "exact zero-padded readout [0, Q_odd]",
            "objective": "identity control; exact pre-softmax scores for every Q and K",
            "causal_mask": "preserved by runtime",
            "softmax": False,
            "value_aware": False,
        },
        "score_frobenius_metrics": {
            "pairwise_rank256": {
                "weighted_relative_squared_error": 0.0,
                "maximum_layer_group_relative_squared_error": 0.0,
            },
            "guarantee": "zero in exact arithmetic for arbitrary post-RoPE Q/K",
        },
        "artifacts": {
            "factors": {
                "file": factor_path.name,
                "sha256": _sha256(factor_path),
                "dtype": "float32 identity/zero or copied source float32",
                "tensors": {name: list(tensor.shape) for name, tensor in factors.items()},
            }
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
        },
    }
    _atomic_json(output_dir / "result.json", payload)
    print(f"[Pairwise-K identity] wrote {output_dir}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-factor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    build(_parser().parse_args())
