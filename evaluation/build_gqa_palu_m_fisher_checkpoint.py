#!/usr/bin/env python3
"""Build a V-only PaLU-M checkpoint from an official Fisher-uniform schedule."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import torch
from torch import Tensor
from transformers import AutoConfig

from evaluation import build_llama31_8b_palu_m_checkpoint as builder
from evaluation.collect_gqa_palu_fisher import FORMAT as FISHER_FORMAT


def _load_fisher(
    path: Path,
    model_metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[list[int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != FISHER_FORMAT or payload.get("status") != "complete":
        raise ValueError("incompatible or incomplete Fisher result")
    if payload["model"] != model_metadata:
        raise ValueError("Fisher result belongs to another base-model snapshot")
    layer_ranks = [
        [int(rank) for rank in ranks]
        for ranks in payload["allocation"]["layer_ranks"]
    ]
    if len(layer_ranks) != builder.NUM_LAYERS:
        raise ValueError("Fisher schedule does not cover every layer")
    for layer_index, ranks in enumerate(layer_ranks):
        if len(ranks) != builder.NUM_KV_HEADS:
            raise ValueError(f"layer {layer_index} has the wrong group count")
        if len(set(ranks)) != 1 or min(ranks) <= 0 or max(ranks) > builder.HEAD_DIM:
            raise ValueError(f"layer {layer_index} has invalid Fisher ranks: {ranks}")
    return payload, layer_ranks


@torch.no_grad()
def build(args: argparse.Namespace) -> None:
    builder.activate_model_profile(args.profile)
    torch.set_num_threads(args.torch_num_threads)
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    builder._validate_config(config)
    model_metadata = builder._model_metadata(model_path)
    fisher_path = Path(args.fisher_result).expanduser().resolve()
    fisher, layer_ranks = _load_fisher(fisher_path, model_metadata)
    whitening_dir = Path(args.whitening_dir).expanduser().resolve()
    whitening, whitening_manifest = builder._load_whitening(
        whitening_dir, model_metadata
    )
    fisher_samples = int(fisher["fisher"]["samples"])
    fisher_sequence_length = int(fisher["fisher"]["sequence_length"])
    if (
        int(whitening_manifest["samples"]) != fisher_samples
        or int(whitening_manifest["sequence_length"]) != fisher_sequence_length
    ):
        raise ValueError("Fisher and whitening calibration shapes do not match")
    if (
        whitening_manifest["windows"]["sha256"]
        != fisher["calibration_windows"]["sha256"]
    ):
        raise ValueError("Fisher and whitening must use the same calibration windows")

    factor_payload: dict[str, Tensor] = {}
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for layer_index, ranks in enumerate(layer_ranks):
        layer_started = time.perf_counter()
        tensor_name = f"model.layers.{layer_index}.self_attn.v_proj.weight"
        dense = builder._load_indexed_tensor(model_path, tensor_name)
        expected_shape = (
            builder.NUM_KV_HEADS * builder.HEAD_DIM,
            builder.HIDDEN_SIZE,
        )
        if tuple(dense.shape) != expected_shape:
            raise ValueError(f"unexpected v_proj shape at layer {layer_index}: {dense.shape}")
        writer, decoder, diagnostics = builder.factorize_v_projection(
            dense,
            whitening[layer_index],
            ranks=ranks,
            output_dtype=torch.bfloat16,
        )
        factor_payload[f"layers.{layer_index}.v_writer.weight"] = writer
        factor_payload[f"layers.{layer_index}.v_decoder.weight"] = decoder
        record = {
            "layer": layer_index,
            "ranks": ranks,
            "source_tensor": tensor_name,
            "source_dtype": str(dense.dtype),
            "writer_shape": list(writer.shape),
            "decoder_shape": list(decoder.shape),
            "factor_dtype": str(writer.dtype),
            **diagnostics,
            "elapsed_seconds": time.perf_counter() - layer_started,
        }
        records.append(record)
        print(
            f"[PaLU Fisher] layer={layer_index}/{builder.NUM_LAYERS - 1} "
            f"rank={ranks[0]} weighted_error="
            f"{diagnostics['relative_activation_weighted_error']:.6f}",
            flush=True,
        )

    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "palu_m_v_factors.safetensors"
    builder._atomic_safetensors(artifact_path, factor_payload)
    rank_sum = sum(sum(ranks) for ranks in layer_ranks)
    total_rank = builder.NUM_LAYERS * builder.NUM_KV_HEADS * builder.HEAD_DIM
    checkpoint_format = builder.CHECKPOINT_FORMAT.replace(
        ".palu_m_v_only.v1", ".palu_m_v_only_fisher.v1"
    )
    manifest = {
        "format": checkpoint_format,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_metadata,
        "compression": {
            "target": "value_cache_only",
            "key_cache": "dense",
            "method": "PaLU M-LRD activation-aware whitened SVD",
            "allocation": "official_palu_fisher_uniform_adapted_to_v_only",
            "rank_block_size": 32,
            "num_query_heads": builder.NUM_QUERY_HEADS,
            "num_physical_kv_heads": builder.NUM_KV_HEADS,
            "head_dim": builder.HEAD_DIM,
            "groups": builder.NUM_KV_HEADS,
            "group_width": builder.HEAD_DIM,
            "layer_ranks": layer_ranks,
            "rank_sum_across_layers": rank_sum,
            "dense_rank_sum_across_layers": total_rank,
            "retained_v_ratio": rank_sum / total_rank,
            "v_cache_compression_ratio": 1.0 - rank_sum / total_rank,
            "requested_v_cache_compression_ratio": fisher["allocation"][
                "requested_v_cache_compression_ratio"
            ],
            "factorization_work_device": "cpu",
            "factorization_work_dtype": "torch.float64",
            "stored_factor_dtype": "torch.bfloat16",
        },
        "fisher": {
            "result": str(fisher_path),
            "result_sha256": builder._sha256(fisher_path),
            "format": fisher["format"],
            "official_palu_commit": fisher["official_palu_commit"],
            "calibration": fisher["fisher"],
            "allocation": fisher["allocation"],
        },
        "calibration": {
            "dataset": "allenai/c4",
            "samples": fisher_samples,
            "sequence_length": fisher_sequence_length,
            "whitening_manifest": str(whitening_dir / "manifest.json"),
            "whitening_manifest_sha256": builder._sha256(
                whitening_dir / "manifest.json"
            ),
            "whitening_artifact_sha256": whitening_manifest["artifact"]["sha256"],
            "windows": whitening_manifest["windows"],
        },
        "artifact": {
            "file": artifact_path.name,
            "sha256": builder._sha256(artifact_path),
            "tensor_count": len(factor_payload),
            "bytes": artifact_path.stat().st_size,
        },
        "layers": records,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    builder._atomic_json(output_dir / "manifest.json", manifest)
    print(
        f"[Result] compression={1.0 - rank_sum / total_rank:.6f} "
        f"artifact={artifact_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(builder.MODEL_PROFILES), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fisher-result", required=True)
    parser.add_argument("--whitening-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--torch-num-threads", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
