#!/usr/bin/env python3
"""Finalize a two-sided-KL C1 allocation as an ICLR matrix checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping

from safetensors import safe_open
import torch
from transformers import AutoConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import build_llama31_8b_palu_m_checkpoint as model_support  # noqa: E402


PROFILES = {
    "qwen3_8b": {
        "checkpoint_format": "basisserve.qwen3_8b.iclr_v_factors.v1",
        "run_prefix": "Q3-8B-C1-R",
    },
    "llama31_8b": {
        "checkpoint_format": "basisserve.llama31_8b.iclr_v_factors.v1",
        "run_prefix": "L31-8B-C1-R",
    },
    "llama2_7b": {
        "checkpoint_format": "basisserve.llama2_7b.iclr_v_factors.v1",
        "run_prefix": "L2-7B-C1-R",
    },
    "qwen3_32b": {
        "checkpoint_format": "basisserve.qwen3_32b.iclr_v_factors.v1",
        "run_prefix": "Q3-32B-C1-R",
    },
    "llama31_70b": {
        "checkpoint_format": "basisserve.llama31_70b.iclr_v_factors.v1",
        "run_prefix": "L31-70B-C1-R",
    },
}


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def _factor_metadata(path: Path) -> dict[str, Any] | None:
    if not _check(path.is_file(), f"missing selected factor: {path}"):
        return None
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if not _check(
            keys
            == {
                "value_coordinate_encoders",
                "head_output_decoders",
                "source_ranks",
            },
            f"unexpected selected-factor tensors: {path}",
        ):
            return None
        shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "tensor_shapes": shapes,
    }


def build(args: argparse.Namespace) -> int:
    profile = PROFILES[args.profile]
    model_support.activate_model_profile(args.profile)
    model_path = Path(args.model).expanduser().resolve()
    checkpoint_dir = Path(args.allocation_dir).expanduser().resolve()
    result_path = checkpoint_dir / "result.json"
    manifest_path = checkpoint_dir / "manifest.json"
    if not all(
        (
            _check(result_path.is_file(), f"missing allocation result: {result_path}"),
            _check(not manifest_path.exists(), f"checkpoint manifest already exists: {manifest_path}"),
        )
    ):
        return 2

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    model_support._validate_config(config)
    model_metadata = model_support._model_metadata(model_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selection = result.get("selection", {})
    factor_stage = result.get("profile", {}).get("factor_stage", {})
    schedule = selection.get("selected_schedule", ())
    geometry = result.get("geometry", {})
    target_rank = int(selection.get("target_average_rank", -1))
    run_id = f"{profile['run_prefix']}{target_rank}"
    num_layers = model_support.NUM_LAYERS
    num_kv_heads = model_support.NUM_KV_HEADS
    head_dim = model_support.HEAD_DIM
    flat_ranks = [int(rank) for layer in schedule for rank in layer]
    valid = all(
        (
            _check(result.get("status") == "complete", "allocation is incomplete"),
            _check(result.get("model_config_sha256") == model_metadata["config_sha256"], "allocation model hash mismatch"),
            _check(result.get("model") == str(model_path), "allocation model path mismatch"),
            _check(selection.get("selected_candidate") == "two_sided_factorized_kl", "checkpoint must export the two-sided-KL schedule"),
            _check(selection.get("forced_selected_candidate") == "two_sided_factorized_kl", "two-sided-KL export must be explicit"),
            _check(target_rank in (64, 80, 96), "target average rank must be 64, 80, or 96"),
            _check(len(schedule) == num_layers, "allocation layer count mismatch"),
            _check(all(len(layer) == num_kv_heads for layer in schedule), "allocation KV-head count mismatch"),
            _check(sum(flat_ranks) == num_layers * num_kv_heads * target_rank, "allocation rank budget mismatch"),
            _check(int(selection.get("target_source_rank_sum", -1)) == sum(flat_ranks), "recorded rank budget mismatch"),
            _check(factor_stage.get("encoder_sweeps") == 6, "C1 factors are not sweep-6 endpoints"),
            _check(
                factor_stage.get("checkpoint_policy")
                == "fixed decoder-refitted endpoint after encoder sweep 6",
                "C1 checkpoint policy mismatch",
            ),
            _check(factor_stage.get("decoder_objective") == "full_layer", "C1 decoder objective mismatch"),
            _check(int(geometry.get("layers", -1)) == num_layers, "allocation geometry layer mismatch"),
            _check(int(geometry.get("physical_kv_heads", -1)) == num_kv_heads, "allocation geometry KV mismatch"),
            _check(int(geometry.get("head_dim", -1)) == head_dim, "allocation head dimension mismatch"),
        )
    )
    if not valid:
        return 2

    selected_artifacts = result.get("selected_artifacts", {})
    if not _check(set(map(int, selected_artifacts)) == set(range(num_layers)), "selected factor set is incomplete"):
        return 2
    layer_records = []
    for layer_index in range(num_layers):
        record = selected_artifacts[str(layer_index)]
        factor_path = checkpoint_dir / str(record.get("file", ""))
        metadata = _factor_metadata(factor_path)
        if metadata is None:
            return 2
        if not _check(model_support._sha256(factor_path) == record.get("sha256"), f"factor hash mismatch at layer {layer_index}"):
            return 2
        layer_records.append(
            {
                "layer": layer_index,
                "ranks": list(map(int, schedule[layer_index])),
                "file": str(factor_path.relative_to(checkpoint_dir)),
                "sha256": record["sha256"],
                "bytes": metadata["bytes"],
                "tensor_shapes": metadata["tensor_shapes"],
            }
        )

    dense_rank_sum = num_layers * num_kv_heads * head_dim
    rank_sum = sum(flat_ranks)
    manifest: Mapping[str, Any] = {
        "format": profile["checkpoint_format"],
        "status": "complete",
        "run_id": run_id,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_metadata,
        "compression": {
            "target": "value_cache_only",
            "key_cache": "dense",
            "value_cache": "low_rank",
            "projection": "joint_v_o",
            "method": "c1-two-sided-kl",
            "method_label": "C1 + Two-Sided KL",
            "allocation": "two_sided_factorized_terminal_kl_alpha1",
            "equivalent_rank_target": target_rank,
            "profiling_anchor_rank": int(selection["profiling_anchor_rank"]),
            "candidate_ranks": list(map(int, selection["candidate_ranks"])),
            "layer_ranks": [list(map(int, layer)) for layer in schedule],
            "rank_sum_across_layers": rank_sum,
            "dense_rank_sum_across_layers": dense_rank_sum,
            "realized_retained_v_ratio": rank_sum / dense_rank_sum,
            "realized_v_cache_compression_ratio": 1.0 - rank_sum / dense_rank_sum,
            "num_query_heads": model_support.NUM_QUERY_HEADS,
            "num_physical_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "encoder_initialization": factor_stage["encoder_initialization"],
            "encoder_sweeps": 6,
            "checkpoint_policy": factor_stage["checkpoint_policy"],
            "decoder_objective": "full_layer",
            "stored_factor_dtype": result.get("numerics", {}).get("deployed_factor_dtype"),
        },
        "artifact": {
            "file": result_path.name,
            "sha256": model_support._sha256(result_path),
            "bytes": result_path.stat().st_size,
            "selected_factor_count": num_layers,
            "selected_factor_bytes": sum(row["bytes"] for row in layer_records),
        },
        "layers": layer_records,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
        },
    }
    model_support._atomic_json(manifest_path, manifest)
    print(f"[Result] checkpoint={checkpoint_dir} run_id={run_id}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--allocation-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(build(parse_args()))
