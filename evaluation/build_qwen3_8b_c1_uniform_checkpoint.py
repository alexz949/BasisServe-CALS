#!/usr/bin/env python3
"""Build a manifest-only Qwen3-8B uniform C1-R80 checkpoint."""

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
from evaluation.build_qwen3_8b_iclr_v_checkpoint import FORMAT  # noqa: E402


RUN_ID = "Q3-8B-C1U-R80"
RANK = 80


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def _tensor_shapes(path: Path) -> dict[str, list[int]] | None:
    if not _check(path.is_file(), f"missing factor: {path}"):
        return None
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if not _check(
            keys == {"value_coordinate_encoders", "head_output_decoders"},
            f"unexpected factor tensors: {path}",
        ):
            return None
        return {key: list(handle.get_slice(key).get_shape()) for key in keys}


def build(args: argparse.Namespace) -> int:
    model_support.activate_model_profile("qwen3_8b")
    model_path = Path(args.model).expanduser().resolve()
    factor_bank_dir = Path(args.factor_bank_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    result_path = factor_bank_dir / "results.json"
    manifest_path = output_dir / "manifest.json"
    if not all(
        (
            _check(result_path.is_file(), f"missing factor-bank result: {result_path}"),
            _check(not manifest_path.exists(), f"checkpoint manifest already exists: {manifest_path}"),
        )
    ):
        return 2

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    model_support._validate_config(config)
    model_metadata = model_support._model_metadata(model_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    fit_config = result.get("fit_config", {})
    records = result.get("records", ())
    valid = all(
        (
            _check(result.get("format") == "basisserve.qwen3_8b.gqa_c1_joint.v1", "factor-bank format mismatch"),
            _check(result.get("status") == "complete", "factor bank is incomplete"),
            _check(fit_config.get("model_config_sha256") == model_metadata["config_sha256"], "factor-bank model hash mismatch"),
            _check(int(fit_config.get("cache_rank_per_head", 0)) == RANK, "factor bank is not uniform R80"),
            _check(int(fit_config.get("encoder_sweeps", 0)) == 6, "factor bank is not sweep 6"),
            _check(
                fit_config.get("checkpoint_policy")
                == "fixed decoder-refitted endpoint after encoder sweep 6",
                "factor-bank checkpoint policy mismatch",
            ),
            _check(fit_config.get("decoder_objective") == "full_layer", "factor-bank decoder objective mismatch"),
            _check(len(records) == model_support.NUM_LAYERS, "factor-bank layer count mismatch"),
        )
    )
    if not valid:
        return 2

    layers = []
    total_bytes = 0
    expected_shapes = {
        "value_coordinate_encoders": [model_support.NUM_KV_HEADS, model_support.HEAD_DIM, RANK],
        "head_output_decoders": [model_support.NUM_QUERY_HEADS, RANK, model_support.HIDDEN_SIZE],
    }
    for layer_index, record in enumerate(records):
        artifact = record.get("artifact", {})
        factor_path = factor_bank_dir / str(artifact.get("file", ""))
        shapes = _tensor_shapes(factor_path)
        row_valid = all(
            (
                _check(int(record.get("layer", -1)) == layer_index, f"layer order mismatch at {layer_index}"),
                _check(shapes == expected_shapes, f"factor shape mismatch at layer {layer_index}"),
                _check(model_support._sha256(factor_path) == artifact.get("sha256"), f"factor hash mismatch at layer {layer_index}"),
                _check(int(record.get("checkpoint", {}).get("sweep", -1)) == 6, f"factor is not sweep 6 at layer {layer_index}"),
            )
        )
        if not row_valid:
            return 2
        factor_bytes = factor_path.stat().st_size
        total_bytes += factor_bytes
        layers.append(
            {
                "layer": layer_index,
                "ranks": [RANK] * model_support.NUM_KV_HEADS,
                "file": os.path.relpath(factor_path, output_dir),
                "sha256": artifact["sha256"],
                "bytes": factor_bytes,
                "tensor_shapes": shapes,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rank_sum = model_support.NUM_LAYERS * model_support.NUM_KV_HEADS * RANK
    dense_rank_sum = (
        model_support.NUM_LAYERS
        * model_support.NUM_KV_HEADS
        * model_support.HEAD_DIM
    )
    manifest: Mapping[str, Any] = {
        "format": FORMAT,
        "status": "complete",
        "run_id": RUN_ID,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_metadata,
        "compression": {
            "target": "value_cache_only",
            "key_cache": "dense",
            "value_cache": "low_rank",
            "projection": "joint_v_o",
            "method": "c1-uniform",
            "method_label": "C1 Uniform",
            "allocation": "uniform_per_layer_per_head",
            "equivalent_rank_target": RANK,
            "layer_ranks": [[RANK] * model_support.NUM_KV_HEADS for _ in range(model_support.NUM_LAYERS)],
            "rank_sum_across_layers": rank_sum,
            "dense_rank_sum_across_layers": dense_rank_sum,
            "realized_retained_v_ratio": rank_sum / dense_rank_sum,
            "realized_v_cache_compression_ratio": 1.0 - rank_sum / dense_rank_sum,
            "num_query_heads": model_support.NUM_QUERY_HEADS,
            "num_physical_kv_heads": model_support.NUM_KV_HEADS,
            "head_dim": model_support.HEAD_DIM,
            "encoder_initialization": fit_config["encoder_initialization"],
            "encoder_sweeps": 6,
            "checkpoint_policy": fit_config["checkpoint_policy"],
            "decoder_objective": "full_layer",
            "stored_factor_dtype": f"torch.{fit_config['factor_dtype']}",
        },
        "artifact": {
            "file": os.path.relpath(result_path, output_dir),
            "sha256": model_support._sha256(result_path),
            "bytes": result_path.stat().st_size,
            "selected_factor_count": model_support.NUM_LAYERS,
            "selected_factor_bytes": total_bytes,
        },
        "layers": layers,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
        },
    }
    model_support._atomic_json(manifest_path, manifest)
    print(f"[Result] checkpoint={output_dir} run_id={RUN_ID}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--factor-bank-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(build(parse_args()))
