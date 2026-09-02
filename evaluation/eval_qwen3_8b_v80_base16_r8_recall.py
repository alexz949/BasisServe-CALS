#!/usr/bin/env python3
"""Evaluate fixed-sink Base16 and Base16+R8 routing recall."""

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

from safetensors.torch import load_file
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (  # noqa: E402
    _load_direct,
    _parse_ints,
)
from evaluation.fit_qwen3_8b_v80_base16_r8_nonsink_page32 import (  # noqa: E402
    BASE_RANK,
    _evaluate_fresh,
    _load_base_maps,
)


FORMAT = "basisserve.qwen3_8b.v80_base16_r8_recall.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fresh-direct-dir", required=True)
    parser.add_argument("--c1-v80", required=True)
    parser.add_argument("--routing-factors", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--pinned-prefix-pages", type=int, default=1)
    parser.add_argument("--physical-token-budget", type=int, default=4096)
    parser.add_argument("--yarn-factor", type=float, required=True)
    parser.add_argument("--original-max-position-embeddings", type=int, default=32768)
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(32 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _factor_file(root: Path, layer: int) -> Path:
    matches = sorted(root.glob(f"shard_*/layer_{layer:03d}.safetensors"))
    assert len(matches) == 1
    return matches[0]


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ("b16_r0", "b16_r8"):
        rows = [record["metrics"][arm] for record in records]
        result[arm] = {
            key: (
                min(float(row[key]) for row in rows)
                if key.endswith("minimum")
                else sum(float(row[key]) for row in rows) / len(rows)
            )
            for key in rows[0]
        }
    return result


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    model_root = Path(args.model).expanduser().resolve()
    capture_root = Path(args.fresh_direct_dir).expanduser().resolve()
    c1_root = Path(args.c1_v80).expanduser().resolve()
    routing_root = Path(args.routing_factors).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layers = _parse_ints(args.layers)
    device = torch.device(args.work_device)
    capture_manifest_path = capture_root / "manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    sequence = int(capture_manifest["calibration"]["sequence_length"])
    page_budget = args.physical_token_budget // args.page_size
    config = AutoConfig.from_pretrained(str(model_root), local_files_only=True)
    rope_theta = float(config.rope_parameters["rope_theta"])
    config.max_position_embeddings = round(
        args.original_max_position_embeddings * args.yarn_factor
    )
    config.rope_parameters = {
        "rope_type": "yarn",
        "factor": args.yarn_factor,
        "original_max_position_embeddings": args.original_max_position_embeddings,
        "rope_theta": rope_theta,
    }
    rotary = Qwen3RotaryEmbedding(config, device=device)
    positions = torch.arange(sequence, device=device).unsqueeze(0)
    cos, sin = rotary(
        torch.empty(1, device=device, dtype=torch.float32),
        positions,
    )
    assert capture_manifest["calibration"]["rope_parameters"] == config.rope_parameters
    c1_manifest_path = c1_root / "results.json"
    c1_manifest = json.loads(c1_manifest_path.read_text(encoding="utf-8"))
    records = []

    for ordinal, layer in enumerate(layers, start=1):
        layer_started = time.monotonic()
        print(f"[64K recall] layer={layer} ({ordinal}/{len(layers)})", flush=True)
        queries, rows = _load_direct(capture_root, capture_manifest, layer)
        c1_artifact = c1_manifest["artifacts"][str(layer)]
        c1_tensors = load_file(str(c1_root / c1_artifact["file"]), device="cpu")
        route_path = _factor_file(routing_root, layer)
        route_tensors = load_file(str(route_path), device="cpu")
        factor_bank = {
            (BASE_RANK, 8): (
                route_tensors["residual_encoder_b16_r8"],
                route_tensors["residual_query_b16_r8"],
            )
        }
        metrics = _evaluate_fresh(
            queries,
            rows,
            value_encoder=c1_tensors["value_coordinate_encoders"],
            output_decoder=c1_tensors["head_output_decoders"],
            base_maps=_load_base_maps(routing_root, layer),
            factor_bank=factor_bank,
            cos=cos,
            sin=sin,
            page_size=args.page_size,
            page_budget=page_budget,
            pinned_prefix_pages=args.pinned_prefix_pages,
            device=device,
        )
        records.append(
            {
                "layer": layer,
                "routing_factor": str(route_path),
                "routing_factor_sha256": _sha256(route_path),
                "metrics": metrics,
                "elapsed_seconds": time.monotonic() - layer_started,
            }
        )
        print(
            "    Base16 mass="
            f"{100.0 * metrics['b16_r0']['attention_mass_recall_mean']:.4f}% "
            "Base16+R8 mass="
            f"{100.0 * metrics['b16_r8']['attention_mass_recall_mean']:.4f}%",
            flush=True,
        )

    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "model": str(model_root),
            "capture": str(capture_root),
            "sequence_length": sequence,
            "query_policy": "last_token_full_prefix",
            "payload": "fixed C1-V80 ALS5",
            "routing": "fixed Page0 plus group-max Base16 or Base16+R8",
            "page_size": args.page_size,
            "pinned_prefix_pages": args.pinned_prefix_pages,
            "physical_token_budget_per_kv_group": args.physical_token_budget,
            "routed_pages_per_kv_group": page_budget - args.pinned_prefix_pages,
            "base_rank": BASE_RANK,
            "residual_rank": 8,
            "rope_parameters": config.rope_parameters,
            "max_position_embeddings": config.max_position_embeddings,
            "position_regime": "static YaRN",
        },
        "layers": records,
        "aggregate": _aggregate(records),
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
        },
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(f"[64K recall] complete output={output_path}", flush=True)


if __name__ == "__main__":
    main()
