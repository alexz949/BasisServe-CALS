#!/usr/bin/env python3
"""Evaluate a K-only output-pullback router on exact 32K direct captures."""

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

from safetensors.torch import load_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.gqa_joint_routing_payload_s80_fisher import (  # noqa: E402
    prepare_softmax_fisher_routing,
    softmax_fisher_routing_diagnostics,
)
from scripts.build_qwen3_gqa_joint_routing_payload_s80 import (  # noqa: E402
    _load_c1_layer,
    _load_direct_layer,
)


FORMAT = "basisserve.qwen3.page_output_pullback_router_diagnostics.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation-direct-capture-dir", required=True)
    parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--c1-v", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--token-budgets", default="2048")
    parser.add_argument(
        "--work-dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _parse_layers(spec: str, total: int) -> tuple[int, ...]:
    if spec == "all":
        return tuple(range(total))
    selected: set[int] = set()
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            selected.update(range(int(left), int(right) + 1))
        else:
            selected.add(int(item))
    return tuple(sorted(selected))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    mean_keys = (
        "raw_score_nmse",
        "softmax_fisher_nmse",
        "physical_page_recall_mean",
        "attention_mass_recall_mean",
        "selected_token_fraction_mean",
        "exact_refined_output_relative_mse",
    )
    minimum_keys = (
        "physical_page_recall_minimum",
        "attention_mass_recall_minimum",
    )
    result: dict[str, float | int] = {"layers": len(rows)}
    for key in mean_keys:
        result[key] = sum(float(row[key]) for row in rows) / len(rows)
    for key in minimum_keys:
        result[key] = min(float(row[key]) for row in rows)
    result["physical_token_budget_mean"] = (
        float(result["selected_token_fraction_mean"]) * int(rows[0]["tokens"])
    )
    return result


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    started = time.monotonic()
    model_root = Path(args.model).expanduser().resolve()
    capture_root = Path(args.validation_direct_capture_dir).expanduser().resolve()
    router_root = Path(args.router_checkpoint).expanduser().resolve()
    c1_root = Path(args.c1_v).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    device = torch.device(args.work_device)
    dtype = torch.float64 if args.work_dtype == "float64" else torch.float32
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    hidden = int(config["hidden_size"])
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim", hidden // query_heads))
    total_layers = int(config["num_hidden_layers"])
    layers = _parse_layers(args.layers, total_layers)
    budgets = tuple(
        sorted({int(item) for item in args.token_budgets.split(",") if item})
    )
    mapping = torch.arange(query_heads, device=device) // (query_heads // kv_heads)
    capture_manifest = json.loads(
        (capture_root / "manifest.json").read_text(encoding="utf-8")
    )
    router_manifest = json.loads(
        (router_root / "result.json").read_text(encoding="utf-8")
    )
    c1_manifest = json.loads(
        (c1_root / "results.json").read_text(encoding="utf-8")
    )
    rows_by_budget: dict[int, list[dict[str, Any]]] = {
        budget: [] for budget in budgets
    }

    for ordinal, layer in enumerate(layers, start=1):
        print(
            f"[Page output diagnostics] layer={layer} ({ordinal}/{len(layers)})",
            flush=True,
        )
        direct = _load_direct_layer(capture_root, capture_manifest, layer)
        routing = prepare_softmax_fisher_routing(
            direct,
            head_to_kv_group=mapping,
            value_dim=head_dim,
            key_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        router_artifact = router_manifest["artifacts"][str(layer)]
        router = load_file(
            str(router_root / router_artifact["file"]),
            device="cpu",
        )
        key_encoder = router["key_routing_encoders"].to(
            device=device,
            dtype=dtype,
        )
        query_factors = router["query_routing_factors"].to(
            device=device,
            dtype=dtype,
        )
        value_encoder, output_decoder = _load_c1_layer(
            c1_root,
            c1_manifest,
            layer,
        )
        value_encoder = value_encoder.to(device=device, dtype=dtype)
        output_decoder = output_decoder.to(device=device, dtype=dtype)
        selection_encoder = torch.cat(
            (
                torch.zeros(
                    kv_heads,
                    head_dim,
                    key_encoder.shape[-1],
                    device=device,
                    dtype=dtype,
                ),
                key_encoder,
            ),
            dim=1,
        )
        payload_encoder = torch.cat(
            (
                value_encoder,
                torch.zeros(
                    kv_heads,
                    head_dim,
                    value_encoder.shape[-1],
                    device=device,
                    dtype=dtype,
                ),
            ),
            dim=1,
        )
        payload_routing = payload_encoder[..., : key_encoder.shape[-1]]
        payload_only = payload_encoder[..., key_encoder.shape[-1] :]
        for budget in budgets:
            row = softmax_fisher_routing_diagnostics(
                routing,
                selection_routing_encoders=selection_encoder,
                payload_routing_encoders=payload_routing,
                payload_only_encoders=payload_only,
                routing_query_factors=query_factors,
                payload_decoders=output_decoder,
                page_size=args.page_size,
                exact_token_budget=budget,
            )
            row["layer"] = layer
            rows_by_budget[budget].append(row)

    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "elapsed_seconds": time.monotonic() - started,
        "configuration": {
            "key_convention": "post_rope",
            "routing_coordinates": "K_only",
            "routing_rank": int(router_manifest["geometry"]["routing_rank"]),
            "value_rank": 80,
            "page_size": args.page_size,
            "token_budgets": list(budgets),
            "layer_coverage": list(layers),
        },
        "sources": {
            "router_checkpoint": str(router_root),
            "c1_v_checkpoint": str(c1_root),
            "validation_direct_capture": str(capture_root),
        },
        "budgets": {
            str(budget): {
                "aggregate": _aggregate(rows),
                "layers": rows,
            }
            for budget, rows in rows_by_budget.items()
        },
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(f"[Page output diagnostics] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
