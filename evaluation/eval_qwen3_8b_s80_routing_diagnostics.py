#!/usr/bin/env python3
"""Compare Store80/Route32 checkpoints on one direct routing capture."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_joint_routing_payload_s80_qwen3 import (  # noqa: E402
    FoldedS80Factors,
    load_s80_factor_bank,
)
from basisserve.core.gqa_joint_routing_payload_s80 import S80Layout  # noqa: E402
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (  # noqa: E402
    prepare_softmax_fisher_routing,
    softmax_fisher_routing_diagnostics,
)
from scripts.build_qwen3_gqa_joint_routing_payload_s80 import (  # noqa: E402
    _SafetensorWeightReader,
    _load_direct_layer,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation-direct-capture-dir", required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--token-budgets", default="1024")
    parser.add_argument(
        "--work-dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


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


def _arm_specs(values: list[str]) -> dict[str, Path]:
    return {
        item.split("=", 1)[0]: Path(item.split("=", 1)[1]).expanduser().resolve()
        for item in values
    }


def _token_budgets(spec: str) -> tuple[int, ...]:
    return tuple(sorted({int(item) for item in spec.split(",") if item.strip()}))


def _recover_factors(
    folded: FoldedS80Factors,
    dense_v_weight: torch.Tensor,
    *,
    layout: S80Layout,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    dense_v = dense_v_weight.to(device=device, dtype=dtype).reshape(
        layout.num_key_value_heads,
        layout.value_dim,
        layout.hidden_size,
    )
    folded_v = folded.v_joint_proj_weight.to(device=device, dtype=dtype).reshape(
        layout.num_key_value_heads,
        layout.joint_rank,
        layout.hidden_size,
    )
    value_encoders = torch.linalg.solve(
        torch.bmm(dense_v, dense_v.mT),
        torch.bmm(dense_v, folded_v.mT),
    )
    reconstructed = torch.bmm(value_encoders.mT, dense_v)
    recovery_relative_mse = float(
        (reconstructed - folded_v).square().sum() / folded_v.square().sum()
    )
    key_encoders = folded.k_joint_encoder.to(device=device, dtype=dtype)
    joint_encoders = torch.cat((value_encoders, key_encoders), dim=1)
    routing_payload_encoders = joint_encoders[..., : layout.routing_rank]
    payload_only_encoders = joint_encoders[..., layout.routing_rank :]
    routing_query_factors = folded.routing_query_factor.to(device=device, dtype=dtype)
    payload_decoders = (
        folded.o_decoder_weight.to(device=device, dtype=dtype)
        .reshape(
            layout.hidden_size,
            layout.num_attention_heads,
            layout.joint_rank,
        )
        .permute(1, 2, 0)
        .contiguous()
    )
    return (
        routing_payload_encoders,
        payload_only_encoders,
        routing_query_factors,
        payload_decoders,
        recovery_relative_mse,
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    mean_keys = (
        "raw_score_nmse",
        "softmax_fisher_nmse",
        "physical_page_recall_mean",
        "attention_mass_recall_mean",
        "selected_token_fraction_mean",
        "exact_refined_output_relative_mse",
        "folded_v_recovery_relative_mse",
    )
    minimum_keys = (
        "physical_page_recall_minimum",
        "attention_mass_recall_minimum",
    )
    aggregate: dict[str, float | int] = {"layers": len(rows)}
    for key in mean_keys:
        aggregate[key] = sum(float(row[key]) for row in rows) / len(rows)
    for key in minimum_keys:
        aggregate[key] = min(float(row[key]) for row in rows)
    aggregate["physical_token_budget_mean"] = (
        float(aggregate["selected_token_fraction_mean"]) * int(rows[0]["tokens"])
    )
    return aggregate


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    started = time.monotonic()
    model_root = Path(args.model).expanduser().resolve()
    capture_root = Path(args.validation_direct_capture_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    device = torch.device(args.work_device)
    dtype = torch.float64 if args.work_dtype == "float64" else torch.float32
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    hidden = int(config["hidden_size"])
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim", hidden // query_heads))
    total_layers = int(config["num_hidden_layers"])
    layout = S80Layout(
        hidden_size=hidden,
        num_attention_heads=query_heads,
        num_key_value_heads=kv_heads,
        value_dim=head_dim,
        key_dim=head_dim,
        joint_rank=80,
        routing_rank=32,
    )
    layers = _parse_layers(args.layers, total_layers)
    token_budgets = _token_budgets(args.token_budgets)
    arm_roots = _arm_specs(args.arm)
    arm_manifests: dict[str, dict[str, Any]] = {}
    arm_factors: dict[str, dict[int, tuple[FoldedS80Factors, dict[str, float]]]] = {}
    for name, root in arm_roots.items():
        manifest, factors = load_s80_factor_bank(root)
        arm_manifests[name] = manifest
        arm_factors[name] = factors
    capture_manifest = json.loads(
        (capture_root / "manifest.json").read_text(encoding="utf-8")
    )
    weights = _SafetensorWeightReader(model_root)
    rows_by_arm: dict[str, dict[int, list[dict[str, Any]]]] = {
        name: {budget: [] for budget in token_budgets} for name in arm_roots
    }
    mapping = layout.head_to_kv_group(device=device)

    for ordinal, layer_index in enumerate(layers, start=1):
        layer_started = time.monotonic()
        direct = _load_direct_layer(capture_root, capture_manifest, layer_index)
        routing = prepare_softmax_fisher_routing(
            direct,
            head_to_kv_group=mapping,
            value_dim=layout.value_dim,
            key_dim=layout.key_dim,
            device=device,
            dtype=dtype,
        )
        dense_v = weights.tensor(
            f"model.layers.{layer_index}.self_attn.v_proj.weight"
        )
        printed: list[str] = []
        for name in arm_roots:
            recovered = _recover_factors(
                arm_factors[name][layer_index][0],
                dense_v,
                layout=layout,
                device=device,
                dtype=dtype,
            )
            arm_printed: list[str] = []
            for budget in token_budgets:
                diagnostics = softmax_fisher_routing_diagnostics(
                    routing,
                    selection_routing_encoders=recovered[0],
                    payload_routing_encoders=recovered[0],
                    payload_only_encoders=recovered[1],
                    routing_query_factors=recovered[2],
                    payload_decoders=recovered[3],
                    page_size=args.page_size,
                    exact_token_budget=budget,
                )
                row = {
                    "layer": layer_index,
                    **diagnostics,
                    "folded_v_recovery_relative_mse": recovered[4],
                }
                rows_by_arm[name][budget].append(row)
                arm_printed.append(
                    f"B{budget}:phys="
                    f"{int(round(float(diagnostics['selected_token_fraction_mean']) * diagnostics['tokens']))} "
                    f"mass={100 * float(diagnostics['attention_mass_recall_mean']):.2f}%"
                )
            printed.append(f"{name}[{', '.join(arm_printed)}]")
        print(
            f"[S80 routing diagnostics] layer={layer_index} "
            f"({ordinal}/{len(layers)}) {' | '.join(printed)} "
            f"seconds={time.monotonic() - layer_started:.2f}",
            flush=True,
        )
        del routing, direct, dense_v
        torch.cuda.empty_cache()

    result = {
        "format": "basisserve.qwen3_8b.s80_routing_diagnostics.v1",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "model": {
            "path": str(model_root),
            "config_sha256": _sha256(model_root / "config.json"),
        },
        "validation_capture": {
            "path": str(capture_root),
            "manifest_sha256": _sha256(capture_root / "manifest.json"),
        },
        "protocol": {
            "layers": list(layers),
            "page_size": args.page_size,
            "nominal_token_budgets_per_query_head": list(token_budgets),
            "gqa_group_union": True,
            "work_dtype": str(dtype).removeprefix("torch."),
            "work_device": str(device),
        },
        "arms": {
            name: {
                "checkpoint": str(arm_roots[name]),
                "manifest_sha256": _sha256(arm_roots[name] / "manifest.json"),
                "checkpoint_routing_metric": arm_manifests[name][
                    "solver_configuration"
                ].get("routing_metric", "raw_qk"),
                "budgets": {
                    str(budget): {
                        "aggregate": _aggregate(rows),
                        "layers": rows,
                    }
                    for budget, rows in rows_by_budget.items()
                },
            }
            for name, rows_by_budget in rows_by_arm.items()
        },
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, rows_by_budget in rows_by_arm.items():
        for budget, rows in rows_by_budget.items():
            aggregate = _aggregate(rows)
            print(
                f"[S80 routing diagnostics] aggregate arm={name} "
                f"nominal_budget={budget} "
                f"physical_budget={float(aggregate['physical_token_budget_mean']):.2f} "
                f"page={100 * float(aggregate['physical_page_recall_mean']):.4f}% "
                f"mass={100 * float(aggregate['attention_mass_recall_mean']):.4f}% "
                f"refined={100 * float(aggregate['exact_refined_output_relative_mse']):.5f}%",
                flush=True,
            )
    print(f"[S80 routing diagnostics] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
