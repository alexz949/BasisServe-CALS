#!/usr/bin/env python3
"""Build a Qwen3-8B C1 rank schedule from PaLU layer Fisher importance.

The allocation keeps the C1 factor banks, held-out local reconstruction
curves, candidate ranks, and exact rank budget fixed.  Only the downstream
layer-importance signal is replaced by PaLU's V-projection Fisher scalar.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.build_qwen3_8b_c1_two_sided_factorized_kl_schedule import (  # noqa: E402
    FORMAT,
    NUM_KV_HEADS,
    NUM_LAYERS,
    SOURCE_FORMAT,
    _accounting,
    _local_error_curves,
    _sha256,
    allocate_layer_schedule,
)


FISHER_FORMAT = "basisserve.gqa.palu_m_v_only_fisher_stats.v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def palu_layer_importances(
    fisher: Mapping[str, Any],
    *,
    normalize: bool = True,
) -> tuple[float, ...]:
    """Return ordered PaLU V-projection Fisher scalars."""

    if fisher.get("format") != FISHER_FORMAT:
        raise ValueError("incompatible PaLU Fisher result format")
    if fisher.get("status") != "complete":
        raise ValueError("PaLU Fisher result is incomplete")
    if fisher["fisher"].get("target") != "v_proj_only":
        raise ValueError("PaLU Fisher result does not target V projections")
    scalars = fisher["fisher"]["scalars"]
    values = tuple(
        float(scalars[f"model.layers.{layer}.self_attn.v_proj"])
        for layer in range(NUM_LAYERS)
    )
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("PaLU Fisher importance must be finite and positive")
    if not normalize:
        return values
    mean = sum(values) / len(values)
    return tuple(value / mean for value in values)


def fisher_weighted_c1_costs(
    local_errors: Sequence[Mapping[int, float]],
    importances: Sequence[float],
    *,
    candidate_ranks: Sequence[int],
    anchor_rank: int,
    exponent: float,
) -> tuple[dict[int, float], ...]:
    """Combine PaLU Fisher importance with held-out C1 error curves."""

    if len(local_errors) != NUM_LAYERS or len(importances) != NUM_LAYERS:
        raise ValueError(f"expected {NUM_LAYERS} layers")
    if not math.isfinite(exponent) or exponent <= 0:
        raise ValueError("exponent must be finite and positive")
    ranks = tuple(map(int, candidate_ranks))
    if anchor_rank not in ranks:
        raise ValueError("anchor rank is missing from candidate ranks")
    costs = []
    for layer, (curve, importance) in enumerate(
        zip(local_errors, importances, strict=True)
    ):
        weight = float(importance)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"layer {layer} Fisher importance is invalid")
        checked = {rank: float(curve[rank]) for rank in ranks}
        if any(not math.isfinite(value) or value < 0 for value in checked.values()):
            raise ValueError(f"layer {layer} C1 errors must be finite and nonnegative")
        anchor_error = checked[anchor_rank] ** exponent
        costs.append(
            {
                rank: weight * (checked[rank] ** exponent - anchor_error)
                for rank in ranks
            }
        )
    return tuple(costs)


def build(args: argparse.Namespace) -> None:
    allocation_path = args.allocation_result.expanduser().resolve()
    fisher_path = args.fisher_result.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    source = json.loads(allocation_path.read_text(encoding="utf-8"))
    fisher = json.loads(fisher_path.read_text(encoding="utf-8"))
    if source.get("format") != SOURCE_FORMAT or source.get("status") != "complete":
        raise ValueError("incompatible or incomplete C1 allocation result")
    c1_config_sha256 = source.get("model_config_sha256")
    if c1_config_sha256 != fisher["model"]["config_sha256"]:
        raise ValueError("C1 and Fisher artifacts belong to different model configs")

    candidate_ranks = tuple(map(int, source["selection"]["candidate_ranks"]))
    anchor_rank = int(source["profile"]["records"][0]["anchor_rank"])
    local_errors = _local_error_curves(
        source,
        candidate_ranks=candidate_ranks,
        error_split=args.local_error_split,
    )
    importances = palu_layer_importances(fisher)
    costs = fisher_weighted_c1_costs(
        local_errors,
        importances,
        candidate_ranks=candidate_ranks,
        anchor_rank=anchor_rank,
        exponent=args.exponent,
    )
    layer_ranks, predicted_cost = allocate_layer_schedule(
        costs,
        candidate_ranks=candidate_ranks,
        anchor_rank=anchor_rank,
    )
    schedule_name = f"palu_fisher_c1_a{format(args.exponent, 'g').replace('.', 'p')}"
    schedule = [[rank] * NUM_KV_HEADS for rank in layer_ranks]
    full_kl_ranks = tuple(
        int(layer[0]) for layer in source["selection"]["selected_schedule"]
    )
    payload = {
        "format": FORMAT,
        "status": "complete",
        "schedule_name": schedule_name,
        "schedule": schedule,
        "accounting": _accounting(layer_ranks, anchor_rank=anchor_rank),
        "method": {
            "name": "palu_fisher_weighted_c1_local_error",
            "formula": "C_lr = normalized_I_l * (e_lr^alpha - e_l_anchor^alpha)",
            "fisher_target": "v_proj_only",
            "fisher_aggregation": fisher["fisher"]["aggregation"],
            "local_error": (
                f"post-ALS factor-dtype {args.local_error_split} relative MSE"
            ),
            "local_error_split": args.local_error_split,
            "candidate_ranks": list(candidate_ranks),
            "anchor_rank": anchor_rank,
            "exponent": args.exponent,
            "normalized_fisher_importances": list(importances),
            "predicted_additive_cost": predicted_cost,
            "predicted_costs": [
                {str(rank): value for rank, value in curve.items()}
                for curve in costs
            ],
        },
        "diagnostics": {
            "full_global_kl_exact_layer_matches": sum(
                left == right
                for left, right in zip(layer_ranks, full_kl_ranks, strict=True)
            ),
            "full_global_kl_mean_absolute_rank_difference": sum(
                abs(left - right)
                for left, right in zip(layer_ranks, full_kl_ranks, strict=True)
            )
            / NUM_LAYERS,
            "fisher_importance_min": min(importances),
            "fisher_importance_max": max(importances),
            "rank_histogram": {
                str(rank): count for rank, count in sorted(Counter(layer_ranks).items())
            },
        },
        "source": {
            "allocation_result": str(allocation_path),
            "allocation_result_sha256": _sha256(allocation_path),
            "palu_fisher_result": str(fisher_path),
            "palu_fisher_result_sha256": _sha256(fisher_path),
            "fisher_dataset": fisher["fisher"]["dataset"],
            "fisher_samples": fisher["fisher"]["samples"],
            "fisher_sequence_length": fisher["fisher"]["sequence_length"],
        },
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, payload)
    print(
        f"[PaLU Fisher -> C1] schedule={list(layer_ranks)} "
        f"rank_sum={sum(layer_ranks)} output={output_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-result", type=Path, required=True)
    parser.add_argument("--fisher-result", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--exponent", type=float, default=1.25)
    parser.add_argument(
        "--local-error-split",
        choices=("heldout", "fit"),
        default="heldout",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
