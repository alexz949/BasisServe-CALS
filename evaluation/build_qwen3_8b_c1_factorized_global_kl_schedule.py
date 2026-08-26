#!/usr/bin/env python3
"""Build a one-terminal-probe C1 layer-rank schedule for Qwen3-8B.

The local post-ALS reconstruction curves are read from the existing rank
banks.  One measured terminal-KL intervention per layer calibrates a scalar
downstream sensitivity.  The resulting separable costs are allocated under
the same exact average-rank budget as the source Global-KL experiment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
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

from basisserve.core.metric_rank_allocation import (  # noqa: E402
    MetricRankOption,
    allocate_metric_rank_exact,
)


FORMAT = "basisserve.c1.factorized_terminal_kl_schedule.v1"
SOURCE_FORMAT = "basisserve.qwen3_8b.gqa_c1.layer_global_kl_allocation.v1"
NUM_LAYERS = 36
NUM_KV_HEADS = 8
HEAD_DIM = 128


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def predict_factorized_costs(
    local_errors: Sequence[Mapping[int, float]],
    terminal_probe_deltas: Sequence[float],
    *,
    candidate_ranks: Sequence[int],
    anchor_rank: int,
    probe_rank: int,
    exponent: float = 1.0,
) -> tuple[tuple[dict[int, float], ...], tuple[float, ...]]:
    """Predict per-layer terminal costs from one non-anchor terminal probe."""

    ranks = tuple(sorted(map(int, candidate_ranks)))
    if not ranks or len(ranks) != len(set(ranks)):
        raise ValueError("candidate_ranks must be non-empty and unique")
    if anchor_rank not in ranks or probe_rank not in ranks:
        raise ValueError("anchor and probe ranks must be candidate ranks")
    if probe_rank == anchor_rank:
        raise ValueError("probe rank must differ from anchor rank")
    if not math.isfinite(exponent) or exponent <= 0:
        raise ValueError("exponent must be finite and positive")
    if len(local_errors) != len(terminal_probe_deltas):
        raise ValueError("local error and terminal probe layer counts differ")

    predicted: list[dict[int, float]] = []
    sensitivities: list[float] = []
    for layer, (curve, observed_delta) in enumerate(
        zip(local_errors, terminal_probe_deltas, strict=True)
    ):
        if set(map(int, curve)) != set(ranks):
            raise ValueError(f"layer {layer} local error curve is incomplete")
        checked = {int(rank): float(value) for rank, value in curve.items()}
        if any(not math.isfinite(value) or value < 0 for value in checked.values()):
            raise ValueError(f"layer {layer} local errors must be finite and nonnegative")
        delta = float(observed_delta)
        if not math.isfinite(delta):
            raise ValueError(f"layer {layer} terminal probe delta is non-finite")
        transformed = {
            rank: checked[rank] ** exponent
            for rank in ranks
        }
        denominator = transformed[probe_rank] - transformed[anchor_rank]
        if denominator == 0:
            raise ValueError(f"layer {layer} probe has zero local-error separation")
        sensitivity = delta / denominator
        if not math.isfinite(sensitivity) or sensitivity <= 0:
            raise ValueError(f"layer {layer} downstream sensitivity is not positive")
        sensitivities.append(sensitivity)
        predicted.append(
            {
                rank: sensitivity
                * (transformed[rank] - transformed[anchor_rank])
                for rank in ranks
            }
        )
    return tuple(predicted), tuple(sensitivities)


def allocate_layer_schedule(
    costs: Sequence[Mapping[int, float]],
    *,
    candidate_ranks: Sequence[int],
    anchor_rank: int,
) -> tuple[tuple[int, ...], float]:
    """Allocate one rank per layer at the exact uniform-anchor rank budget."""

    ranks = tuple(sorted(map(int, candidate_ranks)))
    options = []
    for layer, curve in enumerate(costs):
        options.append(
            tuple(
                MetricRankOption(
                    option_id=f"layer{layer:03d}_rank{rank}",
                    source_family="factorized_terminal_kl",
                    rank=rank,
                    scalar_cost=float(curve[rank]),
                    is_anchor=rank == anchor_rank,
                )
                for rank in ranks
            )
        )
    allocation = allocate_metric_rank_exact(
        options,
        total_rank_budget=len(options) * anchor_rank,
        anchor_rank=anchor_rank,
    )
    return (
        tuple(option.rank for option in allocation.selected_options),
        float(allocation.total_cost),
    )


def _local_error_curves(
    source: Mapping[str, Any],
    *,
    candidate_ranks: Sequence[int],
) -> tuple[dict[int, float], ...]:
    factor_sources = source["factor_sources"]
    curves = [dict() for _ in range(NUM_LAYERS)]
    for rank in candidate_ranks:
        if rank == HEAD_DIM:
            for curve in curves:
                curve[rank] = 0.0
            continue
        record = factor_sources.get(str(rank))
        if record is None:
            raise ValueError(f"missing local factor source for rank {rank}")
        directory = Path(record["path"]).expanduser().resolve()
        result_path = directory / "results.json"
        if _sha256(result_path) != record["results_sha256"]:
            raise ValueError(f"local factor result hash mismatch for rank {rank}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        rows = payload.get("records", ())
        if len(rows) != NUM_LAYERS:
            raise ValueError(f"rank {rank} local factor result is incomplete")
        for layer, row in enumerate(rows):
            configured_rank = int(row["fit_config"]["cache_rank_per_head"])
            if configured_rank != rank:
                raise ValueError(f"rank {rank} local factor result is inconsistent")
            curves[layer][rank] = float(row["fit"]["factor_dtype_relative_mse"])
    return tuple(curves)


def _probe_deltas(
    source: Mapping[str, Any],
    *,
    probe_rank: int,
) -> tuple[float, ...]:
    selected: dict[int, float] = {}
    for row in source["profile"]["records"]:
        if int(row["candidate_rank"]) != probe_rank:
            continue
        layer = int(row["layer"])
        if layer in selected:
            raise ValueError(f"duplicate probe record for layer {layer}")
        selected[layer] = float(row["terminal_kl_delta"]["mean"])
    if set(selected) != set(range(NUM_LAYERS)):
        raise ValueError("terminal probe records do not cover every layer")
    return tuple(selected[layer] for layer in range(NUM_LAYERS))


def _accounting(layer_ranks: Sequence[int], *, anchor_rank: int) -> dict[str, Any]:
    ranks = tuple(map(int, layer_ranks))
    histogram = Counter(ranks)
    source_rank_sum = NUM_KV_HEADS * sum(ranks)
    return {
        "layer_ranks": list(ranks),
        "layer_rank_sum": sum(ranks),
        "layer_rank_histogram": {
            str(rank): count for rank, count in sorted(histogram.items())
        },
        "source_rank_sum": source_rank_sum,
        "source_rank_histogram": {
            str(rank): NUM_KV_HEADS * count
            for rank, count in sorted(histogram.items())
        },
        "anchor_rank": anchor_rank,
        "changed_layers_from_anchor": sum(rank != anchor_rank for rank in ranks),
        "changed_sources_from_anchor": NUM_KV_HEADS
        * sum(rank != anchor_rank for rank in ranks),
        "dense_value_total_width": NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM,
        "dense_reduction": (NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM)
        / source_rank_sum,
        "rectangular_collective_width_by_layer": [
            NUM_KV_HEADS * rank for rank in ranks
        ],
        "rectangular_collective_total_width": source_rank_sum,
        "uniform_anchor_total_width": NUM_LAYERS * NUM_KV_HEADS * anchor_rank,
        "ragged_padding_overhead": 0.0,
    }


def build(args: argparse.Namespace) -> None:
    source_path = args.allocation_result.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("format") != SOURCE_FORMAT or source.get("status") != "complete":
        raise ValueError("source Global-KL allocation is incomplete or incompatible")
    candidate_ranks = tuple(map(int, source["selection"]["candidate_ranks"]))
    anchor_rank = int(source["profile"]["records"][0]["anchor_rank"])
    if int(source["selection"]["target_layer_rank_sum"]) != NUM_LAYERS * anchor_rank:
        raise ValueError("source Global-KL allocation has an unexpected rank budget")
    local_errors = _local_error_curves(source, candidate_ranks=candidate_ranks)
    terminal_deltas = _probe_deltas(source, probe_rank=args.probe_rank)
    costs, sensitivities = predict_factorized_costs(
        local_errors,
        terminal_deltas,
        candidate_ranks=candidate_ranks,
        anchor_rank=anchor_rank,
        probe_rank=args.probe_rank,
        exponent=args.exponent,
    )
    layer_ranks, predicted_cost = allocate_layer_schedule(
        costs,
        candidate_ranks=candidate_ranks,
        anchor_rank=anchor_rank,
    )
    schedule = [[rank] * NUM_KV_HEADS for rank in layer_ranks]
    exponent_label = str(args.exponent).replace(".", "p")
    schedule_name = f"factorized_a{exponent_label}_probe{args.probe_rank}"
    full_schedule = tuple(
        int(layer[0]) for layer in source["schedules"]["mean_dp"]["schedule"]
    )
    payload = {
        "format": FORMAT,
        "status": "complete",
        "schedule_name": schedule_name,
        "schedule": schedule,
        "accounting": _accounting(layer_ranks, anchor_rank=anchor_rank),
        "method": {
            "name": "one_probe_factorized_terminal_kl",
            "formula": (
                "delta_K_lr = s_l * (e_lr^alpha - e_l_anchor^alpha)"
            ),
            "local_error": "post-ALS factor-dtype fit relative MSE",
            "anchor_rank": anchor_rank,
            "probe_rank": args.probe_rank,
            "terminal_interventions_per_layer": 1,
            "exponent": args.exponent,
            "predicted_additive_cost": predicted_cost,
            "sensitivities": list(sensitivities),
            "predicted_costs": [
                {str(rank): curve[rank] for rank in candidate_ranks}
                for curve in costs
            ],
        },
        "diagnostics": {
            "full_mean_dp_exact_layer_matches": sum(
                left == right
                for left, right in zip(layer_ranks, full_schedule, strict=True)
            ),
            "full_mean_dp_mean_absolute_rank_difference": sum(
                abs(left - right)
                for left, right in zip(layer_ranks, full_schedule, strict=True)
            )
            / NUM_LAYERS,
            "full_mean_dp_layer_ranks": list(full_schedule),
        },
        "source": {
            "allocation_result": str(source_path),
            "allocation_result_sha256": _sha256(source_path),
            "profile_dataset": source["profile"]["dataset"],
            "profile_windows": source["profile"]["windows"],
            "profile_sequence_length": source["profile"]["sequence_length"],
        },
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output_path, payload)
    print(
        f"[Factorized Global-KL] schedule={schedule_name} "
        f"matches={payload['diagnostics']['full_mean_dp_exact_layer_matches']}/"
        f"{NUM_LAYERS} output={output_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-result", type=Path, required=True)
    parser.add_argument("--probe-rank", type=int, default=96)
    parser.add_argument("--exponent", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
