#!/usr/bin/env python3
"""Build P0/P1 C1 schedules from an existing Qwen3-8B Full oracle.

P0 reuses every recorded layer-by-rank hard-token NLL intervention.  P1
reuses the rank-96 probe and the held-out local reconstruction curves to
build matched one-probe KL and NLL schedules.  No model forward is required.
"""

from __future__ import annotations

import argparse
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
    predict_factorized_costs,
)


READOUTS = ("terminal_kl", "nll")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def full_readout_costs(
    source: Mapping[str, Any],
    *,
    readout: str,
    candidate_ranks: Sequence[int],
    anchor_rank: int,
) -> tuple[dict[int, float], ...]:
    """Return the measured layer-by-rank cost table for one readout."""

    assert readout in READOUTS
    indexed = {
        (int(row["layer"]), int(row["candidate_rank"])): row
        for row in source["profile"]["records"]
    }
    delta_key = f"{readout}_delta"
    return tuple(
        {
            rank: (
                0.0
                if rank == anchor_rank
                else float(indexed[(layer, rank)][delta_key]["mean"])
            )
            for rank in candidate_ranks
        }
        for layer in range(NUM_LAYERS)
    )


def probe_readout_costs(
    source: Mapping[str, Any],
    local_errors: Sequence[Mapping[int, float]],
    *,
    readout: str,
    candidate_ranks: Sequence[int],
    anchor_rank: int,
    probe_rank: int,
    exponent: float,
) -> tuple[tuple[dict[int, float], ...], tuple[float, ...]]:
    """Predict all rank costs from one recorded terminal readout probe."""

    assert readout in READOUTS
    indexed = {
        int(row["layer"]): row
        for row in source["profile"]["records"]
        if int(row["candidate_rank"]) == probe_rank
    }
    assert set(indexed) == set(range(NUM_LAYERS))
    deltas = tuple(
        float(indexed[layer][f"{readout}_delta"]["mean"])
        for layer in range(NUM_LAYERS)
    )
    return predict_factorized_costs(
        local_errors,
        deltas,
        candidate_ranks=candidate_ranks,
        anchor_rank=anchor_rank,
        probe_rank=probe_rank,
        exponent=exponent,
    )


def _readout_label(readout: str) -> str:
    return "kl" if readout == "terminal_kl" else readout


def _schedule_payload(
    *,
    source_path: Path,
    source: Mapping[str, Any],
    schedule_name: str,
    layer_ranks: Sequence[int],
    costs: Sequence[Mapping[int, float]],
    predicted_cost: float,
    method: Mapping[str, Any],
    reference_schedule: Sequence[int],
) -> dict[str, Any]:
    anchor_rank = int(method["anchor_rank"])
    schedule = [[int(rank)] * NUM_KV_HEADS for rank in layer_ranks]
    return {
        "format": FORMAT,
        "status": "complete",
        "schedule_name": schedule_name,
        "schedule": schedule,
        "accounting": _accounting(layer_ranks, anchor_rank=anchor_rank),
        "method": {
            **dict(method),
            "predicted_additive_cost": float(predicted_cost),
            "predicted_costs": [
                {str(rank): float(value) for rank, value in curve.items()}
                for curve in costs
            ],
        },
        "diagnostics": {
            "reference_full_readout_exact_layer_matches": sum(
                left == right
                for left, right in zip(layer_ranks, reference_schedule, strict=True)
            ),
            "reference_full_readout_mean_absolute_rank_difference": sum(
                abs(left - right)
                for left, right in zip(layer_ranks, reference_schedule, strict=True)
            )
            / NUM_LAYERS,
            "reference_full_readout_layer_ranks": list(reference_schedule),
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


def build(args: argparse.Namespace) -> None:
    source_path = args.allocation_result.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert source.get("format") == SOURCE_FORMAT
    assert source.get("status") == "complete"
    candidate_ranks = tuple(map(int, source["selection"]["candidate_ranks"]))
    anchor_rank = int(source["profile"]["records"][0]["anchor_rank"])
    assert source["selection"]["target_layer_rank_sum"] == NUM_LAYERS * anchor_rank
    assert args.probe_rank in candidate_ranks and args.probe_rank != anchor_rank
    assert math.isfinite(args.exponent) and args.exponent > 0

    local_errors = _local_error_curves(
        source,
        candidate_ranks=candidate_ranks,
        error_split=args.local_error_split,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    exponent_label = format(args.exponent, "g").replace(".", "p")

    full_schedules: dict[str, tuple[tuple[int, ...], tuple[dict[int, float], ...]]] = {}
    for readout in READOUTS:
        costs = full_readout_costs(
            source,
            readout=readout,
            candidate_ranks=candidate_ranks,
            anchor_rank=anchor_rank,
        )
        layer_ranks, predicted_cost = allocate_layer_schedule(
            costs,
            candidate_ranks=candidate_ranks,
            anchor_rank=anchor_rank,
            target_average_rank=anchor_rank,
        )
        full_schedules[readout] = (layer_ranks, costs)
        if readout != "nll":
            continue
        schedule_name = "full_global_nll"
        payload = _schedule_payload(
            source_path=source_path,
            source=source,
            schedule_name=schedule_name,
            layer_ranks=layer_ranks,
            costs=costs,
            predicted_cost=predicted_cost,
            method={
                "name": schedule_name,
                "formula": "C_lr = NLL(M_layer<-rank) - NLL(M_uniform_anchor)",
                "readout": readout,
                "allocation_mode": "full_layer_by_rank_oracle",
                "anchor_rank": anchor_rank,
                "terminal_interventions_per_layer": len(candidate_ranks) - 1,
            },
            reference_schedule=layer_ranks,
        )
        _atomic_json(output_dir / f"{schedule_name}_schedule.json", payload)

    for readout in READOUTS:
        costs, sensitivities = probe_readout_costs(
            source,
            local_errors,
            readout=readout,
            candidate_ranks=candidate_ranks,
            anchor_rank=anchor_rank,
            probe_rank=args.probe_rank,
            exponent=args.exponent,
        )
        layer_ranks, predicted_cost = allocate_layer_schedule(
            costs,
            candidate_ranks=candidate_ranks,
            anchor_rank=anchor_rank,
            target_average_rank=anchor_rank,
        )
        label = _readout_label(readout)
        schedule_name = (
            f"factorized_{label}_a{exponent_label}_probe{args.probe_rank}"
        )
        reference_schedule = full_schedules[readout][0]
        payload = _schedule_payload(
            source_path=source_path,
            source=source,
            schedule_name=schedule_name,
            layer_ranks=layer_ranks,
            costs=costs,
            predicted_cost=predicted_cost,
            method={
                "name": "one_probe_factorized_terminal_readout",
                "formula": (
                    "delta_readout_lr = s_l * "
                    "(e_lr^alpha - e_l_anchor^alpha)"
                ),
                "readout": readout,
                "allocation_mode": "factorized_one_probe",
                "local_error": (
                    f"post-ALS factor-dtype {args.local_error_split} relative MSE"
                ),
                "local_error_split": args.local_error_split,
                "anchor_rank": anchor_rank,
                "probe_rank": args.probe_rank,
                "terminal_interventions_per_layer": 1,
                "exponent": args.exponent,
                "sensitivities": list(sensitivities),
            },
            reference_schedule=reference_schedule,
        )
        _atomic_json(output_dir / f"{schedule_name}_schedule.json", payload)

    print(
        f"[P0/P1 terminal readouts] wrote three schedules to {output_dir}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-rank", type=int, default=96)
    parser.add_argument("--exponent", type=float, default=1.25)
    parser.add_argument(
        "--local-error-split",
        choices=("heldout", "fit"),
        default="heldout",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
