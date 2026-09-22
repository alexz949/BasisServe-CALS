#!/usr/bin/env python3
"""Create paper-table and plotting data for the Wo-C1 collective sweep."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping


FORMAT = "basisserve.qwen3_8b.wo_c1_ag_ar_rank_sweep.v1"
EVAL_FORMAT = "basisserve.qwen3_8b.wo_c1_ag_ar_mcq.v1"
PPL_FORMAT = "basisserve.qwen3_8b.wo_c1_uniform_rank_sweep.v1"
EXPECTED_ARMS = ("c1_allgather", "lr_allreduce")


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def _load(path: Path) -> dict[str, Any] | None:
    if not _check(path.is_file(), f"missing input: {path}"):
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _markdown(payload: Mapping[str, Any]) -> str:
    dense = payload["dense"]
    lines = [
        "# Qwen3-8B Wo-only C1 AllGather vs LR-AllReduce rank sweep",
        "",
        (
            "All headline pairs have equal ideal TP4 ring traffic. Quality is "
            "measured after folding each factorized map into an equivalent BF16 "
            "`o_proj`; V and the KV cache remain dense."
        ),
        "",
        "| AG rank | Retained | AR rank | Bytes/row/rank | AG PPL | AR PPL | AG MCQ | AR MCQ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["ranks"]:
        lines.append(
            f"| {row['source_rank']} | {100 * row['retained_ratio']:.2f}% | "
            f"{row['wire_allreduce_rank']} | {row['ring_bytes_per_row_per_rank']:.0f} | "
            f"{row['c1_allgather']['ppl']:.6f} | {row['lr_allreduce']['ppl']:.6f} | "
            f"{100 * row['c1_allgather']['mcq_average']:.3f} | "
            f"{100 * row['lr_allreduce']['mcq_average']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Dense reference: PPL `{dense['ppl']:.8f}`, MCQ `{100 * dense['mcq_average']:.3f}`.",
            "",
            "MCQ is the unweighted mean of ARC-Easy, ARC-Challenge, HellaSwag, "
            "PIQA, WinoGrande, BoolQ, and OpenBookQA (zero-shot).",
            "",
            "Plot-ready long-form data: `figure_data.csv`.",
            "",
            "## Command",
            "",
            "```bash",
            payload["command"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--existing-ppl", type=Path, required=True)
    parser.add_argument("--dense-quality", type=Path, required=True)
    parser.add_argument("--expected-ranks", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize(args: argparse.Namespace) -> int:
    ranks = tuple(int(part) for part in args.expected_ranks.split(",") if part)
    if not _check(len(ranks) == len(set(ranks)) and ranks == tuple(sorted(ranks)), "expected ranks must be sorted and unique"):
        return 2
    prior = _load(args.existing_ppl.expanduser().resolve())
    dense_quality = _load(args.dense_quality.expanduser().resolve())
    if prior is None or dense_quality is None:
        return 2
    if not _check(prior.get("format") == PPL_FORMAT and prior.get("status") == "complete", "existing PPL sweep is incompatible"):
        return 2
    dense_metrics = dense_quality.get("metrics", {})
    dense_ppl_values = [float(row["dense_ppl"]) for row in prior["ranks"]]
    dense = {
        "ppl": dense_ppl_values[0] if dense_ppl_values else 0.0,
        "mcq_average": float(dense_metrics.get("average_accuracy", 0.0)),
        "task_accuracy": dense_metrics.get("task_accuracy", []),
    }
    if not _check(
        dense["ppl"] > 0
        and dense["mcq_average"] > 0
        and max(dense_ppl_values) == min(dense_ppl_values),
        "dense reference is incomplete or inconsistent",
    ):
        return 2

    prior_by_rank = {int(row["source_rank"]): row for row in prior["ranks"]}
    evaluations: dict[tuple[int, str], dict[str, Any]] = {}
    for path in args.evaluations:
        result = _load(path.expanduser().resolve())
        if result is None:
            return 2
        if not _check(result.get("format") == EVAL_FORMAT and result.get("status") == "complete", f"incompatible evaluation: {path}"):
            return 2
        protocol = result["protocol"]
        rank = int(protocol["source_rank"])
        arm = str(result["arm"])
        key = (rank, arm)
        if not _check(key not in evaluations, f"duplicate evaluation rank/arm: {key}"):
            return 2
        if not _check(result.get("mcq") is not None and protocol.get("limit") is None, f"MCQ is incomplete or limited: {path}"):
            return 2
        evaluations[key] = result

    rows = []
    for rank in ranks:
        arm_results = {arm: evaluations.get((rank, arm)) for arm in EXPECTED_ARMS}
        if not _check(all(arm_results.values()), f"missing MCQ arm at rank {rank}"):
            return 2
        ag = arm_results["c1_allgather"] or {}
        ar = arm_results["lr_allreduce"] or {}
        ag_protocol = ag["protocol"]
        ar_protocol = ar["protocol"]
        if not _check(
            float(ag_protocol["ring_bytes_per_row_per_rank"])
            == float(ar_protocol["ring_bytes_per_row_per_rank"]),
            f"wire mismatch at rank {rank}",
        ):
            return 2
        prior_row = prior_by_rank.get(rank)
        ag_ppl = (
            float(prior_row["joint_c1_ppl"])
            if prior_row is not None
            else float((ag.get("wikitext2") or {}).get("ppl", 0.0))
        )
        ar_ppl = (
            float(prior_row["wire_lr_ppl"])
            if prior_row is not None
            else float((ar.get("wikitext2") or {}).get("ppl", 0.0))
        )
        if not _check(ag_ppl > 0 and ar_ppl > 0, f"missing PPL at rank {rank}"):
            return 2
        rows.append(
            {
                "source_rank": rank,
                "retained_ratio": float(ag_protocol["retained_ratio"]),
                "wire_allreduce_rank": int(ag_protocol["wire_allreduce_rank"]),
                "ring_bytes_per_row_per_rank": float(ag_protocol["ring_bytes_per_row_per_rank"]),
                "c1_allgather": {
                    "ppl": ag_ppl,
                    "mcq_average": float(ag["mcq"]["average_accuracy"]),
                    "task_accuracy": ag["mcq"]["task_accuracy"],
                },
                "lr_allreduce": {
                    "ppl": ar_ppl,
                    "mcq_average": float(ar["mcq"]["average_accuracy"]),
                    "task_accuracy": ar["mcq"]["task_accuracy"],
                },
            }
        )

    payload = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "model": "Qwen3-8B-Base",
        "dense": dense,
        "ranks": rows,
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "results.json", payload)
    (output_dir / "summary.md").write_text(_markdown(payload), encoding="utf-8")

    task_names = [row["task"] for row in rows[0]["c1_allgather"]["task_accuracy"]]
    with (output_dir / "figure_data.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method",
            "allgather_source_rank",
            "method_rank",
            "retained_ratio",
            "ring_bytes_per_row_per_rank",
            "ppl",
            "mcq_average",
            "dense_ppl",
            "dense_mcq_average",
            *task_names,
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for arm in EXPECTED_ARMS:
                metrics = row[arm]
                task_values = {
                    task["task"]: float(task["value"])
                    for task in metrics["task_accuracy"]
                }
                writer.writerow(
                    {
                        "method": arm,
                        "allgather_source_rank": row["source_rank"],
                        "method_rank": row["source_rank"] if arm == "c1_allgather" else row["wire_allreduce_rank"],
                        "retained_ratio": row["retained_ratio"],
                        "ring_bytes_per_row_per_rank": row["ring_bytes_per_row_per_rank"],
                        "ppl": metrics["ppl"],
                        "mcq_average": metrics["mcq_average"],
                        "dense_ppl": dense["ppl"],
                        "dense_mcq_average": dense["mcq_average"],
                        **task_values,
                    }
                )
    print(f"[Complete] {output_dir}", flush=True)
    return 0


def main() -> None:
    status = summarize(parse_args())
    if status:
        sys.exit(status)


if __name__ == "__main__":
    main()
