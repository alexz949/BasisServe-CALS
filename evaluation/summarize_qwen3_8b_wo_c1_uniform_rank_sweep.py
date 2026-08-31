#!/usr/bin/env python3
"""Summarize Qwen3-8B Wo-only uniform-rank quality into a Pareto table."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping


FORMAT = "basisserve.qwen3_8b.wo_c1_uniform_rank_sweep.v1"
QUALITY_FORMAT = "basisserve.qwen3_8b.wo_c1_uniform_rank_quality.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_ranks(raw: str) -> tuple[int, ...]:
    ranks = tuple(
        sorted({int(piece.strip()) for piece in raw.split(",") if piece.strip()})
    )
    if not ranks or min(ranks) <= 0:
        raise ValueError("expected ranks must be positive")
    return ranks


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _arm_value(payload: Mapping[str, Any], arm: str, key: str) -> float:
    return float(payload["arms"][arm][key])


def relative_reduction(value: float, baseline: float) -> float:
    """Return the bounded reduction from a strictly positive baseline."""

    if baseline <= 0.0:
        raise ValueError("relative-reduction baseline must be positive")
    return 1.0 - value / baseline


def quality_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract one validated rank point from a uniform-quality result."""

    if payload.get("format") != QUALITY_FORMAT or payload.get("status") != "complete":
        raise ValueError("uniform-rank quality result is incomplete")
    protocol = payload["protocol"]
    rank = int(protocol["source_rank"])
    arms = payload["arms"]
    if tuple(protocol["arms"]) != (
        "dense",
        "joint_c1",
        "independent_local_c1",
        "wire_matched_lr_allreduce",
        "capacity_matched_lr_allreduce",
    ):
        raise ValueError("uniform quality arms differ")
    c1_bytes = float(
        arms["joint_c1"]["collective_accounting"][
            "ideal_ring_bytes_per_rank_per_activation_row"
        ]
    )
    local_bytes = float(
        arms["independent_local_c1"]["collective_accounting"][
            "ideal_ring_bytes_per_rank_per_activation_row"
        ]
    )
    wire_bytes = float(
        arms["wire_matched_lr_allreduce"]["collective_accounting"][
            "ideal_ring_bytes_per_rank_per_activation_row"
        ]
    )
    if not c1_bytes == local_bytes == wire_bytes:
        raise ValueError("headline arms are not wire matched")
    dense_ppl = float(arms["dense"]["wikitext2"]["ppl"])
    c1_ppl = float(arms["joint_c1"]["wikitext2"]["ppl"])
    local_ppl = float(arms["independent_local_c1"]["wikitext2"]["ppl"])
    wire_ppl = float(arms["wire_matched_lr_allreduce"]["wikitext2"]["ppl"])
    capacity_ppl = float(arms["capacity_matched_lr_allreduce"]["wikitext2"]["ppl"])
    c1_mse = _arm_value(payload, "joint_c1", "mean_heldout_output_relative_mse")
    local_mse = _arm_value(
        payload, "independent_local_c1", "mean_heldout_output_relative_mse"
    )
    wire_mse = _arm_value(
        payload, "wire_matched_lr_allreduce", "mean_heldout_output_relative_mse"
    )
    capacity_mse = _arm_value(
        payload,
        "capacity_matched_lr_allreduce",
        "mean_heldout_output_relative_mse",
    )
    return {
        "source_rank": rank,
        "source_width": int(protocol["source_width"]),
        "retained_ratio": float(protocol["retained_ratio"]),
        "wire_matched_lr_rank": int(
            arms["wire_matched_lr_allreduce"]["collective_accounting"]["shared_rank"]
        ),
        "capacity_matched_lr_rank": int(
            arms["capacity_matched_lr_allreduce"]["collective_accounting"][
                "shared_rank"
            ]
        ),
        "headline_ring_bytes_per_rank_per_row": c1_bytes,
        "dense_ppl": dense_ppl,
        "joint_c1_ppl": c1_ppl,
        "independent_local_c1_ppl": local_ppl,
        "wire_lr_ppl": wire_ppl,
        "capacity_lr_ppl": capacity_ppl,
        "joint_c1_ppl_change_vs_dense": c1_ppl / dense_ppl - 1.0,
        "wire_lr_ppl_change_vs_dense": wire_ppl / dense_ppl - 1.0,
        "joint_c1_ppl_advantage_vs_wire_lr": wire_ppl / c1_ppl - 1.0,
        "joint_c1_ppl_advantage_vs_independent_local": local_ppl / c1_ppl - 1.0,
        "joint_c1_heldout_mse": c1_mse,
        "independent_local_c1_heldout_mse": local_mse,
        "wire_lr_heldout_mse": wire_mse,
        "capacity_lr_heldout_mse": capacity_mse,
        "joint_c1_mse_reduction_vs_wire_lr": relative_reduction(c1_mse, wire_mse),
        "joint_c1_mse_reduction_vs_independent_local": relative_reduction(
            c1_mse, local_mse
        ),
    }


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3-8B Wo-only uniform-rank Pareto sweep",
        "",
        (
            "All headline comparisons use uniform source ranks, identical C4 "
            "calibration covariances, dense V/KV cache, and equal ideal ring bytes "
            "between private C1-AllGather and strong LR-AllReduce."
        ),
        "",
        "| C1 rank | Retained | Wire LR rank | Bytes/row/rank | Joint C1 PPL | Local C1 PPL | Wire LR PPL | Capacity LR PPL |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["ranks"]:
        lines.append(
            f"| {row['source_rank']} | {100.0 * row['retained_ratio']:.1f}% | "
            f"{row['wire_matched_lr_rank']} | {row['headline_ring_bytes_per_rank_per_row']:.0f} | "
            f"{row['joint_c1_ppl']:.8f} | {row['independent_local_c1_ppl']:.8f} | "
            f"{row['wire_lr_ppl']:.8f} | {row['capacity_lr_ppl']:.8f} |"
        )
    lines.extend(
        [
            "",
            "| C1 rank | Joint C1 MSE | Local C1 MSE | Wire LR MSE | Capacity LR MSE | C1 PPL advantage vs wire LR |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["ranks"]:
        lines.append(
            f"| {row['source_rank']} | {row['joint_c1_heldout_mse']:.8g} | "
            f"{row['independent_local_c1_heldout_mse']:.8g} | "
            f"{row['wire_lr_heldout_mse']:.8g} | {row['capacity_lr_heldout_mse']:.8g} | "
            f"{100.0 * row['joint_c1_ppl_advantage_vs_wire_lr']:+.3f}% |"
        )
    lines.extend(
        [
            "",
            (
                "Dense PPL is rerun inside every rank job. Its observed range is "
                f"`{payload['audit']['dense_ppl_minimum']:.8f}`–"
                f"`{payload['audit']['dense_ppl_maximum']:.8f}`."
            ),
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
    parser.add_argument("--quality-results", nargs="+", required=True)
    parser.add_argument("--expected-ranks", default="512,640,768,896,1024")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_ranks = _parse_ranks(args.expected_ranks)
    inputs = []
    rows = []
    model_hash: str | None = None
    protocol_key: tuple[Any, ...] | None = None
    for raw_path in args.quality_results:
        path = Path(raw_path).expanduser().resolve()
        payload = _load_json(path)
        row = quality_row(payload)
        current_model_hash = str(payload["model"]["config_sha256"])
        current_protocol_key = (
            payload["protocol"]["dataset"],
            payload["protocol"]["split"],
            int(payload["protocol"]["sequence_length"]),
            int(payload["protocol"]["batch_size"]),
            payload["protocol"]["model_dtype"],
            payload["protocol"]["attn_implementation"],
        )
        if model_hash is None:
            model_hash = current_model_hash
            protocol_key = current_protocol_key
        elif current_model_hash != model_hash or current_protocol_key != protocol_key:
            raise ValueError("rank results use different model or quality protocol")
        inputs.append({"path": str(path), "sha256": _sha256(path)})
        rows.append(row)
    rows.sort(key=lambda row: int(row["source_rank"]))
    observed_ranks = tuple(int(row["source_rank"]) for row in rows)
    if observed_ranks != expected_ranks:
        raise ValueError(
            f"observed ranks {observed_ranks} differ from expected {expected_ranks}"
        )
    dense_values = [float(row["dense_ppl"]) for row in rows]
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "status": "complete",
        "command": shlex.join((sys.executable, *sys.argv)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_config_sha256": model_hash,
        "quality_protocol": {
            "dataset": protocol_key[0],
            "split": protocol_key[1],
            "sequence_length": protocol_key[2],
            "batch_size": protocol_key[3],
            "model_dtype": protocol_key[4],
            "attn_implementation": protocol_key[5],
        },
        "inputs": inputs,
        "ranks": rows,
        "audit": {
            "expected_ranks": list(expected_ranks),
            "dense_ppl_minimum": min(dense_values),
            "dense_ppl_maximum": max(dense_values),
            "dense_ppl_range": max(dense_values) - min(dense_values),
            "wire_matching_passed_all_ranks": True,
        },
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    _atomic_text(
        output_dir / "results.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(output_dir / "summary.md", _summary_markdown(payload))
    print(f"[Uniform sweep] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
