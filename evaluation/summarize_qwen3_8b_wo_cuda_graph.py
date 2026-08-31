#!/usr/bin/env python3
"""Merge the five matched Qwen3-8B Wo-only CUDA Graph arms."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from basisserve.core.qwen3_8b_wo_tp4 import ARMS
from evaluation.benchmark_qwen3_8b_wo_cuda_graph import FORMAT


SUMMARY_FORMAT = "basisserve.qwen3_8b.wo_cuda_graph_five_arm_summary.tp4.v1"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT or payload.get("status") != "complete":
        raise ValueError(f"incomplete or incompatible result: {path}")
    return payload


def validate_matched_protocol(payloads: Sequence[Mapping[str, Any]]) -> None:
    """Reject summaries that mix hardware, factors, or benchmark settings."""

    if len(payloads) != len(ARMS) or {row["arm"] for row in payloads} != set(ARMS):
        raise ValueError("summary requires exactly one result for every Wo arm")
    reference = payloads[0]
    required_equal = (
        ("model", "config_sha256"),
        ("phase1", "sha256"),
        ("protocol", "configurations"),
        ("protocol", "warmup"),
        ("protocol", "repeats"),
        ("protocol", "dtype"),
        ("environment", "gpu"),
        ("environment", "torch"),
        ("environment", "cuda"),
        ("environment", "nccl"),
    )
    for payload in payloads[1:]:
        for first, second in required_equal:
            if payload[first][second] != reference[first][second]:
                raise ValueError(f"five-arm protocol mismatch at {first}.{second}")


def comparison_rows(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build per-configuration speedups against the dense CUDA Graph."""

    by_arm = {str(payload["arm"]): payload for payload in payloads}
    dense_records = {
        (int(row["batch_size"]), int(row["fixed_context_length"])): row
        for row in by_arm["dense"]["records"]
    }
    output = []
    for arm in ARMS:
        for record in by_arm[arm]["records"]:
            key = (int(record["batch_size"]), int(record["fixed_context_length"]))
            dense = dense_records[key]
            latency = float(record["cuda_graph"]["mean_ms"])
            dense_latency = float(dense["cuda_graph"]["mean_ms"])
            output.append(
                {
                    "arm": arm,
                    "batch_size": key[0],
                    "fixed_context_length": key[1],
                    "cuda_graph_mean_ms": latency,
                    "cuda_graph_median_ms": float(
                        record["cuda_graph"]["median_ms"]
                    ),
                    "cuda_graph_p95_ms": float(record["cuda_graph"]["p95_ms"]),
                    "cuda_graph_std_ms": float(record["cuda_graph"]["std_ms"]),
                    "tokens_per_second": float(
                        record["cuda_graph_tokens_per_second"]
                    ),
                    "speedup_over_dense": dense_latency / latency,
                    "latency_reduction_vs_dense": 1.0 - latency / dense_latency,
                    "peak_allocated_bytes_per_rank": int(
                        record["memory"]["peak_allocated_bytes_per_rank"]
                    ),
                }
            )
    return output


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3-8B Wo-only TP4 five-arm CUDA Graph comparison",
        "",
        "All rows use the same model, factors, shapes, dtype, and replay protocol.",
        "",
        "| Batch | Context | Arm | Graph mean ms | Graph p95 ms | Tokens/s | Speedup vs dense | Peak GiB/rank |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        payload["comparisons"],
        key=lambda item: (
            item["batch_size"],
            item["fixed_context_length"],
            ARMS.index(item["arm"]),
        ),
    ):
        lines.append(
            f"| {row['batch_size']} | {row['fixed_context_length']} | "
            f"{row['arm']} | {row['cuda_graph_mean_ms']:.6f} | "
            f"{row['cuda_graph_p95_ms']:.6f} | {row['tokens_per_second']:.3f} | "
            f"{row['speedup_over_dense']:.4f}x | "
            f"{row['peak_allocated_bytes_per_rank'] / 2**30:.3f} |"
        )
    lines.extend(
        [
            "",
            "`wo_c1_ag` and `wo_c1_local_ar` use exactly the same C1 factors "
            "and approximate function; their difference isolates the collective boundary.",
            "All five arms retain dense V and a dense KV cache.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    inputs = [path.expanduser().resolve() for path in args.inputs]
    payloads = [_load(path) for path in inputs]
    validate_matched_protocol(payloads)
    reference = payloads[0]
    payload = {
        "format": SUMMARY_FORMAT,
        "schema_version": 1,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "arms": list(ARMS),
        "input_results": [
            {"arm": row["arm"], "path": str(path)}
            for row, path in zip(payloads, inputs, strict=True)
        ],
        "model": reference["model"],
        "phase1": reference["phase1"],
        "protocol": reference["protocol"],
        "environment": reference["environment"],
        "quality": {row["arm"]: row["quality"] for row in payloads},
        "communication": {row["arm"]: row["communication"] for row in payloads},
        "comparisons": comparison_rows(payloads),
    }
    output_dir.mkdir(parents=True)
    _atomic_text(
        output_dir / "results.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(output_dir / "summary.md", _summary_markdown(payload))


if __name__ == "__main__":
    main()
