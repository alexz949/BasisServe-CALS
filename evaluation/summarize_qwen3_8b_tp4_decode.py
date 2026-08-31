#!/usr/bin/env python3
"""Merge dense, uniform-C1, and Global-KL-C1 TP4 decode benchmarks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
from typing import Any, Mapping, Sequence


FORMAT = "basisserve.qwen3_8b.tp4_autoregressive_decode_comparison.v1"
INPUT_FORMAT = "basisserve.qwen3_8b.tp4_autoregressive_decode_benchmark.v1"
EXPECTED_ARMS = ("dense", "c1_uniform_r64", "c1_mean_dp")


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != INPUT_FORMAT or payload.get("status") != "complete":
        raise ValueError(f"incomplete or incompatible benchmark {path}")
    return payload


def _record_map(payload: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    output: dict[tuple[int, int], Mapping[str, Any]] = {}
    for record in payload["records"]:
        key = (int(record["batch_size"]), int(record["decode_length"]))
        if key in output:
            raise ValueError(f"duplicate configuration {key}")
        output[key] = record
    return output


def _geometric_mean(values: Sequence[float]) -> float:
    selected = tuple(map(float, values))
    if not selected or any(value <= 0.0 for value in selected):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in selected))


def _aggregate(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, float]:
    speedups = tuple(float(row[f"{name}_speedup"]) for row in rows)
    dense_total = sum(float(row["dense_total_ms"]) for row in rows)
    c1_total = sum(float(row[f"{name}_total_ms"]) for row in rows)
    return {
        "minimum_speedup": min(speedups),
        "median_speedup": statistics.median(speedups),
        "geometric_mean_speedup": _geometric_mean(speedups),
        "maximum_speedup": max(speedups),
        "whole_grid_sequential_speedup": dense_total / c1_total,
        "faster_configuration_fraction": sum(value > 1.0 for value in speedups)
        / len(speedups),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# Qwen3-8B TP4 autoregressive decode speedup",
        "",
        "One prompt token; the prompt forward is excluded. Every row measures the "
        "complete requested decode loop including the transformer, TP collectives, "
        "sharded LM head, and distributed greedy selection.",
        "",
        "| C1 arm | Grid geometric mean | Grid median | Whole-grid sequential | Faster cells |",
        "|:---|---:|---:|---:|---:|",
    ]
    for name, label in (("uniform", "uniform-r64 + ALS5"), ("mean_dp", "mean-DP + ALS5")):
        row = aggregate[name]
        lines.append(
            f"| {label} | {row['geometric_mean_speedup']:.4f}x | "
            f"{row['median_speedup']:.4f}x | "
            f"{row['whole_grid_sequential_speedup']:.4f}x | "
            f"{100.0 * row['faster_configuration_fraction']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "| Batch | Decode tokens | Dense tok/s | Uniform tok/s | Uniform speedup | "
            "Mean-DP tok/s | Mean-DP speedup | Dense GiB | Uniform GiB | Mean-DP GiB |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| {row['batch_size']} | {row['decode_length']} | "
            f"{row['dense_tokens_per_second']:.4f} | "
            f"{row['uniform_tokens_per_second']:.4f} | {row['uniform_speedup']:.4f}x | "
            f"{row['mean_dp_tokens_per_second']:.4f} | {row['mean_dp_speedup']:.4f}x | "
            f"{row['dense_peak_gib']:.3f} | {row['uniform_peak_gib']:.3f} | "
            f"{row['mean_dp_peak_gib']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--uniform", type=Path, required=True)
    parser.add_argument("--mean-dp", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    inputs = {
        "dense": _read(args.dense.expanduser().resolve()),
        "c1_uniform_r64": _read(args.uniform.expanduser().resolve()),
        "c1_mean_dp": _read(args.mean_dp.expanduser().resolve()),
    }
    for expected, observed in zip(EXPECTED_ARMS, inputs.values(), strict=True):
        if observed.get("arm") != expected:
            raise ValueError(f"expected {expected}, got {observed.get('arm')}")
    protocols = [payload["protocol"] for payload in inputs.values()]
    invariant_fields = (
        "tp_size",
        "dtype",
        "prompt_tokens",
        "prompt_token_id",
        "decode_lengths",
        "batch_sizes",
        "warmup_tokens_per_configuration",
        "timed_scope",
        "prompt_forward_timed",
        "early_stopping",
    )
    for field in invariant_fields:
        values = [protocol[field] for protocol in protocols]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"benchmark protocol differs at {field}: {values}")
    maps = {name: _record_map(value) for name, value in inputs.items()}
    keys = set(maps["dense"])
    if any(set(records) != keys for records in maps.values()):
        raise ValueError("benchmark arms cover different configuration grids")

    rows: list[dict[str, Any]] = []
    for batch, decode_length in sorted(keys):
        dense = maps["dense"][(batch, decode_length)]
        uniform = maps["c1_uniform_r64"][(batch, decode_length)]
        mean_dp = maps["c1_mean_dp"][(batch, decode_length)]
        dense_ms = float(dense["critical_path_total_ms"])
        uniform_ms = float(uniform["critical_path_total_ms"])
        mean_ms = float(mean_dp["critical_path_total_ms"])
        rows.append(
            {
                "batch_size": batch,
                "decode_length": decode_length,
                "dense_total_ms": dense_ms,
                "uniform_total_ms": uniform_ms,
                "mean_dp_total_ms": mean_ms,
                "dense_tokens_per_second": float(dense["generated_tokens_per_second"]),
                "uniform_tokens_per_second": float(uniform["generated_tokens_per_second"]),
                "mean_dp_tokens_per_second": float(mean_dp["generated_tokens_per_second"]),
                "uniform_speedup": dense_ms / uniform_ms,
                "mean_dp_speedup": dense_ms / mean_ms,
                "mean_dp_over_uniform": uniform_ms / mean_ms,
                "dense_peak_gib": float(dense["memory"]["peak_allocated_bytes_per_rank"]) / 2**30,
                "uniform_peak_gib": float(uniform["memory"]["peak_allocated_bytes_per_rank"]) / 2**30,
                "mean_dp_peak_gib": float(mean_dp["memory"]["peak_allocated_bytes_per_rank"]) / 2**30,
                "dense_step_latency": dense["step_latency"],
                "uniform_step_latency": uniform["step_latency"],
                "mean_dp_step_latency": mean_dp["step_latency"],
                "dense_segments": dense["segments"],
                "uniform_segments": uniform["segments"],
                "mean_dp_segments": mean_dp["segments"],
            }
        )
    payload = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "inputs": {
            name: {
                "path": str(path.expanduser().resolve()),
                "command": inputs[arm]["command"],
            }
            for name, path, arm in (
                ("dense", args.dense, "dense"),
                ("uniform", args.uniform, "c1_uniform_r64"),
                ("mean_dp", args.mean_dp, "c1_mean_dp"),
            )
        },
        "protocol": protocols[0],
        "communication": {
            name: value["communication"] for name, value in inputs.items()
        },
        "aggregate": {
            "uniform": _aggregate(rows, "uniform"),
            "mean_dp": _aggregate(rows, "mean_dp"),
        },
        "rows": rows,
    }
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    _atomic_text(output_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(output_markdown, _markdown(payload))
    print(json.dumps({"event": "comparison_written", "json": str(output_json), "markdown": str(output_markdown)}))


if __name__ == "__main__":
    main()
