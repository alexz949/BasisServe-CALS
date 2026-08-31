#!/usr/bin/env python3
"""Compare dense, mean-DP avg-R64, and uniform-R64 TP4 decode profiles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import statistics
import sys
from typing import Any, Mapping


DECODE_FORMAT = "basisserve.qwen3_8b.tp4_decode_breakdown.v1"
FORMAT = "basisserve.qwen3_8b.tp4_uniform_comparison.v1"


def _load(path: str | Path) -> tuple[Path, dict[str, Any]]:
    selected = Path(path).expanduser().resolve()
    payload = json.loads(selected.read_text(encoding="utf-8"))
    return selected, payload


def _decode_records(payload: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    if payload.get("format") != DECODE_FORMAT or payload.get("status") != "complete":
        raise ValueError("decode profile is incompatible or incomplete")
    records = {
        (
            int(record["batch_size"]),
            int(record["context_length_including_current_token"]),
        ): record
        for record in payload["records"]
    }
    if len(records) != len(payload["records"]):
        raise ValueError("decode profile contains duplicate configurations")
    return records


def _ppl(payload: Mapping[str, Any]) -> float:
    selected = payload.get("ppl")
    if isinstance(selected, Mapping):
        selected = selected.get("ppl")
    value = float(selected)
    if value <= 0:
        raise ValueError("PPL must be positive")
    return value


def _comparison_row(
    dense: Mapping[str, Any],
    mean_dp: Mapping[str, Any],
    uniform: Mapping[str, Any],
) -> dict[str, Any]:
    batch = int(dense["batch_size"])
    context = int(dense["context_length_including_current_token"])
    key = (batch, context)
    for record in (mean_dp, uniform):
        if key != (
            int(record["batch_size"]),
            int(record["context_length_including_current_token"]),
        ):
            raise ValueError("decode configuration mismatch")
    dense_ms = float(dense["e2e"]["mean_ms"])
    mean_dp_ms = float(mean_dp["e2e"]["mean_ms"])
    uniform_ms = float(uniform["e2e"]["mean_ms"])
    mean_dp_ablation = mean_dp["derived"]["collective_ablation"]
    uniform_ablation = uniform["derived"]["collective_ablation"]
    return {
        "batch_size": batch,
        "context_length": context,
        "dense_e2e_ms": dense_ms,
        "mean_dp_e2e_ms": mean_dp_ms,
        "uniform_e2e_ms": uniform_ms,
        "mean_dp_speedup_vs_dense": dense_ms / mean_dp_ms,
        "uniform_speedup_vs_dense": dense_ms / uniform_ms,
        "uniform_speedup_vs_mean_dp": mean_dp_ms / uniform_ms,
        "uniform_latency_delta_vs_mean_dp_ms": uniform_ms - mean_dp_ms,
        "uniform_latency_delta_vs_mean_dp_percent": 100.0
        * (uniform_ms / mean_dp_ms - 1.0),
        "mean_dp_attention_collective_e2e_ms": float(
            mean_dp_ablation["attention_collective_marginal_e2e_ms"]
        ),
        "uniform_attention_collective_e2e_ms": float(
            uniform_ablation["attention_collective_marginal_e2e_ms"]
        ),
        "mean_dp_all_main_collectives_e2e_ms": float(
            mean_dp_ablation["all_main_collectives_e2e_ms"]
        ),
        "uniform_all_main_collectives_e2e_ms": float(
            uniform_ablation["all_main_collectives_e2e_ms"]
        ),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# Qwen3-8B TP4 uniform-R64 vs mean-DP avg-R64",
        "",
        "Both C1 arms retain exactly 50% of the Value width on average and use "
        "the same direct-slot CUDA/NCCL runtime. The only change is the per-layer "
        "rank schedule.",
        "",
        "| Batch | Context | Dense ms | Mean-DP ms | Uniform ms | Uniform vs mean-DP | Uniform speedup vs dense |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["configurations"]:
        lines.append(
            f"| {row['batch_size']} | {row['context_length']} | "
            f"{row['dense_e2e_ms']:.4f} | {row['mean_dp_e2e_ms']:.4f} | "
            f"{row['uniform_e2e_ms']:.4f} | "
            f"{row['uniform_latency_delta_vs_mean_dp_percent']:+.3f}% | "
            f"{row['uniform_speedup_vs_dense']:.4f}x |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Uniform mean speedup vs dense: `{aggregate['uniform_mean_speedup_vs_dense']:.4f}x`.",
            f"- Mean-DP mean speedup vs dense: `{aggregate['mean_dp_mean_speedup_vs_dense']:.4f}x`.",
            f"- Uniform mean latency delta vs mean-DP: "
            f"`{aggregate['uniform_mean_latency_delta_vs_mean_dp_ms']:+.4f} ms` "
            f"(`{aggregate['uniform_mean_latency_delta_vs_mean_dp_percent']:+.3f}%`).",
            f"- Uniform is faster at `{aggregate['uniform_faster_points']}` of "
            f"`{aggregate['points']}` points.",
            "",
            "## Quality",
            "",
            f"- Dense WikiText-2 PPL: `{payload['quality']['dense_ppl']:.6f}`.",
            f"- Mean-DP avg-R64 PPL: `{payload['quality']['mean_dp_ppl']:.6f}`.",
            f"- Uniform-R64 PPL: `{payload['quality']['uniform_ppl']:.6f}`.",
            "",
            "Latency is measured on real 4xL40S TP4. Instrumented substage data "
            "remains diagnostic; E2E and collective-removal ablations are the "
            "primary comparison.",
        ]
    )
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", required=True)
    parser.add_argument("--mean-dp", required=True)
    parser.add_argument("--uniform", required=True)
    parser.add_argument("--dense-ppl", required=True)
    parser.add_argument("--mean-dp-ppl", required=True)
    parser.add_argument("--uniform-ppl", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dense_path, dense = _load(args.dense)
    mean_dp_path, mean_dp = _load(args.mean_dp)
    uniform_path, uniform = _load(args.uniform)
    if dense.get("arm") != "dense":
        raise ValueError("dense profile has the wrong arm")
    if mean_dp.get("arm") != "c1_mean_dp":
        raise ValueError("mean-DP profile has the wrong arm")
    if uniform.get("arm") != "c1_uniform_r64":
        raise ValueError("uniform profile has the wrong arm")
    dense_records = _decode_records(dense)
    mean_dp_records = _decode_records(mean_dp)
    uniform_records = _decode_records(uniform)
    if not (dense_records.keys() == mean_dp_records.keys() == uniform_records.keys()):
        raise ValueError("decode configuration grids differ")
    rows = [
        _comparison_row(dense_records[key], mean_dp_records[key], uniform_records[key])
        for key in sorted(dense_records)
    ]

    dense_ppl_path, dense_ppl = _load(args.dense_ppl)
    mean_dp_ppl_path, mean_dp_ppl = _load(args.mean_dp_ppl)
    uniform_ppl_path, uniform_ppl = _load(args.uniform_ppl)
    relative_deltas = [
        float(row["uniform_latency_delta_vs_mean_dp_percent"]) for row in rows
    ]
    payload = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "inputs": {
            "dense": str(dense_path),
            "mean_dp": str(mean_dp_path),
            "uniform": str(uniform_path),
            "dense_ppl": str(dense_ppl_path),
            "mean_dp_ppl": str(mean_dp_ppl_path),
            "uniform_ppl": str(uniform_ppl_path),
        },
        "configurations": rows,
        "aggregate": {
            "points": len(rows),
            "uniform_faster_points": sum(
                float(row["uniform_latency_delta_vs_mean_dp_ms"]) < 0
                for row in rows
            ),
            "uniform_mean_speedup_vs_dense": statistics.fmean(
                float(row["uniform_speedup_vs_dense"]) for row in rows
            ),
            "mean_dp_mean_speedup_vs_dense": statistics.fmean(
                float(row["mean_dp_speedup_vs_dense"]) for row in rows
            ),
            "uniform_mean_latency_delta_vs_mean_dp_ms": statistics.fmean(
                float(row["uniform_latency_delta_vs_mean_dp_ms"]) for row in rows
            ),
            "uniform_mean_latency_delta_vs_mean_dp_percent": statistics.fmean(
                relative_deltas
            ),
            "uniform_min_latency_delta_vs_mean_dp_percent": min(relative_deltas),
            "uniform_max_latency_delta_vs_mean_dp_percent": max(relative_deltas),
        },
        "quality": {
            "dense_ppl": _ppl(dense_ppl),
            "mean_dp_ppl": _ppl(mean_dp_ppl),
            "uniform_ppl": _ppl(uniform_ppl),
        },
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    _atomic_text(
        output_dir / "uniform_comparison.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(output_dir / "uniform_comparison.md", _markdown(payload))
    print(
        json.dumps(
            {
                "event": "uniform_comparison_written",
                "json": str(output_dir / "uniform_comparison.json"),
                "markdown": str(output_dir / "uniform_comparison.md"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
