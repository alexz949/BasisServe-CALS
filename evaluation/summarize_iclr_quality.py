#!/usr/bin/env python3
"""Audit one complete ICLR quality matrix and render its result table."""

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


TASKS = (
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
    "boolq",
    "openbookqa",
)
PROFILES = {
    "qwen3-8b": {
        "label": "Qwen3-8B-Base",
        "prefix": "Q3-8B",
        "checkpoint_format": "basisserve.qwen3_8b.iclr_v_factors.v1",
        "quality_format": "basisserve.qwen3_8b.iclr_quality.v1",
        "cuda_devices": ["NVIDIA L40S"] * 4,
    },
    "llama31-8b": {
        "label": "Llama-3.1-8B",
        "prefix": "L31-8B",
        "checkpoint_format": "basisserve.llama31_8b.iclr_v_factors.v1",
        "quality_format": "basisserve.llama31_8b.iclr_quality.v1",
        "cuda_devices": ["NVIDIA L40S"] * 4,
    },
    "llama2-7b": {
        "label": "Llama-2-7B",
        "prefix": "L2-7B",
        "checkpoint_format": "basisserve.llama2_7b.iclr_v_factors.v1",
        "quality_format": "basisserve.llama2_7b.iclr_quality.v1",
        "cuda_devices": ["NVIDIA L40S"] * 4,
    },
    "qwen3-32b": {
        "label": "Qwen3-32B-Base",
        "prefix": "Q3-32B",
        "checkpoint_format": "basisserve.qwen3_32b.iclr_v_factors.v1",
        "quality_format": "basisserve.qwen3_32b.iclr_quality.v1",
        "cuda_devices": ["NVIDIA L40S"] * 4,
    },
    "llama31-70b": {
        "label": "Llama-3.1-70B",
        "prefix": "L31-70B",
        "checkpoint_format": "basisserve.llama31_70b.iclr_v_factors.v1",
        "quality_format": "basisserve.llama31_70b.iclr_quality.v1",
        "cuda_devices": ["NVIDIA H200 NVL"] * 2,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _run_ids(prefix: str) -> list[str]:
    run_ids = [f"{prefix}-Dense"]
    run_ids.extend(f"{prefix}-SVD-R{rank}" for rank in (96, 80, 64))
    for geometry in ("PALUM", "PALUG2", "PALUG4"):
        run_ids.extend(f"{prefix}-{geometry}-R{rank}" for rank in (96, 80, 64))
    run_ids.extend(f"{prefix}-C1-R{rank}" for rank in (96, 80, 64))
    return run_ids


def _task_values(result: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(row["task"]): float(row["value"])
        for row in result["metrics"]["task_accuracy"]
    }


def _audit_result(
    result_path: Path,
    *,
    run_id: str,
    checkpoint_format: str,
    quality_format: str,
    cuda_devices: list[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    failures: list[str] = []
    if not result_path.is_file():
        return None, [f"missing {result_path}"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("format") != quality_format or result.get("status") != "complete":
        failures.append(f"{run_id}: incomplete or wrong quality format")
    if result.get("run_id") != run_id:
        failures.append(f"{run_id}: result run ID mismatch")
    if result.get("checkpoint", {}).get("format") != checkpoint_format:
        failures.append(f"{run_id}: checkpoint format mismatch")
    checkpoint_dir = Path(result.get("checkpoint", {}).get("directory", ""))
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.is_file():
        failures.append(f"{run_id}: checkpoint manifest missing")
    elif _sha256(manifest_path) != result["checkpoint"]["manifest_sha256"]:
        failures.append(f"{run_id}: checkpoint manifest hash mismatch")
    for stage, record in result.get("stages", {}).items():
        stage_path = result_path.parent / record["file"]
        if not stage_path.is_file() or _sha256(stage_path) != record["sha256"]:
            failures.append(f"{run_id}: {stage} stage hash mismatch")
    if set(_task_values(result)) != set(TASKS):
        failures.append(f"{run_id}: task set mismatch")
    environment = result.get("environment", {})
    if environment.get("datasets") != "5.0.0":
        failures.append(f"{run_id}: datasets version is not 5.0.0")
    if environment.get("lm_eval") != "0.4.11":
        failures.append(f"{run_id}: lm-eval version is not 0.4.11")
    if environment.get("cuda_devices") != cuda_devices:
        failures.append(f"{run_id}: evaluation recorded unexpected GPUs")
    return result, failures


def _trend_checks(results: Mapping[str, Mapping[str, Any]], prefix: str) -> dict[str, bool]:
    def metric(run_id: str, name: str) -> float:
        return float(results[run_id]["metrics"][name])

    compressed = [run_id for run_id in results if not run_id.endswith("-Dense")]
    dense = f"{prefix}-Dense"
    checks = {
        "dense_best_wikitext2_ppl": all(
            metric(dense, "wikitext2_ppl") < metric(run_id, "wikitext2_ppl")
            for run_id in compressed
        ),
        "dense_best_c4_ppl": all(
            metric(dense, "c4_validation_128_ppl")
            < metric(run_id, "c4_validation_128_ppl")
            for run_id in compressed
        ),
        "dense_best_average_accuracy": all(
            metric(dense, "average_accuracy") > metric(run_id, "average_accuracy")
            for run_id in compressed
        ),
    }
    for rank in (96, 80, 64):
        svd = f"{prefix}-SVD-R{rank}"
        palu = [
            f"{prefix}-PALUM-R{rank}",
            f"{prefix}-PALUG2-R{rank}",
            f"{prefix}-PALUG4-R{rank}",
        ]
        checks[f"r{rank}_weight_svd_worst"] = (
            all(metric(svd, "wikitext2_ppl") > metric(run_id, "wikitext2_ppl") for run_id in palu)
            and all(metric(svd, "c4_validation_128_ppl") > metric(run_id, "c4_validation_128_ppl") for run_id in palu)
            and all(metric(svd, "average_accuracy") < metric(run_id, "average_accuracy") for run_id in palu)
        )
        checks[f"r{rank}_palu_improves_m_to_g2_to_g4"] = (
            metric(palu[0], "wikitext2_ppl")
            > metric(palu[1], "wikitext2_ppl")
            > metric(palu[2], "wikitext2_ppl")
            and metric(palu[0], "c4_validation_128_ppl")
            > metric(palu[1], "c4_validation_128_ppl")
            > metric(palu[2], "c4_validation_128_ppl")
            and metric(palu[0], "average_accuracy")
            < metric(palu[1], "average_accuracy")
            < metric(palu[2], "average_accuracy")
        )
    return checks


def _markdown(
    *,
    label: str,
    rows: list[Mapping[str, Any]],
    trend_checks: Mapping[str, bool],
) -> str:
    lines = [
        f"# {label} ICLR V-side quality matrix",
        "",
        "| Run ID | Method | Retained V | WT2 PPL | C4 PPL | ARC-E | ARC-C | HS | PIQA | WG | BoolQ | OBQA | Avg |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        tasks = row["tasks"]
        values = " | ".join(f"{tasks[task]:.6f}" for task in TASKS)
        lines.append(
            f"| {row['run_id']} | {row['method']} | {row['retained_ratio']:.6f} | "
            f"{row['wikitext2_ppl']:.6f} | {row['c4_ppl']:.6f} | {values} | "
            f"{row['average_accuracy']:.6f} |"
        )
    lines.extend(["", "## Trend audit", ""])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in trend_checks.items()
    )
    lines.extend(
        [
            "",
            f"Overall expected trend: **{'PASS' if all(trend_checks.values()) else 'FAIL'}**",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> int:
    profile = PROFILES[args.model]
    model_root = Path(args.root).expanduser().resolve() / args.model
    results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for run_id in _run_ids(str(profile["prefix"])):
        result, result_failures = _audit_result(
            model_root / "quality" / run_id / "result.json",
            run_id=run_id,
            checkpoint_format=str(profile["checkpoint_format"]),
            quality_format=str(profile["quality_format"]),
            cuda_devices=list(profile["cuda_devices"]),
        )
        failures.extend(result_failures)
        if result is not None:
            results[run_id] = result
    if failures:
        for failure in failures:
            print(f"[Audit failure] {failure}", file=sys.stderr)
        return 2

    run_ids = _run_ids(str(profile["prefix"]))
    rows = []
    for run_id in run_ids:
        result = results[run_id]
        compression = result["compression"]
        rows.append(
            {
                "run_id": run_id,
                "method": compression["method_label"],
                "retained_ratio": float(compression["realized_retained_v_ratio"]),
                "wikitext2_ppl": float(result["metrics"]["wikitext2_ppl"]),
                "c4_ppl": float(result["metrics"]["c4_validation_128_ppl"]),
                "tasks": _task_values(result),
                "average_accuracy": float(result["metrics"]["average_accuracy"]),
            }
        )
    trends = _trend_checks(results, str(profile["prefix"]))
    payload = {
        "format": "basisserve.iclr_v_quality_summary.v1",
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "model": args.model,
        "rows": rows,
        "trend_checks": trends,
        "expected_trend_pass": all(trends.values()),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_path = output.with_suffix(".md")
    _atomic_text(
        markdown_path,
        _markdown(label=str(profile["label"]), rows=rows, trend_checks=trends),
    )
    print(f"[Summary] expected_trend_pass={all(trends.values())} output={output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(PROFILES), required=True)
    parser.add_argument("--root", default="ICLR-results")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(summarize(parse_args()))
