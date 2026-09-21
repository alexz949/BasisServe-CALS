#!/usr/bin/env python3
"""Audit and summarize the Qwen3.5 reasoning-only V128+Wo run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.qwen35_hybrid_common import atomic_save, load_bank, sha256  # noqa: E402


EXPECTED = {
    "gsm8k": 1319,
    "minerva_math500": 500,
    "ifeval": 541,
    "mbpp_plus_full": 378,
}
SCHEDULE = {3: 80, 7: 96, 11: 96, 15: 96, 19: 192, 23: 160, 27: 192, 31: 112}


def _metrics(result: dict, task: str) -> dict[str, float]:
    metrics = {
        key: float(value)
        for key, value in result["evaluation"]["results"][task].items()
        if "," in key and "stderr" not in key and isinstance(value, (int, float))
    }
    assert metrics and all(math.isfinite(value) for value in metrics.values())
    return metrics


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    bank_path = root / "banks/c1_twosided_v128.pt"
    wo_path = root / "wo/wo_bank.pt"
    data_manifest_path = root / "data/manifest.json"
    bank = load_bank(bank_path)
    wo = __import__("torch").load(wo_path, weights_only=True, map_location="cpu")
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    assert bank["schedule"] == SCHEDULE
    assert bank["nominal_v_rank"] == 128
    assert bank["rank_allocation_source"] == "frozen_c4_two_sided_v128_schedule"
    assert bank["rank_allocation_recomputed_on_reasoning_calibration"] is False
    assert wo["upstream_v_factor_sha256"] == bank["factor_sha256"]
    assert wo["source_rank_by_type"] == {"gdn": 768, "full_attention": 512}
    assert data_manifest["composition"]["fit"] == {"math": 128, "code": 128}

    rows = []
    provenance = None
    for task, count in EXPECTED.items():
        path = root / "eval" / f"{task}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["status"] == "complete"
        assert result["thinking"] is False
        assert result["args"]["task"] == task
        assert result["args"]["limit"] is None
        sample_count = result["evaluation"]["n-samples"][task]
        assert sample_count["original"] == count
        assert sample_count["effective"] == count
        assert len(result["generation_records"]) == count
        assert result["provenance"]["v_bank_sha256"] == sha256(bank_path)
        assert result["provenance"]["wo_bank_sha256"] == sha256(wo_path)
        assert all(
            row["original_prompt_tokens"] == row["retained_prompt_tokens"]
            for row in result["generation_records"]
        )
        shared = {
            key: value
            for key, value in result["provenance"].items()
            if key != "tokenizer_sha256"
        }
        if provenance is None:
            provenance = shared
        assert shared == provenance
        rows.append(
            {
                "task": task,
                "questions": count,
                "metrics": _metrics(result, task),
                "length_capped": result["length_capped"],
                "closing_think_responses": result["closing_think_responses"],
                "elapsed_seconds": result["elapsed_seconds"],
                "source": str(path),
                "source_sha256": sha256(path),
            }
        )

    summary = {
        "format": "basisserve.qwen35_9b.reasoning_v128_wo768_512_summary.v1",
        "status": "complete_and_audited",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Two-Sided V128 frozen-C4 schedule; reasoning-calibrated V factors; reasoning-calibrated Wo768/512",
        "schedule": SCHEDULE,
        "average_v_rank": sum(SCHEDULE.values()) / len(SCHEDULE),
        "calibration": data_manifest["composition"],
        "artifacts": {
            "data_manifest": {"path": str(data_manifest_path), "sha256": sha256(data_manifest_path)},
            "v_bank": {"path": str(bank_path), "sha256": sha256(bank_path)},
            "wo_bank": {"path": str(wo_path), "sha256": sha256(wo_path)},
        },
        "rows": rows,
    }
    atomic_save(root / "summary.json", summary)

    lines = [
        "# Qwen3.5-9B Reasoning-Only Calibration V128 + Wo Results",
        "",
        "- V schedule: `80, 96, 96, 96, 192, 160, 192, 112` (average rank 128).",
        "- Rank allocation is frozen from the C4 Two-Sided run; only V/Wo fitting uses reasoning calibration.",
        "- Fit calibration: `128 Math + 128 Code`, each 2048 tokens.",
        "- Wo ranks: GDN 768 per source; full-attention 512 per source.",
        "- Evaluation: non-thinking, greedy, matched Qwen3.5 vLLM protocol.",
        "",
        "| Task | Questions | Metrics | Length-capped |",
        "|---|---:|---|---:|",
    ]
    for row in rows:
        metric_text = "; ".join(
            f"{key}={100 * value:.2f}%" for key, value in row["metrics"].items()
        )
        lines.append(
            f"| {row['task']} | {row['questions']} | {metric_text} | {row['length_capped']} |"
        )
    lines.extend(
        [
            "",
            "The frozen rank schedule makes this a calibration-domain ablation, not a reasoning-domain allocator result.",
            "",
        ]
    )
    _write_text(root / "summary.md", "\n".join(lines))
    print(json.dumps({"output": str(root / "summary.json"), "rows": rows}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
