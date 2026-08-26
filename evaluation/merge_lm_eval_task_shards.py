#!/usr/bin/env python3
"""Merge disjoint lm-eval task shards emitted by BasisServe evaluators."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping

from lm_eval.utils import make_table


FORMAT = "basisserve.lm_eval_task_shards.v1"
TASK_MAP_FIELDS = (
    "configs",
    "group_subtasks",
    "higher_is_better",
    "n-samples",
    "n-shot",
    "results",
    "versions",
)
GLOBAL_FIELDS = (
    "eot_token_id",
    "git_hash",
    "lm_eval_version",
    "max_length",
    "tokenizer_bos_token",
    "tokenizer_eos_token",
    "tokenizer_pad_token",
    "transformers_version",
    "upper_git_hash",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or not isinstance(
        payload.get("evaluation"), Mapping
    ):
        raise ValueError(f"incomplete lm-eval shard: {path}")
    return payload


def merge_shards(shards: list[tuple[Path, Mapping[str, Any]]]) -> dict[str, Any]:
    if len(shards) < 2:
        raise ValueError("at least two shards are required")
    reference_path, reference = shards[0]
    reference_evaluation = reference["evaluation"]
    for path, shard in shards[1:]:
        for field in ("arm", "model", "checkpoint", "compression"):
            if shard.get(field) != reference.get(field):
                raise ValueError(f"shard metadata mismatch for {field}: {path}")
        for field in GLOBAL_FIELDS:
            if shard["evaluation"].get(field) != reference_evaluation.get(field):
                raise ValueError(f"lm-eval global metadata mismatch for {field}: {path}")

    merged_evaluation = {
        key: value
        for key, value in reference_evaluation.items()
        if key not in TASK_MAP_FIELDS
    }
    for field in TASK_MAP_FIELDS:
        merged: dict[str, Any] = {}
        for path, shard in shards:
            for task, value in shard["evaluation"].get(field, {}).items():
                if task in merged:
                    raise ValueError(f"task {task!r} overlaps across shards at {path}")
                merged[task] = value
        merged_evaluation[field] = dict(sorted(merged.items()))

    tasks = list(merged_evaluation["results"])
    return {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "arm": reference["arm"],
        "model": reference["model"],
        "checkpoint": reference.get("checkpoint"),
        "compression": reference.get("compression"),
        "protocol": {
            **reference.get("protocol", {}),
            "tasks": tasks,
            "execution": "disjoint task shards",
        },
        "evaluation": merged_evaluation,
        "shards": [
            {
                "path": str(path.resolve()),
                "tasks": list(shard["evaluation"]["results"]),
                "elapsed_seconds": shard.get("elapsed_seconds"),
            }
            for path, shard in shards
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [Path(raw).expanduser().resolve() for raw in args.input]
    output = Path(args.output_json).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = merge_shards([(path, _load(path)) for path in inputs])
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(make_table(payload["evaluation"]), flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
