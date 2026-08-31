#!/usr/bin/env python3
"""Merge two Qwen3-32B C4-validation PPL shards and compute paired deltas."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import allocate_qwen3_32b_c1_tp_source_global_kl as common  # noqa: E402
from evaluation.eval_qwen3_32b_c1_c4_ppl_shard import (  # noqa: E402
    ALL_ARMS,
    FORMAT as SHARD_FORMAT,
)
from evaluation.eval_qwen3_32b_c1_wikitext import _atomic_json, _sha256  # noqa: E402


FORMAT = "basisserve.qwen3_32b.gqa_c1.c4_validation_ppl.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _shared(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": payload["model"],
        "allocation": payload["allocation"],
        "windows": payload["windows"],
        "protocol": payload["protocol"],
    }


def merge(args: argparse.Namespace) -> None:
    if len(args.input) != 2:
        raise ValueError("the fixed C4-PPL protocol requires exactly two shards")
    output_path = args.output_json.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    inputs = []
    for raw_path in args.input:
        path = raw_path.expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != SHARD_FORMAT or payload.get("status") != "complete":
            raise ValueError(f"incomplete or incompatible C4-PPL shard: {path}")
        inputs.append((path, payload))
    reference = _shared(inputs[0][1])
    if _shared(inputs[1][1]) != reference:
        raise ValueError("C4-PPL shard protocols or data sources differ")
    arms = {}
    for _, payload in inputs:
        for name, row in payload["arms"].items():
            if name in arms:
                raise ValueError(f"duplicate C4-PPL arm: {name}")
            arms[name] = row
    if set(arms) != set(ALL_ARMS):
        raise ValueError("merged C4-PPL shards do not cover all four arms")

    dense_values = arms["dense"]["metrics"]["document_nll"]["values"]
    comparisons = {}
    for name in ALL_ARMS[1:]:
        values = arms[name]["metrics"]["document_nll"]["values"]
        if len(values) != len(dense_values):
            raise ValueError(f"C4-PPL document count differs for {name}")
        comparisons[name] = {
            "paired_document_delta_nll_vs_dense": common._paired(
                [
                    float(candidate) - float(dense)
                    for candidate, dense in zip(values, dense_values, strict=True)
                ]
            ),
            "ppl_ratio_vs_dense": (
                arms[name]["metrics"]["ppl"]
                / arms["dense"]["metrics"]["ppl"]
            ),
        }
    rankings = sorted(
        (
            {"arm": name, "ppl": arms[name]["metrics"]["ppl"]}
            for name in ALL_ARMS
        ),
        key=lambda row: row["ppl"],
    )
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **reference,
        "arms": {name: arms[name] for name in ALL_ARMS},
        "comparisons": comparisons,
        "rankings_lowest_ppl_first": rankings,
        "shards": [
            {"path": str(path), "sha256": _sha256(path)} for path, _ in inputs
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, payload)
    print(f"[C4 PPL merge] wrote {output_path}", flush=True)


if __name__ == "__main__":
    merge(parse_args())
