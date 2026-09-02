#!/usr/bin/env python3
"""Merge disjoint Qwen3 Store80 Page-Fisher synergy result shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_qwen3_8b_s80_page_fisher_synergy import (  # noqa: E402
    FORMAT,
    _aggregate,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = _parse_args()
    inputs = tuple(Path(item).expanduser().resolve() for item in args.inputs)
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    layers = sorted(
        (record for shard in shards for record in shard["layers"]),
        key=lambda record: int(record["layer"]),
    )
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        output,
        {
            "format": FORMAT,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "source_shards": [str(path) for path in inputs],
            "layers": layers,
            "aggregate": _aggregate(layers),
            "elapsed_seconds_sum": sum(
                float(shard.get("elapsed_seconds", 0.0)) for shard in shards
            ),
        },
    )
    print(f"[S80 Page-Fisher synergy] merged {len(layers)} layers into {output}")


if __name__ == "__main__":
    main()
