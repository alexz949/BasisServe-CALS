#!/usr/bin/env python3
"""Merge disjoint Qwen3 S80 layer banks into one factor bank."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_joint_routing_payload_s80_qwen3 import (  # noqa: E402
    merge_s80_factor_banks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = merge_s80_factor_banks(
        args.output_dir,
        input_dirs=args.inputs,
        command=shlex.join(sys.argv),
    )
    print(f"[S80 merge] wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
