#!/usr/bin/env python3
"""Prepare fixed document-disjoint C4 splits for Qwen3.5-9B C1 ALS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import build_llama31_8b_palu_m_checkpoint as windows_builder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-windows", type=int, default=256)
    parser.add_argument("--heldout-windows", type=int, default=64)
    parser.add_argument("--profile-windows", type=int, default=16)
    parser.add_argument("--confirm-windows", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument(
        "--dataset-revision",
        default="1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-cache", default="results/cache/huggingface/datasets")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = {
        "fit": args.fit_windows,
        "heldout": args.heldout_windows,
        "profile": args.profile_windows,
        "confirm": args.confirm_windows,
    }
    if args.sequence_length <= 0 or any(count <= 0 for count in counts.values()):
        raise ValueError("sequence length and all split sizes must be positive")
    total = sum(counts.values())
    windows_builder.activate_model_profile("qwen35_9b")
    windows_builder._prepare_windows(
        argparse.Namespace(
            model=args.model,
            output_dir=args.output_dir,
            dataset_revision=args.dataset_revision,
            dataset_split=args.dataset_split,
            dataset_cache=args.dataset_cache,
            samples=total,
            sequence_length=args.sequence_length,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    offset = 0
    split_records = {}
    for name, count in counts.items():
        split_records[name] = {
            "offset": offset,
            "count": count,
            "stop": offset + count,
        }
        offset += count
    manifest["splits"] = split_records
    windows_builder._atomic_json(manifest_path, manifest)
    print(
        f"[Qwen3.5 C1 windows] total={total} sequence_length={args.sequence_length} "
        f"splits={split_records}",
        flush=True,
    )


if __name__ == "__main__":
    main()
