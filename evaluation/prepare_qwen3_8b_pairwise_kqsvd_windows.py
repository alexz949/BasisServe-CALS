#!/usr/bin/env python3
"""Prepare the fixed 128x4096 C4-train bank for Pairwise KQ-SVD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import build_llama31_8b_palu_m_checkpoint as windows_builder  # noqa: E402


DATASET_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
SAMPLES = 128
SEQUENCE_LENGTH = 4096


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--dataset-cache", default="results/cache/huggingface/datasets")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.samples != SAMPLES or args.sequence_length != SEQUENCE_LENGTH:
        raise ValueError("the 4K Pairwise KQ-SVD protocol requires exactly 128 x 4096 tokens")
    if args.shuffle_buffer < args.samples:
        raise ValueError("shuffle buffer must cover the requested sample count")

    windows_builder.activate_model_profile("qwen3_8b")
    windows_builder._prepare_windows(
        argparse.Namespace(
            model=args.model,
            output_dir=args.output_dir,
            dataset_revision=args.dataset_revision,
            dataset_split="train",
            dataset_cache=args.dataset_cache,
            samples=args.samples,
            sequence_length=args.sequence_length,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    document_ids = [str(record["document_id"]) for record in manifest["records"]]
    if (
        manifest["dataset"]["split"] != "train"
        or manifest["artifact"]["shape"] != [SAMPLES, SEQUENCE_LENGTH]
        or manifest["sampling"]["sequence_length"] != SEQUENCE_LENGTH
        or len(document_ids) != len(set(document_ids))
    ):
        raise RuntimeError("generated calibration bank violates the fixed protocol")
    print(
        f"[Pairwise KQ-SVD windows] verified {SAMPLES} document-disjoint "
        f"C4-train windows x {SEQUENCE_LENGTH} tokens",
        flush=True,
    )


if __name__ == "__main__":
    main()
