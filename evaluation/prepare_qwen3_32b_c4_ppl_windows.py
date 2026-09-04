#!/usr/bin/env python3
"""Prepare independent C4-validation documents for Qwen3-32B PPL."""

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
MODEL_PROFILE = "qwen3_32b"
DESCRIPTION = __doc__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--dataset-cache", default="results/cache/huggingface/datasets")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples != 128 or args.sequence_length != 2048:
        raise ValueError("the controlled C4-PPL protocol requires 128 x 2048 tokens")
    if args.shuffle_buffer < args.samples:
        raise ValueError("shuffle buffer must cover the requested sample count")
    windows_builder.activate_model_profile(MODEL_PROFILE)
    builder_args = argparse.Namespace(
        model=args.model,
        output_dir=args.output_dir,
        dataset_revision=args.dataset_revision,
        dataset_split="validation",
        dataset_cache=args.dataset_cache,
        samples=args.samples,
        sequence_length=args.sequence_length,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    windows_builder._prepare_windows(builder_args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    document_ids = [str(row["document_id"]) for row in records]
    if (
        manifest["dataset"]["split"] != "validation"
        or manifest["sampling"]["samples"] != 128
        or manifest["sampling"]["sequence_length"] != 2048
        or len(document_ids) != len(set(document_ids))
    ):
        raise RuntimeError("generated C4-validation bank violates the fixed protocol")
    print(
        f"[C4 PPL windows] verified {len(records)} independent validation documents",
        flush=True,
    )


if __name__ == "__main__":
    main()
