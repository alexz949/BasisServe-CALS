#!/usr/bin/env python3
"""Prepare document-disjoint C4 fit and held-out windows for Qwen3-32B C1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from safetensors.torch import load_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import build_llama31_8b_palu_m_checkpoint as windows_builder


FIT_WINDOWS = 256
HELDOUT_WINDOWS = 64
SEQUENCE_LENGTH = 2048
MODEL_PROFILE = "qwen3_32b"


def activate_model_profile(name: str) -> None:
    global MODEL_PROFILE
    if name not in {
        "qwen3_8b",
        "qwen3_32b",
        "llama31_8b",
        "llama31_70b",
        "llama2_7b",
    }:
        raise ValueError(f"unsupported C1 window profile: {name}")
    MODEL_PROFILE = name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix-windows", required=True)
    parser.add_argument("--fit-windows", type=int, default=FIT_WINDOWS)
    parser.add_argument("--heldout-windows", type=int, default=HELDOUT_WINDOWS)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
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
    if min(args.fit_windows, args.heldout_windows, args.sequence_length) <= 0:
        raise ValueError("fit/held-out counts and sequence length must be positive")
    total_windows = args.fit_windows + args.heldout_windows
    windows_builder.activate_model_profile(MODEL_PROFILE)
    builder_args = argparse.Namespace(
        model=args.model,
        output_dir=args.output_dir,
        dataset_revision=args.dataset_revision,
        dataset_split=args.dataset_split,
        dataset_cache=args.dataset_cache,
        samples=total_windows,
        sequence_length=args.sequence_length,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    windows_builder._prepare_windows(builder_args)
    generated_path = Path(args.output_dir).expanduser().resolve() / "windows.safetensors"
    prefix_path = Path(args.prefix_windows).expanduser().resolve()
    generated = load_file(str(generated_path), device="cpu")["input_ids"]
    prefix = load_file(str(prefix_path), device="cpu")["input_ids"]
    if tuple(generated.shape) != (total_windows, args.sequence_length):
        raise ValueError(f"unexpected generated window shape: {tuple(generated.shape)}")
    if prefix.ndim != 2 or int(prefix.shape[1]) != args.sequence_length:
        raise ValueError(f"unexpected prefix shape: {tuple(prefix.shape)}")
    if not 0 < len(prefix) <= total_windows:
        raise ValueError("prefix window count exceeds the requested window bank")
    if not torch.equal(generated[: len(prefix)], prefix):
        raise RuntimeError("generated prefix differs from existing calibration windows")
    print(
        f"[C1 windows] verified prefix={len(prefix)} fit={args.fit_windows} "
        f"heldout={args.heldout_windows} total={total_windows}",
        flush=True,
    )


if __name__ == "__main__":
    main()
