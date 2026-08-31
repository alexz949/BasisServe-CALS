#!/usr/bin/env python3
"""Extend the audited Qwen3-32B C4 bank with 16 Global-KL documents."""

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


PREFIX_WINDOWS = 320
GLOBAL_KL_WINDOWS = 16
SEQUENCE_LENGTH = 2048
MODEL_PROFILE = "qwen3_32b"


def activate_model_profile(name: str) -> None:
    global MODEL_PROFILE
    if name not in {"qwen3_8b", "qwen3_32b"}:
        raise ValueError(f"unsupported Qwen3 Global-KL window profile: {name}")
    MODEL_PROFILE = name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix-windows", required=True)
    parser.add_argument("--prefix-count", type=int, default=PREFIX_WINDOWS)
    parser.add_argument("--global-kl-windows", type=int, default=GLOBAL_KL_WINDOWS)
    parser.add_argument(
        "--dataset-revision",
        default="1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-cache", default="results/cache/huggingface/datasets")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    return parser.parse_args()


def _verify_extension(generated: torch.Tensor, prefix: torch.Tensor, *, extra: int) -> None:
    if generated.ndim != 2 or prefix.ndim != 2:
        raise ValueError("C4 windows must be rank-two tensors")
    if generated.shape[1] != SEQUENCE_LENGTH or prefix.shape[1] != SEQUENCE_LENGTH:
        raise ValueError("C4 windows must contain exactly 2048 tokens")
    if len(generated) != len(prefix) + extra:
        raise ValueError("extended C4 bank has an unexpected window count")
    if not torch.equal(generated[: len(prefix)], prefix):
        raise RuntimeError("extended bank does not preserve the audited 320-window prefix")


def main() -> None:
    args = parse_args()
    if args.prefix_count <= 0 or args.global_kl_windows <= 0:
        raise ValueError("prefix and Global-KL window counts must be positive")
    prefix_path = Path(args.prefix_windows).expanduser().resolve()
    prefix = load_file(str(prefix_path), device="cpu")["input_ids"]
    if len(prefix) != args.prefix_count:
        raise ValueError(
            f"expected prefix_count={args.prefix_count}, found {len(prefix)} windows"
        )

    total_windows = args.prefix_count + args.global_kl_windows
    windows_builder.activate_model_profile(MODEL_PROFILE)
    builder_args = argparse.Namespace(
        model=args.model,
        output_dir=args.output_dir,
        dataset_revision=args.dataset_revision,
        dataset_split=args.dataset_split,
        dataset_cache=args.dataset_cache,
        samples=total_windows,
        sequence_length=SEQUENCE_LENGTH,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    windows_builder._prepare_windows(builder_args)
    generated_path = Path(args.output_dir).expanduser().resolve() / "windows.safetensors"
    generated = load_file(str(generated_path), device="cpu")["input_ids"]
    _verify_extension(generated, prefix, extra=args.global_kl_windows)
    print(
        "[Global-KL windows] verified audited prefix="
        f"{args.prefix_count}; fresh indices="
        f"{args.prefix_count}..{total_windows - 1}",
        flush=True,
    )


if __name__ == "__main__":
    main()
