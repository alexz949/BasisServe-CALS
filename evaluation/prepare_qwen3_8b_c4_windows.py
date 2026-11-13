#!/usr/bin/env python3
"""Sample a document-disjoint C4 window bank for Qwen3-8B-Base.

One entry point for the 2048-token PaLU/C1 calibration banks, the 4096-token
Two-Sided-KL profile bank and the fixed C4-validation PPL bank. The sampler is
the shared one in ``build_llama31_8b_palu_m_checkpoint``.
"""

import argparse
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
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-split", choices=("train", "validation"), default="train")
    parser.add_argument(
        "--dataset-revision",
        default="1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    )
    parser.add_argument("--dataset-cache", default=None)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    return parser.parse_args()


if __name__ == "__main__":
    windows_builder.activate_model_profile("qwen3_8b")
    windows_builder._prepare_windows(parse_args())
