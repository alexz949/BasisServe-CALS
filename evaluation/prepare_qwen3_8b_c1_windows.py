#!/usr/bin/env python3
"""Prepare document-disjoint C4 fit and held-out windows for Qwen3-8B C1."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import prepare_qwen3_32b_c1_windows as prepare


if __name__ == "__main__":
    prepare.activate_model_profile("qwen3_8b")
    prepare.main()
