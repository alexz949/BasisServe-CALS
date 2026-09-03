#!/usr/bin/env python3
"""Prepare document-disjoint C4 windows for Llama-2-7B C1."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import prepare_qwen3_32b_c1_windows as prepare


if __name__ == "__main__":
    prepare.activate_model_profile("llama2_7b")
    prepare.main()
