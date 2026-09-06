#!/usr/bin/env python3
"""Prepare independent C4-validation documents for Llama-3.1-70B PPL."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import prepare_qwen3_32b_c4_ppl_windows as prepare


if __name__ == "__main__":
    prepare.MODEL_PROFILE = "llama31_70b"
    prepare.DESCRIPTION = __doc__
    prepare.main()
