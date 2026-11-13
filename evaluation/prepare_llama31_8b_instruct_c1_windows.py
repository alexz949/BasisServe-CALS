#!/usr/bin/env python3
"""Extend a Llama-3.1-8B-Instruct C4 window bank and verify the shared prefix."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import prepare_qwen3_32b_c1_windows as prepare


if __name__ == "__main__":
    prepare.activate_model_profile("llama31_8b_instruct")
    prepare.main()
