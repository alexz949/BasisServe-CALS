#!/usr/bin/env python3
"""Extend the audited Llama-3.1-70B C4 bank with Global-KL documents."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import prepare_qwen3_32b_global_kl_windows as prepare  # noqa: E402


if __name__ == "__main__":
    prepare.activate_model_profile("llama31_70b")
    prepare.main()
