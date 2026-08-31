#!/usr/bin/env python3
"""Build the matched V-only PaLU-M checkpoint for Qwen3-32B."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import build_llama31_8b_palu_m_checkpoint as builder


if __name__ == "__main__":
    builder.activate_model_profile("qwen3_32b")
    builder.main()
