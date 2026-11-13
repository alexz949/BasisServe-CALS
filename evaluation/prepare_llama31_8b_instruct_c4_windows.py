#!/usr/bin/env python3
"""Sample a document-disjoint C4 window bank for Llama-3.1-8B-Instruct."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import build_llama31_8b_palu_m_checkpoint as windows_builder  # noqa: E402
from evaluation.prepare_qwen3_8b_c4_windows import parse_args  # noqa: E402


if __name__ == "__main__":
    windows_builder.activate_model_profile("llama31_8b_instruct")
    windows_builder._prepare_windows(parse_args())
