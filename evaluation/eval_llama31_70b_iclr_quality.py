#!/usr/bin/env python3
"""Run the complete Llama-3.1-70B ICLR quality protocol on one matrix arm."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import eval_qwen3_8b_iclr_quality as evaluator  # noqa: E402


evaluator.activate_quality_profile("llama31_70b")


if __name__ == "__main__":
    sys.exit(evaluator.evaluate(evaluator.parse_args()))
