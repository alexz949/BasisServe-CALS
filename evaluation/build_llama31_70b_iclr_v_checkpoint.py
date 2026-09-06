#!/usr/bin/env python3
"""Build Llama-3.1-70B V-side checkpoints for the ICLR experiment matrix."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import build_qwen3_8b_iclr_v_checkpoint as builder  # noqa: E402


if __name__ == "__main__":
    builder.activate_experiment_profile("llama31_70b")
    sys.exit(builder.build(builder.parse_args()))
