#!/usr/bin/env python3
"""Allocate Qwen3-8B C1 ranks with post-ALS whole-layer Global-KL."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import run_qwen3_32b_c1_layer_global_kl_sharded as runner


if __name__ == "__main__":
    runner.activate_model_profile("qwen3_8b")
    runner.__doc__ = __doc__
    runner.main()
