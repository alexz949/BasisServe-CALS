#!/usr/bin/env python3
"""Compare frozen Qwen3-8B per-layer C1 schedules on full C4 documents."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import eval_qwen3_32b_c1_layer_schedules_c4 as evaluator


if __name__ == "__main__":
    evaluator.activate_model_profile("qwen3_8b")
    evaluator.__doc__ = __doc__
    evaluator.evaluate(evaluator.parse_args())
