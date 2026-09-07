#!/usr/bin/env python3
"""Qwen3-32B hard-generation protocol on two L40S GPUs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.eval_qwen3_32b_hard_generation_vllm import evaluator

evaluator.EXPECTED_GPU = "NVIDIA L40S"
evaluator.TENSOR_PARALLEL_SIZE = 2

if __name__ == "__main__":
    sys.exit(evaluator.evaluate(evaluator.parse_args()))
