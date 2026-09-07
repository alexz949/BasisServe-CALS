#!/usr/bin/env python3
"""Matched Qwen3-32B R80 MATH500, full MBPP+, and BBH generation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.eval_qwen3_32b_r80_generation_vllm import evaluator

evaluator.FORMAT = "basisserve.qwen3_32b.hard_generation_vllm.v1"
evaluator.STAGE_FORMAT = "basisserve.qwen3_32b.hard_generation_vllm.task.v1"
evaluator.TASKS = ("minerva_math500", "mbpp_plus_full", "bbh_cot_fewshot")
evaluator.TASK_FEWSHOT = dict(zip(evaluator.TASKS, (4, 0, 3)))
evaluator.TASK_MAX_GEN_TOKS = dict(zip(evaluator.TASKS, (4096, 2048, 1024)))
evaluator.REQUIRED_MAX_LENGTH = 8192
evaluator.TASK_INCLUDE_PATH = str(Path(__file__).parent / "tasks" / "mbpp_plus_full")
evaluator.TASK_GEN_KWARGS = {
    task: {"max_gen_toks": tokens, "do_sample": False, "temperature": 0}
    for task, tokens in evaluator.TASK_MAX_GEN_TOKS.items()
}
evaluator.UNSAFE_TASKS = frozenset(("mbpp_plus_full",))


def summarize_task(task, evaluation):
    metrics = evaluation.get("results", {}).get(task)
    if metrics is None:
        metrics = evaluation.get("groups", {}).get(task, {})
    selected = {
        key: float(value) for key, value in metrics.items()
        if "," in key and "stderr" not in key and isinstance(value, (float, int))
    }
    if not evaluator._check(bool(selected), f"missing metrics for {task}"):
        return None
    return selected


evaluator._summarize_task = summarize_task

if __name__ == "__main__":
    sys.exit(evaluator.evaluate(evaluator.parse_args()))
