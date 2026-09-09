#!/usr/bin/env python3
"""Matched L40S TP2 BBH stop fix and MBPP+ pass@3 comparison."""

from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.eval_qwen3_32b_hard_generation_vllm import evaluator, summarize_task

evaluator.FORMAT = "basisserve.qwen3_32b.bbh_mbpp3_vllm.v1"
evaluator.STAGE_FORMAT = evaluator.FORMAT + ".task"
evaluator.EXPECTED_GPU = "NVIDIA L40S"
evaluator.TENSOR_PARALLEL_SIZE = 2
evaluator.TASKS = ("bbh_cot_fewshot", "mbpp_plus_pass3")
evaluator.TASK_FEWSHOT = {"bbh_cot_fewshot": 3, "mbpp_plus_pass3": 0}
evaluator.TASK_MAX_GEN_TOKS = {"bbh_cot_fewshot": 1024, "mbpp_plus_pass3": 2048}
evaluator.TASK_INCLUDE_PATH = str(Path(__file__).parent / "tasks" / "mbpp_plus_pass3")
evaluator.UNSAFE_TASKS = frozenset(("mbpp_plus_pass3",))
evaluator.TASK_GEN_KWARGS = {
    "bbh_cot_fewshot": {"until": ["\nQ:"], "max_gen_toks": 1024, "do_sample": False, "temperature": 0},
    "mbpp_plus_pass3": {"max_gen_toks": 2048, "do_sample": True, "temperature": 0.2, "top_p": 0.95},
}


def summarize(task, result):
    metrics = summarize_task(task, result)
    if metrics is not None and task == "bbh_cot_fewshot":
        samples = [s for rows in result["samples"].values() for s in rows]
        metrics["answer_extraction_rate"] = sum(
            bool(re.search(r"(?<=the answer is )(.*)(?=.)", s["resps"][0][0]))
            for s in samples
        ) / len(samples)
    return metrics


evaluator._summarize_task = summarize

if __name__ == "__main__":
    sys.exit(evaluator.evaluate(evaluator.parse_args()))
