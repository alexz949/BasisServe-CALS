#!/usr/bin/env python3
"""Run on a Slurm worker: verify exported augmented tests against references."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset
from evaluation.tasks.mbpp_plus_full.utils import process_results, problems


def main():
    data = load_dataset("evalplus/mbppplus", split="test")
    if {"Mbpp/" + str(doc["task_id"]) for doc in data} != set(problems()):
        print("Dataset task IDs do not match official EvalPlus v0.2.0", flush=True)
        return 1
    failures = []
    for index, doc in enumerate(data):
        problem = problems()["Mbpp/" + str(doc["task_id"])]
        result = process_results(doc, [problem["prompt"] + problem["canonical_solution"]])
        if result["plus_pass_at_1"] != 1:
            failures.append({"task_id": doc["task_id"], **result})
        print(json.dumps({"index": index, "task_id": doc["task_id"], **result}), flush=True)
    print(json.dumps({"count": len(data), "failures": failures}), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
