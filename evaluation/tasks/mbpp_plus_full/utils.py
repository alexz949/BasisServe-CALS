"""lm-eval generation with EvalPlus 0.3.1 base and augmented scoring."""

from functools import lru_cache
import re
import pickle
from pathlib import Path
import json
import subprocess
import sys
import os

from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
from evalplus.data.utils import CACHE_DIR
from evalplus.eval import PASS, untrusted_check
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec


@lru_cache(maxsize=1)
def problems():
    return get_mbpp_plus(version="v0.2.0")


@lru_cache(maxsize=1)
def oracle_directory():
    directory = Path(CACHE_DIR) / ("basisserve-oracles-0.3.1-" + get_mbpp_plus_hash(version="v0.2.0"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@lru_cache(maxsize=378)
def oracle(task_id):
    path = oracle_directory() / (task_id.replace("/", "-") + ".pkl")
    if path.exists():
        with path.open("rb") as stream:
            return pickle.load(stream)
    problem = problems()[task_id]
    expected = {}
    for kind in ("base", "plus"):
        expected[kind] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem[kind + "_input"], problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS,
        )
    with path.open("wb") as stream:
        pickle.dump(expected, stream)
    return expected


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return "\n\n".join(blocks) if blocks else text.strip()


def score(task_id, code, kind):
    problem = problems()[task_id]
    expected, times = oracle(task_id)[kind]
    status, _ = untrusted_check(
        "mbpp", code, problem[kind + "_input"], problem["entry_point"],
        expected=expected, atol=problem["atol"], ref_time=times,
        fast_check=True,
    )
    return float(status == PASS)


def score_solution(task_id, code):
    base = score(task_id, code, "base")
    plus = score(task_id, code, "plus") if base else 0.0
    return {"base_pass_at_1": base, "plus_pass_at_1": plus}


def process_results(doc, results):
    # A clean CPU interpreter keeps vLLM/torch mappings outside EvalPlus's
    # address-space limit and never forks the active CUDA evaluator.
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "Mbpp/" + str(doc["task_id"])],
        input=extract_code(results[0]), text=True, capture_output=True, check=True,
    )
    return json.loads(completed.stdout)


if __name__ == "__main__":
    # Includes the large reference-output mapping for extreme MBPP/255 inputs.
    os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = str(16 * 1024**3)
    print(json.dumps(score_solution(sys.argv[1], sys.stdin.read())))
