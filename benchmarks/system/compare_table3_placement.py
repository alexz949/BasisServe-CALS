"""Compare two teacher-forced check records from bench_table3_k_offload.py."""

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file


def compare(reference, candidate):
    first = json.loads((reference / "result.json").read_text())
    second = json.loads((candidate / "result.json").read_text())
    first_tensors = load_file(str(reference / "check.safetensors"))
    second_tensors = load_file(str(candidate / "check.safetensors"))
    steps = len(first["support_sha256"])
    same_support = [first["support_sha256"][s] == second["support_sha256"][s] for s in range(steps)]
    same_keys = [first["selected_key_sha256"][s] == second["selected_key_sha256"][s]
                 for s in range(steps)]
    logits = (first_tensors["logits"] - second_tensors["logits"]).abs()
    attention = (first_tensors["attention"] - second_tensors["attention"]).abs()
    agree = first_tensors["logits"].argmax(-1) == second_tensors["logits"].argmax(-1)
    # Trial directories are named check_<arm>[_p<page size>]_t<context>_c<cohort>.
    names = [path.name.removeprefix("check_").rsplit("_t", 1)[0] for path in (reference, candidate)]
    return dict(reference=names[0], candidate=names[1], context_tokens=first["context_tokens"],
        cohort=first["cohort"], forced_steps=steps,
        same_forced_tokens=first["forced_tokens"] == second["forced_tokens"],
        same_prompt=first["environment"]["prompt_sha256"] == second["environment"]["prompt_sha256"],
        cache_tokens_match=first["cache_tokens"] == second["cache_tokens"],
        steps_with_identical_support=sum(same_support),
        steps_with_bitwise_identical_selected_keys=sum(same_keys),
        cache_identical={name: first["cache_sha256"][name] == second["cache_sha256"][name]
                         for name in first["cache_sha256"]},
        attention_max_abs_difference=float(attention.max()),
        logits_max_abs_difference=float(logits.max()),
        logits_mean_abs_difference=float(logits.mean()),
        argmax_agreement=float(agree.float().mean()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(compare(arguments.reference, arguments.candidate), indent=2))
