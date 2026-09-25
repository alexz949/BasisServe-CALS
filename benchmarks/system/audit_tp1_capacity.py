"""Check trial completeness, active batch, metrics and exact-Dense parity."""

import argparse
import collections
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    trials = json.loads((args.input / "outcomes.json").read_text())
    settings = json.loads((args.input / "manifest.json").read_text())["settings"]
    assert json.loads((args.input / "source_check.json").read_text())["changed_files"] == []
    successes = [t for t in trials if t["status"] == "success"]
    assert all(t["status"] in ("success", "gpu_oom") for t in trials)
    groups = collections.defaultdict(list)
    for trial in trials:
        groups[trial["context"], trial["method"], trial["batch"]].append(trial)
    for group in groups.values():
        if group[0]["status"] == "success":
            assert len(group) == settings["repeats"]
            assert sorted(t["repeat"] for t in group) == list(range(settings["repeats"]))
            assert all(t["status"] == "success" for t in group)
        else:
            assert len(group) == 1 and group[0]["phase"] in ("cache_allocation", "prefill", "warmup", "decode", "model_load")
    for context in settings["contexts"]:
        for method in settings["methods"]:
            subset = [t for t in trials if (t["context"], t["method"]) == (context, method)]
            batches = sorted({t["batch"] for t in subset})
            assert batches == settings["batches"][:len(batches)]
            assert subset[-1]["status"] == "gpu_oom" or batches == settings["batches"]
    for trial in successes:
        r = trial["result"]
        assert (r["batch"], r["context"]) == (trial["batch"], trial["context"])
        assert r["decode_steps"] == settings["steps"] and r["warmup_steps"] == settings["warmup"]
        assert r["active_batch"] == [r["batch"]] * settings["steps"]
        assert r["all_logits_finite"] and len(r["tokens"]) == r["batch"]
        assert all(len(row) == settings["steps"] for row in r["tokens"])
        assert len(r["step_ms"]) == settings["steps"]
        assert all(math.isfinite(x) and x > 0 for x in r["step_ms"])
        assert math.isclose(r["mean_step_ms"], statistics.fmean(r["step_ms"]))
        assert math.isclose(r["aggregate_tokens_per_second"], r["batch"] * r["decode_steps"] / r["decode_seconds"])
        expected_host = 0 if r["method"] == "dense_local" else 32 * r["batch"] * 8 * (r["context"] + max(r["warmup_steps"], r["decode_steps"])) * 128 * 2
        assert r["host_key_bytes"] == expected_host
    by_key = {(t["method"], t["context"], t["batch"], t["repeat"]): t["result"] for t in successes}
    pairs = []
    for key, result in by_key.items():
        if key[0] == "dense_local":
            other = by_key[("dense_k_offload", *key[1:])]
            assert result["tokens"] == other["tokens"]
            pairs.append(list(key[1:]))
    repeat_token_agreement = {
        f"{context}:{method}:b{batch}": all(t["result"]["tokens"] == group[0]["result"]["tokens"] for t in group)
        for (context, method, batch), group in groups.items() if group[0]["status"] == "success"
    }
    report = dict(status="passed", trials=len(trials), successful=len(successes),
        gpu_oom=sum(t["status"] == "gpu_oom" for t in trials),
        measured_decode_tokens=sum(t["batch"] * settings["steps"] for t in successes),
        exact_dense_matching_pairs=pairs, repeat_token_agreement=repeat_token_agreement,
        runtime_sources_unchanged=True, sha256_checks=False)
    (args.input / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
