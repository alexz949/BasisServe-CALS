"""TP port consistency on real text, teacher-forced: for each prompt (TP4 inputs, row 0 of cohorts 0/1/2, 8192 tokens)
an arm generates 128 greedy tokens at W=1; the same arm at W=4 is then fed exactly those tokens. At every step the
W=4 argmax is compared with the W=1 token, and the log-probability of the W=1 token is compared between W=1 and W=4.
Dense attention's own W=1 vs W=4 numbers are the reference for TP rounding noise. Every arm, BasisKV included, goes
through the same check.

Why teacher forcing: greedy free-running first divergence (round 1 and the first round-2 check) depends on one
near-tie and then on everything after it; sparse top-k selection turns BF16 rounding differences into such ties
more often than dense attention does, so it measures sensitivity, not port correctness.

Criterion, fixed before these runs: an arm passes when its pooled top-1 agreement (3 prompts x 128 steps) is at least
0.90. Agreement and log-probability differences are reported next to dense's; the criterion only catches a broken
port, it does not rank arms.

Usage: python benchmarks/system/check_v2q8_tp_consistency.py
"""

import json
import statistics

import run_v2q8_exp_smoke as smoke

ARMS = ("dense_v128", "basis_koff_v128", "shadowkv_upstream_v128", "lrqk_upstream_koff_v128")
COHORTS = (0, 1, 2)
THRESHOLD = 0.90


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main(arms=ARMS, cohorts=COHORTS):
    out = smoke.OUT / "checks" / "tp_consistency"
    records, rows = [], {}
    for cohort in cohorts:
        for arm in arms:
            base = ["benchmarks/system/bench_v2q8_tp.py", "--arm", arm, "--mode", "request", "--length", "8192",
                    "--batch", "1", "--cohort", str(cohort), "--inputs", str(smoke.INPUTS[4]), *smoke.COMMON]
            free = out / f"{arm}_c{cohort}_w1"
            records.append(smoke.run(f"checks_tp_{arm}_c{cohort}_w1", smoke.torchrun(base + ["--output", str(free)],
                                     [0]), [0], free, extra=smoke.arm_env(arm)))
            reference = load(free / "result.json")
            if reference.get("status") != "complete":
                continue
            measurement = reference["ranks"][0]["measurement"]
            tokens = out / f"{arm}_c{cohort}_w1_tokens.json"
            tokens.write_text(json.dumps(measurement["generated_tokens"]) + "\n")
            forced = out / f"{arm}_c{cohort}_w4_forced"
            records.append(smoke.run(f"checks_tp_{arm}_c{cohort}_w4_forced", smoke.torchrun(
                base + ["--output", str(forced), "--force-tokens", str(tokens)], [0, 1, 2, 3]), [0, 1, 2, 3], forced,
                extra=smoke.arm_env(arm)))
            result = load(forced / "result.json")
            if result.get("status") != "complete":
                continue
            other = result["ranks"][0]["measurement"]
            w1, w4 = measurement["generated_tokens"][0], other["generated_tokens"][0]
            lp1, lp4 = measurement["fed_token_logprobs"][0], other["fed_token_logprobs"][0]
            rows[(arm, cohort)] = dict(steps=len(w1), top1_agree=sum(a == b for a, b in zip(w1, w4)),
                                       logprob_abs_diff=[abs(a - b) for a, b in zip(lp1, lp4)],
                                       w1_distinct_tokens=len(set(w1)))
    summary = dict(method="teacher-forced: W=4 fed the W=1 greedy tokens", prompt_tokens=8192, generated_tokens=128,
                   cohorts=list(cohorts), threshold_top1_agreement=THRESHOLD, arms={})
    for arm in arms:
        parts = [rows.get((arm, c)) for c in cohorts]
        if not all(parts):
            summary["arms"][arm] = dict(passed=False, missing_runs=[c for c, p in zip(cohorts, parts) if not p])
            continue
        steps = sum(p["steps"] for p in parts)
        diffs = [d for p in parts for d in p["logprob_abs_diff"]]
        agreement = sum(p["top1_agree"] for p in parts) / steps
        summary["arms"][arm] = dict(
            top1_agreement=agreement, top1_agreement_per_cohort=[p["top1_agree"] / p["steps"] for p in parts],
            logprob_abs_diff_mean=statistics.fmean(diffs), logprob_abs_diff_median=statistics.median(diffs),
            logprob_abs_diff_max=max(diffs), w1_distinct_tokens_per_cohort=[p["w1_distinct_tokens"] for p in parts],
            passed=agreement >= THRESHOLD)
    summary["status"] = "complete" if all(r["passed"] for r in summary["arms"].values()) else "mismatch"
    (out / "result_teacher_forced.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return records


if __name__ == "__main__":
    main()
