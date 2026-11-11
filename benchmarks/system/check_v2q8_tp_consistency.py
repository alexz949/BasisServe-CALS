"""TP port consistency on real text, teacher-forced: for each prompt (TP4 inputs, row 0 of cohorts 0/1/2, 8192 tokens;
or the first `batch` rows in one batched run) an arm generates 128 greedy tokens at W=1; the same arm at W=4 is then fed exactly those tokens. At every step the
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

ARMS = ("dense_v128", "basis_koff_v128", "basis_koff_v128_opt", "shadowkv_upstream_v128", "lrqk_upstream_koff_v128")
COHORTS = (0, 1, 2)
THRESHOLD = 0.90


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main(arms=ARMS, cohorts=COHORTS, batch=1, reference=None):
    """batch: requests per run (rows 0..batch-1 of the cohort; upstream ShadowKV prefills them in groups of 4).
    reference: {arm: arm whose W=1 run supplies the forced tokens}, for arms that only differ under TP (the
    ShadowKV SVD modes); such an arm is also compared with its reference arm's W=4 run."""
    reference = reference or {}
    out = smoke.OUT / "checks" / "tp_consistency"
    suffix = "" if batch == 1 else f"_b{batch}"
    records, rows, free_runs, forced_runs = [], {}, {}, {}
    for cohort in cohorts:
        for arm in arms:
            source = reference.get(arm, arm)
            common = ["--mode", "request", "--length", "8192", "--batch", str(batch), "--cohort", str(cohort),
                      "--inputs", str(smoke.INPUTS[4]), *smoke.COMMON]
            free = out / f"{source}_c{cohort}_w1{suffix}"
            if (source, cohort) not in free_runs:
                records.append(smoke.run(f"checks_tp_{source}_c{cohort}_w1{suffix}", smoke.torchrun(
                    ["benchmarks/system/bench_v2q8_tp.py", "--arm", source, *common, "--output", str(free)], [0]),
                    [0], free, extra=smoke.arm_env(source)))
                free_runs[(source, cohort)] = load(free / "result.json")
            if free_runs[(source, cohort)].get("status") != "complete":
                continue
            measurement = free_runs[(source, cohort)]["ranks"][0]["measurement"]
            tokens = out / f"{source}_c{cohort}_w1{suffix}_tokens.json"
            tokens.write_text(json.dumps(measurement["generated_tokens"]) + "\n")
            forced = out / f"{arm}_c{cohort}_w4{suffix}_forced"
            records.append(smoke.run(f"checks_tp_{arm}_c{cohort}_w4{suffix}_forced", smoke.torchrun(
                ["benchmarks/system/bench_v2q8_tp.py", "--arm", arm, *common, "--output", str(forced),
                 "--force-tokens", str(tokens)], [0, 1, 2, 3]), [0, 1, 2, 3], forced, extra=smoke.arm_env(arm)))
            result = load(forced / "result.json")
            if result.get("status") != "complete":
                continue
            other = result["ranks"][0]["measurement"]
            forced_runs[(arm, cohort)] = other
            rows[(arm, cohort)] = [
                dict(steps=len(w1), top1_agree=sum(a == b for a, b in zip(w1, w4)),
                     logprob_abs_diff=[abs(a - b) for a, b in zip(lp1, lp4)], w1_distinct_tokens=len(set(w1)))
                for w1, w4, lp1, lp4 in zip(measurement["generated_tokens"], other["generated_tokens"],
                                            measurement["fed_token_logprobs"], other["fed_token_logprobs"])]
    summary = dict(method="teacher-forced: W=4 fed the W=1 greedy tokens", prompt_tokens=8192, generated_tokens=128,
                   cohorts=list(cohorts), requests_per_cohort=batch, threshold_top1_agreement=THRESHOLD, arms={})
    for arm in arms:
        parts = [rows.get((arm, c)) for c in cohorts]
        if not all(parts):
            summary["arms"][arm] = dict(passed=False, missing_runs=[c for c, p in zip(cohorts, parts) if not p])
            continue
        steps = sum(r["steps"] for p in parts for r in p)
        diffs = [d for p in parts for r in p for d in r["logprob_abs_diff"]]
        agreement = sum(r["top1_agree"] for p in parts for r in p) / steps
        summary["arms"][arm] = dict(
            top1_agreement=agreement,
            top1_agreement_per_cohort=[sum(r["top1_agree"] for r in p) / sum(r["steps"] for r in p) for p in parts],
            logprob_abs_diff_mean=statistics.fmean(diffs), logprob_abs_diff_median=statistics.median(diffs),
            logprob_abs_diff_max=max(diffs),
            w1_distinct_tokens_per_cohort=[[r["w1_distinct_tokens"] for r in p] for p in parts] if batch > 1 else
            [p[0]["w1_distinct_tokens"] for p in parts],
            passed=agreement >= THRESHOLD)
        if batch > 1:
            summary["arms"][arm]["top1_agreement_per_request"] = [[r["top1_agree"] / r["steps"] for r in p]
                                                                  for p in parts]
        if arm in reference:
            summary["arms"][arm]["forced_tokens_from"] = reference[arm]
        if arm in reference and all((reference[arm], c) in forced_runs for c in cohorts):
            # Same forced tokens, both at W=4: this arm against its reference arm, step by step.
            same, total, gaps = 0, 0, []
            for cohort in cohorts:
                a, b = forced_runs[(arm, cohort)], forced_runs[(reference[arm], cohort)]
                for x, y, lx, ly in zip(a["generated_tokens"], b["generated_tokens"], a["fed_token_logprobs"],
                                        b["fed_token_logprobs"]):
                    same += sum(i == j for i, j in zip(x, y))
                    total += len(x)
                    gaps += [abs(i - j) for i, j in zip(lx, ly)]
            summary["arms"][arm][f"against_{reference[arm]}_w4"] = dict(
                argmax_agreement=same / total, logprob_abs_diff_mean=statistics.fmean(gaps),
                logprob_abs_diff_median=statistics.median(gaps), logprob_abs_diff_max=max(gaps))
    # Arms checked earlier and not re-run here are kept (the criterion is per arm and absolute).
    path = out / f"result_teacher_forced{suffix}.json"
    earlier = load(path).get("arms", {})
    summary["arms"] = {**{arm: row for arm, row in earlier.items() if arm not in summary["arms"]}, **summary["arms"]}
    summary["status"] = "complete" if all(r["passed"] for r in summary["arms"].values()) else "mismatch"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return records


if __name__ == "__main__":
    main()
