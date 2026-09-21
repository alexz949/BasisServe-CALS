"""Export the Fisher-free Section 4 query-sampling ablation."""

import argparse
import csv
import json
from pathlib import Path
import random


METHOD_LABELS = {
    "dense_full": "Dense Full",
    "exact_k": "Exact-K",
    "page_fisher": "Page-Fisher",
    "score_mse": "Score-MSE (Page-Fisher init)",
    "score_only": "Fixed Score-only",
    "qgram_score_only": "Query-Gram Score-only",
}
DIAGNOSTIC_METHODS = ("exact_k", "page_fisher", "score_mse", "score_only", "qgram_score_only")
HARD_METHODS = ("page_fisher", "score_mse", "score_only", "qgram_score_only")
LONG_METHODS = ("dense_full", "exact_k", "page_fisher", "score_mse", "score_only", "qgram_score_only")
HARD_TASKS = ("niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2")
LONG_TASKS = ("qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "qmsum")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260918


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def load_results(folder, method, count):
    records = [read_json(folder / method / "evaluate" / f"sample_{index:03d}.json") for index in range(count)]
    assert all(record["status"] == "complete" for record in records)
    assert [record["sample"]["index"] for record in records] == list(range(count))
    return records


def summarize(records, tasks, samples_per_task, *, macro):
    task_scores = {}
    for task in tasks:
        selected = [record["result"]["score"] for record in records if record["sample"]["task"] == task]
        assert len(selected) == samples_per_task
        task_scores[task] = 100 * sum(selected) / samples_per_task
    mean = sum(task_scores.values()) / len(task_scores) if macro else 100 * sum(
        record["result"]["score"] for record in records
    ) / len(records)
    return {"mean": mean, "tasks": task_scores}


def paired_bootstrap(left, right, tasks, samples_per_task):
    assert len(left) == len(right) == len(tasks) * samples_per_task
    differences = [
        candidate["result"]["score"] - baseline["result"]["score"]
        for baseline, candidate in zip(left, right, strict=True)
    ]
    generator = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_REPLICATES):
        task_means = []
        for task_index in range(len(tasks)):
            group = differences[task_index * samples_per_task : (task_index + 1) * samples_per_task]
            task_means.append(
                sum(group[generator.randrange(samples_per_task)] for _ in range(samples_per_task))
                / samples_per_task
            )
        estimates.append(100 * sum(task_means) / len(task_means))
    estimates.sort()
    return {
        "delta_query_gram_minus_fixed": 100 * sum(differences) / len(differences),
        "wins": sum(value > 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "stratified_paired_bootstrap_95_ci": [
            estimates[int(0.025 * (BOOTSTRAP_REPLICATES - 1))],
            estimates[int(0.975 * (BOOTSTRAP_REPLICATES - 1))],
        ],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def write_csv(path, result):
    fields = (
        "method",
        "routed_page_recall",
        "page_kl_exact_to_proxy",
        "post_wo_rel_mse",
        "hard_ruler",
        "longbench",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in LONG_METHODS:
            diagnostics = result["diagnostics"].get(method, {})
            writer.writerow(
                {
                    "method": method,
                    "routed_page_recall": diagnostics.get("routed_page_recall", ""),
                    "page_kl_exact_to_proxy": diagnostics.get("page_kl_exact_to_proxy", ""),
                    "post_wo_rel_mse": diagnostics.get("post_wo_rel_mse", ""),
                    "hard_ruler": result["hard_ruler"].get(method, {}).get("mean", ""),
                    "longbench": result["longbench"].get(method, {}).get("mean", ""),
                }
            )


def markdown(result):
    lines = [
        "# Section 4 Fisher-Free Score Routing: Query-Sampling Ablation",
        "",
        "Llama-3.1-8B-Instruct with Dense V128 and original W_O. All routed methods use",
        "B16R16, Page32, B2048 with sink32/recent64 inside the budget, and exact selected",
        "post-RoPE keys for final attention. Fitting uses 40 ALS sweeps and PCG100.",
        "",
        "## Routing diagnostics",
        "",
        "| Method | Routed page recall | Page KL ↓ | Post-W_O rel-MSE ↓ |",
        "|---|---:|---:|---:|",
    ]
    for method in DIAGNOSTIC_METHODS:
        metrics = result["diagnostics"][method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {metrics['routed_page_recall']:.6f} | "
            f"{metrics['page_kl_exact_to_proxy']:.6f} | {metrics['post_wo_rel_mse']:.6f} |"
        )
    lines += [
        "",
        "## Downstream evaluation",
        "",
        "| Method | Hard RULER-64K | LongBench-32K |",
        "|---|---:|---:|",
    ]
    for method in LONG_METHODS:
        hard = result["hard_ruler"].get(method, {}).get("mean")
        long = result["longbench"].get(method, {}).get("mean")
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{hard:.4f} | {long:.4f} |" if hard is not None else
            f"| {METHOD_LABELS[method]} | — | {long:.4f} |"
        )
    hard_pair = result["paired"]["hard_ruler_query_gram_vs_fixed"]
    long_pair = result["paired"]["longbench_query_gram_vs_fixed"]
    fixed_kl = result["diagnostics"]["score_only"]["page_kl_exact_to_proxy"]
    qgram_kl = result["diagnostics"]["qgram_score_only"]["page_kl_exact_to_proxy"]
    kl_reduction = 100 * (fixed_kl - qgram_kl) / fixed_kl
    lines += [
        "",
        "## Fixed sampling versus Query-Gram selection",
        "",
        f"Query-Gram selection reduces page KL by **{kl_reduction:.2f}%** relative to fixed",
        "length-stratified positions. Its downstream point estimate is higher by",
        f"**{hard_pair['delta_query_gram_minus_fixed']:.4f}** on Hard RULER and",
        f"**{long_pair['delta_query_gram_minus_fixed']:.4f}** on LongBench.",
        "",
        f"- Hard RULER: {hard_pair['wins']} wins / {hard_pair['losses']} losses / "
        f"{hard_pair['ties']} ties; paired 95% CI "
        f"[{hard_pair['stratified_paired_bootstrap_95_ci'][0]:.4f}, "
        f"{hard_pair['stratified_paired_bootstrap_95_ci'][1]:.4f}].",
        f"- LongBench: {long_pair['wins']} wins / {long_pair['losses']} losses / "
        f"{long_pair['ties']} ties; paired 95% CI "
        f"[{long_pair['stratified_paired_bootstrap_95_ci'][0]:.4f}, "
        f"{long_pair['stratified_paired_bootstrap_95_ci'][1]:.4f}].",
        "",
        "The confidence intervals include zero. We therefore treat Query-Gram selection as",
        "an inference-free calibration refinement rather than an essential component. The",
        "primary result is that unweighted causal QK-score fitting works without loss",
        "gradients, Fisher weights, or task labels.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--section4", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/section4_ablation"))
    args = parser.parse_args()

    diagnostics = read_json(args.section4 / "diagnostics/qgram_score_only/summary.json")
    assert diagnostics["status"] == "complete"
    diagnostic_means = {}
    for method in DIAGNOSTIC_METHODS:
        values = dict(diagnostics["means"][method])
        values["attention_rel_mse"] = values.get(
            "pooled_attention_rel_mse", values["attention_rel_mse"]
        )
        values["post_wo_rel_mse"] = values.get(
            "pooled_post_wo_rel_mse", values["post_wo_rel_mse"]
        )
        diagnostic_means[method] = values

    hard_folder = args.section4 / "ruler64k_hard_n50_seed43"
    hard_records = {method: load_results(hard_folder, method, 300) for method in HARD_METHODS}
    hard = {
        method: summarize(records, HARD_TASKS, 50, macro=False)
        for method, records in hard_records.items()
    }

    long_folder = args.section4 / "longbench32k_seed43"
    long_records = {method: load_results(long_folder, method, 192) for method in LONG_METHODS}
    long = {
        method: summarize(records, LONG_TASKS, 32, macro=True)
        for method, records in long_records.items()
    }

    result = {
        "status": "complete",
        "protocol": {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "value_mode": "Dense V128 and original W_O",
            "router": "B16R16",
            "page_size": 32,
            "token_budget": 2048,
            "sink_tokens_inside_budget": 32,
            "recent_tokens_inside_budget": 64,
            "hard_ruler_samples": 300,
            "longbench_samples": 192,
            "fit_sweeps": 40,
            "pcg_max_iterations": 100,
        },
        "diagnostics": diagnostic_means,
        "hard_ruler": hard,
        "longbench": long,
        "paired": {
            "hard_ruler_query_gram_vs_fixed": paired_bootstrap(
                hard_records["score_only"], hard_records["qgram_score_only"], HARD_TASKS, 50
            ),
            "longbench_query_gram_vs_fixed": paired_bootstrap(
                long_records["score_only"], long_records["qgram_score_only"], LONG_TASKS, 32
            ),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "score_sampling_ablation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    write_csv(args.output / "score_sampling_ablation.csv", result)
    (args.output / "score_sampling_ablation.md").write_text(markdown(result))
    print({"status": "complete", "output": str(args.output)}, flush=True)


if __name__ == "__main__":
    main()
