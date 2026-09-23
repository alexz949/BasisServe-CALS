"""Summarize the specialized TP8 Basis-joint decode optimization grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


MATRIX = {
    4096: (1, 8, 32, 128),
    16384: (1, 4, 8, 16),
    65536: (1, 4, 8, 16),
    130048: (1, 4, 8, 16),
}
TRIAL_FIELDS = (
    "prompt_tokens", "batch", "cohort", "status", "failure_stage",
    "rank_files", "rank_outputs_identical", "baseline_mean_ms",
    "optimized_mean_ms", "speedup_vs_baseline", "optimized_p50_ms",
    "optimized_p95_ms", "optimized_tokens_s", "sequence_count",
    "exact_sequences_vs_baseline", "token_agreement_vs_baseline",
    "median_exact_prefix_tokens", "minimum_exact_prefix_tokens",
    "prefill_peak_maxrank_bytes", "decode_resident_maxrank_bytes",
    "decode_peak_maxrank_bytes", "host_exact_k_total_bytes", "slot_hit_rate",
    "result_directory", "launcher_log",
)
SUMMARY_FIELDS = (
    "prompt_tokens", "batch", "status", "successful_trials",
    "baseline_basis_median_ms", "optimized_basis_median_ms",
    "speedup_vs_baseline_basis", "speedup_vs_dense",
    "speedup_vs_als_full", "optimized_tokens_s", "optimized_p50_ms",
    "optimized_p95_ms", "prefill_peak_maxrank_bytes",
    "decode_resident_maxrank_bytes", "decode_peak_maxrank_bytes",
    "host_exact_k_total_bytes", "slot_hit_rate", "exact_trial_trajectories",
    "sequence_count", "exact_sequences_vs_baseline",
    "token_agreement_vs_baseline", "median_exact_prefix_tokens",
    "cohort_optimized_mean_ms", "failure_stage",
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def failure_stage(log_path: Path) -> str:
    text = log_path.read_text(errors="replace")
    if "torch.OutOfMemoryError" in text:
        return "gpu_oom_prefill" if '"status": "model_ready"' in text else "gpu_oom_setup"
    return "process_failure"


def trial_folder(root: Path, prompt_tokens: int, batch: int, cohort: int) -> Path:
    return root / (
        f"formal_decode_c{cohort}_basis_joint_p{prompt_tokens}_b{batch}_r{cohort}"
    )


def load_ranks(root: Path, prompt_tokens: int, batch: int, cohort: int) -> list[dict]:
    folder = trial_folder(root, prompt_tokens, batch, cohort)
    paths = [folder / f"rank{rank}.json" for rank in range(8)]
    assert all(path.is_file() for path in paths), folder
    rows = [json.loads(path.read_text()) for path in paths]
    assert {row["rank"] for row in rows} == set(range(8))
    assert all(row["status"] == "complete" for row in rows)
    assert all(row["prompt_tokens"] == prompt_tokens for row in rows)
    assert all(row["batch"] == batch and row["repeat"] == cohort for row in rows)
    assert all(len(row["decode_step_ms"]) == 128 for row in rows)
    assert all(row["decode_step_ms"] == rows[0]["decode_step_ms"] for row in rows)
    assert all(row["generated_token_ids"] == rows[0]["generated_token_ids"] for row in rows)
    return rows


def comparison_metrics(baseline: list[list[int]], optimized: list[list[int]]) -> dict:
    assert len(baseline) == len(optimized)
    prefixes = []
    agreements = []
    exact = 0
    for before, after in zip(baseline, optimized, strict=True):
        assert len(before) == len(after)
        prefix = len(before)
        matches = 0
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            matches += left == right
            if prefix == len(before) and left != right:
                prefix = index
        prefixes.append(prefix)
        agreements.append(matches / len(before))
        exact += before == after
    return {
        "sequence_count": len(baseline),
        "exact_sequences": exact,
        "token_agreement": statistics.fmean(agreements),
        "median_prefix": statistics.median(prefixes),
        "minimum_prefix": min(prefixes),
    }


def median(rows: list[dict], field: str) -> float:
    return statistics.median(row[field] for row in rows)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def gib(value: float | int | None) -> str:
    return "-" if value is None else f"{value / (1 << 30):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root", type=Path,
        default=Path("results/system_benchmarks/llama31_8b_tp8_combined"),
    )
    parser.add_argument(
        "--optimized-root", type=Path,
        default=Path("results/system_benchmarks/llama31_8b_tp8_combined_optimized"),
    )
    args = parser.parse_args()
    baseline_root = args.baseline_root
    optimized_root = args.optimized_root
    manifest = json.loads((optimized_root / "decode_grid_trials.json").read_text())
    assert manifest["status"] == "complete_with_failures"
    assert len(manifest["trials"]) == 48
    trial_index = {
        (row["prompt_tokens"], row["batch"], row["cohort"]): row
        for row in manifest["trials"]
    }
    assert len(trial_index) == 48
    baseline_summary = list(csv.DictReader((baseline_root / "summary.csv").open()))
    baseline_index = {
        (int(row["prompt_tokens"]), int(row["batch"]), row["arm"]): row
        for row in baseline_summary
    }

    trials = []
    first_raw = None
    for prompt_tokens, batches in MATRIX.items():
        for batch in batches:
            for cohort in range(3):
                launch = trial_index[prompt_tokens, batch, cohort]
                if launch["returncode"] != 0:
                    trials.append({
                        "prompt_tokens": prompt_tokens,
                        "batch": batch,
                        "cohort": cohort,
                        "status": "failed",
                        "failure_stage": failure_stage(Path(launch["log"])),
                        "rank_files": 0,
                        "rank_outputs_identical": None,
                        "result_directory": None,
                        "launcher_log": launch["log"],
                    })
                    continue
                optimized_ranks = load_ranks(
                    optimized_root, prompt_tokens, batch, cohort
                )
                baseline_ranks = load_ranks(
                    baseline_root, prompt_tokens, batch, cohort
                )
                optimized = optimized_ranks[0]
                baseline = baseline_ranks[0]
                first_raw = first_raw or optimized
                comparison = comparison_metrics(
                    baseline["generated_token_ids"], optimized["generated_token_ids"]
                )
                steps = optimized["decode_step_ms"]
                slots = [row["slot_counts"] for row in optimized_ranks]
                hits = sum(row["last_step_hits"] for row in slots)
                valid = sum(row["last_step_valid"] for row in slots)
                trials.append({
                    "prompt_tokens": prompt_tokens,
                    "batch": batch,
                    "cohort": cohort,
                    "status": "complete",
                    "failure_stage": None,
                    "rank_files": 8,
                    "rank_outputs_identical": True,
                    "baseline_mean_ms": baseline["decode_step_mean_ms"],
                    "optimized_mean_ms": statistics.fmean(steps),
                    "speedup_vs_baseline": (
                        baseline["decode_step_mean_ms"] / statistics.fmean(steps)
                    ),
                    "optimized_p50_ms": statistics.median(steps),
                    "optimized_p95_ms": percentile(steps, 0.95),
                    "optimized_tokens_s": optimized["decode_tokens_per_second"],
                    "sequence_count": comparison["sequence_count"],
                    "exact_sequences_vs_baseline": comparison["exact_sequences"],
                    "token_agreement_vs_baseline": comparison["token_agreement"],
                    "median_exact_prefix_tokens": comparison["median_prefix"],
                    "minimum_exact_prefix_tokens": comparison["minimum_prefix"],
                    "prefill_peak_maxrank_bytes": max(
                        row["prefill_peak_allocated_bytes"] for row in optimized_ranks
                    ),
                    "decode_resident_maxrank_bytes": max(
                        row["decode_resident_allocated_bytes"] for row in optimized_ranks
                    ),
                    "decode_peak_maxrank_bytes": max(
                        row["decode_peak_allocated_bytes"] for row in optimized_ranks
                    ),
                    "host_exact_k_total_bytes": sum(
                        row["host_persistent_exact_key_bytes"] for row in optimized_ranks
                    ),
                    "slot_hit_rate": hits / valid,
                    "result_directory": str(
                        trial_folder(optimized_root, prompt_tokens, batch, cohort)
                    ),
                    "launcher_log": launch["log"],
                })
    assert first_raw is not None
    assert sum(row["status"] == "complete" for row in trials) == 45
    assert sum(row["status"] == "failed" for row in trials) == 3

    summaries = []
    for prompt_tokens, batches in MATRIX.items():
        for batch in batches:
            rows = [
                row for row in trials
                if row["prompt_tokens"] == prompt_tokens and row["batch"] == batch
            ]
            complete = [row for row in rows if row["status"] == "complete"]
            if not complete:
                summaries.append({
                    "prompt_tokens": prompt_tokens,
                    "batch": batch,
                    "status": "failed",
                    "successful_trials": 0,
                    "failure_stage": ";".join(sorted({row["failure_stage"] for row in rows})),
                })
                continue
            assert len(complete) == 3
            baseline_basis = baseline_index[prompt_tokens, batch, "basis_joint"]
            baseline_dense = baseline_index[prompt_tokens, batch, "dense"]
            baseline_als = baseline_index[prompt_tokens, batch, "als_full"]
            optimized_ms = median(complete, "optimized_mean_ms")
            sequence_count = sum(row["sequence_count"] for row in complete)
            exact_sequences = sum(row["exact_sequences_vs_baseline"] for row in complete)
            summaries.append({
                "prompt_tokens": prompt_tokens,
                "batch": batch,
                "status": "complete",
                "successful_trials": 3,
                "baseline_basis_median_ms": float(baseline_basis["decode_step_mean_ms"]),
                "optimized_basis_median_ms": optimized_ms,
                "speedup_vs_baseline_basis": (
                    float(baseline_basis["decode_step_mean_ms"]) / optimized_ms
                ),
                "speedup_vs_dense": (
                    float(baseline_dense["decode_step_mean_ms"]) / optimized_ms
                    if baseline_dense["status"] == "complete" else None
                ),
                "speedup_vs_als_full": (
                    float(baseline_als["decode_step_mean_ms"]) / optimized_ms
                    if baseline_als["status"] == "complete" else None
                ),
                "optimized_tokens_s": batch * 1000 / optimized_ms,
                "optimized_p50_ms": median(complete, "optimized_p50_ms"),
                "optimized_p95_ms": median(complete, "optimized_p95_ms"),
                "prefill_peak_maxrank_bytes": median(complete, "prefill_peak_maxrank_bytes"),
                "decode_resident_maxrank_bytes": median(
                    complete, "decode_resident_maxrank_bytes"
                ),
                "decode_peak_maxrank_bytes": median(complete, "decode_peak_maxrank_bytes"),
                "host_exact_k_total_bytes": median(complete, "host_exact_k_total_bytes"),
                "slot_hit_rate": median(complete, "slot_hit_rate"),
                "exact_trial_trajectories": sum(
                    row["exact_sequences_vs_baseline"] == row["sequence_count"]
                    for row in complete
                ),
                "sequence_count": sequence_count,
                "exact_sequences_vs_baseline": exact_sequences,
                "token_agreement_vs_baseline": (
                    sum(
                        row["token_agreement_vs_baseline"] * row["sequence_count"]
                        for row in complete
                    ) / sequence_count
                ),
                "median_exact_prefix_tokens": statistics.median(
                    row["median_exact_prefix_tokens"] for row in complete
                ),
                "cohort_optimized_mean_ms": ";".join(
                    f"{row['optimized_mean_ms']:.6f}" for row in complete
                ),
                "failure_stage": None,
            })

    write_csv(optimized_root / "optimization_trial_summary.csv", TRIAL_FIELDS, trials)
    write_csv(optimized_root / "optimization_summary.csv", SUMMARY_FIELDS, summaries)
    valid = [row for row in summaries if row["status"] == "complete"]
    basis_speedups = [row["speedup_vs_baseline_basis"] for row in valid]
    dense_speedups = [row["speedup_vs_dense"] for row in valid if row["speedup_vs_dense"]]
    als_speedups = [row["speedup_vs_als_full"] for row in valid if row["speedup_vs_als_full"]]
    successful_trials = [row for row in trials if row["status"] == "complete"]
    total_sequences = sum(row["sequence_count"] for row in successful_trials)
    exact_sequences = sum(row["exact_sequences_vs_baseline"] for row in successful_trials)
    token_agreement = sum(
        row["token_agreement_vs_baseline"] * row["sequence_count"]
        for row in successful_trials
    ) / total_sequences
    full_exact_trials = sum(
        row["exact_sequences_vs_baseline"] == row["sequence_count"]
        for row in successful_trials
    )
    prefixes = [row["median_exact_prefix_tokens"] for row in successful_trials]
    source_hashes = first_raw["metadata"]["source_sha256"]
    core_hash = source_hashes["basisserve/core/llama31_8b_tp8_combined.py"]
    kernel_hash = source_hashes["basisserve/kernels/csrc/fused_decode_postprocess.cu"]

    lines = [
        "# Llama-3.1-8B-Instruct TP8 Basis-joint specialized decode optimization",
        "",
        "## Outcome",
        "",
        "The optimized Basis-joint grid completed **45/48 trials**. The three preserved "
        "failures are P=130048/B=16 prefill CUDA OOMs, matching the baseline grid; all "
        "decode-capable workloads completed three preselected cohorts.",
        "",
        f"Across the 15 valid workloads, the geometric-mean speedup over the previous "
        f"Basis-joint implementation is **{math.exp(statistics.fmean(math.log(value) for value in basis_speedups)):.3f}x** "
        f"(range {min(basis_speedups):.3f}x–{max(basis_speedups):.3f}x). The optimized "
        f"path is faster than the matched Dense baseline at **{sum(value > 1 for value in dense_speedups)}/{len(dense_speedups)}** "
        f"valid points (geomean {math.exp(statistics.fmean(math.log(value) for value in dense_speedups)):.3f}x), "
        f"and faster than ALS-full at **{sum(value > 1 for value in als_speedups)}/{len(als_speedups)}** points.",
        "",
        "## Specialized implementation",
        "",
        "- Decode Q/K/V projection is one local `4096→736` BF16 GEMM for fixed "
        "`Hq=4/Hkv=1/V=96`, followed by one joint Q/K RoPE call. This removes two GEMM "
        "launches per layer without adding persistent model memory.",
        "- Final page selection, 2048-token support packing, and persistent-slot planning "
        "share one SM89 CUDA CTA; missing exact-K fetch remains a second kernel. This removes "
        "one launch per layer and the separate slot-planner extension call.",
        "- Split-KV remains 16 and output communication remains prepared uniform NCCL. "
        "Microbenchmarks found split tuning below 0.5 µs/layer and IPC AllGather slower than NCCL.",
        f"- Core SHA256: `{core_hash}`; postprocess CUDA SHA256: `{kernel_hash}`.",
        "",
        "## Median results",
        "",
        "Each row is the median of three cohorts. Latency is full-model ms per decode step; "
        "tokens/s is aggregate fixed-batch throughput.",
        "",
        "| Context | B | Old Basis ms | Optimized ms | Speedup | tokens/s | vs Dense | vs ALS-full | GPU prefill GiB | GPU resident GiB | Host K GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        context = "~128K" if row["prompt_tokens"] == 130048 else f"{row['prompt_tokens'] // 1024}K"
        if row["status"] != "complete":
            lines.append(
                f"| {context} | {row['batch']} | - | {row['failure_stage']} | - | - | - | - | - | - | - |"
            )
            continue
        dense = "-" if row["speedup_vs_dense"] is None else f"{row['speedup_vs_dense']:.3f}x"
        als = "-" if row["speedup_vs_als_full"] is None else f"{row['speedup_vs_als_full']:.3f}x"
        lines.append(
            f"| {context} | {row['batch']} | {row['baseline_basis_median_ms']:.2f} | "
            f"{row['optimized_basis_median_ms']:.2f} | {row['speedup_vs_baseline_basis']:.3f}x | "
            f"{row['optimized_tokens_s']:.2f} | {dense} | {als} | "
            f"{gib(row['prefill_peak_maxrank_bytes'])} | {gib(row['decode_resident_maxrank_bytes'])} | "
            f"{gib(row['host_exact_k_total_bytes'])} |"
        )
    lines.extend([
        "",
        "## Correctness and numerical behavior",
        "",
        "- Every successful trial has all eight rank JSON files. The 128 timing samples and "
        "generated IDs are identical across ranks inside each TP8 replica.",
        "- Stateful kernel validation over realistic 512-candidate routing confirmed exact "
        "selected pages, support IDs, slot hit counts, and fetched exact K versus the old "
        "two-call postprocess/slot path.",
        f"- The fused BF16 projection is not bitwise equivalent to three separate GEMMs: "
        f"**{full_exact_trials}/45 trials** and **{exact_sequences}/{total_sequences} sequences** "
        f"retain the complete 145-token greedy trajectory. Position-wise token agreement is "
        f"**{100 * token_agreement:.2f}%** and the median per-trial median exact prefix is "
        f"**{statistics.median(prefixes):.1f} tokens**. Autoregressive greedy divergence is not "
        "a task-quality metric, so this speed grid must not be presented as a quality evaluation.",
        "- The existing limited RULER checks were not expanded in this run, following the "
        "decision to stop enlarging quality experiments and focus on speed.",
        "",
        "## Command and warnings",
        "",
        "Environment: `basis`.",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda \\",
        "conda run --no-capture-output -n basis python \\",
        "  benchmarks/system/run_llama31_8b_tp8_decode_grid.py \\",
        "  --arms basis_joint --contexts 4096 16384 65536 130048 \\",
        "  --cohorts 0 1 2 --conditioning-steps 16 --measure-steps 128 \\",
        "  --output-root results/system_benchmarks/llama31_8b_tp8_combined_optimized",
        "```",
        "",
        "Pinned-memory `set_mempolicy` returned `Operation not permitted`; GPU-local CPU "
        "affinity still succeeded. P=130048/B=16 failed during one-shot prefill when PyTorch "
        "requested an additional 15.88 GiB with only about 9.4 GiB free per rank.",
        "",
        "## Artifacts",
        "",
        "- `optimization_summary.csv`: 16 workload summaries.",
        "- `optimization_trial_summary.csv`: all 48 cohort trials, including the three OOMs.",
        "- `decode_grid_trials.json`: commands, timestamps, return codes, and launcher logs.",
        "- `formal_decode_*`: raw per-rank JSON and logs.",
        "",
    ])
    report_path = optimized_root / "OPTIMIZATION_SUMMARY.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "successful_trials": 45,
        "failed_trials": 3,
        "valid_workloads": len(valid),
        "geomean_speedup": math.exp(
            statistics.fmean(math.log(value) for value in basis_speedups)
        ),
        "report": str(report_path),
        "summary_csv": str(optimized_root / "optimization_summary.csv"),
        "trial_csv": str(optimized_root / "optimization_trial_summary.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
