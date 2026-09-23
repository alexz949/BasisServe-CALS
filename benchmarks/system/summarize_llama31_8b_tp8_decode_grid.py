"""Validate and summarize the Llama-3.1-8B TP8 combined decode grid."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPOSITORY_ROOT / "results/system_benchmarks/llama31_8b_tp8_combined"
FACTOR_MANIFEST = Path("/workspace/runs/l31-cal128/tp8-combined-v96/manifest.json")
ROUTER_ROOT = Path("/workspace/runs/l31-cal128/router/ours_b16r16")
ARMS = ("dense", "als_full", "basis_joint")
ARM_LABELS = {
    "dense": "Dense",
    "als_full": "ALS-full V96",
    "basis_joint": "Basis-joint V96+B16R16",
}
MATRIX = {
    4096: (1, 8, 32, 128),
    16384: (1, 4, 8, 16),
    65536: (1, 4, 8, 16),
    130048: (1, 4, 8, 16),
}
TRIAL_FIELDS = (
    "experiment", "arm", "tp", "dp", "prompt_tokens", "nominal_context",
    "batch", "output_tokens", "cohort", "prompt_cohort_hash", "status",
    "failure_stage", "failure_detail", "rank_files", "rank_outputs_identical",
    "cache_length_measurement_start", "cache_length_measurement_end",
    "prefill_cuda_ms_maxrank", "prefill_wall_s_maxrank", "decode_step_mean_ms",
    "decode_step_p50_ms", "decode_step_p95_ms", "decode_tokens_s",
    "gpu_peak_prefill_maxrank_bytes", "gpu_peak_prefill_sumrank_bytes",
    "gpu_decode_resident_maxrank_bytes", "gpu_decode_resident_sumrank_bytes",
    "gpu_peak_decode_maxrank_bytes", "gpu_peak_decode_sumrank_bytes",
    "host_persistent_total_bytes", "process_max_rss_sumrank_bytes",
    "k_slot_hits_last_step", "k_slot_valid_last_step", "k_slot_hit_rate_last_step",
    "generated_token_ids_sha256", "result_directory", "launcher_log",
)
SUMMARY_FIELDS = (
    "experiment", "arm", "tp", "dp", "prompt_tokens", "nominal_context",
    "batch", "output_tokens", "value_rank_schedule_hash", "router_hash",
    "implementation_commit", "prompt_cohort_hash", "key_placement", "routing_mode",
    "k_slots", "dtype", "engine", "graph_mode", "prefill_policy", "trial_count",
    "successful_trials", "status", "ttft_mean_ms", "ttft_p50_ms", "tpot_mean_ms",
    "cohort_e2e_s", "cohort_output_tokens_s", "e2e_speedup_vs_dense",
    "decode_step_mean_ms", "decode_step_p50_ms", "decode_step_p95_ms",
    "decode_tokens_s", "decode_speedup_vs_dense", "prefill_cuda_ms_maxrank",
    "prefill_wall_s_maxrank", "gpu_peak_prefill_maxrank_bytes",
    "gpu_peak_prefill_sumrank_bytes", "gpu_decode_resident_maxrank_bytes",
    "gpu_decode_resident_sumrank_bytes", "gpu_peak_decode_maxrank_bytes",
    "gpu_peak_decode_sumrank_bytes", "gpu_pool_capacity_maxrank_bytes",
    "gpu_active_cache_maxrank_bytes", "host_persistent_total_bytes",
    "host_peak_total_bytes", "selected_unique_tokens_mean", "k_slot_hit_rate",
    "preemptions", "recomputed_tokens", "actual_active_decode_batch", "b_max",
    "failure_stage", "cohort_decode_step_mean_ms", "cohort_decode_tokens_s",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path, pattern: str) -> str:
    digest = hashlib.sha256()
    paths = sorted(root.glob(pattern))
    assert paths
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def classify_failure(log_path: Path) -> tuple[str, str]:
    text = log_path.read_text(errors="replace")
    oom = re.search(r"torch\.OutOfMemoryError: ([^\n]+)", text)
    if oom:
        if '"status": "prefill_complete"' in text:
            stage = "gpu_oom_decode"
        elif '"status": "model_ready"' in text:
            stage = "gpu_oom_prefill"
        else:
            stage = "gpu_oom_setup"
        return stage, oom.group(1).strip()
    host = re.search(r"(Cannot allocate memory|MemoryError[^\n]*)", text)
    if host:
        return "host_oom", host.group(1).strip()
    return "process_failure", "nonzero launcher return code; inspect launcher log"


def load_success(root: Path, trial: dict) -> tuple[dict, dict]:
    arm = trial["arm"]
    prompt_tokens = trial["prompt_tokens"]
    batch = trial["batch"]
    cohort = trial["cohort"]
    folder = root / f"formal_decode_c{cohort}_{arm}_p{prompt_tokens}_b{batch}_r{cohort}"
    rank_paths = [folder / f"rank{rank}.json" for rank in range(8)]
    assert all(path.is_file() for path in rank_paths), folder
    ranks = [json.loads(path.read_text()) for path in rank_paths]
    assert {row["rank"] for row in ranks} == set(range(8))
    assert all(row["status"] == "complete" for row in ranks)
    assert all(row["arm"] == arm for row in ranks)
    assert all(row["prompt_tokens"] == prompt_tokens for row in ranks)
    assert all(row["batch"] == batch for row in ranks)
    assert all(row["repeat"] == cohort for row in ranks)
    assert all(row["measured_steps"] == 128 for row in ranks)
    assert all(len(row["decode_step_ms"]) == 128 for row in ranks)
    step_times = ranks[0]["decode_step_ms"]
    assert all(row["decode_step_ms"] == step_times for row in ranks)
    token_ids = ranks[0]["generated_token_ids"]
    assert all(row["generated_token_ids"] == token_ids for row in ranks)
    token_digest = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()
    ).hexdigest()
    slots = [row["slot_counts"] for row in ranks]
    slot_hits = sum(row["last_step_hits"] for row in slots) if slots[0] else None
    slot_valid = sum(row["last_step_valid"] for row in slots) if slots[0] else None
    slot_rate = slot_hits / slot_valid if slot_valid else None
    result = {
        "experiment": "fixed_batch_decode",
        "arm": arm,
        "tp": 8,
        "dp": 1,
        "prompt_tokens": prompt_tokens,
        "nominal_context": "~128K" if prompt_tokens == 130048 else f"{prompt_tokens // 1024}K",
        "batch": batch,
        "output_tokens": 128,
        "cohort": cohort,
        "prompt_cohort_hash": ranks[0]["prompt_cohort"]["cohort_hash"],
        "status": "complete",
        "failure_stage": None,
        "failure_detail": None,
        "rank_files": 8,
        "rank_outputs_identical": True,
        "cache_length_measurement_start": prompt_tokens + 16,
        "cache_length_measurement_end": prompt_tokens + 16 + 128,
        "prefill_cuda_ms_maxrank": max(row["prefill_cuda_ms"] for row in ranks),
        "prefill_wall_s_maxrank": max(row["prefill_wall_seconds"] for row in ranks),
        "decode_step_mean_ms": statistics.fmean(step_times),
        "decode_step_p50_ms": statistics.median(step_times),
        "decode_step_p95_ms": percentile(step_times, 0.95),
        "decode_tokens_s": statistics.median(
            row["decode_tokens_per_second"] for row in ranks
        ),
        "gpu_peak_prefill_maxrank_bytes": max(
            row["prefill_peak_allocated_bytes"] for row in ranks
        ),
        "gpu_peak_prefill_sumrank_bytes": sum(
            row["prefill_peak_allocated_bytes"] for row in ranks
        ),
        "gpu_decode_resident_maxrank_bytes": max(
            row["decode_resident_allocated_bytes"] for row in ranks
        ),
        "gpu_decode_resident_sumrank_bytes": sum(
            row["decode_resident_allocated_bytes"] for row in ranks
        ),
        "gpu_peak_decode_maxrank_bytes": max(
            row["decode_peak_allocated_bytes"] for row in ranks
        ),
        "gpu_peak_decode_sumrank_bytes": sum(
            row["decode_peak_allocated_bytes"] for row in ranks
        ),
        "host_persistent_total_bytes": sum(
            row["host_persistent_exact_key_bytes"] for row in ranks
        ),
        "process_max_rss_sumrank_bytes": sum(row["process_max_rss_bytes"] for row in ranks),
        "k_slot_hits_last_step": slot_hits,
        "k_slot_valid_last_step": slot_valid,
        "k_slot_hit_rate_last_step": slot_rate,
        "generated_token_ids_sha256": token_digest,
        "result_directory": str(folder),
        "launcher_log": trial["log"],
    }
    return result, ranks[0]


def failure_row(trial: dict) -> dict:
    stage, detail = classify_failure(Path(trial["log"]))
    return {
        "experiment": "fixed_batch_decode",
        "arm": trial["arm"],
        "tp": 8,
        "dp": 1,
        "prompt_tokens": trial["prompt_tokens"],
        "nominal_context": (
            "~128K" if trial["prompt_tokens"] == 130048
            else f"{trial['prompt_tokens'] // 1024}K"
        ),
        "batch": trial["batch"],
        "output_tokens": 128,
        "cohort": trial["cohort"],
        "prompt_cohort_hash": None,
        "status": "failed",
        "failure_stage": stage,
        "failure_detail": detail,
        "rank_files": 0,
        "rank_outputs_identical": None,
        "cache_length_measurement_start": None,
        "cache_length_measurement_end": None,
        "prefill_cuda_ms_maxrank": None,
        "prefill_wall_s_maxrank": None,
        "decode_step_mean_ms": None,
        "decode_step_p50_ms": None,
        "decode_step_p95_ms": None,
        "decode_tokens_s": None,
        "gpu_peak_prefill_maxrank_bytes": None,
        "gpu_peak_prefill_sumrank_bytes": None,
        "gpu_decode_resident_maxrank_bytes": None,
        "gpu_decode_resident_sumrank_bytes": None,
        "gpu_peak_decode_maxrank_bytes": None,
        "gpu_peak_decode_sumrank_bytes": None,
        "host_persistent_total_bytes": None,
        "process_max_rss_sumrank_bytes": None,
        "k_slot_hits_last_step": None,
        "k_slot_valid_last_step": None,
        "k_slot_hit_rate_last_step": None,
        "generated_token_ids_sha256": None,
        "result_directory": None,
        "launcher_log": trial["log"],
    }


def median(rows: list[dict], field: str) -> float | None:
    values = [row[field] for row in rows if row[field] is not None]
    return statistics.median(values) if values else None


def make_summaries(trials: list[dict], metadata: dict) -> list[dict]:
    factor_hash = sha256_file(FACTOR_MANIFEST)
    router_hash = sha256_tree(ROUTER_ROOT, "layer_*.json")
    summaries = []
    for prompt_tokens, batches in MATRIX.items():
        for batch in batches:
            for arm in ARMS:
                rows = [
                    row for row in trials
                    if row["prompt_tokens"] == prompt_tokens
                    and row["batch"] == batch and row["arm"] == arm
                ]
                assert len(rows) == 3
                completed = [row for row in rows if row["status"] == "complete"]
                status = "complete" if len(completed) == 3 else (
                    "failed" if not completed else "partial"
                )
                stages = sorted({row["failure_stage"] for row in rows if row["failure_stage"]})
                summaries.append({
                    "experiment": "fixed_batch_decode",
                    "arm": arm,
                    "tp": 8,
                    "dp": 1,
                    "prompt_tokens": prompt_tokens,
                    "nominal_context": rows[0]["nominal_context"],
                    "batch": batch,
                    "output_tokens": 128,
                    "value_rank_schedule_hash": factor_hash if arm != "dense" else None,
                    "router_hash": router_hash if arm == "basis_joint" else None,
                    "implementation_commit": metadata["git_commit"],
                    "prompt_cohort_hash": ";".join(
                        row["prompt_cohort_hash"] or "unavailable" for row in rows
                    ),
                    "key_placement": (
                        "pinned_host_exact_k_with_gpu_slots" if arm == "basis_joint" else "gpu"
                    ),
                    "routing_mode": (
                        "two_stage_512_persistent_slots" if arm == "basis_joint" else "full"
                    ),
                    "k_slots": 2048 if arm == "basis_joint" else None,
                    "dtype": "bfloat16",
                    "engine": "custom_fixed_batch_transformers",
                    "graph_mode": "eager",
                    "prefill_policy": "full_context_causal_arm_value_representation",
                    "trial_count": 3,
                    "successful_trials": len(completed),
                    "status": status,
                    "ttft_mean_ms": None,
                    "ttft_p50_ms": None,
                    "tpot_mean_ms": None,
                    "cohort_e2e_s": None,
                    "cohort_output_tokens_s": None,
                    "e2e_speedup_vs_dense": None,
                    "decode_step_mean_ms": median(completed, "decode_step_mean_ms"),
                    "decode_step_p50_ms": median(completed, "decode_step_p50_ms"),
                    "decode_step_p95_ms": median(completed, "decode_step_p95_ms"),
                    "decode_tokens_s": median(completed, "decode_tokens_s"),
                    "decode_speedup_vs_dense": None,
                    "prefill_cuda_ms_maxrank": median(completed, "prefill_cuda_ms_maxrank"),
                    "prefill_wall_s_maxrank": median(completed, "prefill_wall_s_maxrank"),
                    "gpu_peak_prefill_maxrank_bytes": median(
                        completed, "gpu_peak_prefill_maxrank_bytes"
                    ),
                    "gpu_peak_prefill_sumrank_bytes": median(
                        completed, "gpu_peak_prefill_sumrank_bytes"
                    ),
                    "gpu_decode_resident_maxrank_bytes": median(
                        completed, "gpu_decode_resident_maxrank_bytes"
                    ),
                    "gpu_decode_resident_sumrank_bytes": median(
                        completed, "gpu_decode_resident_sumrank_bytes"
                    ),
                    "gpu_peak_decode_maxrank_bytes": median(
                        completed, "gpu_peak_decode_maxrank_bytes"
                    ),
                    "gpu_peak_decode_sumrank_bytes": median(
                        completed, "gpu_peak_decode_sumrank_bytes"
                    ),
                    "gpu_pool_capacity_maxrank_bytes": None,
                    "gpu_active_cache_maxrank_bytes": None,
                    "host_persistent_total_bytes": median(
                        completed, "host_persistent_total_bytes"
                    ),
                    "host_peak_total_bytes": None,
                    "selected_unique_tokens_mean": None,
                    "k_slot_hit_rate": median(completed, "k_slot_hit_rate_last_step"),
                    "preemptions": 0 if status == "complete" else None,
                    "recomputed_tokens": 0 if status == "complete" else None,
                    "actual_active_decode_batch": batch if status == "complete" else None,
                    "b_max": None,
                    "failure_stage": ";".join(stages) if stages else None,
                    "cohort_decode_step_mean_ms": ";".join(
                        f"{row['decode_step_mean_ms']:.6f}" for row in rows
                        if row["decode_step_mean_ms"] is not None
                    ),
                    "cohort_decode_tokens_s": ";".join(
                        f"{row['decode_tokens_s']:.6f}" for row in rows
                        if row["decode_tokens_s"] is not None
                    ),
                })
    by_key = {(row["prompt_tokens"], row["batch"], row["arm"]): row for row in summaries}
    for row in summaries:
        dense = by_key[(row["prompt_tokens"], row["batch"], "dense")]
        if row["status"] == "complete" and dense["status"] == "complete":
            row["decode_speedup_vs_dense"] = (
                dense["decode_step_mean_ms"] / row["decode_step_mean_ms"]
            )
    return summaries


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def fnum(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def gib(value: float | None) -> str:
    return "-" if value is None else f"{value / (1 << 30):.2f}"


def markdown_report(
    summaries: list[dict], *, successes: int, failures: int, metadata: dict,
    factor_hash: str, router_hash: str,
) -> str:
    matched = [row for row in summaries if row["decode_speedup_vs_dense"] is not None]
    basis = [row for row in matched if row["arm"] == "basis_joint"]
    als = [row for row in matched if row["arm"] == "als_full"]
    basis_wins = sum(row["decode_speedup_vs_dense"] > 1 for row in basis)
    als_wins = sum(row["decode_speedup_vs_dense"] > 1 for row in als)
    lines = [
        "# Llama-3.1-8B-Instruct TP8 Joint-ALS + key-routing decode benchmark",
        "",
        "## Outcome",
        "",
        f"The frozen Experiment B grid completed with **{successes}/144 successful trials** "
        f"and **{failures} preserved failures**. Each summary row is the median of three "
        "preselected real-text cohorts; each successful trial contains 16 conditioning "
        "steps followed by 128 measured autoregressive decode steps.",
        "",
        f"ALS-full is faster than Dense at **{als_wins}/{len(als)}** valid matched "
        f"workloads; Basis-joint is faster than Dense at **{basis_wins}/{len(basis)}**. "
        "A speedup is omitted wherever Dense did not complete.",
        "",
        "This report covers the pure fixed-batch decode grid only. It is not a complete "
        "request/TTFT benchmark, capacity search, or profiler breakdown.",
        "",
        "## Configuration",
        "",
        "- Model: Llama-3.1-8B-Instruct, local snapshot "
        "`0e9e39f249a16976918f6564b8830bc894c89659`.",
        "- Replica: TP=8, DP=1, PP=1 on 8x NVIDIA L40S (46,068 MiB each); PCIe-only "
        "topology with cross-socket `SYS` links and no NVLink.",
        "- Software: PyTorch " + str(metadata["pytorch"]) + ", CUDA "
        + str(metadata["cuda"]) + ", NCCL " + ".".join(map(str, metadata["nccl"])) + ".",
        "- Precision: BF16; TF32 disabled; eager execution.",
        "- Basis route: V96 + Base16/Residual16, Page32, two-stage512, 62 routed pages "
        "+ recent64 = hard physical support 2048, persistent exact-K GPU slots.",
        "- Basis placement: V96/R16/metadata/K slots on GPU; historical exact K in "
        "pinned host memory. ALS-full and Dense keep their complete cache on GPU.",
        f"- Factor manifest SHA256: `{factor_hash}`.",
        f"- Router layer-bank SHA256: `{router_hash}`.",
        f"- Implementation commit recorded by the trials: `{metadata['git_commit']}` "
        "(the worktree was dirty; per-trial source hashes are retained in raw JSON).",
        "- Command: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 conda run "
        "--no-capture-output -n basis python "
        "benchmarks/system/run_llama31_8b_tp8_decode_grid.py`.",
        "",
        "## Median results",
        "",
        "`GPU prefill` and `GPU resident` are max-rank PyTorch allocated bytes, not the "
        "sum across the replica. `Host K` is the sum of rank-private exact-K capacity.",
        "",
    ]
    for prompt_tokens in MATRIX:
        label = "~128K (P=130048)" if prompt_tokens == 130048 else f"{prompt_tokens // 1024}K"
        lines.extend([
            f"### {label}",
            "",
            "| B | Arm | Status | Mean ms | P50 ms | P95 ms | tokens/s | vs Dense | "
            "GPU prefill GiB | GPU resident GiB | Host K GiB | slot hit |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for batch in MATRIX[prompt_tokens]:
            for arm in ARMS:
                row = next(
                    item for item in summaries
                    if item["prompt_tokens"] == prompt_tokens
                    and item["batch"] == batch and item["arm"] == arm
                )
                status = "OK" if row["status"] == "complete" else (
                    row["failure_stage"] or row["status"]
                )
                speedup = (
                    "-" if row["decode_speedup_vs_dense"] is None
                    else f"{row['decode_speedup_vs_dense']:.3f}x"
                )
                hit = (
                    "-" if row["k_slot_hit_rate"] is None
                    else f"{100 * row['k_slot_hit_rate']:.1f}%"
                )
                lines.append(
                    f"| {batch} | {ARM_LABELS[arm]} | {status} | "
                    f"{fnum(row['decode_step_mean_ms'])} | {fnum(row['decode_step_p50_ms'])} | "
                    f"{fnum(row['decode_step_p95_ms'])} | {fnum(row['decode_tokens_s'])} | "
                    f"{speedup} | {gib(row['gpu_peak_prefill_maxrank_bytes'])} | "
                    f"{gib(row['gpu_decode_resident_maxrank_bytes'])} | "
                    f"{gib(row['host_persistent_total_bytes'])} | {hit} |"
                )
        lines.append("")
    lines.extend([
        "## Capacity observations",
        "",
        "- P=65536, B=16: Dense failed all three cohorts during prefill; ALS-full and "
        "Basis-joint completed all three. This is the clean measured point where Dense "
        "cannot run but both compressed paths can.",
        "- P=130048, B=8: all three arms completed all cohorts.",
        "- P=130048, B=16: all three arms failed during full-context prefill. The immediate "
        "allocation was a 15.88 GiB prompt/hidden-state tensor, so this identifies the "
        "current one-shot fixed-runner prefill peak, not a decode-cache-only limit.",
        "- These are workload-grid observations, not Experiment C `B_max`: no integer "
        "capacity search or three-trial boundary verification was performed beyond the grid.",
        "",
        "## Validation and caveats",
        "",
        "- All 132 successful trials contain all eight rank JSON files. Synchronized "
        "128-step timing arrays and generated token IDs are identical across ranks within "
        "each TP8 replica.",
        "- Step latency is full-model replica latency and includes projections, routing, "
        "K-slot planning/fetch, attention, TP collectives, decoder, MLP, LM head, and greedy "
        "token choice. It is not divided by batch size.",
        "- CPU affinity was bound to the GPU-local NUMA node, but `set_mempolicy` returned "
        "`Operation not permitted`; strict NUMA placement of pinned host pages was therefore "
        "not enforceable in this container.",
        "- Host K is allocation capacity, not sampled physical RSS. NVML phase samples, "
        "active-cache byte itemization, host peak, and mean/P95/max unique selected-token "
        "counts were not collected by this runner. The slot metric is the weighted hit rate "
        "from the final measured step only.",
        "- `prefill_cuda_ms` is retained in CSV/raw JSON for diagnostic context, but this "
        "Experiment B report does not present it as TTFT or request E2E.",
        "",
        "## Artifacts",
        "",
        "- `summary.csv`: one median row per arm/P/B (48 rows).",
        "- `decode_trial_summary.csv`: all 144 cohort trials, including failure stage/log.",
        "- `decode_grid_trials.json`: launcher commands, timestamps, and return codes.",
        "- `formal_decode_*`: per-rank raw JSON, 128 step samples, token IDs, memory, and logs.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    root = DEFAULT_ROOT
    manifest = json.loads((root / "decode_grid_trials.json").read_text())
    assert manifest["status"] == "complete_with_failures"
    assert len(manifest["trials"]) == 144
    keys = {
        (row["arm"], row["prompt_tokens"], row["batch"], row["cohort"])
        for row in manifest["trials"]
    }
    assert len(keys) == 144
    expected = {
        (arm, prompt_tokens, batch, cohort)
        for prompt_tokens, batches in MATRIX.items()
        for batch in batches for cohort in range(3) for arm in ARMS
    }
    assert keys == expected
    trial_rows = []
    first_success = None
    for trial in manifest["trials"]:
        if trial["returncode"] == 0:
            row, raw = load_success(root, trial)
            first_success = first_success or raw
        else:
            row = failure_row(trial)
        trial_rows.append(row)
    assert first_success is not None
    successes = sum(row["status"] == "complete" for row in trial_rows)
    failures = len(trial_rows) - successes
    assert successes == 132 and failures == 12
    summaries = make_summaries(trial_rows, first_success["metadata"])
    assert len(summaries) == 48
    write_csv(root / "decode_trial_summary.csv", TRIAL_FIELDS, trial_rows)
    write_csv(root / "summary.csv", SUMMARY_FIELDS, summaries)
    factor_hash = sha256_file(FACTOR_MANIFEST)
    router_hash = sha256_tree(ROUTER_ROOT, "layer_*.json")
    report = markdown_report(
        summaries, successes=successes, failures=failures,
        metadata=first_success["metadata"], factor_hash=factor_hash, router_hash=router_hash,
    )
    (root / "SUMMARY.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "successful_trials": successes,
        "failed_trials": failures,
        "summary_rows": len(summaries),
        "summary_csv": str(root / "summary.csv"),
        "trial_csv": str(root / "decode_trial_summary.csv"),
        "report": str(root / "SUMMARY.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
