"""Validate raw TP8 ShadowKV trials and prepare request, decode, and memory tables."""

from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path
import statistics

from benchmarks.system.run_llama31_8b_tp8_request_grid import DEFAULT_OUTPUT, FULL_GRID


REQUEST_FIELDS = (
    "prefill_ms", "representation_build_ms", "k_gather_ms", "svd_ms",
    "factor_redistribution_ms", "other_prepare_ms", "first_token_ms",
    "representation_ready_ms", "decode_after_first_token_ms",
    "decode_after_representation_ready_ms", "total_request_ms",
    "output_tokens_per_second",
)


def write_csv(path, rows, fieldnames=None):
    assert rows or fieldnames
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_validated_trial(attempt):
    folder = Path(attempt["output_dir"])
    replica = json.loads((folder / "replica.json").read_text())
    ranks = [json.loads((folder / f"rank{rank}.json").read_text()) for rank in range(8)]
    assert replica["status"] == "complete" and replica["successful_ranks"] == 8
    assert {row["rank"] for row in ranks} == set(range(8))
    assert len({row["metadata"]["git_commit"] for row in ranks}) == 1
    assert all(row["status"] == "complete" and row["schema"] == "basisserve.llama31_8b.tp8_request.v1"
               for row in ranks)
    assert all(row["generated_token_ids"] == ranks[0]["generated_token_ids"] for row in ranks)
    for row in ranks:
        for field in ("arm", "mode", "prompt_tokens", "batch", "cohort"):
            assert row[field] == attempt[field] == replica[field]
        assert len(row["generated_token_ids"][0]) == (128 if attempt["mode"] == "request" else 145)
    if attempt["mode"] == "request":
        request = replica["request"]
        expected_phases = (32 * (2 + 3 * attempt["batch"]) + 1
                           if attempt["arm"] == "shadowkv" else 0)
        assert len(request["phase_timeline"]) == expected_phases
        assert abs(sum(request[name] for name in request["figure_stack_fields"])
                   - request["total_request_ms"]) < 1e-5
        assert request["total_request_ms"] >= request["representation_ready_ms"]
    else:
        steady = replica["steady_decode"]
        assert steady["conditioning_steps"] == 16 and steady["measured_steps"] == 128
        assert len(steady["step_ms_max_rank"]) == 128
    return replica, ranks


def group_status(rows):
    if all(row["status"] == "complete" for row in rows):
        return "complete"
    if any(row["status"] == "failed_oom" for row in rows):
        return "oom"
    if any(row["status"] == "failed" for row in rows):
        return "failed"
    return "incomplete"


def figure_rows(request_rows):
    return [row for row in request_rows
            if row["arm"] in ("basis_joint", "shadowkv")
            and (row["prompt_tokens"], row["batch"]) in (
                (65536, 1), (65536, 8), (130048, 1), (130048, 8))]


def plot_request_breakdown(rows, output):
    chosen = figure_rows(rows)
    assert len(chosen) == 8
    if any(row["status"] == "incomplete" for row in chosen):
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    positions = [(65536, 1), (65536, 8), (130048, 1), (130048, 8)]
    labels = ["64K\nB1", "64K\nB8", "~128K\nB1", "~128K\nB8"]
    colors = {"prefill_ms": "#40729a", "representation_build_ms": "#da9550",
              "decode_after_representation_ready_ms": "#368b73"}
    names = {"prefill_ms": "Prefill", "representation_build_ms": "Online construction",
             "decode_after_representation_ready_ms": "Decode after ready"}
    x = np.arange(4)
    width = 0.34
    present = [row for row in chosen if row["status"] == "complete"]
    maximum = max((row["total_request_ms"] for row in present), default=1000)
    top = maximum * 1.12
    fig, ax = plt.subplots(figsize=(9.0, 4.7), layout="constrained")
    for arm, offset, arm_name in (("basis_joint", -width / 2, "BasisKV"),
                                   ("shadowkv", width / 2, "ShadowKV")):
        for index, (length, batch) in enumerate(positions):
            row = next(item for item in chosen if item["arm"] == arm
                       and item["prompt_tokens"] == length and item["batch"] == batch)
            xpos = x[index] + offset
            if row["status"] != "complete":
                ax.scatter([xpos], [top], marker="x", s=70, color="#b54046", zorder=5)
                ax.text(xpos, top * 1.025, "OOM" if row["status"] == "oom" else "failed",
                        ha="center", va="bottom", fontsize=8, color="#b54046")
                continue
            bottom = 0
            for field in colors:
                ax.bar(xpos, row[field], width, bottom=bottom, color=colors[field],
                       label=names[field] if index == 0 and arm == "basis_joint" else None)
                bottom += row[field]
            ax.errorbar(xpos, bottom, yerr=[[max(0.0, bottom - row["cohort_min_total_ms"])],
                                                 [max(0.0, row["cohort_max_total_ms"] - bottom)]],
                        fmt="none", ecolor="#30343a", capsize=2, lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("128-token request latency (ms)")
    ax.set_title("TP8 request latency")
    ax.set_ylim(0, top * 1.14)
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.01, -0.20,
            "Left: BasisKV   Right: ShadowKV. Bars use the median-total cohort.\n"
            "BasisKV projection, encoding, and cache writes count in prefill; online construction is ShadowKV-specific.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "shadowkv_request_breakdown.pdf", bbox_inches="tight")
    fig.savefig(output / "shadowkv_request_breakdown.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.output_root
    manifest = json.loads((root / "grid_trials.json").read_text())
    assert manifest["grid"] == FULL_GRID
    latest = {}
    failures = []
    for attempt in manifest["attempts"]:
        key = tuple(attempt[field] for field in ("arm", "mode", "prompt_tokens", "batch", "cohort"))
        latest[key] = attempt
        if attempt["status"] != "complete":
            failure = attempt.get("failure", {})
            failures.append({"arm": attempt["arm"], "mode": attempt["mode"],
                             "prompt_tokens": attempt["prompt_tokens"], "batch": attempt["batch"],
                             "cohort": attempt["cohort"], "attempt": attempt["attempt"],
                             "oom": failure.get("oom"),
                             "allocation_request": failure.get("allocation_request"),
                             "missing_rank_results": json.dumps(failure.get("missing_rank_results")),
                             "last_rank_stages": json.dumps(failure.get("last_rank_stages")),
                             "truncated_rank_logs": json.dumps(failure.get("truncated_rank_logs")),
                             "returncode": attempt.get("returncode"),
                             "launcher_log": attempt["launcher_log"]})
    trials = []
    validated = {}
    for arm, mode, length, batch, cohort in product(
            FULL_GRID["arms"], FULL_GRID["modes"], FULL_GRID["contexts"],
            FULL_GRID["batches"], FULL_GRID["cohorts"]):
        key = (arm, mode, length, batch, cohort)
        attempt = latest.get(key)
        status = ("not_run" if attempt is None else
                  "complete" if attempt["status"] == "complete" else
                  "failed_oom" if attempt.get("failure", {}).get("oom") else "failed")
        row = {"arm": arm, "mode": mode, "prompt_tokens": length, "batch": batch,
               "cohort": cohort, "status": status,
               "attempt": attempt["attempt"] if attempt else None,
               "raw_dir": attempt["output_dir"] if attempt else None,
               **{field: None for field in REQUEST_FIELDS},
               "mean_ms_per_step": None, "p50_ms": None, "p95_ms": None,
               "aggregate_tokens_per_second": None,
               "gpu_allocated_ready_max_rank_bytes": None,
               "gpu_reserved_ready_max_rank_bytes": None,
               "gpu_allocated_peak_request_max_rank_bytes": None,
               "gpu_allocated_peak_decode_max_rank_bytes": None,
               "nvml_process_ready_max_rank_bytes": None,
               "cpu_value_sum_bytes": None, "cpu_key_sum_bytes": None}
        if status == "complete":
            replica, ranks = load_validated_trial(attempt)
            validated[key] = (replica, ranks)
            if mode == "request":
                row.update({field: replica["request"][field] for field in REQUEST_FIELDS})
            else:
                row.update({field: replica["steady_decode"][field] for field in (
                    "mean_ms_per_step", "p50_ms", "p95_ms", "aggregate_tokens_per_second")})
            row["gpu_allocated_ready_max_rank_bytes"] = replica["memory_max_rank"]["representation_ready"]
            row["gpu_reserved_ready_max_rank_bytes"] = max(
                item["memory"]["representation_ready"]["reserved_bytes"] for item in ranks)
            row["gpu_allocated_peak_request_max_rank_bytes"] = (
                replica["peak_request_allocated_max_rank_bytes"] if mode == "request" else None)
            row["gpu_allocated_peak_decode_max_rank_bytes"] = max(
                item["memory"]["decode_end"]["peak_allocated_bytes"] for item in ranks)
            row["nvml_process_ready_max_rank_bytes"] = max(
                item["memory"]["representation_ready"]["nvml_process_bytes"] for item in ranks)
            row["cpu_value_sum_bytes"] = replica["pinned_host_value_sum_bytes"]
            row["cpu_key_sum_bytes"] = replica["pinned_host_exact_key_sum_bytes"]
        trials.append(row)
    write_csv(root / "summary.csv", trials)
    write_csv(root / "failures.csv", failures, fieldnames=(
        "arm", "mode", "prompt_tokens", "batch", "cohort", "attempt", "oom",
        "allocation_request", "missing_rank_results", "last_rank_stages",
        "truncated_rank_logs",
        "returncode", "launcher_log"))

    request_rows = []
    steady_rows = []
    memory_rows = []
    for arm, length, batch in product(FULL_GRID["arms"], FULL_GRID["contexts"], FULL_GRID["batches"]):
        for mode in FULL_GRID["modes"]:
            cohort_rows = [row for row in trials if row["arm"] == arm and row["mode"] == mode
                           and row["prompt_tokens"] == length and row["batch"] == batch]
            status = group_status(cohort_rows)
            complete = [row for row in cohort_rows if row["status"] == "complete"]
            common = {"arm": arm, "prompt_tokens": length, "batch": batch,
                      "status": status, "successful_cohorts": len(complete),
                      "non_complete_cohorts": 3 - len(complete)}
            memory = {**common, "mode": mode,
                      "gpu_allocated_ready_median_gib": None,
                      "gpu_reserved_ready_median_gib": None,
                      "gpu_peak_request_median_gib": None,
                      "gpu_peak_decode_median_gib": None,
                      "nvml_process_ready_median_gib": None,
                      "cpu_value_sum_median_gib": None,
                      "cpu_key_sum_median_gib": None}
            if status == "complete":
                for source, target in (
                    ("gpu_allocated_ready_max_rank_bytes", "gpu_allocated_ready_median_gib"),
                    ("gpu_reserved_ready_max_rank_bytes", "gpu_reserved_ready_median_gib"),
                    ("gpu_allocated_peak_decode_max_rank_bytes", "gpu_peak_decode_median_gib"),
                    ("nvml_process_ready_max_rank_bytes", "nvml_process_ready_median_gib"),
                    ("cpu_value_sum_bytes", "cpu_value_sum_median_gib"),
                    ("cpu_key_sum_bytes", "cpu_key_sum_median_gib")):
                    memory[target] = statistics.median(row[source] for row in complete) / 2**30
                if mode == "request":
                    memory["gpu_peak_request_median_gib"] = statistics.median(
                        row["gpu_allocated_peak_request_max_rank_bytes"] for row in complete) / 2**30
            memory_rows.append(memory)
            if mode == "request":
                selected = sorted(complete, key=lambda row: row["total_request_ms"])[1] if status == "complete" else None
                result = {**common, "representative_cohort": selected["cohort"] if selected else None,
                          "cohort_min_total_ms": min(row["total_request_ms"] for row in complete) if selected else None,
                          "cohort_max_total_ms": max(row["total_request_ms"] for row in complete) if selected else None,
                          **{field: selected[field] if selected else None for field in REQUEST_FIELDS}}
                request_rows.append(result)
            else:
                result = {**common, **{field: statistics.median(row[field] for row in complete)
                                      if status == "complete" else None
                                      for field in ("mean_ms_per_step", "p50_ms", "p95_ms",
                                                    "aggregate_tokens_per_second")}}
                steady_rows.append(result)
    write_csv(root / "request_breakdown.csv", request_rows)
    write_csv(root / "decode_summary.csv", steady_rows)
    write_csv(root / "memory_summary.csv", memory_rows)
    plotted = plot_request_breakdown(request_rows, root.parent / "plots")
    counts = {status: sum(row["status"] == status for row in trials)
              for status in ("complete", "not_run", "failed_oom", "failed")}
    samples = [ranks[0] for replica, ranks in validated.values()]
    provenance = samples[0] if samples else None
    lines = ["# ShadowKV TP8 Request Benchmark", "",
             f"Grid status: `{manifest['status']}`. Trial statuses: `{json.dumps(counts, sort_keys=True)}`.",
             "",
             "Three frozen real-text cohorts per method/context/batch/mode are required for a group result.",
             "Request components come from the cohort with median total request time, so the stacked bars add exactly.",
             "The request generates 128 tokens in total; steady decode measures 128 steps after 16 conditioning steps.",
             "Every rank uses a common host monotonic clock and synchronized GPU phase boundaries.",
             "The separate ShadowKV construction category includes gather, SVD, redistribution, and official preparation.",
             "BasisKV projection, encoding, and cache writes remain inside prefill and total request time.",
             "The result is an instrumented synchronized request; timing instrumentation overhead is included.",
             "", "## Provenance", "",
             "- Hardware: 8 x NVIDIA L40S, TP8 / DP1 / PP1; topology in `docs/hardware/l40s_tp8_topology.md`.",
             "- Environment: `basis`; BF16, TF32 off, eager; CPU affinity policy from the frozen runner.",
             "- ShadowKV: rank 160, chunk 8, sparse budget 2048, owner layer % 8, pinned CPU V.",
             "- BasisKV: frozen Joint-ALS V96 / B16R16 / Page32 / support 2048 full-scan path.",
             "- Inputs: frozen `p{length}_c{cohort}` real-text windows and the pinned Llama-3.1-8B-Instruct revision.",
             "- Full trial commands, failures, timestamps, and output paths: `grid_trials.json`.",
             "- Dense is the repository's matched TP8 eager control, not a tuned serving engine."]
    if provenance is not None:
        meta = provenance["metadata"]
        lines += [f"- Code base commit: `{meta['git_commit']}`; worktree dirty: `{bool(meta['git_status'])}`.",
                  f"- Model path/revision: `{provenance['model']}`.",
                  f"- ShadowKV source commit (when that arm runs): `{next((row['upstream_commit'] for row in samples if row['upstream_commit']), 'pending')}`.",
                  f"- PyTorch `{meta['pytorch']}`, CUDA `{meta['cuda']}`, NCCL `{meta['nccl']}`, Transformers `{meta['transformers']}`, FlashAttention `{meta['flash_attn']}`."]
    lines += ["", "## Outputs", "",
              "- `summary.csv`: every planned trial, including missing and failed configurations.",
              "- `request_breakdown.csv`: one additive representative request per complete three-cohort group.",
              "- `decode_summary.csv`: three-cohort medians from independent 16+128 windows.",
              "- `memory_summary.csv`: median max-rank allocated/reserved/NVML snapshots; active state bytes remain in raw rank JSON.",
              "- `failures.csv`: every failed attempt, including earlier failures after a successful retry.",
              f"- Primary request plot generated: `{plotted}`.", "",
              "No number from the 4K smoke directories is a formal benchmark result.",
              "OOM during prefill must be reported as prefill OOM, not decode-cache capacity.", ""]
    (root / "SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps({"grid_status": manifest["status"], "trial_statuses": counts,
                      "plot_generated": plotted}), flush=True)


if __name__ == "__main__":
    main()
