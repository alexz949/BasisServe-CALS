"""Validate and summarize the Qwen3 TP8 V-only GPU memory/OOM grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.system.run_qwen3_8b_tp8_v_only_memory_grid import FULL_GRID, DEFAULT_OUTPUT


FIELDS = ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes",
          "peak_reserved_bytes", "nvml_process_bytes")
GIB = 2 ** 30


def write_csv(path: Path, rows: list[dict], fieldnames: tuple[str, ...] | None = None) -> None:
    columns = fieldnames or tuple(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def theoretical_state_bytes(arm: str, length: int, batch: int, star_ranks: list[int] | None) -> dict:
    capacity = length + 1
    key_width = 36 * 128
    if arm == "basis_v64":
        value_width = 36 * 64
    else:
        assert star_ranks is not None and len(star_ranks) == 36
        value_width = sum(128 if index in (0, 1, 31) else width
                          for index, width in enumerate(star_ranks))
    return {
        "dense_key_cache": batch * capacity * key_width * 2,
        "value_cache": batch * capacity * value_width * 2,
    }


def load_validated_trial(attempt: dict) -> list[dict]:
    root = Path(attempt["output_dir"])
    ranks = [json.loads((root / f"rank{index}.json").read_text()) for index in range(8)]
    for index, row in enumerate(ranks):
        assert row["schema"] == "basisserve.qwen3_8b.tp8_v_only_memory.v1"
        assert row["status"] == "complete" and row["rank"] == index
        assert row["tp"] == 8 and row["dp"] == row["pp"] == 1
        assert row["arm"] == attempt["arm"]
        assert row["prompt_tokens"] == attempt["prompt_tokens"]
        assert row["batch"] == attempt["batch"] and row["cohort"] == attempt["cohort"]
        assert row["chunk_size"] == FULL_GRID["chunk_size"]
        assert row["dense_key"] and row["full_attention"]
        assert not row["key_offload"] and not row["value_offload"]
        assert row["dtype"] == "bfloat16" and len(row["generated_token_ids"]) == 2
        assert all(len(tokens) == row["batch"] for tokens in row["generated_token_ids"])
        expected = theoretical_state_bytes(row["arm"], row["prompt_tokens"], row["batch"],
                                           row["star_value_ranks"])
        assert all(row["state_bytes"][name] == value for name, value in expected.items())
        assert all(row["memory"][phase][name] >= 0
                   for phase in ("cache_allocated", "prefill_complete", "decode_ready", "decode_complete")
                   for name in FIELDS)
    assert all(row["state_bytes"] == ranks[0]["state_bytes"] for row in ranks)
    assert all(row["star_value_ranks"] == ranks[0]["star_value_ranks"] for row in ranks)
    assert all(row["generated_token_ids"] == ranks[0]["generated_token_ids"] for row in ranks)
    return ranks


def status_of(attempt: dict | None) -> str:
    if attempt is None:
        return "not_run"
    if attempt["status"] == "complete":
        return "complete"
    failure = attempt.get("failure", {})
    return ("failed_oom_" + failure.get("phase", "unknown")) if failure.get("oom") else "failed"


def group_status(rows: list[dict]) -> str:
    if all(row["status"] == "complete" for row in rows):
        return "complete"
    if all(row["status"] == "not_run" for row in rows):
        return "not_run"
    if any(row["status"].startswith("failed_oom") for row in rows):
        return "oom_or_mixed"
    return "partial_or_failed"


def plot_memory(rows: list[dict], output: Path) -> bool:
    if not any(row["status"] == "complete" for row in rows):
        return False
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = {"basis_v64": "#1c7c73", "star_v_adaptive": "#c45830"}
    labels = {"basis_v64": "BasisKV V64", "star_v_adaptive": "STAR-KV V-only (54.05% actual)"}
    successful = [row["decode_ready_nvml_max_rank_gib"] for row in rows
                  if row["status"] == "complete"]
    top = max(successful) * 1.1 + 1 if successful else 1
    for axis, batch in zip(axes, (1, 8)):
        for arm in FULL_GRID["arms"]:
            arm_rows = sorted((row for row in rows if row["batch"] == batch and row["arm"] == arm),
                              key=lambda item: item["prompt_tokens"])
            x = [row["prompt_tokens"] / 1024 for row in arm_rows if row["status"] == "complete"]
            y = [row["decode_ready_nvml_max_rank_gib"] for row in arm_rows if row["status"] == "complete"]
            axis.plot(x, y, marker="o", linewidth=2, color=colors[arm], label=labels[arm])
            for row in arm_rows:
                if row["oom_cohorts"]:
                    axis.scatter(row["prompt_tokens"] / 1024, top, marker="x", s=100,
                                 linewidths=2, color=colors[arm])
        axis.set_title(f"Batch {batch}")
        axis.set_xlabel("Context length (K tokens)")
        axis.set_xticks([length / 1024 for length in FULL_GRID["contexts"]])
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(0, top * 1.06)
    axes[0].set_ylabel("Decode-ready NVML process GPU memory (GiB / GPU)")
    axes[0].legend(loc="upper left")
    figure.suptitle("Qwen3-8B-Base TP8 V-only GPU residency")
    figure.text(0.5, 0.005, "Crosses mark recorded OOM configurations; actual STAR V retention is 54.05%.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure.savefig(output / "starkv_tp8_memory.pdf", bbox_inches="tight")
    figure.savefig(output / "starkv_tp8_memory.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.output_root
    manifest = json.loads((root / "grid_trials.json").read_text())
    assert manifest["grid"] == FULL_GRID
    latest = {}
    failures = []
    for attempt in manifest["attempts"]:
        key = tuple(attempt[field] for field in ("arm", "prompt_tokens", "batch", "cohort"))
        latest[key] = attempt
        if attempt["status"] != "complete":
            failure = attempt.get("failure", {})
            failures.append({
                "arm": attempt["arm"], "prompt_tokens": attempt["prompt_tokens"],
                "batch": attempt["batch"], "cohort": attempt["cohort"],
                "attempt": attempt["attempt"], "oom": failure.get("oom"),
                "oom_ranks": json.dumps(failure.get("oom_ranks")),
                "oom_rank_stages": json.dumps(failure.get("oom_rank_stages")),
                "phase": failure.get("phase"),
                "allocation_request": failure.get("allocation_request"),
                "missing_rank_results": json.dumps(failure.get("missing_rank_results")),
                "last_rank_stages": json.dumps(failure.get("last_rank_stages")),
                "truncated_rank_logs": json.dumps(failure.get("truncated_rank_logs")),
                "returncode": attempt.get("returncode"),
                "launcher_log": attempt["launcher_log"],
            })
    trials = []
    samples = []
    for length in FULL_GRID["contexts"]:
        for batch in FULL_GRID["batches"]:
            for cohort in FULL_GRID["cohorts"]:
                for arm in FULL_GRID["arms"]:
                    attempt = latest.get((arm, length, batch, cohort))
                    status = status_of(attempt)
                    row = {
                        "arm": arm, "prompt_tokens": length, "batch": batch, "cohort": cohort,
                        "status": status, "attempt": attempt["attempt"] if attempt else None,
                        "raw_dir": attempt["output_dir"] if attempt else None,
                        "failure_phase": attempt.get("failure", {}).get("phase") if attempt else None,
                        "theoretical_value_gib": None,
                        "theoretical_key_gib": None,
                        "value_factor_max_rank_gib": None,
                        "decode_ready_allocated_max_rank_gib": None,
                        "decode_ready_reserved_max_rank_gib": None,
                        "decode_ready_nvml_max_rank_gib": None,
                        "peak_prefill_allocated_max_rank_gib": None,
                        "peak_decode_allocated_max_rank_gib": None,
                    }
                    if status == "complete":
                        ranks = load_validated_trial(attempt)
                        samples.append(ranks[0])
                        state = ranks[0]["state_bytes"]
                        row["theoretical_value_gib"] = state["value_cache"] / GIB
                        row["theoretical_key_gib"] = state["dense_key_cache"] / GIB
                        row["value_factor_max_rank_gib"] = max(
                            item["state_bytes"]["value_factors"] for item in ranks) / GIB
                        for phase, metric, target in (
                            ("decode_ready", "allocated_bytes", "decode_ready_allocated_max_rank_gib"),
                            ("decode_ready", "reserved_bytes", "decode_ready_reserved_max_rank_gib"),
                            ("decode_ready", "nvml_process_bytes", "decode_ready_nvml_max_rank_gib"),
                            ("prefill_complete", "peak_allocated_bytes", "peak_prefill_allocated_max_rank_gib"),
                            ("decode_complete", "peak_allocated_bytes", "peak_decode_allocated_max_rank_gib"),
                        ):
                            row[target] = max(item["memory"][phase][metric] for item in ranks) / GIB
                    trials.append(row)
    write_csv(root / "summary.csv", trials)
    write_csv(root / "failures.csv", failures, fieldnames=(
        "arm", "prompt_tokens", "batch", "cohort", "attempt", "oom", "oom_ranks",
        "oom_rank_stages", "phase",
        "allocation_request", "missing_rank_results", "last_rank_stages", "truncated_rank_logs",
        "returncode", "launcher_log"))

    grouped = []
    frontier = []
    for arm in FULL_GRID["arms"]:
        for batch in FULL_GRID["batches"]:
            per_batch = []
            for length in FULL_GRID["contexts"]:
                cohort_rows = [row for row in trials if row["arm"] == arm
                               and row["batch"] == batch and row["prompt_tokens"] == length]
                status = group_status(cohort_rows)
                complete = [row for row in cohort_rows if row["status"] == "complete"]
                row = {
                    "arm": arm, "prompt_tokens": length, "batch": batch,
                    "status": status, "successful_cohorts": len(complete),
                    "oom_cohorts": sum(item["status"].startswith("failed_oom") for item in cohort_rows),
                    "other_failed_cohorts": sum(item["status"] == "failed" for item in cohort_rows),
                }
                for field in ("theoretical_value_gib", "theoretical_key_gib",
                              "value_factor_max_rank_gib", "decode_ready_allocated_max_rank_gib",
                              "decode_ready_reserved_max_rank_gib", "decode_ready_nvml_max_rank_gib",
                              "peak_prefill_allocated_max_rank_gib", "peak_decode_allocated_max_rank_gib"):
                    row[field] = complete[0][field] if status == "complete" else None
                grouped.append(row)
                per_batch.append(row)
            oom_lengths = [row["prompt_tokens"] for row in per_batch if row["oom_cohorts"]]
            cache_oom_lengths = [row["prompt_tokens"] for row in trials
                                 if row["arm"] == arm and row["batch"] == batch
                                 and row["status"] == "failed_oom_cache_state_allocation"]
            workspace_oom_lengths = [row["prompt_tokens"] for row in trials
                                     if row["arm"] == arm and row["batch"] == batch
                                     and row["status"] in ("failed_oom_transient_prefill_workspace",
                                                           "failed_oom_decode_workspace")]
            other_oom_lengths = [row["prompt_tokens"] for row in trials
                                 if row["arm"] == arm and row["batch"] == batch
                                 and row["status"].startswith("failed_oom_")
                                 and row["status"] not in (
                                     "failed_oom_cache_state_allocation",
                                     "failed_oom_transient_prefill_workspace",
                                     "failed_oom_decode_workspace")]
            successes = [row for row in per_batch if row["status"] == "complete"]
            largest = max(successes, key=lambda item: item["prompt_tokens"]) if successes else None
            frontier.append({
                "arm": arm, "batch": batch,
                "first_any_oom_context": min(oom_lengths) if oom_lengths else None,
                "first_cache_state_oom_context": min(cache_oom_lengths) if cache_oom_lengths else None,
                "first_workspace_oom_context": min(workspace_oom_lengths) if workspace_oom_lengths else None,
                "first_other_oom_context": min(other_oom_lengths) if other_oom_lengths else None,
                "largest_success_context": largest["prompt_tokens"] if largest else None,
                "decode_ready_nvml_gib_at_largest_success": (
                    largest["decode_ready_nvml_max_rank_gib"] if largest else None),
                "theoretical_value_gib_at_largest_success": (
                    largest["theoretical_value_gib"] if largest else None),
            })
    write_csv(root / "memory_summary.csv", grouped)
    write_csv(root / "oom_frontier.csv", frontier)
    plotted = plot_memory(grouped, root.parent / "plots")
    counts = {name: sum(row["status"] == name for row in trials)
              for name in sorted({row["status"] for row in trials})}
    sample = samples[0] if samples else None
    lines = [
        "# STAR-KV V-only TP8 Memory Benchmark", "",
        f"Grid status: `{manifest['status']}`. Trial statuses: `{json.dumps(counts, sort_keys=True)}`.",
        "", "Qwen3-8B-Base, TP8 / DP1 / PP1, BF16, exact dense K, full Flash SDPA attention,",
        "fixed 4096-token chunked prefill, one decode step after the full prompt.",
        "One preselected Qwen-tokenized LongBench-v2 cohort (cohort 0, eight real-text requests) is used per configuration.",
        "The primary decode-ready GPU-resident metric is the maximum NVML process memory across the eight TP ranks for that single trial; no cohort median or repeatability estimate is available.",
        "PyTorch allocated/reserved and prefill/decode peaks are reported separately.",
        "Actual STAR V retention is 54.0473%, not a strict 50% result; layers 0, 1, and 31 use dense local V.",
        "STAR's global V latent is replicated per TP rank; Basis V64 is source-local.",
        "The 65536+ contexts exceed Qwen3's native 32768-token context and are memory-only stress tests, not quality claims.",
        "No CPU V offload, K routing, K compression, or substituted checkpoint is used.",
        "", "## Provenance", "",
        "- Hardware: 8 x NVIDIA L40S; topology in `docs/hardware/l40s_tp8_topology.md`.",
        "- Environment: `basis`; BF16, TF32 disabled, eager, same CPU affinity policy as the frozen TP8 runner.",
        "- STAR full fused checkpoint: HF `alexz949/BasisServe-CALS`, revision `0ef83dff27205b131c82df6d62636129e9dac7b9`, upstream STAR-KV `c9f0f36e7e386eaf93099c9c9796e168ca1e6504`.",
        "- Basis V64 factors: HF `alexz949/BasisServe-CALS`, revision `0872566b1da66eb4c813d7a1cb3313325f22b287`.",
        "- Qwen3-8B-Base model revision: `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.",
        "- Every trial command, status, and output path is in `grid_trials.json`; all failures are in `failures.csv`.",
    ]
    if sample:
        meta = sample["metadata"]
        lines += [
            f"- Code base commit: `{meta['git_commit']}`; worktree dirty: `{bool(meta['git_status'])}`.",
            f"- PyTorch `{meta['pytorch']}`, CUDA `{meta['cuda']}`, NCCL `{meta['nccl']}`.",
            f"- Transformers `{meta['transformers']}`, FlashAttention `{meta['flash_attn']}`, GPU `{meta['gpu_name']}`.",
        ]
    lines += [
        "", "## Outputs", "",
        "- `summary.csv`: every trial, including failed and missing configurations.",
        "- `memory_summary.csv`: one max-rank observation per successful configuration; OOM groups have no measured resident value.",
        "- `oom_frontier.csv`: first observed cache/workspace/other OOM and largest success per batch.",
        "- `failures.csv`: every failed attempt, its last rank stages, allocation request when available, and launcher log.",
        f"- Memory plot generated: `{plotted}`.",
        "", "Theoretical active V/K bytes are computed from actual stored tensor widths and do not equal total process memory.",
        "A prefill-workspace OOM is not a KV-cache capacity limit.",
        "No 4K smoke measurement is a formal grid result.", "",
    ]
    (root / "SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps({"grid_status": manifest["status"], "trial_statuses": counts,
                      "plot_generated": plotted}), flush=True)


if __name__ == "__main__":
    main()
