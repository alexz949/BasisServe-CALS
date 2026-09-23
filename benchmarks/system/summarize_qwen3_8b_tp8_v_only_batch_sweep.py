"""Validate and summarize the fixed 16K TP8 V-only batch sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.system.run_qwen3_8b_tp8_v_only_batch_sweep import DEFAULT_OUTPUT, GRID
from benchmarks.system.summarize_qwen3_8b_tp8_v_only_memory import GIB, FIELDS, status_of, write_csv


def expected_cache_bytes(arm: str, batch: int, star_ranks: list[int] | None) -> dict[str, int]:
    capacity = GRID["prompt_tokens"] + GRID["reserve_decode_tokens"]
    if arm == "basis_v64":
        value_width = 36 * 64
    else:
        assert star_ranks is not None and len(star_ranks) == 36
        value_width = sum(128 if index in (0, 1, 31) else width
                          for index, width in enumerate(star_ranks))
    return {
        "dense_key_cache": batch * capacity * 36 * 128 * 2,
        "value_cache": batch * capacity * value_width * 2,
    }


def load_trial(attempt: dict) -> list[dict]:
    root = Path(attempt["output_dir"])
    ranks = [json.loads((root / f"rank{index}.json").read_text()) for index in range(8)]
    for index, row in enumerate(ranks):
        assert row["schema"] == "basisserve.qwen3_8b.tp8_v_only_memory.v1"
        assert row["status"] == "complete" and row["rank"] == index
        assert row["tp"] == 8 and row["dp"] == row["pp"] == 1
        assert row["arm"] == attempt["arm"] and row["batch"] == attempt["batch"]
        assert row["prompt_tokens"] == GRID["prompt_tokens"] and row["cohort"] == GRID["cohort"]
        assert row["chunk_size"] == GRID["chunk_size"]
        assert row["output_tokens"] == row["reserve_decode_tokens"] == 128
        assert row["prompt_source_rows"] == 8
        assert row["prompt_repeated"] == (row["batch"] > 8)
        assert row["dense_key"] and row["full_attention"]
        assert not row["key_offload"] and not row["value_offload"]
        assert row["dtype"] == "bfloat16" and len(row["generated_token_ids"]) == 128
        assert all(len(tokens) == row["batch"] for tokens in row["generated_token_ids"])
        expected = expected_cache_bytes(row["arm"], row["batch"], row["star_value_ranks"])
        assert all(row["state_bytes"][name] == value for name, value in expected.items())
        assert all(row["memory"][phase][name] >= 0
                   for phase in ("cache_allocated", "prefill_complete", "decode_ready", "decode_complete")
                   for name in FIELDS)
    assert all(row["state_bytes"] == ranks[0]["state_bytes"] for row in ranks)
    assert all(row["star_value_ranks"] == ranks[0]["star_value_ranks"] for row in ranks)
    assert all(row["generated_token_ids"] == ranks[0]["generated_token_ids"] for row in ranks)
    return ranks


def plot_memory(rows: list[dict], output: Path) -> bool:
    if not any(row["status"] == "complete" for row in rows):
        return False
    output.mkdir(parents=True, exist_ok=True)
    colors = {"basis_v64": "#1c7c73", "star_v_adaptive": "#c45830"}
    labels = {"basis_v64": "BasisKV V64", "star_v_adaptive": "STAR-KV V-only (54.05% actual)"}
    top = max(row["decode_ready_nvml_max_rank_gib"] for row in rows
              if row["status"] == "complete") * 1.1 + 1
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for arm in GRID["arms"]:
        arm_rows = [row for row in rows if row["arm"] == arm]
        x = [row["batch"] for row in arm_rows if row["status"] == "complete"]
        y = [row["decode_ready_nvml_max_rank_gib"] for row in arm_rows
             if row["status"] == "complete"]
        axis.plot(x, y, marker="o", linewidth=2, color=colors[arm], label=labels[arm])
        for row in arm_rows:
            if row["status"].startswith("failed_oom_"):
                offset = 0.96 if arm == "basis_v64" else 1.04
                axis.scatter(row["batch"] * offset, top, marker="x", s=110,
                             linewidths=2, color=colors[arm])
    axis.set_xscale("log", base=2)
    axis.set_xticks(GRID["batches"], [str(batch) for batch in GRID["batches"]])
    axis.set_xlim(0.8, 310)
    axis.set_ylim(0, top * 1.07)
    axis.set_xlabel("Batch size")
    axis.set_ylabel("Decode-ready NVML process memory (GiB / GPU)")
    axis.set_title("Qwen3-8B-Base TP8: 16K prefill + 128 output tokens")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left")
    figure.text(0.5, 0.005, "Crosses mark recorded OOM; no resident value is imputed to a failed trial.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output / "batch_sweep_16k_memory.pdf", bbox_inches="tight")
    figure.savefig(output / "batch_sweep_16k_memory.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.output_root
    manifest = json.loads((root / "grid_trials.json").read_text())
    assert manifest["grid"] == GRID
    latest = {}
    failures = []
    for attempt in manifest["attempts"]:
        latest[(attempt["arm"], attempt["batch"])] = attempt
        if attempt["status"] != "complete":
            failure = attempt.get("failure", {})
            failures.append({
                "arm": attempt["arm"], "batch": attempt["batch"],
                "attempt": attempt["attempt"], "oom": failure.get("oom"),
                "phase": failure.get("phase"),
                "oom_ranks": json.dumps(failure.get("oom_ranks")),
                "oom_rank_stages": json.dumps(failure.get("oom_rank_stages")),
                "allocation_request": failure.get("allocation_request"),
                "last_rank_stages": json.dumps(failure.get("last_rank_stages")),
                "missing_rank_results": json.dumps(failure.get("missing_rank_results")),
                "returncode": attempt.get("returncode"),
                "launcher_log": attempt["launcher_log"],
            })
    rows = []
    samples = []
    for batch in GRID["batches"]:
        for arm in GRID["arms"]:
            attempt = latest.get((arm, batch))
            status = status_of(attempt)
            row = {
                "arm": arm, "prompt_tokens": GRID["prompt_tokens"], "batch": batch,
                "output_tokens": GRID["output_tokens"], "status": status,
                "attempt": attempt["attempt"] if attempt else None,
                "raw_dir": attempt["output_dir"] if attempt else None,
                "failure_phase": attempt.get("failure", {}).get("phase") if attempt else None,
                "theoretical_key_gib": None, "theoretical_value_gib": None,
                "value_factor_max_rank_gib": None,
                "decode_ready_allocated_max_rank_gib": None,
                "decode_ready_reserved_max_rank_gib": None,
                "decode_ready_nvml_max_rank_gib": None,
                "decode_end_nvml_max_rank_gib": None,
                "peak_prefill_allocated_max_rank_gib": None,
                "peak_decode_allocated_max_rank_gib": None,
            }
            if status == "complete":
                ranks = load_trial(attempt)
                samples.append(ranks[0])
                state = ranks[0]["state_bytes"]
                row["theoretical_key_gib"] = state["dense_key_cache"] / GIB
                row["theoretical_value_gib"] = state["value_cache"] / GIB
                row["value_factor_max_rank_gib"] = max(
                    item["state_bytes"]["value_factors"] for item in ranks) / GIB
                for phase, metric, target in (
                    ("decode_ready", "allocated_bytes", "decode_ready_allocated_max_rank_gib"),
                    ("decode_ready", "reserved_bytes", "decode_ready_reserved_max_rank_gib"),
                    ("decode_ready", "nvml_process_bytes", "decode_ready_nvml_max_rank_gib"),
                    ("decode_complete", "nvml_process_bytes", "decode_end_nvml_max_rank_gib"),
                    ("prefill_complete", "peak_allocated_bytes", "peak_prefill_allocated_max_rank_gib"),
                    ("decode_complete", "peak_allocated_bytes", "peak_decode_allocated_max_rank_gib"),
                ):
                    row[target] = max(item["memory"][phase][metric] for item in ranks) / GIB
            rows.append(row)
    write_csv(root / "summary.csv", rows)
    write_csv(root / "failures.csv", failures, fieldnames=(
        "arm", "batch", "attempt", "oom", "phase", "oom_ranks", "oom_rank_stages",
        "allocation_request", "last_rank_stages", "missing_rank_results",
        "returncode", "launcher_log"))
    frontier = []
    for arm in GRID["arms"]:
        arm_rows = [row for row in rows if row["arm"] == arm]
        successes = [row for row in arm_rows if row["status"] == "complete"]
        ooms = [row for row in arm_rows if row["status"].startswith("failed_oom_")]
        largest = max(successes, key=lambda row: row["batch"]) if successes else None
        first = min(ooms, key=lambda row: row["batch"]) if ooms else None
        frontier.append({
            "arm": arm,
            "largest_success_batch": largest["batch"] if largest else None,
            "first_oom_batch": first["batch"] if first else None,
            "first_oom_phase": first["failure_phase"] if first else None,
            "decode_ready_nvml_gib_at_largest_success": (
                largest["decode_ready_nvml_max_rank_gib"] if largest else None),
        })
    write_csv(root / "batch_frontier.csv", frontier)
    plotted = plot_memory(rows, root / "plots")
    counts = {name: sum(row["status"] == name for row in rows)
              for name in sorted({row["status"] for row in rows})}
    lines = [
        "# Qwen3 TP8 V-only 16K Batch Sweep", "",
        f"Grid status: `{manifest['status']}`; trial statuses: `{json.dumps(counts, sort_keys=True)}`.",
        "", "BasisKV V64 versus STAR-KV V-only adaptive export, exact dense K, full attention,",
        "BF16, TP8 / DP1 / PP1 on eight NVIDIA L40S GPUs. One preselected real-text",
        "cohort supplies eight prompts; batches above eight repeat those prompts in order.",
        "Each trial prefills 16384 tokens in 256-token chunks, reserves 128 decode slots,",
        "and actually generates 128 greedy output tokens without EOS stopping.",
        "The first output token comes from prefill; the remaining 127 require decode forwards.",
        "The model's native context is 32768 tokens, so the 16512-token request fits.",
        "No CPU V offload, K routing, K compression, or substituted checkpoint is used.",
        "The STAR export retains 54.0473% actual global V rank, not a strict 50%.",
        "", "## Outcomes", "",
    ]
    for row in frontier:
        lines.append(
            f"- {row['arm']}: largest successful batch `{row['largest_success_batch']}`; "
            f"first observed OOM batch `{row['first_oom_batch']}` "
            f"(phase `{row['first_oom_phase']}`)."
        )
    lines += [
        "", "The main plotted value is max-rank decode-ready NVML process GPU memory.",
        "An OOM has no measured decode-ready value. Decode-end NVML memory and peak",
        "prefill/decode PyTorch allocation are separate columns in `summary.csv`.",
        "Theoretical K/V bytes use actual stored widths and are not process memory.",
        "Every attempt and its command are in `grid_trials.json`; all failures are in `failures.csv`.",
        f"Memory plot generated: `{plotted}`.",
    ]
    if samples:
        meta = samples[0]["metadata"]
        lines += [
            "", "## Provenance", "",
            f"- Code base commit: `{meta['git_commit']}`; worktree dirty: `{bool(meta['git_status'])}`.",
            f"- PyTorch `{meta['pytorch']}`, CUDA `{meta['cuda']}`, NCCL `{meta['nccl']}`.",
            f"- Transformers `{meta['transformers']}`, FlashAttention `{meta['flash_attn']}`.",
            "- Qwen3-8B-Base revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.",
            "- STAR fused checkpoint HF revision `0ef83dff27205b131c82df6d62636129e9dac7b9`.",
            "- Basis V64 factor HF revision `0872566b1da66eb4c813d7a1cb3313325f22b287`.",
        ]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"grid_status": manifest["status"], "trial_statuses": counts,
                      "plot_generated": plotted}), flush=True)


if __name__ == "__main__":
    main()
