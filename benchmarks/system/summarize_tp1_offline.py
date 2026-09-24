"""Three-cohort TP1 tables and figures; keep request and profile scopes separate."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("dense", "basis", "shadowkv", "lrqk")
LABELS = dict(dense="Dense", basis="BasisKV", shadowkv="ShadowKV", lrqk="LRQK")
CONTEXTS = (16384, 32768, 65536, 130048)


def aggregate(outcomes):
    groups = defaultdict(list)
    for item in outcomes:
        groups[(item["mode"], item["context"], item["output_tokens"], item["method"])].append(item)
    rows = []
    for (mode, context, count, method), trials in sorted(groups.items()):
        successful = [t for t in trials if t["status"] == "complete"]
        row = dict(mode=mode, context=context, output_tokens=count, method=method,
                   trials=len(trials), complete=len(successful), status="incomplete")
        if len(trials) == 3 and len(successful) == 3:
            assert sorted(t["cohort"] for t in trials) == [0, 1, 2]
            row["status"] = "complete"
            data = [json.loads((Path(t["directory"]) / "result.json").read_text()) for t in trials]
            values = defaultdict(list)
            for result in data:
                measured = result["measurement"]
                for key in ("request_seconds", "prefill_inclusive_seconds", "post_prefill_phase_seconds",
                            "decode_phase_seconds", "post_prefill_ready_seconds", "steady_tail_wall_mean_ms",
                            "cuda_median_ms", "cuda_mean_ms", "wall_median_ms"):
                    if measured.get(key) is not None:
                        values[key].append(measured[key])
                values["peak_allocated_gib"].append(result["peak_allocated_gib"])
                build = result.get("build_profile", {})
                construction = build.get("construction_including_preparation_seconds", build.get("prompt_specific_fit_seconds"))
                if construction is not None:
                    values["profile_construction_seconds"].append(construction)
                components = result.get("attention_profile", {}).get("components", [])
                if components:
                    values["profile_attention_block_mean_ms"].append(statistics.fmean(c["attention_block"] for c in components))
            for key, numbers in values.items():
                assert len(numbers) == 3
                row[key] = statistics.median(numbers)
                row[key + "_min"] = min(numbers)
                row[key + "_max"] = max(numbers)
                row[key + "_mean"] = statistics.fmean(numbers)
        elif len(trials) == 3 and all(t["status"] == "gpu_oom" for t in trials):
            row["status"] = "GPU OOM (3/3)"
        else:
            row["status"] = f"{len(successful)}/{len(trials)} complete"
        rows.append(row)
    return rows


def figures(root, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(dense="#326da8", basis="#18836d", shadowkv="#b84b45", lrqk="#765a9d")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for method in METHODS[:2]:
        group = sorted((r for r in rows if r["mode"] == "steady" and r["method"] == method and r["status"] == "complete"), key=lambda r: r["context"])
        axes[0].plot([CONTEXTS.index(r["context"]) for r in group], [r["cuda_median_ms"] for r in group],
                     marker="o", color=colors[method], label=LABELS[method])
    axes[0].set(xticks=range(4), xticklabels=["16K", "32K", "64K", "130048"],
                xlabel="Prompt tokens", ylabel="Steady decode (ms/token)", title="(a) GPU-local sparse decode")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    for method in METHODS[1:]:
        group = sorted((r for r in rows if r["mode"] == "request" and r["context"] == 130048 and r["method"] == method and r["status"] == "complete"), key=lambda r: r["output_tokens"])
        if group:
            axes[1].plot([r["output_tokens"] for r in group], [r["request_seconds"] for r in group],
                         marker="o", color=colors[method], label=LABELS[method])
        else:
            axes[1].plot([], [], color=colors[method], label=LABELS[method] + " (no complete point)")
    axes[1].set(xticks=[32, 128, 512], xlabel="Output tokens", ylabel="Request latency (s)",
                title="(b) 130048-token prompt, B1")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.text(0.5, 0.015, "Three-cohort medians. Warm runtime, fresh request state. Default external storage policies; not quality-matched.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(root / "tp1_two_panel.png", dpi=180)
    fig.savefig(root / "tp1_two_panel.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ymax = 1.0
    for index, method in enumerate(METHODS):
        row = next(r for r in rows if r["mode"] == "request" and r["context"] == 130048 and r["output_tokens"] == 128 and r["method"] == method)
        if row["status"] != "complete":
            ax.text(index, 0, row["status"], ha="center", va="bottom", fontsize=8)
            continue
        bottom = 0
        for key, color in (("prefill_inclusive_seconds_mean", "#b8c3cc"),
                           ("post_prefill_phase_seconds_mean", "#ca9145"),
                           ("decode_phase_seconds_mean", "#326da8")):
            ax.bar(index, row[key], bottom=bottom, color=color, width=.65)
            bottom += row[key]
        ymax = max(ymax, bottom)
        ax.annotate(f"Build profile: {row['profile_construction_seconds']:.2f}s",
                    (index, bottom), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#b8c3cc", label="Prefill (includes in-prefill build)"),
                       Patch(color="#ca9145", label="Post-prefill phase"),
                       Patch(color="#326da8", label="Decode (includes lazy restore)")], frameon=False, fontsize=8)
    ax.set(xticks=range(4), xticklabels=[LABELS[m] for m in METHODS], ylabel="Mean request latency (s)",
           title="130048-token prompt, 128 output tokens, TP1/B1", ylim=(0, ymax*1.4))
    fig.text(.5, .015, "Stacked phases are additive wall times. Build labels are separate profile medians, already included, not extra segments.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(root / "request_phases.png", dpi=180)
    fig.savefig(root / "request_phases.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/system_benchmarks/tp1_offline")
    args = parser.parse_args()
    root = args.root
    outcomes = json.loads((root / "formal_outcomes.json").read_text())
    assert len(outcomes) == 78
    rows = aggregate(outcomes)
    with (root / "aggregate.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    success = sum(t["status"] == "complete" for t in outcomes)
    oom = sum(t["status"] == "gpu_oom" for t in outcomes)
    lines = ["# TP1 Offline-Representation Benchmark", "", "## Outcome", "",
             f"{len(outcomes)} formal trials: **{success} complete**, **{oom} GPU OOM**, {len(outcomes)-success-oom} other failures.",
             "Environment: `basis`, one NVIDIA L40S (GPU 0), serial TP1/B1, Llama-3.1-8B-Instruct, BF16, TF32 off.",
             "All primary measurements follow same-shape runtime warmup and reset request-specific state.",
             "BasisKV uses GPU-local K/V, Dense V128, the frozen 16-warp B16R16/Page32 full-scan path.",
             "Neither the page-cache candidate nor two-stage routing is used.", "",
             "Three fixed prompt cohorts use calibration windows 0/1/2. This is a timing study, not a held-out quality evaluation.",
             "Tables report three-cohort medians only when all three trials complete. All raw failures remain recorded.", "",
             "## GPU-Local Sparse Decode", "",
             "Full-model values are CUDA-event steady medians: 16 conditioning calls, 128 measured calls, greedy feedback.",
             "Attention-block values are separate 32-step component-profile means per cohort, then median across cohorts.",
             "The block is after QKV/RoPE and before output projection, including cache update, routing, selection and attention.", "",
             "| Prompt | Method | Full model ms/token | Attention block ms/token | Status |",
             "|---:|---|---:|---:|---|"]
    def fmt(row, key):
        return f"{row[key]:.3f}" if key in row else "-"
    for row in rows:
        if row["mode"] == "steady":
            lines.append(f"| {row['context']} | {LABELS[row['method']]} | {fmt(row, 'cuda_median_ms')} | {fmt(row, 'profile_attention_block_mean_ms')} | {row['status']} |")
    lines += ["", "## Requests: 128 Output Tokens", "",
        "Build is a separate synchronized profile of fitting plus required preparation/placement; nested timers are not added twice.",
        "Ready delay is measured in the request path. LRQK restores per layer during the first decode, so its delay includes preceding decode work.",
        "The steady tail is wall mean after the first 16 decode calls, including greedy selection; it is NOT the CUDA median above.", "",
        "| Prompt | Method | Profiled build s | Ready delay s | Request s | Steady tail ms/token | Status |",
        "|---:|---|---:|---:|---:|---:|---|"]
    for row in rows:
        if row["mode"] == "request" and row["output_tokens"] == 128:
            lines.append(f"| {row['context']} | {LABELS[row['method']]} | {fmt(row, 'profile_construction_seconds')} | {fmt(row, 'post_prefill_ready_seconds')} | {fmt(row, 'request_seconds')} | {fmt(row, 'steady_tail_wall_mean_ms')} | {row['status']} |")
    lines += ["", "## Output-Length Sweep", "", "Prompt length is fixed at 130,048, leaving generation headroom.", "",
              "| Output tokens | Method | Request s | Status |", "|---:|---|---:|---|"]
    for row in sorted(rows, key=lambda r: (r["output_tokens"], r["method"])):
        if row["mode"] == "request" and row["context"] == 130048 and row["method"] != "dense":
            lines.append(f"| {row['output_tokens']} | {LABELS[row['method']]} | {fmt(row, 'request_seconds')} | {row['status']} |")
    lines += ["", "## Interpretation and Limits", "",
        "- BasisKV's zero prompt-specific fit does not mean free preparation: projection, encoding and cache writes are inside prefill.",
        "- Model/factor loading, upfront cache allocation, input transfer, tokenization and network transport are excluded. This is warmed device-side request latency, not user-visible network latency.",
        "- ShadowKV uses its default CPU V / rank-160 reconstructed K path; LRQK uses CPU exact K/V, rank 32, active 2048 plus lite 64. These are whole-runtime comparisons, not an offline-only ablation or quality-matched comparison.",
        "- ShadowKV generation buffers are provisioned for actual initial support plus requested output length; selection and reconstruction are unchanged. LRQK retains exact tokenwise prefill MLP chunking at the long context; decode dispatch is unchanged.",
        "- Fixed-length greedy generation ignores EOS. N outputs correspond to one prefill token and N-1 decode calls. Primary request validation checks final logits; separate smoke/steady checks cover additional intermediate logits.",
        "- Construction profiles are separate instrumented passes. Do not subtract them from primary request time to invent exact build-free prefills or add them again to E2E.",
        "- Each plotted stacked phase uses cohort means, so phases sum to mean E2E. Construction annotations are separate profile medians. Other tables and line plots use cohort medians.",
        "- GPU OOM during warmup is still a failure of this requested runtime/configuration, not a timed latency or proof of an algorithmic capacity limit. Failure phases/log paths are in formal_outcomes.json.", "",
        "- The LRQK environment emitted PyTorch graph-break and recompile-limit warnings during screening. Raw logs retain these warnings; results describe this adapted runtime, not an optimized performance bound for LRQK.", "",
        "## Reproduction", "", "Executed serially, directly under the user's authorization:", "", "```bash",
        "export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9",
        "/workspace/miniforge3/bin/conda run --no-capture-output -n basis \\",
        "  python benchmarks/system/run_tp1_offline_grid.py --phase formal \\",
        "  > results/system_benchmarks/tp1_offline/formal.log 2>&1",
        "/workspace/miniforge3/bin/conda run --no-capture-output -n basis \\",
        "  python benchmarks/system/summarize_tp1_offline.py", "```", "",
        "The same launcher ran `--phase smoke` (6 trials) and `--phase capacity` (5 trials) before the formal grid.",
        "Exact child commands, logs and results are retained per trial. Inputs are frozen in inputs/; source snapshots and dependencies are retained under formal/source/ and freeze/.",
        "No SHA256 checks were performed. No files from this grid have been uploaded or committed.", "",
        "![TP1 two-panel figure](tp1_two_panel.png)", "", "![Actual request phases](request_phases.png)", ""]
    (root / "SUMMARY.md").write_text("\n".join(lines))
    figures(root, rows)
    print(json.dumps(dict(trials=len(outcomes), complete=success, gpu_oom=oom, summary=str(root / "SUMMARY.md"))))


if __name__ == "__main__":
    main()
