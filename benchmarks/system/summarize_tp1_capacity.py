"""Summarize measured active-batch throughput without interpolating OOM points."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    trials = json.loads((args.input / "outcomes.json").read_text())
    manifest = json.loads((args.input / "manifest.json").read_text())
    repeats = manifest["settings"]["repeats"]
    contexts = sorted({t["context"] for t in trials})
    methods = ["dense_local", "dense_k_offload", "basis_k_offload"]
    labels = dict(dense_local="Dense-local", dense_k_offload="Dense-K-offload", basis_k_offload="BasisKV-K-offload")
    rows = []
    for context in contexts:
        for method in methods:
            batches = sorted({t["batch"] for t in trials if t["method"] == method and t["context"] == context})
            for batch in batches:
                selected = [t for t in trials if (t["context"], t["method"], t["batch"]) == (context, method, batch)]
                runs = [t["result"] for t in selected if t["status"] == "success"]
                complete = len(runs) == repeats
                state = "success" if complete else next((t["status"] for t in selected if t["status"] != "success"), "incomplete")
                row = dict(context=context, method=method, batch=batch, status=state,
                    successful_repeats=len(runs), oom_phase=next((t.get("phase", "") for t in selected if t["status"] != "success"), ""))
                if runs:
                    for run in runs:
                        assert run["active_batch"] == [batch] * run["decode_steps"]
                        assert run["all_logits_finite"]
                    row.update(aggregate_tokens_per_second=statistics.median(r["aggregate_tokens_per_second"] for r in runs),
                        mean_step_ms=statistics.median(r["mean_step_ms"] for r in runs),
                        gpu_peak_gib=max(max(r["decode_resident"]["gpu_peak_allocated_bytes"],
                                             r["final_memory"]["gpu_peak_allocated_bytes"]) for r in runs) / 2**30,
                        decode_resident_gib=max(r["decode_resident"]["gpu_allocated_bytes"] for r in runs) / 2**30,
                        host_peak_rss_gib=max(r["final_memory"]["host_peak_rss_bytes"] for r in runs) / 2**30,
                        host_key_gib=max(r["host_key_bytes"] for r in runs) / 2**30)
                rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.input / "summary.csv").open("w") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# TP1 capacity and K-offload throughput", "",
        f"Completed trials: {len(trials)}; successful: {sum(t['status'] == 'success' for t in trials)}; "
        f"GPU OOM: {sum(t['status'] == 'gpu_oom' for t in trials)}. "
        "OOM is an observed outcome, not a missing throughput estimate.", "",
        "Llama-3.1-8B-Instruct BF16, one L40S, environment `basis`. Full V128 remains on GPU in every arm. "
        "Basis uses full-scan B16R16 Page32 routing, 1,984 routed tokens plus 64 recent tokens, and persistent exact-K slots. "
        "No two-stage routing. Sparse and dense attention are different algorithms; this is not a quality evaluation.", "",
        "Prompts are admitted sequentially into independent preallocated KV rows. All B requests then decode together "
        "with no EOS stopping or batch reduction. Reported throughput excludes prefill, includes logits/argmax, "
        "finite checks and synchronization. The default 128 measured decode forwards exclude the first token produced by prefill. "
        "Warmup lengths and K-slot state are reset. This is a fixed-active-batch benchmark, not scheduler throughput or E2E latency.", "",
        "All methods use the same prompt rows. Repeats are fresh processes; table values are medians across successful repeats. "
        "GPU peak is PyTorch allocated memory, not total device usage. Host peak RSS includes model loading; explicit pinned K bytes are separate in CSV. "
        "OOM during allocation or prefill is not labelled decode OOM. Other errors are retained as failures, not OOM.", "",
        "| Context | Method | Batch | Status | Repeats | Decode ms/step | Aggregate tok/s | GPU peak GiB |", "|---:|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        status = row["status"] + (f" ({row['oom_phase']})" if row["oom_phase"] else "")
        values = [f"{row[key]:.2f}" if key in row else "-" for key in ("mean_step_ms", "aggregate_tokens_per_second", "gpu_peak_gib")]
        lines.append(f"| {row['context']//1024}K | {labels[row['method']]} | {row['batch']} | {status} | {row['successful_repeats']} | " + " | ".join(values) + " |")
    lines += ["", "## Largest successful tested batch", "",
        "Only configurations completing all requested repeats count. These are tested powers of two, not exact maximum capacities.", "",
        "| Context | Dense-local | Dense-K-offload | BasisKV-K-offload |", "|---:|---:|---:|---:|"]
    for context in contexts:
        values = [str(max((r["batch"] for r in rows if r["context"] == context and r["method"] == method and r["status"] == "success"), default=0)) for method in methods]
        lines.append(f"| {context//1024}K | " + " | ".join(values) + " |")
    lines += ["", "## Prefill capacity caveat", "",
        "The 128K/B2 Basis run failed during prefill in `apply_rotary_pos_emb`, "
        "while allocating a 1 GiB temporary for `rotate_half(q) * sin`. It did not reach decode. "
        "The CUDA diagnostic reported 42.31 GiB allocated by PyTorch and 1.28 GiB reserved but "
        "unallocated, with 277 MiB device memory free. This result does not establish a "
        "fundamental decode-resident capacity limit: reducing prefill temporaries or allocator "
        "fragmentation would require a separately validated run. Dense-K-offload completed "
        "128K/B2; the current data do not demonstrate a Basis capacity advantage there.", "",
        "All other observed OOMs occurred during cache allocation. The 64K/B4 comparison "
        "is successful for both offload methods while Dense-local OOMs. Do not extrapolate "
        "the 64K result to an unmeasured successful Basis 128K/B2 decode."]
    lines += ["", "## Reproduction", "", "Environment: `basis`. Raw per-trial commands, logs, tokens and memory statistics are retained in trial directories. "
        "Source snapshot: `source.tar.gz`; run settings: `manifest.json`; end-of-run byte comparison: `source_check.json`. "
        "The final report generator (`summary_source.py`) adds the observed OOM caveat and figure formatting after the run. "
        "No runtime source changed during the run. No SHA256 checks.",
        "", "```bash", manifest["command"], "```", ""]
    (args.input / "SUMMARY.md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(contexts), figsize=(6 * len(contexts), 4), squeeze=False)
    colors = dict(dense_local="#555555", dense_k_offload="#bb5544", basis_k_offload="#23866d")
    for ax, context in zip(axes[0], contexts, strict=True):
        for method in methods:
            points = [r for r in rows if r["context"] == context and r["method"] == method and r["status"] == "success"]
            ax.plot([r["batch"] for r in points], [r["aggregate_tokens_per_second"] for r in points], marker="o", label=labels[method], color=colors[method])
            for row in rows:
                if (row["context"], row["method"], row["status"]) == (context, method, "gpu_oom"):
                    height = .04 + .065 * methods.index(method)
                    ax.plot(row["batch"], height, marker="x", color=colors[method], transform=ax.get_xaxis_transform())
                    label = "Prefill OOM" if row["oom_phase"] == "prefill" else "OOM"
                    ax.annotate(label, (row["batch"], height), xycoords=ax.get_xaxis_transform(), xytext=(5, 0), textcoords="offset points", color=colors[method], fontsize=8)
        ax.set(xscale="log", xlabel="Active batch", ylabel="Aggregate decode throughput (tokens/s)", title=f"{context//1024}K context", ylim=(0, None))
        ticks = sorted({r["batch"] for r in rows if r["context"] == context})
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(.85, max(ticks) * 1.55)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.text(.5, .01, "OOM markers are vertically offset labels, not throughput measurements.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(args.input / "throughput.png", dpi=180)
    fig.savefig(args.input / "throughput.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
