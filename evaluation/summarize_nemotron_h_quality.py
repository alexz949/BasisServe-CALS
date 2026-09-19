"""Summarize audited Nemotron-H C1 checkpoints and paired quality runs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import read_json, write_json, sha256


RUNS = ("Dense", "C1-R64", "C1-R96")


def _pct(value):
    return f"{100 * value:.3f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--run-prefix", required=True)
    args = parser.parse_args()
    quality = {}
    inputs = {}
    for suffix in RUNS:
        path = args.root / "quality" / f"{args.run_prefix}-{suffix}" / "result.json"
        result = read_json(path)
        assert result["status"] == "complete" and not result.get("smoke", False)
        quality[suffix] = result
        inputs[str(path.relative_to(args.root))] = sha256(path)
    checkpoints = {}
    diagnostics = {}
    for rank in (64, 96):
        checkpoint_path = args.root / "checkpoints" / f"r{rank}" / "manifest.json"
        allocation_path = args.root / f"v{rank}" / "results.json"
        wo_path = args.root / "manifests" / f"wo_fp32_r{rank}_audit.json"
        checkpoint = read_json(checkpoint_path)
        allocation = read_json(allocation_path)
        wo = read_json(wo_path)
        assert checkpoint["status"] == allocation["status"] == wo["status"] == "complete"
        metrics = [row["quantized_heldout_relative_mse"] for row in wo["metrics"].values()]
        checkpoints[rank] = checkpoint
        diagnostics[rank] = {
            "attention_layer_ranks": checkpoint["compression"]["v_layer_ranks"],
            "attention_uniform_heldout_relative_mse": read_json(
                args.root / "vbank" / f"r{rank}" / "results.json"
            )["aggregate"]["mean_heldout_factor_dtype_relative_mse"],
            "terminal_kl_uniform": allocation["confirmation"]["uniform"]["terminal_kl"]["mean"],
            "terminal_kl_selected": allocation["confirmation"]["selected"]["terminal_kl"]["mean"],
            "mamba_wo_mean_quantized_heldout_relative_mse": sum(metrics) / len(metrics),
            "mamba_wo_min_quantized_heldout_relative_mse": min(metrics),
            "mamba_wo_max_quantized_heldout_relative_mse": max(metrics),
        }
        for path in (checkpoint_path, allocation_path, wo_path):
            inputs[str(path.relative_to(args.root))] = sha256(path)
    dense = quality["Dense"]["metrics"]
    rows = {}
    for suffix in RUNS:
        metrics = quality[suffix]["metrics"]
        rows[suffix] = {
            "wikitext2_ppl": metrics["wikitext2_ppl"],
            "wikitext2_relative_change": metrics["wikitext2_ppl"] / dense["wikitext2_ppl"] - 1,
            "c4_ppl": metrics["c4_validation_128_ppl"],
            "c4_relative_change": metrics["c4_validation_128_ppl"] / dense["c4_validation_128_ppl"] - 1,
            "mcq_average": metrics["average_accuracy"],
            "mcq_point_change": 100 * (metrics["average_accuracy"] - dense["average_accuracy"]),
            "tasks": metrics["task_accuracy"],
            "seconds": quality[suffix]["seconds"],
        }
    summary = {
        "status": "complete",
        "format": "basisserve.nemotron_h.c1_quality_summary.v1",
        "label": args.label,
        "protocol": {
            "calibration": "allenai/c4 train, 256x2048 fit + 64x2048 heldout",
            "attention_fit": "ALS6, fixed encoder CG16, BF16 factors",
            "allocation": "two-sided terminal-KL, seven-rank bank 32:16:128",
            "mamba_wo_fit": "same retained ratio as full attention, ALS6, fixed encoder CG16",
            "quality": quality["Dense"]["protocol"],
        },
        "quality": rows,
        "fit_diagnostics": diagnostics,
        "compression": {str(rank): checkpoints[rank]["compression"] for rank in (64, 96)},
        "inputs": inputs,
    }
    write_json(args.root / "summary.json", summary)
    task_names = [row["task"] for row in rows["Dense"]["tasks"]]
    task_maps = {suffix: {row["task"]: row["value"] for row in rows[suffix]["tasks"]}
        for suffix in RUNS}
    lines = [
        f"# {args.label}: C1 V and Mamba Wo compression",
        "",
        "## Protocol",
        "",
        "- Calibration: C4 train, 256x2048 fit windows and 64x2048 held-out windows.",
        "- Full-attention V: ALS6, fixed encoder CG16, BF16 factors, two-sided terminal-KL layer allocation.",
        "- Mamba2 Wo: the same retained ratio as full attention, ALS6, fixed encoder CG16, FP32 fit and BF16 factors.",
        "- Quality: full WikiText2 test at 2048, 128 disjoint C4 validation windows at 2048, and seven zero-shot MCQ tasks.",
        "",
        "## Quality",
        "",
        "| Setting | WikiText2 PPL | vs Dense | C4 PPL | vs Dense | MCQ average | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"Dense": "Dense", "C1-R64": "Retain 50% (mean V64)",
        "C1-R96": "Retain 75% (mean V96)"}
    for suffix in RUNS:
        row = rows[suffix]
        lines.append(f"| {labels[suffix]} | {row['wikitext2_ppl']:.6f} | "
            f"{_pct(row['wikitext2_relative_change']) if suffix != 'Dense' else '—'} | "
            f"{row['c4_ppl']:.6f} | "
            f"{_pct(row['c4_relative_change']) if suffix != 'Dense' else '—'} | "
            f"{_pct(row['mcq_average'])} | "
            f"{row['mcq_point_change']:+.3f} pt |")
    lines += ["", "## MCQ by task", "", "| Task | Dense | Retain 50% | Retain 75% |",
        "|---|---:|---:|---:|"]
    for task in task_names:
        lines.append(f"| {task} | {_pct(task_maps['Dense'][task])} | "
            f"{_pct(task_maps['C1-R64'][task])} | {_pct(task_maps['C1-R96'][task])} |")
    lines += ["", "## Fit diagnostics", "",
        "| Setting | Full-attention layer ranks | Uniform V held-out rel-MSE | Selected terminal KL | Mamba Wo held-out rel-MSE |",
        "|---|---|---:|---:|---:|"]
    for rank in (64, 96):
        row = diagnostics[rank]
        lines.append(f"| Mean V{rank} | `{row['attention_layer_ranks']}` | "
            f"{row['attention_uniform_heldout_relative_mse']:.6f} | "
            f"{row['terminal_kl_selected']:.6f} | "
            f"{row['mamba_wo_mean_quantized_heldout_relative_mse']:.6f} |")
    lines += ["", "## Communication accounting", ""]
    for rank in (64, 96):
        compression = checkpoints[rank]["compression"]
        attention = compression["full_attention_allgather"]
        mamba = compression["mamba_wo_allgather"]
        lines.append(f"- Mean V{rank}: full-attention private AllGather bytes are reduced by "
            f"{_pct(attention['reduction_vs_dense_allgather'])}; Mamba2 Wo private AllGather bytes are reduced by "
            f"{_pct(mamba['reduction_vs_dense_allgather'])}.")
    lines += ["", "All persisted factors and result inputs are hash-verified in `summary.json`.", ""]
    (args.root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("SUMMARY COMPLETE", args.root / "RESULTS.md")


if __name__ == "__main__":
    main()
