"""Audit and summarize Nemotron-H PaLU M-LRD/G-LRD4 quality results."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.nemotron_h_palu import CHECKPOINT_FORMAT, FISHER_FORMAT, V_COVARIANCE_FORMAT
from evaluation.v96kl_common import read_json, sha256, write_json


METHODS = ("mlrd", "glrd4")
RANKS = (64, 96)


def _pct(value):
    return f"{100 * value:.3f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--run-prefix", required=True)
    args = parser.parse_args()
    palu_root = args.root / "palu"
    dense_path = args.root / "quality" / f"{args.run_prefix}-Dense" / "result.json"
    dense = read_json(dense_path)
    assert dense["status"] == "complete" and not dense.get("smoke", False)
    fisher_path = palu_root / "fisher.json"
    covariance_path = palu_root / "v_covariances" / "manifest.json"
    fisher = read_json(fisher_path)
    covariance = read_json(covariance_path)
    assert fisher["status"] == covariance["status"] == "complete"
    assert fisher["format"] == FISHER_FORMAT
    assert covariance["format"] == V_COVARIANCE_FORMAT
    assert fisher["model"]["config_sha256"] == covariance["model"]["config_sha256"]
    assert fisher["fisher"]["samples"] == 256
    assert fisher["fisher"]["sequence_length"] == 2048
    assert covariance["calibration"]["fit_windows"] == 256
    assert covariance["calibration"]["heldout_windows"] == 64
    assert covariance["calibration"]["sequence_length"] == 2048
    dense_metrics = dense["metrics"]
    inputs = {
        str(dense_path.relative_to(args.root)): sha256(dense_path),
        str(fisher_path.relative_to(args.root)): sha256(fisher_path),
        str(covariance_path.relative_to(args.root)): sha256(covariance_path),
    }
    rows = {}
    diagnostics = {}
    compressions = {}
    for method in METHODS:
        for rank in RANKS:
            slug = f"{method}_r{rank}"
            checkpoint_dir = palu_root / "checkpoints" / slug
            checkpoint_path = checkpoint_dir / "manifest.json"
            checkpoint = read_json(checkpoint_path)
            assert checkpoint["status"] == "complete"
            assert checkpoint["format"] == CHECKPOINT_FORMAT
            assert checkpoint["compression"]["method_slug"] == method
            assert checkpoint["compression"]["equivalent_mean_rank_per_head"] == rank
            assert checkpoint["compression"]["mamba2_wo"] == "dense_unchanged"
            assert checkpoint["compression"]["attention_o_proj"] == "dense_unchanged"
            artifact = checkpoint_dir / checkpoint["artifact"]["file"]
            assert sha256(artifact) == checkpoint["artifact"]["sha256"]
            run_name = f"{args.run_prefix}-PaLU-{method.upper()}-R{rank}"
            smoke_path = palu_root / "quality" / run_name / "smoke.json"
            result_path = palu_root / "quality" / run_name / "result.json"
            smoke = read_json(smoke_path)
            result = read_json(result_path)
            assert smoke["status"] == result["status"] == "complete"
            assert smoke["smoke"] and not result.get("smoke", False)
            assert smoke["checkpoint_manifest_sha256"] == sha256(checkpoint_path)
            assert result["checkpoint"]["manifest_sha256"] == sha256(checkpoint_path)
            metrics = result["metrics"]
            rows[slug] = {
                "wikitext2_ppl": metrics["wikitext2_ppl"],
                "wikitext2_relative_change": metrics["wikitext2_ppl"]
                / dense_metrics["wikitext2_ppl"]
                - 1,
                "c4_ppl": metrics["c4_validation_128_ppl"],
                "c4_relative_change": metrics["c4_validation_128_ppl"]
                / dense_metrics["c4_validation_128_ppl"]
                - 1,
                "mcq_average": metrics["average_accuracy"],
                "mcq_point_change": 100
                * (metrics["average_accuracy"] - dense_metrics["average_accuracy"]),
                "tasks": metrics["task_accuracy"],
                "seconds": result["seconds"],
            }
            layer_diagnostics = checkpoint["layers"]
            diagnostics[slug] = {
                "layer_ranks": checkpoint["compression"]["layer_ranks"],
                "realized_retained_v_ratio": checkpoint["compression"][
                    "realized_retained_v_ratio"
                ],
                "mean_fit_relative_error": sum(
                    row["quantized_fit_relative_error"] for row in layer_diagnostics
                )
                / len(layer_diagnostics),
                "mean_heldout_relative_error": sum(
                    row["quantized_heldout_relative_error"]
                    for row in layer_diagnostics
                )
                / len(layer_diagnostics),
            }
            compressions[slug] = checkpoint["compression"]
            for path in (checkpoint_path, artifact, smoke_path, result_path):
                inputs[str(path.relative_to(args.root))] = sha256(path)
    summary = {
        "status": "complete",
        "format": "basisserve.nemotron_h.palu_quality_summary.v1",
        "label": args.label,
        "protocol": {
            "calibration": "allenai/c4 train, 256x2048 Fisher and V-input covariance fit + 64x2048 covariance heldout",
            "allocation": "PaLU double-shift Fisher weighting with an exact global mean-rank budget in blocks of 32",
            "factorization": "PaLU activation-aware balanced whitened SVD, BF16 factors",
            "targets": "full-attention V only; attention o_proj and Mamba2 Wo dense",
            "quality": dense["protocol"],
        },
        "dense": dense_metrics,
        "quality": rows,
        "fit_diagnostics": diagnostics,
        "compression": compressions,
        "inputs": inputs,
    }
    write_json(palu_root / "summary.json", summary)
    lines = [
        f"# {args.label}: PaLU V-only compression",
        "",
        "## Protocol",
        "",
        "- Calibration: C4 train; 256x2048 windows for PaLU double-shift Fisher weighting and V-input covariance, plus 64x2048 held-out covariance windows.",
        "- Methods: M-LRD (one group per KV head) and G-LRD4 (four KV heads per group), with an exact global mean-rank budget allocated in blocks of 32.",
        "- Targets: full-attention V only. Attention o_proj and every Mamba2 Wo remain dense.",
        "- Quality: full WikiText2 test at 2048, 128 disjoint C4 validation windows at 2048, and seven zero-shot MCQ tasks.",
        "",
        "## Quality",
        "",
        "| Setting | WikiText2 PPL | vs Dense | C4 PPL | vs Dense | MCQ average | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Dense | {dense_metrics['wikitext2_ppl']:.6f} | — | {dense_metrics['c4_validation_128_ppl']:.6f} | — | {_pct(dense_metrics['average_accuracy'])} | — |",
    ]
    names = {
        "mlrd_r64": "M-LRD retain 50%",
        "mlrd_r96": "M-LRD retain 75%",
        "glrd4_r64": "G-LRD4 retain 50%",
        "glrd4_r96": "G-LRD4 retain 75%",
    }
    for slug in ("mlrd_r64", "mlrd_r96", "glrd4_r64", "glrd4_r96"):
        row = rows[slug]
        lines.append(
            f"| {names[slug]} | {row['wikitext2_ppl']:.6f} | {_pct(row['wikitext2_relative_change'])} | "
            f"{row['c4_ppl']:.6f} | {_pct(row['c4_relative_change'])} | "
            f"{_pct(row['mcq_average'])} | {row['mcq_point_change']:+.3f} pt |"
        )
    lines += [
        "",
        "## Fit diagnostics",
        "",
        "| Setting | Layer group ranks | Realized retention | Fit rel-error | Held-out rel-error |",
        "|---|---|---:|---:|---:|",
    ]
    for slug in ("mlrd_r64", "mlrd_r96", "glrd4_r64", "glrd4_r96"):
        row = diagnostics[slug]
        lines.append(
            f"| {names[slug]} | `{row['layer_ranks']}` | "
            f"{_pct(row['realized_retained_v_ratio'])} | "
            f"{row['mean_fit_relative_error']:.6f} | "
            f"{row['mean_heldout_relative_error']:.6f} |"
        )
    lines += [
        "",
        "All factors, checkpoint manifests, smoke runs, and quality results are hash-verified in `summary.json`.",
        "",
    ]
    (palu_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("PALU SUMMARY COMPLETE", palu_root / "RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
