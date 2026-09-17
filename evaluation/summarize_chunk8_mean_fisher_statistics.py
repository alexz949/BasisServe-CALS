"""Summarize the all-layer Mean-128 Chunk8 Fisher fit."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics

from evaluation.v96kl_common import read_json, sha256, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--fit-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def markdown(summary: dict) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# All-Layer Mean-128 Chunk8 Fisher Fit",
        "",
        "Llama-3.1-8B-Instruct; Dense V128; frozen Base16; direct Mean-128 Chunk8 R16; 64×64K fit and 16×64K held-out windows; eight ALS sweeps followed by a final query closure; PCG capped at 50 iterations.",
        "",
        "## Aggregate",
        "",
        f"- Initial mean held-out Fisher NMSE: `{aggregate['initial_heldout_mean']:.6f}`",
        f"- Final mean held-out Fisher NMSE: `{aggregate['final_heldout_mean']:.6f}`",
        f"- Relative reduction: `{100 * aggregate['heldout_relative_reduction']:.2f}%`",
        f"- Layers improved over initialization: `{aggregate['layers_improved_over_initial']}/32`",
        f"- Layers whose sweep-8 encoder endpoint beats sweep 4: `{aggregate['sweep8_better_than_sweep4']}/32`",
        "",
        "| Sweep | Mean train Fisher NMSE | Mean held-out Fisher NMSE | Median held-out Fisher NMSE |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate_sweeps"]:
        lines.append(
            f"| {row['sweep']} | {row['train_mean']:.6f} | "
            f"{row['heldout_mean']:.6f} | {row['heldout_median']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per layer",
            "",
            "`Best` is a held-out convergence diagnostic only. Saved factors always use the fixed sweep-8 endpoint plus final query closure.",
            "",
            "| Layer | Initial held-out | Sweep 4 held-out | Sweep 8 held-out | Final closure held-out | Final train | Best |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["layers"]:
        lines.append(
            f"| {row['layer']} | {row['initial_heldout']:.6f} | "
            f"{row['sweep4_heldout']:.6f} | {row['sweep8_heldout']:.6f} | "
            f"{row['final_heldout']:.6f} | {row['final_train']:.6f} | "
            f"S{row['best_sweep']} ({row['best_heldout']:.6f}) |"
        )
    lines.extend(
        [
            "",
            "## Timing and storage",
            "",
            f"- Mean statistics load time per layer: `{summary['timing']['load_mean_seconds']:.2f}s` average, `{summary['timing']['load_max_seconds']:.2f}s` maximum.",
            f"- Mean eight-sweep fit plus diagnostics and closure: `{summary['timing']['fit_mean_seconds']:.2f}s` average, `{summary['timing']['fit_max_seconds']:.2f}s` maximum per layer.",
            f"- Compact statistics bank: `{summary['statistics_bank_bytes'] / (1024 ** 3):.2f} GiB` from layer manifests.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parser().parse_args()
    records = []
    layer_rows = []
    artifacts = {}
    for layer in range(32):
        path = args.fit_root / f"layer_{layer:03d}.json"
        record = read_json(path)
        assert record["status"] == "complete" and all(record["audits"].values())
        assert record["protocol"]["layer"] == layer
        assert record["protocol"]["feature"] == "mean"
        assert record["protocol"]["als_sweeps"] == 8
        assert len(record["sweeps"]) == 8
        best = min(record["sweeps"], key=lambda row: row["heldout_fisher_nmse"])
        layer_rows.append(
            {
                "layer": layer,
                "initial_heldout": record["metrics"]["initial"]["heldout"]["fisher_nmse"],
                "sweep4_heldout": record["sweeps"][3]["heldout_fisher_nmse"],
                "sweep8_heldout": record["sweeps"][7]["heldout_fisher_nmse"],
                "final_heldout": record["metrics"]["final"]["heldout"]["fisher_nmse"],
                "final_train": record["metrics"]["final"]["fit"]["fisher_nmse"],
                "best_sweep": best["sweep"],
                "best_heldout": best["heldout_fisher_nmse"],
            }
        )
        artifacts[str(layer)] = {"file": path.name, "sha256": sha256(path)}
        records.append(record)

    aggregate_sweeps = []
    for sweep in range(8):
        train = [record["sweeps"][sweep]["train_fisher_nmse"] for record in records]
        heldout = [
            record["sweeps"][sweep]["heldout_fisher_nmse"] for record in records
        ]
        aggregate_sweeps.append(
            {
                "sweep": sweep + 1,
                "train_mean": statistics.mean(train),
                "heldout_mean": statistics.mean(heldout),
                "heldout_median": statistics.median(heldout),
            }
        )
    initial_heldout = [row["initial_heldout"] for row in layer_rows]
    final_heldout = [row["final_heldout"] for row in layer_rows]
    initial_mean = statistics.mean(initial_heldout)
    final_mean = statistics.mean(final_heldout)
    statistics_root = Path(records[0]["protocol"]["statistics_source"]).parent
    statistics_bytes = sum(
        read_json(statistics_root / f"layer_{layer:03d}.json")["total_bytes"]
        for layer in range(32)
    )
    summary = {
        "status": "complete",
        "protocol": {
            "model_variant": "Llama-3.1-8B-Instruct",
            "feature": "mean",
            "feature_dim": 128,
            "layers": list(range(32)),
            "als_sweeps": 8,
            "final_query_closure": True,
            "cg_iterations": 50,
        },
        "aggregate": {
            "initial_heldout_mean": initial_mean,
            "final_heldout_mean": final_mean,
            "heldout_relative_reduction": (initial_mean - final_mean) / initial_mean,
            "layers_improved_over_initial": sum(
                final < initial
                for initial, final in zip(initial_heldout, final_heldout, strict=True)
            ),
            "sweep8_better_than_sweep4": sum(
                row["sweep8_heldout"] < row["sweep4_heldout"] for row in layer_rows
            ),
            "best_sweep_counts": {
                str(sweep): sum(row["best_sweep"] == sweep for row in layer_rows)
                for sweep in range(1, 9)
            },
        },
        "aggregate_sweeps": aggregate_sweeps,
        "layers": layer_rows,
        "timing": {
            "load_mean_seconds": statistics.mean(
                record["timing"]["statistics_load_wall_seconds"]
                for record in records
            ),
            "load_max_seconds": max(
                record["timing"]["statistics_load_wall_seconds"]
                for record in records
            ),
            "fit_mean_seconds": statistics.mean(
                record["timing"]["fit_wall_seconds_including_diagnostics_and_final_closure"]
                for record in records
            ),
            "fit_max_seconds": max(
                record["timing"]["fit_wall_seconds_including_diagnostics_and_final_closure"]
                for record in records
            ),
        },
        "statistics_bank_bytes": statistics_bytes,
        "artifacts": artifacts,
    }
    write_json(args.output / "summary.json", summary)
    text = markdown(summary)
    path = args.output / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
