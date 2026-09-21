"""Export the Section 4 mechanism ablations as paper-facing CSV, plots, and Markdown."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from evaluation.v96kl_common import read_json


RANKS = (4, 8, 16, 24, 32, 48, 64, 80, 96)
COMPONENTS = {
    "r16_only": (0, 16, 16, 16),
    "r20_only": (0, 20, 20, 20),
    "b4r16": (4, 16, 20, 16),
    "b16_only": (16, 0, 16, 0),
    "b16r16": (16, 16, 32, 16),
    "r32_only": (0, 32, 32, 32),
    "exact_k": (128, 0, 128, 128),
}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(row):
    return {
        "attention_mass_recall": row["attention_mass_recall"],
        "nonsink_mass_recall": row["nonsink_mass_recall"],
        "page_recall": row["page_recall"],
        "routed_page_recall": row["routed_page_recall"],
        "page_kl": row["page_kl_exact_to_proxy"],
        "attention_rel_mse": row["pooled_attention_rel_mse"] if "pooled_attention_rel_mse" in row else row["attention_rel_mse"],
        "post_wo_rel_mse": row["pooled_post_wo_rel_mse"] if "pooled_post_wo_rel_mse" in row else row["post_wo_rel_mse"],
    }


def component_tables(section4, output):
    result = read_json(section4 / "diagnostics/components/summary.json")
    rows = []
    for method, capacities in COMPONENTS.items():
        base, residual, width, state = capacities
        rows.append(
            {
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "method": method,
                "base_rank": base,
                "residual_rank": residual,
                "route_scan_width": width,
                "additional_routing_state": state,
                "budget": 2048,
                "page_size": 32,
                **metrics(result["means"][method]),
                "ruler_average": "",
                "seed": 20260828,
            }
        )
    write_csv(output / "base_residual_components.csv", rows)
    layer_rows = []
    for layer in result["layers"]:
        for method, capacities in COMPONENTS.items():
            base, residual, width, state = capacities
            layer_rows.append(
                {
                    "layer": layer["layer"],
                    "method": method,
                    "base_rank": base,
                    "residual_rank": residual,
                    "route_scan_width": width,
                    "additional_routing_state": state,
                    **metrics(layer["means"][method]),
                }
            )
    write_csv(output / "base_residual_components_layers.csv", layer_rows)
    return rows


def base_tables(section4, output):
    result = read_json(section4 / "diagnostics/base_rank_sweep/summary.json")
    rows = []
    layer_rows = []
    for rank in RANKS:
        fit_records = []
        for layer in range(32):
            fit = read_json(section4 / f"base_rank_sweep/b{rank}r0/layer_{layer:03d}.json")
            fit_records.append(fit["metrics"]["heldout"])
            layer_rows.append(
                {
                    "layer": layer,
                    "base_rank": rank,
                    "centered_explained_fraction": fit["metrics"]["heldout"]["centered_explained_fraction"],
                    "predictable_energy_captured_fraction": fit["metrics"]["heldout"]["predictable_energy_captured_fraction"],
                    **metrics(result["layers"][layer]["means"][f"b{rank}_only"]),
                }
            )
        rows.append(
            {
                "base_rank": rank,
                "residual_rank": 0,
                "centered_explained_fraction": sum(row["centered_explained_fraction"] for row in fit_records) / 32,
                "predictable_energy_captured_fraction": sum(row["predictable_energy_captured_fraction"] for row in fit_records) / 32,
                **metrics(result["means"][f"b{rank}_only"]),
            }
        )
    write_csv(output / "base_rank_sweep.csv", rows)
    write_csv(output / "base_rank_sweep_layers.csv", layer_rows)
    return rows, result["means"]["exact_k"]["attention_mass_recall"]


def objective_tables(section4, root, output):
    result = read_json(section4 / "diagnostics/objectives/summary.json")
    paths = {
        "residual_mse": section4 / "objectives/residual_mse/ours_b16r16",
        "score_mse": section4 / "objectives/score_mse/ours_b16r16",
        "page_fisher": root / "ours_b16r16",
    }
    rows = []
    layer_rows = []
    for objective, path in paths.items():
        wall = 0.0
        for layer in range(32):
            fit = read_json(path / f"layer_{layer:03d}.json")
            wall += float(fit.get("fit_wall_time_seconds", 0.0))
            layer_rows.append(
                {
                    "layer": layer,
                    "objective": objective,
                    "base_rank": 16,
                    "residual_rank": 16,
                    "heldout_training_objective": (
                        fit["losses"].get("heldout_normalized", "")
                        if objective != "page_fisher"
                        else fit["losses"]["b16_r16"]["validation_page_fisher_nmse"]
                    ),
                    **metrics(result["layers"][layer]["means"][objective]),
                }
            )
        rows.append(
            {
                "objective": objective,
                "base_rank": 16,
                "residual_rank": 16,
                **metrics(result["means"][objective]),
                "ruler_average": "",
                "fit_wall_time_seconds": wall if wall else "",
                "seed": 20260828,
            }
        )
    write_csv(output / "residual_objectives.csv", rows)
    write_csv(output / "residual_objectives_layers.csv", layer_rows)
    return rows


def plot_base(rows, exact, output):
    ranks = [row["base_rank"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    axes[0].plot(ranks, [row["centered_explained_fraction"] for row in rows], marker="o", label="Centered K energy")
    axes[0].plot(ranks, [row["predictable_energy_captured_fraction"] for row in rows], marker="s", label="Full-RRR predictable energy")
    axes[0].set(xlabel="Base rank", ylabel="Explained fraction", ylim=(0, 1.03))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(ranks, [row["attention_mass_recall"] for row in rows], marker="o", label="Base-only")
    axes[1].axhline(exact, color="black", linestyle="--", linewidth=1, label="Exact-K oracle")
    axes[1].set(xlabel="Base rank", ylabel="Retained attention mass")
    axes[1].legend(frameon=False, fontsize=8)
    for axis, label in zip(axes, ("(a)", "(b)")):
        axis.grid(alpha=0.2)
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    figure.savefig(plots / "base_rank_sweep.pdf")
    figure.savefig(plots / "base_rank_sweep.png", dpi=240)
    plt.close(figure)


def markdown(components, base, objectives, output):
    def best(rows, key, reverse=False):
        return sorted(rows, key=lambda row: float(row[key]), reverse=reverse)[0]

    b4 = next(row for row in components if row["method"] == "b4r16")
    r20 = next(row for row in components if row["method"] == "r20_only")
    page = next(row for row in objectives if row["objective"] == "page_fisher")
    score = next(row for row in objectives if row["objective"] == "score_mse")
    residual = next(row for row in objectives if row["objective"] == "residual_mse")
    lines = [
        "# Section 4 Routing Ablation: Mechanism Results",
        "",
        "Model: Llama-3.1-8B-Instruct. All arms use Dense V128, original W_O, Page32,",
        "sink32/recent64 inside B2048, and exact selected post-RoPE K for final attention.",
        "",
        "## Matched logical width",
        "",
        "| Method | Width | Extra state | Mass | Routed page recall | Page KL | Post-W_O rel-MSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| B4R16 | 20 | 16 | {b4['attention_mass_recall']:.6f} | {b4['routed_page_recall']:.6f} | {b4['page_kl']:.6f} | {b4['post_wo_rel_mse']:.6f} |",
        f"| R20-only | 20 | 20 | {r20['attention_mass_recall']:.6f} | {r20['routed_page_recall']:.6f} | {r20['page_kl']:.6f} | {r20['post_wo_rel_mse']:.6f} |",
        "",
        "## Residual objective",
        "",
        "| Objective | Mass | Routed page recall | Page KL | Post-W_O rel-MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in (residual, score, page):
        lines.append(
            f"| {row['objective']} | {row['attention_mass_recall']:.6f} | {row['routed_page_recall']:.6f} | {row['page_kl']:.6f} | {row['post_wo_rel_mse']:.6f} |"
        )
    lines += [
        "",
        "## Base-rank sweep",
        "",
        f"Best Base-only retained mass: rank {best(base, 'attention_mass_recall', True)['base_rank']}.",
        "See `plots/base_rank_sweep.pdf` and the layer-level CSV for the complete trajectory.",
        "",
        "RULER columns remain intentionally empty until the mechanism results have been inspected.",
    ]
    (output / "SECTION4_ABLATION_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--section4", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/section4_ablation"))
    args = parser.parse_args()
    components = component_tables(args.section4, args.output)
    base, exact = base_tables(args.section4, args.output)
    objectives = objective_tables(args.section4, args.root, args.output)
    plot_base(base, exact, args.output)
    markdown(components, base, objectives, args.output)
    print({"status": "complete", "output": str(args.output)}, flush=True)


if __name__ == "__main__":
    main()
