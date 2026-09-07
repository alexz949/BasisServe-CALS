#!/usr/bin/env python3
"""Validate and summarize the completed four-ceiling experiment; no model forward."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
from statistics import mean
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ARMS = ("local_c1", "local_raw", "all_c1", "all_raw")
LABELS = ("Local C1-V80", "Local raw V128", "All C1-V640", "All raw V1024")
COLORS = ("#D55E00", "#E69F00", "#0072B2", "#009E73")


def _fit_values(layer: dict) -> list[float]:
    return [
        layer["local_c1_v80"]["fit_predictable_total_fraction"],
        layer["local_raw_v128"]["fit_predictable_total_fraction"],
        layer["all_c1_v640"]["fit"]["aggregate"]["predictable_total_fraction"],
        layer["all_raw_v1024"]["fit"]["aggregate"]["predictable_total_fraction"],
    ]


def _heldout_values(layer: dict) -> list[float]:
    return [
        1.0 - layer[name]["heldout"]["aggregate"]["key_centered_relative_mse"]
        for name in ("local_c1_v80", "local_raw_v128", "all_c1_v640", "all_raw_v1024")
    ]


def _table(rows: list[dict], split: str) -> list[str]:
    return [
        "| Layer | Local C1-V80 | Local raw V128 | All C1-V640 | All raw V1024 |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {row['layer']} | "
            + " | ".join(f"{value * 100:.3f}%" for value in row[split]) + " |"
            for row in rows
        ],
        "",
    ]


def _plot(rows: list[dict], root: Path) -> None:
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    layer_ids = [row["layer"] for row in rows]
    for ax, split, title in zip(
        axes,
        ("fit", "heldout"),
        ("Fit affine ceiling: 64 x 32K", "Held-out prediction: 16 x 32K"),
    ):
        ax.axvspan(23.5, 35.5, color="#F0F2F4", zorder=0)
        for i, (label, color) in enumerate(zip(LABELS, COLORS)):
            values = [row[split][i] for row in rows]
            ax.plot(layer_ids, values, color=color, label=label, linewidth=1.9,
                    linestyle="--" if i < 2 else "-", marker="o", markersize=2.7)
            ax.scatter([33], [values[33]], color=color, s=28, zorder=4)
        ax.axvline(33, color="#AAAAAA", linewidth=0.8, linestyle=":")
        ax.set(xlim=(-0.5, 35.5), ylim=(0, 1), xlabel="Layer index", title=title)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Explained centered pre-K variance (mean over 8 KV groups)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Qwen3-8B-Base: local versus all-group Value prediction of Key", y=0.99)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    for extension in ("png", "svg", "pdf"):
        fig.savefig(root / f"four_ceilings.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = sorted(root.glob("shard_*/result.json"))
    shards = [json.loads(path.read_text()) for path in paths]
    assert len(shards) == 4 and all(s["status"] == "complete" for s in shards)
    protocol = {k: v for k, v in shards[0]["protocol"].items() if k != "shard_index"}
    assert (protocol["fit_documents"], protocol["validation_documents"],
            protocol["sequence_length"]) == (64, 16, 32768)
    for shard in shards:
        assert {k: v for k, v in shard["protocol"].items() if k != "shard_index"} == protocol
    layers = sorted((layer for shard in shards for layer in shard["layers"]),
                    key=lambda layer: layer["layer"])
    assert [layer["layer"] for layer in layers] == list(range(36))

    local_root = Path(protocol["local_ceilings_root"])
    local_shards = [json.loads(p.read_text()) for p in sorted(local_root.glob("shard_*/result.json"))]
    locals_by_layer = {layer["layer"]: layer for shard in local_shards for layer in shard["layers"]}
    for local_shard in local_shards:
        for key in ("model", "calibration_root", "fit_documents", "validation_documents", "sequence_length"):
            assert local_shard["protocol"][key] == protocol[key]
    target_energy_relative_difference = 0.0
    rows = []
    for layer in layers:
        fit, heldout = _fit_values(layer), _heldout_values(layer)
        assert all(math.isfinite(x) for x in fit + heldout)
        assert all(-1e-6 <= x <= 1 + 1e-6 for x in fit)
        assert fit[0] <= fit[1] + 1e-6 <= fit[3] + 2e-6
        assert fit[0] <= fit[2] + 1e-6 <= fit[3] + 2e-6
        source = locals_by_layer[layer["layer"]]
        raw_groups = layer["all_raw_v1024"]["fit"]["groups"]
        c1_groups = layer["all_c1_v640"]["fit"]["groups"]
        local_groups = source["raw_v128"]["fit_spectra"]["groups"]
        local_c1_groups = source["c1_v80"]["fit_groups"]
        assert len(raw_groups) == len(c1_groups) == len(local_groups) == len(local_c1_groups) == 8
        for g, (raw, c1, local, local_c1) in enumerate(zip(raw_groups, c1_groups, local_groups, local_c1_groups)):
            assert raw["group"] == c1["group"] == local["group"] == local_c1["group"] == g
            assert raw["predictable_total_fraction"] + 1e-5 >= local["predictable_total_fraction"]
            assert raw["predictable_total_fraction"] + 1e-5 >= c1["predictable_total_fraction"]
            assert c1["predictable_total_fraction"] + 1e-5 >= local_c1["predictable_total_fraction"]
            difference = abs(raw["target_centered_energy"] - local["target_centered_energy"])
            relative = difference / max(abs(local["target_centered_energy"]), 1e-300)
            target_energy_relative_difference = max(target_energy_relative_difference, relative)
            assert relative < 1e-4
        rows.append({"layer": layer["layer"], "fit": fit, "heldout": heldout})

    cohorts = {}
    for label, subset in (("0--11", rows[:12]), ("12--23", rows[12:24]),
                          ("24--35", rows[24:]), ("All", rows)):
        cohorts[label] = {split: [mean(row[split][i] for row in subset) for i in range(4)]
                          for split in ("fit", "heldout")}
    checks = {
        "completed_layers": len(rows),
        "fit_nested_subspaces_checked_per_group": True,
        "maximum_relative_target_energy_difference_vs_local": target_energy_relative_difference,
        "heldout_all_c1_better_than_local_raw_layers": sum(row["heldout"][2] > row["heldout"][1] for row in rows),
        "maximum_all_raw_fit_minus_heldout": max(row["fit"][3] - row["heldout"][3] for row in rows),
    }
    result = {
        "format": "basisserve.qwen3_8b.four_v_pre_k_ceilings_summary.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "protocol": protocol,
        "arm_order": list(ARMS),
        "aggregation": "arithmetic mean of per-KV-group R-squared; equal weight to each layer in cohorts",
        "checks": checks,
        "cohorts": cohorts,
        "layers": rows,
        "sources": [{"path": str(path.relative_to(root)),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "command": shard["command"], "environment": shard["environment"]}
                    for path, shard in zip(paths, shards)],
    }
    (root / "aggregate.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (root / "ceilings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", *[f"{split}_{arm}" for split in ("fit", "heldout") for arm in ARMS]])
        writer.writerows([row["layer"], *row["fit"], *row["heldout"]] for row in rows)

    l33 = rows[33]
    c, _, ca, va = l33["fit"]
    lines = [
        "# Qwen3-8B: Four Value-to-Pre-Key Affine Ceilings",
        "",
        "All 36 layers completed. The all-group control changes the interpretation of the earlier local-group result: substantial Key variance is linearly predictable from other Value groups, including their existing C1 latents.",
        "",
        "![Fit and held-out four-ceiling curves](four_ceilings.png)",
        "",
        "Vector exports: [SVG](four_ceilings.svg), [PDF](four_ceilings.pdf). Exact values: [CSV](ceilings.csv), [aggregate JSON](aggregate.json).",
        "",
        "## Protocol and metric",
        "",
        "Qwen3-8B-Base; all 36 layers; 8 KV groups per layer. The regression fit uses 64 C4 windows of 32,768 tokens; evaluation uses the same 16 separate 32,768-token windows as the local controls. The C1-V80 checkpoint is fixed. All-group activation rows are concatenated across groups at the same token position. No neighboring token, future token, Query, or Key is used as an input feature.",
        "",
        "Inputs are local C1-V80, local raw V128, all C1-V640, and all raw V1024. Each target is that layer's per-group pre-RoPE K128. Each predictor is an unrestricted affine map, without a Base16/24 rank constraint. Existing activation captures are streamed; this experiment performs no model forward.",
        "",
        r"For centered fit activations, $\eta_g=1-\mathrm{tr}(G_{KK,g}-G_{KX,g}G_{XX}^{\dagger}G_{XK,g})/\mathrm{tr}(G_{KK,g})$. Held-out values are $R_g^2=1-\|K_g-\widehat K_g\|_F^2/\|K_g-\overline K_{g,\mathrm{heldout}}\|_F^2$, using the fit-split map and bias without refitting. Layer values are arithmetic means over the 8 group-specific ratios, not one ratio of pooled energies. Cohort means weight layers equally.",
        "",
        "## Representative held-out results",
        "",
        *_table([rows[i] for i in (0, 13, 29, 33, 35)], "heldout"),
        "## Depth averages",
        "",
        "| Layers | Split | Local C1-V80 | Local raw V128 | All C1-V640 | All raw V1024 |",
        "|:---|:---|---:|---:|---:|---:|",
    ]
    for label, metrics in cohorts.items():
        for split in ("fit", "heldout"):
            lines.append(f"| {label} | {split} | " + " | ".join(f"{v * 100:.3f}%" for v in metrics[split]) + " |")
    lines.extend([
        "",
        "## Layer 33 decomposition and interpretation",
        "",
        f"The local raw-V fit ceiling is {l33['fit'][1]*100:.2f}%, while the all-raw-V ceiling is {va*100:.2f}%. Access to the other groups therefore adds {(va-l33['fit'][1])*100:.2f} percentage points. Held-out prediction shows the same effect: {l33['heldout'][1]*100:.2f}% to {l33['heldout'][3]*100:.2f}%.",
        "",
        f"Local C1-V80 explains {c*100:.2f}% in fit; all C1-V640 explains {ca*100:.2f}%. Their held-out values are {l33['heldout'][0]*100:.2f}% and {l33['heldout'][2]*100:.2f}%. This establishes a substantial cross-group linear prediction opportunity within the existing C1 features.",
        "",
        r"Along the nested fit spaces $C_g\subseteq C_{\mathrm{all}}\subseteq V_{\mathrm{all}}$, centered Key energy admits the following normalized projection partition:",
        "",
        "| Component | Fraction of centered K energy, mean over groups |",
        "|:---|---:|",
        f"| Predictable from local C1 | {c*100:.3f}% |",
        f"| Additional prediction from other C1 groups | {(ca-c)*100:.3f}% |",
        f"| Predictable from all raw V but lost by all C1 | {(va-ca)*100:.3f}% |",
        f"| Unexplained by all raw V under affine prediction | {(1-va)*100:.3f}% |",
        "",
        "The earlier statement that most layer-33 Key variance is absent from Values was too broad: the approximately 74% unexplained local-raw-V fraction included variance recoverable from other groups. The all-raw-V fit residual is approximately 33%, not 74%. This is a correction to the interpretation; the previous local measurements remain valid.",
        "",
        "Both effects coexist. Cross-group access produces a large gain, but all-raw-V held-out explained variance still falls from about 84% in layers 0--23 to about 71% in layers 24--35. The results support a combination of group locality limits and a remaining late-layer affine residual; they do not establish statistical independence or a literal movement of information between groups.",
        "",
        "All-C1 held-out prediction exceeds local-raw-V prediction in all 36 layers. These are unrestricted reconstruction results. They do not measure query-weighted error, page recall, PPL, RULER accuracy, or decode speed. They also do not establish that a low-rank cross-group router attains this ceiling. Reusing resident C1 features avoids new per-token inputs, but cross-group computation and tensor-parallel communication costs remain unmeasured.",
        "",
        "## Complete per-layer fit results",
        "",
        *_table(rows, "fit"),
        "## Complete per-layer held-out results",
        "",
        *_table(rows, "heldout"),
        "## Verification and execution",
        "",
        f"All shard protocols match, all 36 layers are present once, all metrics are finite, and the expected nested-space inequalities hold per KV group. The maximum relative difference in target centered energy between new and reused raw-V controls is {target_energy_relative_difference:.3g}. All-group input Grams reported full numerical rank (640 and 1024).",
        "",
        "The fit script uses FP32 products with TF32 disabled, accumulated into FP64 Grams; the centered covariance pseudoinverse is computed in FP64. The reused local control uses the same capture discovery and dimensions. The largest all-raw-V fit-to-held-out difference is " + f"{checks['maximum_all_raw_fit_minus_heldout']*100:.3f} percentage points.",
        "",
        "Slurm array 8300105: four NVIDIA L40S GPUs, one per shard, 2 CPU threads and 40 GiB host allocation per shard. All tasks completed with exit code 0 in 6:00--6:03. Reported MaxRSS: 36.55--37.46 GiB. GPU experiments used `/home/zhangal/.conda/envs/basis/bin/python`, PyTorch 2.6.0+cu124. Stderr contains only the Transformers RoPE `device` deprecation warning.",
        "",
        "The synthetic cross-group test passed before submission: targets generated from another Value group are recovered with essentially zero held-out error. The aggregation checks above additionally validate the completed experiment outputs.",
        "",
        "Exact GPU script commands and Python executable paths are retained in each [shard result](shard_0/result.json) and in the `sources` field of [aggregate.json](aggregate.json); the latter includes source SHA-256 hashes. Only `--shard-index` and the shard output suffix differ across the four commands.",
        "",
        "Summary and plot command:",
        "",
        "```bash",
        shlex.join([sys.executable, *sys.argv]),
        "```",
        "",
        "Plotting uses Matplotlib from an isolated temporary dependency directory; the shared conda environments are not modified.",
        "",
    ])
    (root / "summary.md").write_text("\n".join(lines))
    _plot(rows, root)
    print(json.dumps({"output": str(root), "checks": checks, "cohorts": cohorts,
                      "layer33": l33}, indent=2))


if __name__ == "__main__":
    main()
