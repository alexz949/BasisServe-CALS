#!/usr/bin/env python3
"""Analyze real Qwen3-8B Wo-C1 TP-source decoder subspaces."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import scipy.stats
from safetensors.torch import load_file
import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.analysis.c1_rigorous import (  # noqa: E402
    decoder_row_basis,
    pairwise_subspace_metrics,
    subspace_union_energy_ranks,
)


FORMAT = "basisserve.qwen3_8b.wo_c1_source_subspaces.v1"
PHASE1_FORMAT = "basisserve.qwen3_8b.wo_c1_lr_ar_phase1.v2"


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() or None


def _load_phase1(phase1_dir: Path) -> dict[str, Any]:
    results_path = phase1_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if payload.get("format") != PHASE1_FORMAT or payload.get("status") != "complete":
        raise ValueError("input is not a completed Qwen3-8B Wo Phase-1 result")
    model = payload["model"]
    expected = {
        "model_type": "qwen3",
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    observed = {key: model[key] for key in expected}
    if observed != expected:
        raise ValueError(f"unexpected Qwen3-8B geometry: {observed}")
    layers = sorted(payload["layers"], key=lambda row: int(row["layer"]))
    if [int(row["layer"]) for row in layers] != list(range(36)):
        raise ValueError("Phase-1 result does not contain exactly 36 ordered layers")
    payload["layers"] = layers
    return payload


def _aggregate_pairwise(
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, float]:
    angle_minima = [float(row["minimum_angle_degrees"]) for row in rows]
    angle_medians = [float(row["median_angle_degrees"]) for row in rows]
    angle_means = [float(row["mean_angle_degrees"]) for row in rows]
    angle_maxima = [float(row["maximum_angle_degrees"]) for row in rows]
    overlaps = [float(row["projection_overlap"]) for row in rows]
    chordal_squared = [
        float(row["normalized_chordal_distance_squared"]) for row in rows
    ]
    return {
        "minimum_angle_degrees": min(angle_minima),
        "median_angle_degrees": statistics.median(angle_medians),
        "mean_angle_degrees": statistics.fmean(angle_means),
        "maximum_angle_degrees": max(angle_maxima),
        "mean_projection_overlap": statistics.fmean(overlaps),
        "mean_normalized_chordal_distance_squared": statistics.fmean(
            chordal_squared
        ),
        "mean_normalized_chordal_distance": statistics.fmean(
            math.sqrt(max(value, 0.0)) for value in chordal_squared
        ),
    }


def _maximum_orthonormality_residual(bases: Sequence[Tensor]) -> float:
    maximum = 0.0
    for basis in bases:
        identity = torch.eye(
            basis.shape[1],
            dtype=basis.dtype,
            device=basis.device,
        )
        maximum = max(
            maximum,
            float(
                (
                    basis.transpose(0, 1) @ basis - identity
                ).abs().max()
            ),
        )
    return maximum


def _precision_check(
    source_decoders: Tensor,
    fp32_pairwise: Sequence[Mapping[str, float | int]],
) -> dict[str, Any]:
    decoders64 = source_decoders.to(device="cpu", dtype=torch.float64)
    bases64 = tuple(decoder_row_basis(decoder) for decoder in decoders64)
    pairwise64 = pairwise_subspace_metrics(bases64)
    fields = (
        "minimum_angle_degrees",
        "median_angle_degrees",
        "mean_angle_degrees",
        "maximum_angle_degrees",
        "projection_overlap",
        "normalized_chordal_distance_squared",
    )
    differences = {
        field: max(
            abs(float(left[field]) - float(right[field]))
            for left, right in zip(fp32_pairwise, pairwise64)
        )
        for field in fields
    }
    return {
        "scope": "same stored BF16 decoder values; FP32 GPU versus FP64 CPU decomposition",
        "maximum_absolute_differences": differences,
        "fp64_maximum_orthonormality_residual": _maximum_orthonormality_residual(
            bases64
        ),
        "passed": (
            max(
                differences["minimum_angle_degrees"],
                differences["median_angle_degrees"],
                differences["mean_angle_degrees"],
                differences["maximum_angle_degrees"],
            )
            <= 0.05
            and differences["projection_overlap"] <= 1.0e-3
            and differences["normalized_chordal_distance_squared"] <= 1.0e-3
        ),
    }


def _correlation(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    pearson = scipy.stats.pearsonr(x, y)
    spearman = scipy.stats.spearmanr(x, y)
    return {
        "samples": len(x),
        "pearson": {
            "statistic": float(pearson.statistic),
            "pvalue": float(pearson.pvalue),
        },
        "spearman": {
            "statistic": float(spearman.statistic),
            "pvalue": float(spearman.pvalue),
        },
    }


def _correlations(layers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    diversity = {
        "mean_angle_degrees": [
            float(row["subspace"]["aggregate"]["mean_angle_degrees"])
            for row in layers
        ],
        "mean_normalized_chordal_distance_squared": [
            float(
                row["subspace"]["aggregate"][
                    "mean_normalized_chordal_distance_squared"
                ]
            )
            for row in layers
        ],
        "union_energy_rank_95": [
            float(row["subspace"]["union"]["energy_ranks"]["95%"])  # type: ignore[index]
            for row in layers
        ],
    }
    advantages = {
        "heldout_relative_mse": [
            float(row["quality"]["c1_advantage_heldout_relative_mse"])
            for row in layers
        ],
        "heldout_relative_l2": [
            float(row["quality"]["c1_advantage_heldout_relative_l2"])
            for row in layers
        ],
    }
    return {
        diversity_name: {
            advantage_name: _correlation(values, advantages[advantage_name])
            for advantage_name in advantages
        }
        for diversity_name, values in diversity.items()
    }


def _gnuplot_quote(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace("'", "\\'")


def _plot(
    layers: Sequence[Mapping[str, Any]],
    *,
    data_path: Path,
    plot_path: Path,
) -> None:
    if shutil.which("gnuplot") is None:
        raise RuntimeError("gnuplot is required to generate the scatter plot")
    data_lines = ["layer\tchordal_squared\tunion_rank_95\tadvantage_mse"]
    for row in layers:
        data_lines.append(
            "\t".join(
                (
                    str(row["layer"]),
                    str(
                        row["subspace"]["aggregate"][
                            "mean_normalized_chordal_distance_squared"
                        ]
                    ),
                    str(row["subspace"]["union"]["energy_ranks"]["95%"]),
                    str(row["quality"]["c1_advantage_heldout_relative_mse"]),
                )
            )
        )
    _atomic_text(data_path, "\n".join(data_lines) + "\n")
    data = _gnuplot_quote(data_path)
    output = _gnuplot_quote(plot_path)
    script = f"""
set terminal svg size 1500,620 dynamic enhanced font 'Arial,12'
set output '{output}'
set datafile separator '\\t'
set key off
set grid
set multiplot layout 1,2 title 'Qwen3-8B TP4 source diversity vs C1 equal-wire advantage'
set xlabel 'Mean normalized chordal distance squared'
set ylabel 'Wire LR MSE - C1 MSE'
plot '{data}' every ::1 using 2:4 with points pt 7 ps 0.8, \\
     '' every ::1 using 2:4:1 with labels offset char 0.5,0.5 font ',8'
set xlabel 'Union 95% energy rank'
set ylabel 'Wire LR MSE - C1 MSE'
plot '{data}' every ::1 using 3:4 with points pt 7 ps 0.8, \\
     '' every ::1 using 3:4:1 with labels offset char 0.5,0.5 font ',8'
unset multiplot
"""
    completed = subprocess.run(
        ("gnuplot",),
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"gnuplot failed: {completed.stderr.strip()}")
    if not plot_path.is_file() or plot_path.stat().st_size == 0:
        raise RuntimeError("gnuplot did not produce the expected SVG")


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3-8B Wo-C1 TP-source subspaces",
        "",
        (
            "This analysis measures the four real TP-source decoder row spaces "
            "in every Qwen3-8B layer and compares their diversity with the "
            "existing equal-wire LR-AllReduce minus C1 heldout error."
        ),
        "",
        "## Protocol",
        "",
        "- Model: `Qwen3-8B-Base`",
        "- TP degree and sources: `TP4`, four sources per layer",
        "- Decoder per source: `[512, 4096]`",
        "- Layers and source pairs: `36`, six pairs per layer",
        "- Stored factors: `BF16`; primary decomposition: `FP32`",
        "- Metrics are invariant to invertible latent-basis rotations.",
        "",
        "## Per-layer results",
        "",
        "| Layer | Mean angle | Mean overlap | Mean chordal d^2 | Union r90 | Union r95 | Union r99 | C1 MSE | Wire LR MSE | C1 advantage |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["layers"]:
        aggregate = row["subspace"]["aggregate"]
        energy = row["subspace"]["union"]["energy_ranks"]
        quality = row["quality"]
        lines.append(
            f"| {row['layer']} | {aggregate['mean_angle_degrees']:.3f}° | "
            f"{aggregate['mean_projection_overlap']:.4f} | "
            f"{aggregate['mean_normalized_chordal_distance_squared']:.4f} | "
            f"{energy['90%']} | {energy['95%']} | {energy['99%']} | "
            f"{quality['c1_heldout_relative_mse']:.6g} | "
            f"{quality['wire_lr_heldout_relative_mse']:.6g} | "
            f"{quality['c1_advantage_heldout_relative_mse']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Diversity/advantage correlations",
            "",
            "| Diversity | Advantage | Pearson r | Pearson p | Spearman rho | Spearman p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for diversity, targets in payload["correlations"].items():
        for target, correlation in targets.items():
            lines.append(
                f"| `{diversity}` | `{target}` | "
                f"{correlation['pearson']['statistic']:.6f} | "
                f"{correlation['pearson']['pvalue']:.6g} | "
                f"{correlation['spearman']['statistic']:.6f} | "
                f"{correlation['spearman']['pvalue']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Numerical validation",
            "",
            (
                f"- Maximum FP32 row-basis orthonormality residual: "
                f"`{payload['validation']['maximum_fp32_orthonormality_residual']:.6g}`"
            ),
        ]
    )
    for check in payload["validation"]["fp64_precision_checks"]:
        differences = check["maximum_absolute_differences"]
        lines.append(
            f"- Layer `{check['layer']}` FP32/FP64 maximum mean-angle delta "
            f"`{differences['mean_angle_degrees']:.6g}°`, overlap delta "
            f"`{differences['projection_overlap']:.6g}`: "
            f"`{'pass' if check['passed'] else 'FAIL'}`"
        )
    lines.extend(
        [
            "",
            "The analysis is descriptive. A weak or absent correlation is a "
            "valid negative result and must not be hidden.",
            "",
            "## Command",
            "",
            "```bash",
            payload["command"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_layers(raw: str, *, num_layers: int) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return tuple(range(num_layers))
    selected = tuple(sorted({int(piece.strip()) for piece in raw.split(",") if piece.strip()}))
    if not selected or min(selected) < 0 or max(selected) >= num_layers:
        raise ValueError("selected layers are outside the checkpoint")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--precision-check-layers", default="0,17,35")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work-dtype", choices=("float32",), default="float32")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    phase1_dir = Path(args.phase1_dir).expanduser().resolve()
    phase1 = _load_phase1(phase1_dir)
    selected_layers = _parse_layers(args.layers, num_layers=36)
    precision_layers = set(
        _parse_layers(args.precision_check_layers, num_layers=36)
    )
    if not precision_layers.issubset(selected_layers):
        raise ValueError("precision-check layers must be analyzed layers")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the full real-model subspace analysis requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    layer_records = []
    precision_checks = []
    maximum_orthonormality = 0.0
    phase1_by_layer = {int(row["layer"]): row for row in phase1["layers"]}
    for ordinal, layer in enumerate(selected_layers, start=1):
        phase1_record = phase1_by_layer[layer]
        artifact_path = phase1_dir / phase1_record["artifact"]["file"]
        if _sha256(artifact_path) != phase1_record["artifact"]["sha256"]:
            raise RuntimeError(f"artifact hash mismatch for layer {layer}")
        stored = load_file(str(artifact_path), device="cpu")["c1_source_decoders"]
        if tuple(stored.shape) != (4, 512, 4096) or stored.dtype != torch.bfloat16:
            raise ValueError(f"unexpected source decoders at layer {layer}")
        decoders = stored.to(device=device, dtype=torch.float32)
        bases = tuple(decoder_row_basis(decoder) for decoder in decoders)
        orthonormality = _maximum_orthonormality_residual(bases)
        maximum_orthonormality = max(maximum_orthonormality, orthonormality)
        pairwise = pairwise_subspace_metrics(bases)
        aggregate = _aggregate_pairwise(pairwise)
        union = subspace_union_energy_ranks(bases)
        c1_mse = float(
            phase1_record["c1_allgather"][
                "factor_dtype_heldout_relative_mse"
            ]
        )
        lr_mse = float(
            phase1_record["lr_allreduce"]["wire_matched"][
                "factor_dtype_heldout_relative_mse"
            ]
        )
        c1_l2 = math.sqrt(max(c1_mse, 0.0))
        lr_l2 = math.sqrt(max(lr_mse, 0.0))
        layer_record = {
            "layer": layer,
            "artifact": {
                "path": str(artifact_path),
                "sha256": phase1_record["artifact"]["sha256"],
                "stored_dtype": "bfloat16",
                "source_decoder_shape": [4, 512, 4096],
            },
            "subspace": {
                "basis_work_dtype": "float32",
                "pairwise": list(pairwise),
                "aggregate": aggregate,
                "union": union,
                "maximum_orthonormality_residual": orthonormality,
            },
            "quality": {
                "metric_scope": "heldout attention layer-output error; not terminal logits",
                "c1_heldout_relative_mse": c1_mse,
                "wire_lr_heldout_relative_mse": lr_mse,
                "c1_advantage_heldout_relative_mse": lr_mse - c1_mse,
                "c1_heldout_relative_l2": c1_l2,
                "wire_lr_heldout_relative_l2": lr_l2,
                "c1_advantage_heldout_relative_l2": lr_l2 - c1_l2,
            },
        }
        layer_records.append(layer_record)
        if layer in precision_layers:
            check = _precision_check(stored, pairwise)
            check["layer"] = layer
            precision_checks.append(check)
        print(
            json.dumps(
                {
                    "event": "layer_complete",
                    "layer": layer,
                    "progress": f"{ordinal}/{len(selected_layers)}",
                    "mean_angle_degrees": aggregate["mean_angle_degrees"],
                    "mean_chordal_squared": aggregate[
                        "mean_normalized_chordal_distance_squared"
                    ],
                    "union_rank_95": union["energy_ranks"]["95%"],  # type: ignore[index]
                    "c1_advantage_mse": lr_mse - c1_mse,
                }
            ),
            flush=True,
        )
        del decoders, bases
        torch.cuda.empty_cache()

    correlations = _correlations(layer_records)
    plot_data_path = output_dir / "diversity_vs_advantage.tsv"
    plot_path = output_dir / "diversity_vs_advantage.svg"
    _plot(
        layer_records,
        data_path=plot_data_path,
        plot_path=plot_path,
    )
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "status": "complete",
        "command": shlex.join((sys.executable, *sys.argv)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "work_dtype": args.work_dtype,
            "torch_num_threads": args.torch_num_threads,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "model": phase1["model"],
        "source": {
            "phase1_dir": str(phase1_dir),
            "phase1_results": str(phase1_dir / "results.json"),
            "phase1_results_sha256": _sha256(phase1_dir / "results.json"),
        },
        "protocol": {
            "tp_size": 4,
            "sources_per_layer": 4,
            "source_decoder_shape": [512, 4096],
            "source_pairs_per_layer": 6,
            "layers": list(selected_layers),
            "stored_factor_dtype": "bfloat16",
            "primary_decomposition_dtype": "float32",
            "union_energy_thresholds": [0.9, 0.95, 0.99, 0.999],
            "precision_check_layers": sorted(precision_layers),
        },
        "correlations": correlations,
        "validation": {
            "maximum_fp32_orthonormality_residual": maximum_orthonormality,
            "fp64_precision_checks": precision_checks,
            "all_precision_checks_passed": all(
                check["passed"] for check in precision_checks
            ),
        },
        "plot": {
            "path": str(plot_path),
            "data_path": str(plot_data_path),
            "backend": "gnuplot svg",
        },
        "layers": layer_records,
    }
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.md"
    _atomic_text(
        results_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(summary_path, _summary_markdown(payload))
    if not payload["validation"]["all_precision_checks_passed"]:
        raise RuntimeError(f"FP32/FP64 precision check failed; inspect {results_path}")
    print(
        json.dumps(
            {
                "event": "result_written",
                "results": str(results_path),
                "results_sha256": _sha256(results_path),
                "summary": str(summary_path),
                "plot": str(plot_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
