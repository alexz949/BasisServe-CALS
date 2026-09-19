#!/usr/bin/env python3
"""Build Section-3 appendix CSVs and plots from immutable result JSONs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = REPO_ROOT / "results/section3_ablation/qwen3_8b_base"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/section3_ablation"
DENSE_RESULT = REPO_ROOT / "ICLR-results/qwen3-8b/quality/Q3-8B-Dense/result.json"
TASKS = (
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
    "boolq",
    "openbookqa",
)
ALS_CONFIGS = (
    (64, "activation_aware", "r64_activation", 0),
    (64, "random_orthogonal", "r64_random", 73),
    (96, "activation_aware", "r96_activation", 0),
    (96, "random_orthogonal", "r96_random", 73),
)
ALS_SWEEPS = (0, 1, 2, 4, 6, 8, 12, 16)
LONG_CONTEXTS = (
    (2048, "2k"),
    (8192, "8k"),
    (32768, "32k"),
    (131072, "128k"),
)
LONG_BUCKETS = ("0_8k", "8_32k", "32_64k", "64_128k")


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def _load(path: Path) -> dict[str, Any] | None:
    if not _check(path.is_file(), f"missing result: {path}"):
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _check(payload.get("status") == "complete", f"incomplete result: {path}"):
        return None
    return payload


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quality(payload: Mapping[str, Any]) -> tuple[float, float | None, dict[str, float]]:
    if "wikitext2" in payload:
        ppl = float(payload["wikitext2"]["ppl"])
        mcq = payload.get("mcq")
        if mcq is None:
            return ppl, None, {}
        scores = {
            str(row["task"]): float(row["value"])
            for row in mcq["task_accuracy"]
        }
        return ppl, float(mcq["average_accuracy"]), scores
    metrics = payload["metrics"]
    scores = {
        str(row["task"]): float(row["value"])
        for row in metrics["task_accuracy"]
    }
    return float(metrics["wikitext2_ppl"]), float(metrics["average_accuracy"]), scores


def _cg_metrics(
    layer_records: Sequence[Mapping[str, Any]],
    sweep: int,
) -> tuple[float, int, float, int, int]:
    if sweep == 0:
        return 0.0, 0, 0.0, 0, 0
    steps: list[Mapping[str, Any]] = []
    max_iterations = 0
    for record in layer_records:
        cg = record["solver"]["cg"]
        max_iterations = max(max_iterations, int(cg["max_iterations"]))
        for row in cg["per_sweep"]:
            if int(row["sweep"]) <= sweep:
                steps.extend(row["encoder_steps"])
    iterations = [int(row["iterations"]) for row in steps]
    cap_hits = sum(
        int(value >= max_iterations and not bool(row["converged"]))
        for value, row in zip(iterations, steps)
    )
    converged = sum(int(bool(row["converged"])) for row in steps)
    mean_iterations = sum(iterations) / len(iterations) if iterations else 0.0
    maximum = max(iterations, default=0)
    cap_fraction = cap_hits / len(iterations) if iterations else 0.0
    return mean_iterations, maximum, cap_fraction, converged, len(iterations)


def _group_rows(
    model_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    dense = _load(DENSE_RESULT)
    if dense is None:
        return None
    dense_ppl, dense_mcq, _ = _quality(dense)
    rows: list[dict[str, Any]] = [
        {
            "method": "Dense",
            "rank": 128,
            "ppl": dense_ppl,
            "heldout_rel_mse": 0.0,
            "mcq_average": dense_mcq,
            "checkpoint_path": str(DENSE_RESULT.parent),
            "evaluation_path": str(DENSE_RESULT),
        }
    ]
    layer_rows: list[dict[str, Any]] = []
    methods = (
        (
            "Activation-aware group SVD",
            64,
            model_root / "formal/group_svd/r64",
            model_root / "formal/evaluation/group_svd_r64/results.json",
            "group_svd",
        ),
        (
            "Activation-aware group SVD",
            96,
            model_root / "formal/group_svd/r96",
            model_root / "formal/evaluation/group_svd_r96/results.json",
            "group_svd",
        ),
        (
            "Joint C1",
            64,
            model_root / "formal/als_s16/r64_activation/sweep_016",
            model_root
            / "formal/evaluation/als_s16/r64_activation/sweep_016/results.json",
            "joint",
        ),
        (
            "Joint C1",
            96,
            model_root / "formal/als_s16/r96_activation/sweep_016",
            model_root
            / "formal/evaluation/als_s16/r96_activation/sweep_016/results.json",
            "joint",
        ),
    )
    for method, rank, bank_dir, evaluation_path, kind in methods:
        bank = _load(bank_dir / "results.json")
        quality = _load(evaluation_path)
        if bank is None or quality is None:
            return None
        ppl, mcq, _ = _quality(quality)
        rows.append(
            {
                "method": method,
                "rank": rank,
                "ppl": ppl,
                "heldout_rel_mse": bank["aggregate"][
                    "mean_heldout_factor_dtype_relative_mse"
                ],
                "mcq_average": mcq,
                "checkpoint_path": str(bank_dir),
                "evaluation_path": str(evaluation_path),
            }
        )
        for record in bank["records"]:
            if kind == "group_svd":
                fit_mse = record["fit_factor_dtype_relative_mse"]
                heldout_mse = record["heldout_factor_dtype_relative_mse"]
            else:
                fit_mse = record["fit"]["factor_dtype_relative_mse"]
                heldout_mse = record["heldout"]["factor_dtype_relative_mse"]
            layer_rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "layer": int(record["layer"]),
                    "fit_rel_mse": fit_mse,
                    "heldout_rel_mse": heldout_mse,
                }
            )
    return rows, layer_rows


def _als_rows(
    model_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for rank, initialization, name, seed in ALS_CONFIGS:
        bank_dir = model_root / "formal/als_s16" / name
        main = _load(bank_dir / "results.json")
        if main is None:
            return None
        main_records = main["records"]
        for sweep in ALS_SWEEPS:
            if sweep == 6:
                checkpoint_dir = model_root / "formal/als_s6" / name
                quality_path = (
                    model_root
                    / "formal/evaluation/als_s6"
                    / name
                    / "results.json"
                )
                sweep_main = _load(checkpoint_dir / "results.json")
                if sweep_main is None:
                    return None
                sweep_records = sweep_main["records"]
            else:
                checkpoint_dir = bank_dir / f"sweep_{sweep:03d}"
                quality_path = (
                    model_root
                    / "formal/evaluation/als_s16"
                    / name
                    / f"sweep_{sweep:03d}/results.json"
                )
                sweep_main = main
                sweep_records = main_records
            checkpoint = _load(checkpoint_dir / "results.json")
            quality = _load(quality_path)
            if checkpoint is None or quality is None:
                return None
            ppl, mcq, _ = _quality(quality)
            mean_cg, max_cg, cap_fraction, converged, solves = _cg_metrics(
                sweep_records, sweep
            )
            rows.append(
                {
                    "initialization": initialization,
                    "sweeps": sweep,
                    "rank": rank,
                    "heldout_rel_mse": checkpoint["aggregate"][
                        "mean_heldout_factor_dtype_relative_mse"
                    ],
                    "ppl": ppl,
                    "mcq_average": mcq,
                    "mean_cg_iters": mean_cg,
                    "max_cg_iters": max_cg,
                    "cg_cap_hit_fraction": cap_fraction,
                    "cg_converged_count": converged,
                    "cg_solve_count": solves,
                    "cg_mode": sweep_main["fit_config"]["encoder_cg_mode"],
                    "seed": seed,
                    "checkpoint_path": str(checkpoint_dir),
                    "evaluation_path": str(quality_path),
                }
            )
            by_layer = {int(record["layer"]): record for record in sweep_records}
            for record in checkpoint["records"]:
                layer = int(record["layer"])
                layer_mean, layer_max, layer_cap, layer_converged, layer_solves = (
                    _cg_metrics([by_layer[layer]], sweep)
                )
                layer_rows.append(
                    {
                        "initialization": initialization,
                        "sweeps": sweep,
                        "rank": rank,
                        "layer": layer,
                        "fit_rel_mse": record["fit"][
                            "factor_dtype_relative_mse"
                        ],
                        "heldout_rel_mse": record["heldout"][
                            "factor_dtype_relative_mse"
                        ],
                        "mean_cg_iters": layer_mean,
                        "max_cg_iters": layer_max,
                        "cg_cap_hit_fraction": layer_cap,
                        "cg_converged_count": layer_converged,
                        "cg_solve_count": layer_solves,
                    }
                )
    return rows, layer_rows


def _plot_als(rows: Sequence[Mapping[str, Any]], output_root: Path) -> None:
    colors = {64: "#2ca02c", 96: "#d62728"}
    styles = {"activation_aware": "-", "random_orthogonal": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    for rank, initialization, _, _ in ALS_CONFIGS:
        selected = sorted(
            (
                row
                for row in rows
                if int(row["rank"]) == rank
                and row["initialization"] == initialization
            ),
            key=lambda row: int(row["sweeps"]),
        )
        label = f"R{rank} " + (
            "activation-aware" if initialization == "activation_aware" else "random"
        )
        x = [int(row["sweeps"]) for row in selected]
        axes[0].plot(
            x,
            [float(row["heldout_rel_mse"]) for row in selected],
            marker="o",
            color=colors[rank],
            linestyle=styles[initialization],
            label=label,
        )
        axes[1].plot(
            x,
            [float(row["ppl"]) for row in selected],
            marker="o",
            color=colors[rank],
            linestyle=styles[initialization],
            label=label,
        )
    axes[0].set_xlabel("Decoder-closed ALS sweeps")
    axes[0].set_ylabel("Held-out relative MSE")
    axes[0].set_title("(a) Reconstruction")
    axes[1].set_xlabel("Decoder-closed ALS sweeps")
    axes[1].set_ylabel("WikiText-2 PPL")
    axes[1].set_title("(b) Language modeling")
    for axis in axes:
        axis.set_xticks(ALS_SWEEPS)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "als_init_sweep.pdf", bbox_inches="tight")
    fig.savefig(plot_dir / "als_init_sweep.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def summarize_phase2(model_root: Path, output_root: Path) -> int:
    group = _group_rows(model_root)
    als = _als_rows(model_root)
    if group is None or als is None:
        return 2
    group_rows, group_layer_rows = group
    als_rows, als_layer_rows = als
    _write_csv(
        output_root / "group_svd_vs_joint.csv",
        group_rows,
        (
            "method",
            "rank",
            "ppl",
            "heldout_rel_mse",
            "mcq_average",
            "checkpoint_path",
            "evaluation_path",
        ),
    )
    _write_csv(
        output_root / "group_svd_vs_joint_layers.csv",
        group_layer_rows,
        ("method", "rank", "layer", "fit_rel_mse", "heldout_rel_mse"),
    )
    _write_csv(
        output_root / "als_init_sweep.csv",
        als_rows,
        (
            "initialization",
            "sweeps",
            "rank",
            "heldout_rel_mse",
            "ppl",
            "mcq_average",
            "mean_cg_iters",
            "max_cg_iters",
            "cg_cap_hit_fraction",
            "cg_converged_count",
            "cg_solve_count",
            "cg_mode",
            "seed",
            "checkpoint_path",
            "evaluation_path",
        ),
    )
    _write_csv(
        output_root / "als_init_sweep_layers.csv",
        als_layer_rows,
        (
            "initialization",
            "sweeps",
            "rank",
            "layer",
            "fit_rel_mse",
            "heldout_rel_mse",
            "mean_cg_iters",
            "max_cg_iters",
            "cg_cap_hit_fraction",
            "cg_converged_count",
            "cg_solve_count",
        ),
    )
    _plot_als(als_rows, output_root)
    print(
        f"[Complete] Phase 2 CSVs and ALS plots under {output_root}",
        flush=True,
    )
    return 0


def summarize_wt2(model_root: Path, output_root: Path) -> int:
    dense = _load(DENSE_RESULT)
    if dense is None:
        return 2
    calibration_manifest_path = model_root / "data/wikitext2/manifest.json"
    if not _check(
        calibration_manifest_path.is_file(),
        f"missing calibration manifest: {calibration_manifest_path}",
    ):
        return 2
    calibration_manifest = json.loads(
        calibration_manifest_path.read_text(encoding="utf-8")
    )
    fit_partition = next(
        partition
        for partition in calibration_manifest["artifact"]["ordered_partitions"]
        if partition["name"] == "fit"
    )
    fit_windows = int(fit_partition["stop"]) - int(fit_partition["start"])
    sequence_length = int(calibration_manifest["artifact"]["shape"][1])
    calibration_tokens = fit_windows * sequence_length

    wt2_root = model_root / "formal/wt2"
    specs = (
        ("Joint C1", 64, 1, wt2_root / "c1/r64", "c1_r64"),
        ("Joint C1", 96, 1, wt2_root / "c1/r96", "c1_r96"),
        ("PaLU M-LRD", 64, 1, wt2_root / "palu/m/r64", "palu_m_r64"),
        ("PaLU M-LRD", 96, 1, wt2_root / "palu/m/r96", "palu_m_r96"),
        ("PaLU G4-LRD", 64, 4, wt2_root / "palu/g4/r64", "palu_g4_r64"),
        ("PaLU G4-LRD", 96, 4, wt2_root / "palu/g4/r96", "palu_g4_r96"),
    )
    rows: list[dict[str, Any]] = []
    source_results: list[dict[str, Any]] = []

    dense_ppl, dense_mcq, dense_scores = _quality(dense)
    if not _check(
        set(dense_scores) == set(TASKS),
        "Dense reference does not contain the complete seven-task MCQ suite",
    ):
        return 2
    rows.append(
        {
            "model": "Qwen3-8B-Base",
            "method": "Dense",
            "rank": 128,
            "rank_definition": "dense head dimension",
            "group_size": 1,
            "group_rank": 128,
            "calibration_dataset": "none (pretrained reference)",
            "calibration_tokens": 0,
            "ppl": dense_ppl,
            "mcq_average": dense_mcq,
            **dense_scores,
            "seed": "",
            "checkpoint_path": str(DENSE_RESULT.parent),
            "evaluation_path": str(DENSE_RESULT),
        }
    )
    source_results.append(
        {
            "method": "Dense",
            "path": str(DENSE_RESULT),
            "sha256": _sha256(DENSE_RESULT),
            "slurm_job_id": dense.get("environment", {}).get("slurm_job_id"),
            "command": dense.get("command"),
            "reused_existing_result": True,
        }
    )

    for method, rank, group_size, checkpoint_path, evaluation_name in specs:
        evaluation_path = wt2_root / f"evaluation/{evaluation_name}/results.json"
        quality = _load(evaluation_path)
        if quality is None:
            return 2
        ppl, mcq, scores = _quality(quality)
        if not _check(
            set(scores) == set(TASKS),
            f"{method} R{rank} does not contain the complete seven-task MCQ suite",
        ):
            return 2
        rows.append(
            {
                "model": "Qwen3-8B-Base",
                "method": method,
                "rank": rank,
                "rank_definition": "equivalent rank per physical KV head",
                "group_size": group_size,
                "group_rank": group_size * rank,
                "calibration_dataset": "WikiText-2 train",
                "calibration_tokens": calibration_tokens,
                "ppl": ppl,
                "mcq_average": mcq,
                **scores,
                "seed": 20260918,
                "checkpoint_path": str(checkpoint_path),
                "evaluation_path": str(evaluation_path),
            }
        )
        source_results.append(
            {
                "method": method,
                "rank": rank,
                "group_size": group_size,
                "group_rank": group_size * rank,
                "path": str(evaluation_path),
                "sha256": _sha256(evaluation_path),
                "slurm_job_id": quality.get("environment", {}).get("slurm_job_id"),
                "command": quality.get("command"),
                "reused_existing_result": False,
            }
        )

    csv_path = output_root / "wt2_calibration.csv"
    _write_csv(
        csv_path,
        rows,
        (
            "model",
            "method",
            "rank",
            "rank_definition",
            "group_size",
            "group_rank",
            "calibration_dataset",
            "calibration_tokens",
            "ppl",
            "mcq_average",
            *TASKS,
            "seed",
            "checkpoint_path",
            "evaluation_path",
        ),
    )
    manifest_path = output_root / "wt2_calibration_manifest.json"
    _write_json(
        manifest_path,
        {
            "format": "basisserve.qwen3_8b.section3.wt2_calibration_summary.v1",
            "status": "complete",
            "model": calibration_manifest["model"],
            "calibration": {
                "dataset": calibration_manifest["dataset"],
                "fit_windows": fit_windows,
                "heldout_windows": int(calibration_manifest["artifact"]["shape"][0])
                - fit_windows,
                "sequence_length": sequence_length,
                "fit_tokens": calibration_tokens,
                "seed": 20260918,
                "source_manifest": str(calibration_manifest_path),
                "source_manifest_sha256": _sha256(calibration_manifest_path),
                "window_artifact_sha256": calibration_manifest["artifact"]["sha256"],
            },
            "rank_definition": {
                "rank": "equivalent retained V rank per physical KV head",
                "palu_g4_group_rank": "4 * rank",
                "palu_g4_group_ranks": {"64": 256, "96": 384},
            },
            "evaluation": {
                "ppl_batch_size": 2,
                "lm_eval_batch_size": 8,
                "tasks": list(TASKS),
                "dense_reference_reused": str(DENSE_RESULT),
            },
            "source_results": source_results,
            "artifacts": {
                "csv": str(csv_path),
                "csv_sha256": _sha256(csv_path),
            },
        },
    )
    print(f"[Complete] WT2 calibration CSV and manifest under {output_root}", flush=True)
    return 0


def _plot_long_context(
    rows: Sequence[Mapping[str, Any]], output_root: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    colors = {
        2048: "#1f77b4",
        8192: "#ff7f0e",
        32768: "#2ca02c",
        131072: "#d62728",
    }
    bucket_labels = ("0–8K", "8–32K", "32–64K", "64–128K")
    x = list(range(len(LONG_BUCKETS)))
    for axis, rank in zip(axes, (64, 96)):
        selected = sorted(
            (
                row
                for row in rows
                if row["method"] == "Joint C1" and int(row["rank"]) == rank
            ),
            key=lambda row: int(row["calibration_context"]),
        )
        for row in selected:
            context = int(row["calibration_context"])
            y = [float(row[f"delta_nll_{bucket}"]) for bucket in LONG_BUCKETS]
            yerr = [
                float(row[f"delta_nll_se_{bucket}"]) for bucket in LONG_BUCKETS
            ]
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                color=colors[context],
                marker="o",
                linewidth=1.5,
                capsize=2.5,
                label=f"{context // 1024}K calibration",
            )
        axis.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        axis.set_title(f"({'a' if rank == 64 else 'b'}) Joint C1 R{rank}")
        axis.set_xticks(x)
        axis.set_xticklabels(bucket_labels)
        axis.set_xlabel("Evaluation-token position")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"$\Delta$ NLL vs. Dense")
    axes[0].legend(frameon=False, fontsize=8)
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "long_context_calibration.pdf", bbox_inches="tight")
    fig.savefig(
        plot_dir / "long_context_calibration.png", bbox_inches="tight", dpi=300
    )
    plt.close(fig)


def summarize_long_context(model_root: Path, output_root: Path) -> int:
    long_root = model_root / "formal/long_context"
    data_manifest_path = model_root / "data/long_context/manifest.json"
    source_manifest_path = (
        REPO_ROOT / "results/calibration/qwen3_8b_c4_512f128h_s4096/manifest.json"
    )
    if not _check(data_manifest_path.is_file(), f"missing {data_manifest_path}"):
        return 2
    if not _check(source_manifest_path.is_file(), f"missing {source_manifest_path}"):
        return 2
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    sampling_seed = int(source_manifest["sampling"]["seed"])
    sequence_counts = {
        int(condition["context_length"]): int(condition["fit_windows"])
        for condition in data_manifest["conditions"]
    }
    dense_path = long_root / "evaluation/dense/results.json"
    dense = _load(dense_path)
    if dense is None:
        return 2
    dense_metrics = dense["metrics"]
    dense_documents = {
        int(document["document"]): document
        for document in dense_metrics["documents"]
    }
    rows: list[dict[str, Any]] = [
        {
            "model": "Qwen3-8B-Base",
            "method": "Dense",
            "rank": 128,
            "calibration_context": "",
            "num_calibration_sequences": 0,
            "calibration_tokens": 0,
            "evaluation_context_length": 131072,
            "evaluation_documents": len(dense_metrics["documents"]),
            "overall_ppl": float(dense_metrics["ppl"]),
            "overall_nll": float(dense_metrics["mean_nll"]),
            **{
                f"ppl_{bucket}": float(
                    dense_metrics["position_buckets"][bucket]["ppl"]
                )
                for bucket in LONG_BUCKETS
            },
            **{
                f"nll_{bucket}": float(
                    dense_metrics["position_buckets"][bucket]["mean_nll"]
                )
                for bucket in LONG_BUCKETS
            },
            **{f"delta_nll_{bucket}": 0.0 for bucket in LONG_BUCKETS},
            **{f"delta_nll_se_{bucket}": 0.0 for bucket in LONG_BUCKETS},
            "seed": sampling_seed,
            "yarn_factor": 4.0,
            "checkpoint_path": str(dense["model"]["path"]),
            "evaluation_path": str(dense_path),
        }
    ]
    document_rows: list[dict[str, Any]] = []
    for document_id, document in sorted(dense_documents.items()):
        document_rows.append(
            {
                "model": "Qwen3-8B-Base",
                "method": "Dense",
                "rank": 128,
                "calibration_context": "",
                "document": document_id,
                "overall_nll": float(document["mean_nll"]),
                "delta_nll": 0.0,
                **{
                    f"nll_{bucket}": float(
                        document["position_buckets"][bucket]["mean_nll"]
                    )
                    for bucket in LONG_BUCKETS
                },
                **{f"delta_nll_{bucket}": 0.0 for bucket in LONG_BUCKETS},
            }
        )
    source_results: list[dict[str, Any]] = [
        {
            "method": "Dense",
            "path": str(dense_path),
            "sha256": _sha256(dense_path),
            "slurm_job_id": dense.get("environment", {}).get("slurm_job_id"),
            "command": dense.get("command"),
        }
    ]
    for context_length, label in LONG_CONTEXTS:
        for rank in (64, 96):
            checkpoint_path = long_root / f"calibration/{label}/c1/r{rank}"
            checkpoint = _load(checkpoint_path / "results.json")
            evaluation_path = (
                long_root
                / f"evaluation/c1_cal_{label}_r{rank}/results.json"
            )
            quality = _load(evaluation_path)
            if checkpoint is None or quality is None:
                return 2
            metrics = quality["metrics"]
            quality_documents = {
                int(document["document"]): document
                for document in metrics["documents"]
            }
            if not _check(
                set(quality_documents) == set(dense_documents),
                f"document IDs differ from Dense for {label} R{rank}",
            ):
                return 2
            delta_by_bucket = {
                bucket: float(metrics["position_buckets"][bucket]["mean_nll"])
                - float(dense_metrics["position_buckets"][bucket]["mean_nll"])
                for bucket in LONG_BUCKETS
            }
            delta_se_by_bucket: dict[str, float] = {}
            for bucket in LONG_BUCKETS:
                paired_deltas = [
                    float(
                        quality_documents[document_id]["position_buckets"][bucket][
                            "mean_nll"
                        ]
                    )
                    - float(
                        dense_documents[document_id]["position_buckets"][bucket][
                            "mean_nll"
                        ]
                    )
                    for document_id in sorted(dense_documents)
                ]
                mean_delta = sum(paired_deltas) / len(paired_deltas)
                variance = sum(
                    (value - mean_delta) ** 2 for value in paired_deltas
                ) / (len(paired_deltas) - 1)
                delta_se_by_bucket[bucket] = math.sqrt(
                    variance / len(paired_deltas)
                )
            rows.append(
                {
                    "model": "Qwen3-8B-Base",
                    "method": "Joint C1",
                    "rank": rank,
                    "calibration_context": context_length,
                    "num_calibration_sequences": sequence_counts[context_length],
                    "calibration_tokens": 1048576,
                    "evaluation_context_length": 131072,
                    "evaluation_documents": len(metrics["documents"]),
                    "overall_ppl": float(metrics["ppl"]),
                    "overall_nll": float(metrics["mean_nll"]),
                    **{
                        f"ppl_{bucket}": float(
                            metrics["position_buckets"][bucket]["ppl"]
                        )
                        for bucket in LONG_BUCKETS
                    },
                    **{
                        f"nll_{bucket}": float(
                            metrics["position_buckets"][bucket]["mean_nll"]
                        )
                        for bucket in LONG_BUCKETS
                    },
                    **{
                        f"delta_nll_{bucket}": delta_by_bucket[bucket]
                        for bucket in LONG_BUCKETS
                    },
                    **{
                        f"delta_nll_se_{bucket}": delta_se_by_bucket[bucket]
                        for bucket in LONG_BUCKETS
                    },
                    "seed": sampling_seed,
                    "yarn_factor": 4.0,
                    "checkpoint_path": str(checkpoint_path),
                    "evaluation_path": str(evaluation_path),
                }
            )
            for document_id in sorted(dense_documents):
                document = quality_documents[document_id]
                dense_document = dense_documents[document_id]
                document_rows.append(
                    {
                        "model": "Qwen3-8B-Base",
                        "method": "Joint C1",
                        "rank": rank,
                        "calibration_context": context_length,
                        "document": document_id,
                        "overall_nll": float(document["mean_nll"]),
                        "delta_nll": float(document["mean_nll"])
                        - float(dense_document["mean_nll"]),
                        **{
                            f"nll_{bucket}": float(
                                document["position_buckets"][bucket]["mean_nll"]
                            )
                            for bucket in LONG_BUCKETS
                        },
                        **{
                            f"delta_nll_{bucket}": float(
                                document["position_buckets"][bucket]["mean_nll"]
                            )
                            - float(
                                dense_document["position_buckets"][bucket][
                                    "mean_nll"
                                ]
                            )
                            for bucket in LONG_BUCKETS
                        },
                    }
                )
            source_results.append(
                {
                    "method": "Joint C1",
                    "rank": rank,
                    "calibration_context_length": context_length,
                    "checkpoint_sha256": _sha256(checkpoint_path / "results.json"),
                    "path": str(evaluation_path),
                    "sha256": _sha256(evaluation_path),
                    "slurm_job_id": quality.get("environment", {}).get(
                        "slurm_job_id"
                    ),
                    "command": quality.get("command"),
                }
            )
    csv_path = output_root / "long_context_calibration.csv"
    _write_csv(
        csv_path,
        rows,
        (
            "model",
            "method",
            "rank",
            "calibration_context",
            "num_calibration_sequences",
            "calibration_tokens",
            "evaluation_context_length",
            "evaluation_documents",
            "overall_ppl",
            "overall_nll",
            *(f"ppl_{bucket}" for bucket in LONG_BUCKETS),
            *(f"nll_{bucket}" for bucket in LONG_BUCKETS),
            *(f"delta_nll_{bucket}" for bucket in LONG_BUCKETS),
            *(f"delta_nll_se_{bucket}" for bucket in LONG_BUCKETS),
            "seed",
            "yarn_factor",
            "checkpoint_path",
            "evaluation_path",
        ),
    )
    document_csv_path = output_root / "long_context_calibration_documents.csv"
    _write_csv(
        document_csv_path,
        document_rows,
        (
            "model",
            "method",
            "rank",
            "calibration_context",
            "document",
            "overall_nll",
            "delta_nll",
            *(f"nll_{bucket}" for bucket in LONG_BUCKETS),
            *(f"delta_nll_{bucket}" for bucket in LONG_BUCKETS),
        ),
    )
    _plot_long_context(rows, output_root)
    manifest_path = output_root / "long_context_manifest.json"
    _write_json(
        manifest_path,
        {
            "format": "basisserve.qwen3_8b.section3.long_context_summary.v1",
            "status": "complete",
            "protocol": {
                "calibration_context_lengths": [
                    context for context, _ in LONG_CONTEXTS
                ],
                "ranks_per_physical_kv_head": [64, 96],
                "calibration_tokens_per_condition": 1048576,
                "heldout_tokens_per_condition": 262144,
                "evaluation_context_length": 131072,
                "evaluation_documents": 8,
                "evaluation_chunk_size": 128,
                "full_dense_attention": True,
                "dense_key_cache": True,
                "yarn_factor": 4.0,
                "position_buckets": list(LONG_BUCKETS),
                "sampling_seed": sampling_seed,
                "paired_document_standard_errors": True,
            },
            "source_results": source_results,
            "artifacts": {
                "csv": str(csv_path),
                "csv_sha256": _sha256(csv_path),
                "document_csv": str(document_csv_path),
                "document_csv_sha256": _sha256(document_csv_path),
                "plot_pdf": str(
                    output_root / "plots/long_context_calibration.pdf"
                ),
                "plot_png": str(
                    output_root / "plots/long_context_calibration.png"
                ),
            },
        },
    )
    print(f"[Complete] Long-context CSV, manifest, and plots under {output_root}", flush=True)
    return 0


def _plot_long_calibration_short_wt2(
    rows: Sequence[Mapping[str, Any]], output_root: Path
) -> None:
    labels = ("2K", "8K", "32K", "128K")
    x = list(range(len(labels)))
    dense_ppl = float(
        next(row["ppl"] for row in rows if row["method"] == "Dense")
    )
    colors = {64: "#2ca02c", 96: "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), constrained_layout=True)
    for axis, rank in zip(axes, (64, 96)):
        selected = sorted(
            (
                row
                for row in rows
                if row["method"] == "Joint C1" and int(row["rank"]) == rank
            ),
            key=lambda row: int(row["calibration_context"]),
        )
        axis.plot(
            x,
            [float(row["ppl"]) for row in selected],
            color=colors[rank],
            marker="o",
            linewidth=1.8,
            label=f"Joint C1 R{rank}",
        )
        axis.axhline(
            dense_ppl,
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            label=f"Dense ({dense_ppl:.4f})",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_xlabel("Calibration sequence length")
        axis.set_title(f"({'a' if rank == 64 else 'b'}) R{rank}")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("WikiText-2 PPL")
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        plot_dir / "long_calibration_short_wt2.pdf", bbox_inches="tight"
    )
    fig.savefig(
        plot_dir / "long_calibration_short_wt2.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def summarize_long_calibration_short_wt2(
    model_root: Path, output_root: Path
) -> int:
    dense = _load(DENSE_RESULT)
    if dense is None:
        return 2
    dense_ppl, _, _ = _quality(dense)
    long_root = model_root / "formal/long_context"
    sequence_counts = {2048: 512, 8192: 128, 32768: 32, 131072: 8}
    rows: list[dict[str, Any]] = [
        {
            "model": "Qwen3-8B-Base",
            "method": "Dense",
            "rank": 128,
            "calibration_context": "",
            "num_calibration_sequences": 0,
            "calibration_tokens": 0,
            "evaluation_dataset": "WikiText-2",
            "evaluation_split": "test",
            "evaluation_sequence_length": 2048,
            "evaluation_windows": 146,
            "evaluation_tokens": 298862,
            "rope_scaling": "none",
            "ppl": dense_ppl,
            "delta_ppl_vs_dense": 0.0,
            "delta_nll_vs_dense": 0.0,
            "checkpoint_path": str(DENSE_RESULT.parent),
            "evaluation_path": str(DENSE_RESULT),
        }
    ]
    source_results: list[dict[str, Any]] = [
        {
            "method": "Dense",
            "path": str(DENSE_RESULT),
            "sha256": _sha256(DENSE_RESULT),
            "reused_existing_result": True,
        }
    ]
    for context_length, label in LONG_CONTEXTS:
        for rank in (64, 96):
            checkpoint_path = long_root / f"calibration/{label}/c1/r{rank}"
            evaluation_path = (
                long_root / f"short_wt2/c1_cal_{label}_r{rank}/results.json"
            )
            quality = _load(evaluation_path)
            if quality is None:
                return 2
            metric = quality["wikitext2"]
            protocol_ok = all(
                (
                    metric["dataset"] == "wikitext2",
                    metric["split"] == "test",
                    int(metric["seqlen"]) == 2048,
                    int(metric["batch_size"]) == 2,
                    int(metric["chunks"]) == 146,
                    int(metric["tokens"]) == 298862,
                    quality["environment"]["device"] == "NVIDIA L40S",
                )
            )
            if not _check(protocol_ok, f"protocol mismatch for {label} R{rank}"):
                return 2
            ppl = float(metric["ppl"])
            rows.append(
                {
                    "model": "Qwen3-8B-Base",
                    "method": "Joint C1",
                    "rank": rank,
                    "calibration_context": context_length,
                    "num_calibration_sequences": sequence_counts[context_length],
                    "calibration_tokens": 1048576,
                    "evaluation_dataset": "WikiText-2",
                    "evaluation_split": "test",
                    "evaluation_sequence_length": 2048,
                    "evaluation_windows": int(metric["chunks"]),
                    "evaluation_tokens": int(metric["tokens"]),
                    "rope_scaling": "none",
                    "ppl": ppl,
                    "delta_ppl_vs_dense": ppl - dense_ppl,
                    "delta_nll_vs_dense": math.log(ppl) - math.log(dense_ppl),
                    "checkpoint_path": str(checkpoint_path),
                    "evaluation_path": str(evaluation_path),
                }
            )
            source_results.append(
                {
                    "method": "Joint C1",
                    "rank": rank,
                    "calibration_context_length": context_length,
                    "checkpoint_sha256": _sha256(checkpoint_path / "results.json"),
                    "path": str(evaluation_path),
                    "sha256": _sha256(evaluation_path),
                    "slurm_job_id": quality["environment"].get("slurm_job_id"),
                    "command": quality.get("command"),
                    "reused_existing_result": False,
                }
            )
    csv_path = output_root / "long_calibration_short_wt2.csv"
    _write_csv(
        csv_path,
        rows,
        (
            "model",
            "method",
            "rank",
            "calibration_context",
            "num_calibration_sequences",
            "calibration_tokens",
            "evaluation_dataset",
            "evaluation_split",
            "evaluation_sequence_length",
            "evaluation_windows",
            "evaluation_tokens",
            "rope_scaling",
            "ppl",
            "delta_ppl_vs_dense",
            "delta_nll_vs_dense",
            "checkpoint_path",
            "evaluation_path",
        ),
    )
    _plot_long_calibration_short_wt2(rows, output_root)
    manifest_path = output_root / "long_calibration_short_wt2_manifest.json"
    _write_json(
        manifest_path,
        {
            "format": "basisserve.qwen3_8b.section3.long_calibration_short_wt2.v1",
            "status": "complete",
            "protocol": {
                "calibration_context_lengths": [
                    context for context, _ in LONG_CONTEXTS
                ],
                "ranks_per_physical_kv_head": [64, 96],
                "calibration_tokens_per_condition": 1048576,
                "evaluation_dataset": "WikiText-2",
                "evaluation_split": "test",
                "evaluation_sequence_length": 2048,
                "evaluation_windows": 146,
                "evaluation_tokens": 298862,
                "ppl_batch_size": 2,
                "rope_scaling": None,
                "dense_reference_ppl": dense_ppl,
                "slurm_job_id": "8340320",
                "device": "NVIDIA L40S",
            },
            "source_results": source_results,
            "artifacts": {
                "csv": str(csv_path),
                "csv_sha256": _sha256(csv_path),
                "plot_pdf": str(
                    output_root / "plots/long_calibration_short_wt2.pdf"
                ),
                "plot_png": str(
                    output_root / "plots/long_calibration_short_wt2.png"
                ),
            },
        },
    )
    print(
        f"[Complete] Short-context robustness CSV, manifest, and plots under {output_root}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("phase2", "wt2", "long", "long-short-wt2"),
        default="phase2",
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = args.model_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.stage == "phase2":
        status = summarize_phase2(model_root, output_root)
    elif args.stage == "wt2":
        status = summarize_wt2(model_root, output_root)
    elif args.stage == "long":
        status = summarize_long_context(model_root, output_root)
    else:
        status = summarize_long_calibration_short_wt2(model_root, output_root)
    if status:
        sys.exit(status)


if __name__ == "__main__":
    main()
