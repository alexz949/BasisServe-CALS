#!/usr/bin/env python3
"""Check whether FP32 roundoff explains the layer-23 QR(0) failure."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.pairwise_qr import (  # noqa: E402
    solve_square_root_least_squares,
)
from evaluation.compare_qwen3_32b_c1_pairwise_qr import (  # noqa: E402
    EXPERIMENT_FORMAT as FP32_EXPERIMENT_FORMAT,
    HIDDEN_SIZE,
    NUM_QUERY_HEADS,
    SNAPSHOT_FORMAT,
    _activation_pairwise_r,
    _atomic_write_json,
    _atomic_write_text,
    _dense_block_basis,
    _evaluate_decoder,
    _load_and_validate_layer,
    _output_energy,
    _read_json,
    _sha256,
)


FORMAT = "basisserve.qwen3_32b.c1_layer23_qr0_fp64_check.v1"
LAYER = 23
PRIMARY_RELATIVE_TOLERANCE = 1.0e-2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("results/calibration/qwen3_32b_c1_128f64h_p128"),
    )
    parser.add_argument(
        "--factor-dir",
        type=Path,
        default=Path("results/checkpoints/qwen3_32b_c1_v96_als_d1e5"),
    )
    parser.add_argument(
        "--fp32-reference-json",
        type=Path,
        default=Path(
            "results/evaluation/qwen3_32b_c1_fixed_a_pairwise_qr_l23_l0.json"
        ),
    )
    parser.add_argument("--qr-block-rows", type=int, default=8192)
    parser.add_argument("--evaluation-row-chunk-size", type=int, default=256)
    parser.add_argument("--output-column-chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "results/evaluation/qwen3_32b_c1_layer23_qr0_fp64_check.json"
        ),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path(
            "results/evaluation/qwen3_32b_c1_layer23_qr0_fp64_check.md"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _relative_change(value: float, reference: float) -> float:
    return value / max(abs(reference), torch.finfo(torch.float64).tiny) - 1.0


def _reference_layer(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("format") != FP32_EXPERIMENT_FORMAT:
        raise ValueError("incompatible FP32 reference format")
    if payload.get("status") != "complete":
        raise ValueError("FP32 reference experiment is incomplete")
    if payload.get("environment", {}).get("work_dtype") != "float32":
        raise ValueError("reference experiment did not use float32")
    matches = [item for item in payload["layers"] if int(item["layer"]) == LAYER]
    if len(matches) != 1:
        raise ValueError("FP32 reference must contain exactly one layer-23 result")
    return matches[0]


def _validate_sources(
    current: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> None:
    for key in (
        "snapshot_sha256",
        "factor_sha256",
        "fit_rows",
        "heldout_row_start",
        "heldout_rows",
        "rank_per_head",
    ):
        if current[key] != reference[key]:
            raise ValueError(f"FP32/FP64 source mismatch for {key}")


def _render_markdown(payload: Mapping[str, Any]) -> str:
    fp32 = payload["fp32_reference"]["metrics"]
    fp64 = payload["fp64"]["metrics"]
    changes = payload["fp64_vs_fp32_relative_change"]
    baseline = payload["fp64_vs_cholesky_baseline"]
    decision = payload["decision"]
    solve32 = payload["fp32_reference"]["solve"]
    solve64 = payload["fp64"]["solve"]
    lines = [
        "# Qwen3-32B layer-23 QR(0) FP64 precision check",
        "",
        "This check reruns only the unregularized layer-23 square-root solve in FP64. "
        "It uses the same raw BF16 activation snapshot, fixed BF16 encoder `A`, fit rows, "
        "and held-out rows as the saved FP32 experiment.",
        "",
        "## Command",
        "",
        "```bash",
        str(payload["command"]),
        "```",
        "",
        "## FP32 versus FP64",
        "",
        "| Metric | FP32 | FP64 | FP64 relative change |",
        "|---|---:|---:|---:|",
        f"| Fit relative MSE | {fp32['fit_relative_mse']:.12e} | "
        f"{fp64['fit_relative_mse']:.12e} | {changes['fit_relative_mse']:.6%} |",
        f"| Held-out relative MSE | {fp32['heldout_relative_mse']:.12e} | "
        f"{fp64['heldout_relative_mse']:.12e} | {changes['heldout_relative_mse']:.6%} |",
        f"| Decoder Frobenius norm | {fp32['decoder_frobenius_norm']:.12e} | "
        f"{fp64['decoder_frobenius_norm']:.12e} | "
        f"{changes['decoder_frobenius_norm']:.6%} |",
        f"| Decoder maximum absolute value | "
        f"{fp32['decoder_maximum_absolute_value']:.12e} | "
        f"{fp64['decoder_maximum_absolute_value']:.12e} | "
        f"{changes['decoder_maximum_absolute_value']:.6%} |",
        f"| Minimum abs R diagonal | {solve32['minimum_abs_r_diagonal']:.12e} | "
        f"{solve64['minimum_abs_r_diagonal']:.12e} | "
        f"{changes['minimum_abs_r_diagonal']:.6%} |",
        f"| R diagonal ratio | {solve32['r_diagonal_ratio']:.12e} | "
        f"{solve64['r_diagonal_ratio']:.12e} | "
        f"{changes['r_diagonal_ratio']:.6%} |",
        "",
        "## Decision",
        "",
        f"- Maximum FP32/FP64 relative difference over fit MSE, held-out MSE, and "
        f"decoder norm: {decision['maximum_primary_relative_difference']:.6%}.",
        f"- FP64 held-out MSE remains {baseline['heldout_mse_ratio']:.3f}x the "
        "damped Cholesky baseline.",
        f"- FP64 decoder norm remains {baseline['decoder_norm_ratio']:.3f}x the "
        "damped Cholesky baseline.",
        f"- FP32 numerical error excluded as the primary explanation: "
        f"**{decision['fp32_roundoff_excluded_as_primary_explanation']}**.",
        "",
        str(decision["interpretation"]),
        "",
        "The original FP32 decoder tensor was not persisted, so this check compares "
        "output metrics and decoder norms rather than an elementwise decoder distance. "
        "That is sufficient for testing whether FP32 roundoff caused the catastrophic "
        "held-out behavior.",
        "",
    ]
    return "\n".join(lines)


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    for name in (
        "qr_block_rows",
        "evaluation_row_chunk_size",
        "output_column_chunk_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', ' ')} must be positive")
    for output in (args.output_json, args.output_markdown):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists: {output}; pass --overwrite")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the FP64 precision check requires a CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    fp32_reference_sha256 = _sha256(args.fp32_reference_json)
    fp32_payload = _read_json(args.fp32_reference_json)
    fp32_layer = _reference_layer(fp32_payload)
    relative_damping = float(fp32_payload["configuration"]["relative_damping"])

    snapshot_manifest_path = args.snapshot_dir / "manifest.json"
    snapshot_manifest_sha256 = _sha256(snapshot_manifest_path)
    snapshot_manifest = _read_json(snapshot_manifest_path)
    if snapshot_manifest.get("format") != SNAPSHOT_FORMAT:
        raise ValueError("FP64 precision check requires the raw-activation snapshot")
    if (
        snapshot_manifest_sha256
        != fp32_payload["configuration"]["snapshot_manifest_sha256"]
    ):
        raise ValueError("FP32 reference used a different snapshot manifest")

    print("[layer 23] validating hashes and loading fixed inputs", flush=True)
    activation, weight, fixed_a_cpu, _, source = _load_and_validate_layer(
        layer=LAYER,
        snapshot_dir=args.snapshot_dir,
        snapshot_manifest=snapshot_manifest,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
        factor_dir=args.factor_dir,
        relative_damping=relative_damping,
    )
    _validate_sources(source, fp32_layer["source"])
    fixed_a = fixed_a_cpu.to(device=device, dtype=torch.float64)
    fit_rows = int(source["fit_rows"])
    heldout_start = int(source["heldout_row_start"])
    heldout_rows = int(source["heldout_rows"])
    rank = int(source["rank_per_head"])

    torch.cuda.reset_peak_memory_stats(device)
    experiment_started = time.monotonic()
    print("[layer 23] pairwise activation TSQR in FP64", flush=True)
    r_x, pairwise_diagnostics = _activation_pairwise_r(
        activation,
        fit_rows=fit_rows,
        block_rows=args.qr_block_rows,
        device=device,
        dtype=torch.float64,
    )
    basis = _dense_block_basis(
        fixed_a,
        device=device,
        dtype=torch.float64,
    )
    target_weight = (
        weight.to(device=device, dtype=torch.float64).transpose(0, 1).contiguous()
    )

    print("[layer 23] square-root QR, lambda=0, FP64", flush=True)
    decoder_flat, solve_diagnostics = solve_square_root_least_squares(
        left_factor=r_x,
        basis=basis,
        target_weight=target_weight,
        absolute_product_damping=0.0,
        output_chunk_size=args.output_column_chunk_size,
    )
    decoder = decoder_flat.reshape(NUM_QUERY_HEADS, rank, HIDDEN_SIZE)

    print("[layer 23] evaluating fit and held-out rows in FP64", flush=True)
    evaluation_started = time.monotonic()
    fit_target_energy = _output_energy(
        activation,
        target_weight,
        row_start=0,
        row_count=fit_rows,
        row_chunk_size=args.evaluation_row_chunk_size,
        device=device,
        dtype=torch.float64,
    )
    heldout_target_energy = _output_energy(
        activation,
        target_weight,
        row_start=heldout_start,
        row_count=heldout_rows,
        row_chunk_size=args.evaluation_row_chunk_size,
        device=device,
        dtype=torch.float64,
    )
    fp64_metrics = _evaluate_decoder(
        activation,
        target_weight,
        fixed_a,
        decoder,
        fit_rows=fit_rows,
        heldout_start=heldout_start,
        heldout_rows=heldout_rows,
        fit_target_energy=fit_target_energy,
        heldout_target_energy=heldout_target_energy,
        row_chunk_size=args.evaluation_row_chunk_size,
        device=device,
        dtype=torch.float64,
    )
    torch.cuda.synchronize(device)
    evaluation_seconds = time.monotonic() - evaluation_started
    experiment_seconds = time.monotonic() - experiment_started
    peak_memory_gib = torch.cuda.max_memory_allocated(device) / 1024**3

    fp32_method = fp32_layer["methods"]["pairwise_qr_unregularized"]
    fp32_metrics = fp32_method["metrics"]
    fp32_solve = fp32_method["solve"]
    fp64_solve = asdict(solve_diagnostics)
    comparison_keys = (
        "fit_relative_mse",
        "heldout_relative_mse",
        "decoder_frobenius_norm",
        "decoder_maximum_absolute_value",
    )
    changes = {
        key: _relative_change(float(fp64_metrics[key]), float(fp32_metrics[key]))
        for key in comparison_keys
    }
    for key in ("minimum_abs_r_diagonal", "r_diagonal_ratio"):
        changes[key] = _relative_change(
            float(fp64_solve[key]),
            float(fp32_solve[key]),
        )
    primary_keys = (
        "fit_relative_mse",
        "heldout_relative_mse",
        "decoder_frobenius_norm",
    )
    maximum_primary_difference = max(abs(changes[key]) for key in primary_keys)
    baseline_metrics = fp32_layer["methods"]["cholesky_damped"]["metrics"]
    heldout_ratio = (
        fp64_metrics["heldout_relative_mse"]
        / baseline_metrics["heldout_relative_mse"]
    )
    norm_ratio = (
        fp64_metrics["decoder_frobenius_norm"]
        / baseline_metrics["decoder_frobenius_norm"]
    )
    excluded = (
        maximum_primary_difference <= PRIMARY_RELATIVE_TOLERANCE
        and heldout_ratio >= 10.0
        and norm_ratio >= 5.0
    )
    interpretation = (
        "FP64 reproduces the large-norm, poor-held-out QR(0) solution. The failure "
        "therefore comes from the unregularized statistical objective rather than "
        "FP32 roundoff."
        if excluded
        else "FP64 differs materially from FP32; numerical precision remains a plausible "
        "contributor and requires deeper decoder-level comparison."
    )

    payload: dict[str, Any] = {
        "format": FORMAT,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "environment": {
            "required_conda_environment": "lowrank or basis",
            "active_conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_memory_gib": (
                torch.cuda.get_device_properties(device).total_memory / 1024**3
            ),
            "device": str(device),
            "work_dtype": "float64",
            "tf32_enabled": False,
        },
        "source": source,
        "fp32_reference_json": str(args.fp32_reference_json),
        "fp32_reference_sha256": fp32_reference_sha256,
        "fp32_reference": fp32_method,
        "fp64": {
            "metrics": fp64_metrics,
            "solve": fp64_solve,
            "pairwise_activation_qr": pairwise_diagnostics,
            "evaluation_seconds": evaluation_seconds,
            "experiment_seconds": experiment_seconds,
            "peak_gpu_memory_gib": peak_memory_gib,
        },
        "fp64_vs_fp32_relative_change": changes,
        "fp64_vs_cholesky_baseline": {
            "heldout_mse_ratio": heldout_ratio,
            "decoder_norm_ratio": norm_ratio,
        },
        "decision": {
            "primary_relative_tolerance": PRIMARY_RELATIVE_TOLERANCE,
            "maximum_primary_relative_difference": maximum_primary_difference,
            "fp32_roundoff_excluded_as_primary_explanation": excluded,
            "interpretation": interpretation,
        },
        "warnings": [
            "The FP32 decoder tensor was not persisted, so no elementwise decoder "
            "distance is available.",
            "Both precision runs start from the same BF16 snapshot and BF16 fixed A.",
        ],
    }
    _atomic_write_json(args.output_json, payload)
    _atomic_write_text(args.output_markdown, _render_markdown(payload))
    print(f"wrote {args.output_json}", flush=True)
    print(f"wrote {args.output_markdown}", flush=True)


if __name__ == "__main__":
    main()
