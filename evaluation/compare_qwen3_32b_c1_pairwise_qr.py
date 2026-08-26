#!/usr/bin/env python3
"""Compare normal-equation and pairwise-QR C1 decoder solves at fixed A."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Iterable, Mapping

import torch
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.pairwise_qr import (  # noqa: E402
    pairwise_qr_r,
    solve_square_root_least_squares,
)


SNAPSHOT_FORMAT = "basisserve.attention_o_proj_ppl_snapshots.v1"
EXPERIMENT_FORMAT = "basisserve.qwen3_32b.c1_fixed_a_pairwise_qr.v1"
NUM_QUERY_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 5120
QUERY_WIDTH = NUM_QUERY_HEADS * HEAD_DIM


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
        help="Checkpoint supplying the one fixed encoder A per layer.",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=(23, 0))
    parser.add_argument("--relative-damping", type=float, default=1.0e-5)
    parser.add_argument("--qr-block-rows", type=int, default=8192)
    parser.add_argument("--evaluation-row-chunk-size", type=int, default=256)
    parser.add_argument("--output-column-chunk-size", type=int, default=256)
    parser.add_argument(
        "--work-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "results/evaluation/qwen3_32b_c1_fixed_a_pairwise_qr_l23_l0.json"
        ),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path(
            "results/evaluation/qwen3_32b_c1_fixed_a_pairwise_qr_l23_l0.md"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _validate_positive_arguments(args: argparse.Namespace) -> None:
    if args.relative_damping < 0:
        raise ValueError("relative damping must be non-negative")
    for name in (
        "qr_block_rows",
        "evaluation_row_chunk_size",
        "output_column_chunk_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', ' ')} must be positive")
    if len(args.layers) != 2 or set(args.layers) != {0, 23}:
        raise ValueError("this controlled experiment requires exactly layers 23 and 0")


def _load_and_validate_layer(
    *,
    layer: int,
    snapshot_dir: Path,
    snapshot_manifest: Mapping[str, Any],
    snapshot_manifest_sha256: str,
    factor_dir: Path,
    relative_damping: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    snapshot_record = snapshot_manifest["artifacts"][str(layer)]
    snapshot_path = snapshot_dir / snapshot_record["file"]
    observed_snapshot_sha256 = _sha256(snapshot_path)
    if observed_snapshot_sha256 != snapshot_record["sha256"]:
        raise ValueError(f"snapshot hash mismatch at layer {layer}")

    factor_record_path = factor_dir / f"layer_{layer:03d}.json"
    factor_record = _read_json(factor_record_path)
    if int(factor_record.get("layer", -1)) != layer:
        raise ValueError(f"factor record layer mismatch at layer {layer}")
    fit_config = factor_record["fit_config"]
    expected_geometry = {
        "num_query_heads": NUM_QUERY_HEADS,
        "num_physical_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "hidden_size": HIDDEN_SIZE,
    }
    for key, expected in expected_geometry.items():
        if int(fit_config.get(key, -1)) != expected:
            raise ValueError(f"factor geometry mismatch for {key} at layer {layer}")
    if fit_config.get("decoder_objective") != "full_layer":
        raise ValueError("baseline checkpoint must use the full-layer decoder objective")
    if not math.isclose(
        float(fit_config.get("covariance_damping", float("nan"))),
        relative_damping,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("requested damping does not match the baseline checkpoint")
    if fit_config.get("snapshot_manifest_sha256") != snapshot_manifest_sha256:
        raise ValueError(f"factor snapshot manifest mismatch at layer {layer}")
    if (
        factor_record["source_snapshot"]["fit"]["sha256"]
        != observed_snapshot_sha256
    ):
        raise ValueError(f"factor source snapshot mismatch at layer {layer}")

    factor_path = factor_dir / factor_record["artifact"]["file"]
    observed_factor_sha256 = _sha256(factor_path)
    if observed_factor_sha256 != factor_record["artifact"]["sha256"]:
        raise ValueError(f"factor hash mismatch at layer {layer}")

    snapshot_payload = load_file(str(snapshot_path), device="cpu")
    factor_payload = load_file(str(factor_path), device="cpu")
    activation = snapshot_payload["activation"].contiguous()
    weight = snapshot_payload["weight"].contiguous()
    fixed_a = factor_payload["value_coordinate_encoders"].contiguous()
    checkpoint_decoder = factor_payload["head_output_decoders"].contiguous()

    fit_rows = int(fit_config["fit_rows"])
    heldout_rows = int(fit_config["validation_rows"])
    heldout_start = int(fit_config.get("validation_row_start", fit_rows))
    rank = int(fit_config["cache_rank_per_head"])
    expected_rows = int(snapshot_manifest["calibration"]["rows_per_layer"])
    expected_shapes = {
        "activation": (expected_rows, QUERY_WIDTH),
        "weight": (HIDDEN_SIZE, QUERY_WIDTH),
        "fixed_a": (NUM_KV_HEADS, HEAD_DIM, rank),
        "checkpoint_decoder": (NUM_QUERY_HEADS, rank, HIDDEN_SIZE),
    }
    observed_tensors = {
        "activation": activation,
        "weight": weight,
        "fixed_a": fixed_a,
        "checkpoint_decoder": checkpoint_decoder,
    }
    for name, expected in expected_shapes.items():
        if tuple(observed_tensors[name].shape) != expected:
            raise ValueError(
                f"unexpected {name} shape at layer {layer}: "
                f"{tuple(observed_tensors[name].shape)}"
            )
    if heldout_start < fit_rows or heldout_start + heldout_rows > expected_rows:
        raise ValueError(f"invalid fit/held-out row split at layer {layer}")

    source = {
        "snapshot_file": str(snapshot_path),
        "snapshot_sha256": observed_snapshot_sha256,
        "factor_record_file": str(factor_record_path),
        "factor_file": str(factor_path),
        "factor_sha256": observed_factor_sha256,
        "fit_rows": fit_rows,
        "heldout_row_start": heldout_start,
        "heldout_rows": heldout_rows,
        "rank_per_head": rank,
        "checkpoint_regularized_fit_relative_mse": float(
            factor_record["fit"]["factor_dtype_relative_mse"]
        ),
        "checkpoint_heldout_relative_mse": float(
            factor_record["heldout"]["factor_dtype_relative_mse"]
        ),
        "checkpoint_absolute_trace_damping": float(
            factor_record["covariance"]["fit_absolute_trace_damping"]
        ),
    }
    return activation, weight, fixed_a, checkpoint_decoder, source


def _row_ranges(
    rows: int,
    block_rows: int,
    minimum_rows: int,
) -> tuple[tuple[int, int], ...]:
    if rows < minimum_rows:
        raise ValueError("TSQR input has fewer rows than columns")
    if block_rows < minimum_rows:
        raise ValueError("TSQR block rows must be at least the input column count")
    ranges = [
        (start, min(start + block_rows, rows))
        for start in range(0, rows, block_rows)
    ]
    if len(ranges) > 1 and ranges[-1][1] - ranges[-1][0] < minimum_rows:
        previous_start, _ = ranges[-2]
        ranges[-2:] = [(previous_start, rows)]
    return tuple(ranges)


@torch.no_grad()
def _activation_pairwise_r(
    activation: torch.Tensor,
    *,
    fit_rows: int,
    block_rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, Any]]:
    ranges = _row_ranges(fit_rows, block_rows, QUERY_WIDTH)
    scale = fit_rows**-0.5

    def blocks() -> Iterable[torch.Tensor]:
        for start, stop in ranges:
            work = activation[start:stop].to(device=device, dtype=dtype)
            work.mul_(scale)
            yield work

    _synchronize(device)
    started = time.monotonic()
    r_x, diagnostics = pairwise_qr_r(blocks())
    _synchronize(device)
    elapsed = time.monotonic() - started
    return r_x, {**asdict(diagnostics), "wall_time_seconds": elapsed}


def _head_mapping(device: torch.device) -> torch.Tensor:
    query_heads_per_group = NUM_QUERY_HEADS // NUM_KV_HEADS
    return torch.arange(NUM_QUERY_HEADS, device=device) // query_heads_per_group


@torch.no_grad()
def _dense_block_basis(
    fixed_a: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rank = int(fixed_a.shape[2])
    mapping = _head_mapping(device)
    by_head = fixed_a.index_select(0, mapping)
    basis = torch.zeros(
        QUERY_WIDTH,
        NUM_QUERY_HEADS * rank,
        device=device,
        dtype=dtype,
    )
    for head in range(NUM_QUERY_HEADS):
        basis[
            head * HEAD_DIM : (head + 1) * HEAD_DIM,
            head * rank : (head + 1) * rank,
        ] = by_head[head]
    return basis


@torch.no_grad()
def _output_energy(
    activation: torch.Tensor,
    matrix: torch.Tensor,
    *,
    row_start: int,
    row_count: int,
    row_chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    energy = 0.0
    row_stop = row_start + row_count
    for start in range(row_start, row_stop, row_chunk_size):
        stop = min(start + row_chunk_size, row_stop)
        rows = activation[start:stop].to(device=device, dtype=dtype)
        output = rows @ matrix
        energy += float(output.square().sum(dtype=torch.float64))
    return energy


@torch.no_grad()
def _evaluate_decoder(
    activation: torch.Tensor,
    target_weight: torch.Tensor,
    fixed_a: torch.Tensor,
    decoder: torch.Tensor,
    *,
    fit_rows: int,
    heldout_start: int,
    heldout_rows: int,
    fit_target_energy: float,
    heldout_target_energy: float,
    row_chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    mapping = _head_mapping(device)
    by_head = fixed_a.index_select(0, mapping)
    approximation = torch.einsum("hdr,hro->hdo", by_head, decoder)
    residual_weight = approximation.reshape(QUERY_WIDTH, HIDDEN_SIZE) - target_weight
    fit_error = _output_energy(
        activation,
        residual_weight,
        row_start=0,
        row_count=fit_rows,
        row_chunk_size=row_chunk_size,
        device=device,
        dtype=dtype,
    )
    heldout_error = _output_energy(
        activation,
        residual_weight,
        row_start=heldout_start,
        row_count=heldout_rows,
        row_chunk_size=row_chunk_size,
        device=device,
        dtype=dtype,
    )
    return {
        "fit_relative_mse": fit_error / fit_target_energy,
        "heldout_relative_mse": heldout_error / heldout_target_energy,
        "decoder_frobenius_norm": math.sqrt(
            float(decoder.square().sum(dtype=torch.float64))
        ),
        "decoder_maximum_absolute_value": float(decoder.abs().max()),
    }


def _relative_tensor_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = float((left - right).square().sum(dtype=torch.float64))
    denominator = float(right.square().sum(dtype=torch.float64))
    return math.sqrt(difference / max(denominator, torch.finfo(torch.float64).tiny))


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3-32B C1 fixed-A pairwise QR experiment",
        "",
        "This experiment changes only the decoder solver. Each layer uses the same "
        "BF16 encoder `A` loaded from the existing V96/damping-1e-5 checkpoint.",
        "",
        "The damped square-root problem uses `delta = 1e-5 * tr(C) / 8192`:",
        "",
        "```text",
        "[R_X B; sqrt(delta) B] D ~= [R_X W; sqrt(delta) W]",
        "```",
        "",
        "It is therefore objective-equivalent to the current covariance damping, "
        "while avoiding decoder normal equations.",
        "",
        "## Command",
        "",
        "```bash",
        str(payload["command"]),
        "```",
        "",
        "## Accuracy",
        "",
        "| Layer | Solver | Fit relative MSE | Held-out relative MSE | Decoder norm |",
        "|---:|---|---:|---:|---:|",
    ]
    method_labels = {
        "cholesky_damped": "Cholesky + 1e-5",
        "pairwise_qr_unregularized": "Pairwise QR + lambda=0",
        "pairwise_qr_damped": "Pairwise QR + 1e-5 augmentation",
    }
    for layer in payload["layers"]:
        for method, label in method_labels.items():
            metrics = layer["methods"][method]["metrics"]
            lines.append(
                f"| {layer['layer']} | {label} | "
                f"{metrics['fit_relative_mse']:.9e} | "
                f"{metrics['heldout_relative_mse']:.9e} | "
                f"{metrics['decoder_frobenius_norm']:.6e} |"
            )
    lines.extend(
        [
            "",
            "## Solver agreement and timing",
            "",
            "| Layer | Absolute delta | QR(damped) vs Cholesky decoder | "
            "QR(0) vs Cholesky decoder | Peak GPU GiB |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for layer in payload["layers"]:
        agreement = layer["decoder_agreement"]
        lines.append(
            f"| {layer['layer']} | {layer['absolute_damping']:.9e} | "
            f"{agreement['pairwise_qr_damped_vs_cholesky']:.9e} | "
            f"{agreement['pairwise_qr_unregularized_vs_cholesky']:.9e} | "
            f"{layer['peak_gpu_memory_gib']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Layer | Cholesky baseline | Activation TSQR (s) | "
            "QR lambda=0 (s) | QR augmented (s) |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for layer in payload["layers"]:
        lines.append(
            f"| {layer['layer']} | existing checkpoint (no solve) | "
            f"{layer['pairwise_activation_qr']['wall_time_seconds']:.3f} | "
            f"{layer['methods']['pairwise_qr_unregularized']['solve']['wall_time_seconds']:.3f} | "
            f"{layer['methods']['pairwise_qr_damped']['solve']['wall_time_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
        ]
    )
    by_layer = {int(layer["layer"]): layer for layer in payload["layers"]}
    if 23 in by_layer and 0 in by_layer:
        pathological = by_layer[23]
        control = by_layer[0]
        pathological_baseline = pathological["methods"]["cholesky_damped"]["metrics"]
        pathological_qr0 = pathological["methods"]["pairwise_qr_unregularized"][
            "metrics"
        ]
        pathological_qr_damped = pathological["methods"]["pairwise_qr_damped"][
            "metrics"
        ]
        control_baseline = control["methods"]["cholesky_damped"]["metrics"]
        control_qr0 = control["methods"]["pairwise_qr_unregularized"]["metrics"]
        lines.extend(
            [
                "- On layer 23, QR without damping lowers raw fit MSE by "
                f"{100 * (1 - pathological_qr0['fit_relative_mse'] / pathological_baseline['fit_relative_mse']):.3f}%, "
                "but held-out MSE becomes "
                f"{pathological_qr0['heldout_relative_mse'] / pathological_baseline['heldout_relative_mse']:.3f}x "
                "the baseline and decoder norm becomes "
                f"{pathological_qr0['decoder_frobenius_norm'] / pathological_baseline['decoder_frobenius_norm']:.3f}x larger.",
                "- Matched damping restores layer-23 held-out MSE to within "
                f"{100 * abs(pathological_qr_damped['heldout_relative_mse'] / pathological_baseline['heldout_relative_mse'] - 1):.3f}% "
                "of the checkpoint baseline.",
                "- On control layer 0, removing damping changes held-out MSE by only "
                f"{100 * (control_qr0['heldout_relative_mse'] / control_baseline['heldout_relative_mse'] - 1):.3f}%.",
                "- Therefore square-root/TSQR removes normal equations as a numerical "
                "liability, but layer 23 still needs ridge as statistical regularization. "
                "The clean formulation is: QR provides stability; lambda controls generalization.",
                "",
                "The baseline decoder is the stored BF16 checkpoint artifact. Decoder-space "
                "distances also include checkpoint factor quantization, so output MSE is the "
                "primary solver comparison.",
                "",
            ]
        )
    return "\n".join(lines)


def _write_outputs(
    payload: Mapping[str, Any],
    *,
    output_json: Path,
    output_markdown: Path,
) -> None:
    _atomic_write_json(output_json, payload)
    _atomic_write_text(output_markdown, _render_markdown(payload))


@torch.no_grad()
def _run_layer(
    *,
    layer: int,
    args: argparse.Namespace,
    snapshot_manifest: Mapping[str, Any],
    snapshot_manifest_sha256: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    print(f"[layer {layer}] validating hashes and loading tensors", flush=True)
    activation, weight, fixed_a_cpu, checkpoint_decoder_cpu, source = (
        _load_and_validate_layer(
            layer=layer,
            snapshot_dir=args.snapshot_dir,
            snapshot_manifest=snapshot_manifest,
            snapshot_manifest_sha256=snapshot_manifest_sha256,
            factor_dir=args.factor_dir,
            relative_damping=args.relative_damping,
        )
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fixed_a = fixed_a_cpu.to(device=device, dtype=dtype)
    checkpoint_decoder = checkpoint_decoder_cpu.to(device=device, dtype=dtype)
    fit_rows = int(source["fit_rows"])
    heldout_start = int(source["heldout_row_start"])
    heldout_rows = int(source["heldout_rows"])

    print(f"[layer {layer}] pairwise activation TSQR", flush=True)
    r_x, pairwise_diagnostics = _activation_pairwise_r(
        activation,
        fit_rows=fit_rows,
        block_rows=args.qr_block_rows,
        device=device,
        dtype=dtype,
    )
    covariance_trace = float(r_x.square().sum(dtype=torch.float64))
    absolute_damping = args.relative_damping * covariance_trace / QUERY_WIDTH
    recorded_delta = float(source["checkpoint_absolute_trace_damping"])
    relative_delta_difference = abs(absolute_damping - recorded_delta) / max(
        abs(recorded_delta), torch.finfo(torch.float64).tiny
    )
    basis = _dense_block_basis(fixed_a, device=device, dtype=dtype)
    target_weight = weight.to(device=device, dtype=dtype).transpose(0, 1).contiguous()

    print(f"[layer {layer}] square-root QR, lambda=0", flush=True)
    qr_unregularized_flat, qr_unregularized_diagnostics = (
        solve_square_root_least_squares(
            left_factor=r_x,
            basis=basis,
            target_weight=target_weight,
            absolute_product_damping=0.0,
            output_chunk_size=args.output_column_chunk_size,
        )
    )
    rank = int(fixed_a.shape[2])
    qr_unregularized_decoder = qr_unregularized_flat.reshape(
        NUM_QUERY_HEADS,
        rank,
        HIDDEN_SIZE,
    )
    qr_unregularized_solve = asdict(qr_unregularized_diagnostics)
    print(f"[layer {layer}] square-root QR, matched 1e-5 augmentation", flush=True)
    qr_damped_flat, qr_damped_diagnostics = solve_square_root_least_squares(
        left_factor=r_x,
        basis=basis,
        target_weight=target_weight,
        absolute_product_damping=absolute_damping,
        output_chunk_size=args.output_column_chunk_size,
    )
    qr_damped_decoder = qr_damped_flat.reshape(
        NUM_QUERY_HEADS,
        rank,
        HIDDEN_SIZE,
    )
    qr_damped_solve = asdict(qr_damped_diagnostics)

    print(f"[layer {layer}] evaluating identical fit and held-out rows", flush=True)
    fit_target_energy = _output_energy(
        activation,
        target_weight,
        row_start=0,
        row_count=fit_rows,
        row_chunk_size=args.evaluation_row_chunk_size,
        device=device,
        dtype=dtype,
    )
    heldout_target_energy = _output_energy(
        activation,
        target_weight,
        row_start=heldout_start,
        row_count=heldout_rows,
        row_chunk_size=args.evaluation_row_chunk_size,
        device=device,
        dtype=dtype,
    )
    method_decoders = {
        "cholesky_damped": checkpoint_decoder,
        "pairwise_qr_unregularized": qr_unregularized_decoder,
        "pairwise_qr_damped": qr_damped_decoder,
    }
    methods: dict[str, Any] = {}
    solves = {
        "cholesky_damped": {
            "source": "existing bfloat16 checkpoint; no new solve",
            "wall_time_seconds": 0.0,
        },
        "pairwise_qr_unregularized": qr_unregularized_solve,
        "pairwise_qr_damped": qr_damped_solve,
    }
    for name, decoder in method_decoders.items():
        metrics = _evaluate_decoder(
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
            dtype=dtype,
        )
        methods[name] = {"metrics": metrics, "solve": solves[name]}

    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "layer": layer,
        "source": source,
        "absolute_damping": absolute_damping,
        "recorded_checkpoint_absolute_damping": recorded_delta,
        "relative_difference_from_recorded_absolute_damping": relative_delta_difference,
        "target_energy": {
            "fit": fit_target_energy,
            "heldout": heldout_target_energy,
        },
        "pairwise_activation_qr": pairwise_diagnostics,
        "methods": methods,
        "baseline_validation": {
            "heldout_relative_difference": abs(
                methods["cholesky_damped"]["metrics"]["heldout_relative_mse"]
                - source["checkpoint_heldout_relative_mse"]
            )
            / max(
                abs(source["checkpoint_heldout_relative_mse"]),
                torch.finfo(torch.float64).tiny,
            ),
            "fit_comparison_omitted": (
                "direct fit_relative_mse is unregularized, whereas the checkpoint "
                "fit metric includes covariance damping"
            ),
        },
        "decoder_agreement": {
            "pairwise_qr_damped_vs_cholesky": _relative_tensor_distance(
                qr_damped_decoder, checkpoint_decoder
            ),
            "pairwise_qr_unregularized_vs_cholesky": _relative_tensor_distance(
                qr_unregularized_decoder, checkpoint_decoder
            ),
        },
        "peak_gpu_memory_gib": peak_memory,
    }
    del (
        activation,
        weight,
        fixed_a,
        checkpoint_decoder,
        qr_unregularized_decoder,
        qr_damped_decoder,
        r_x,
        basis,
        target_weight,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[layer {layer}] complete", flush=True)
    return result


def main() -> None:
    args = _parse_args()
    _validate_positive_arguments(args)
    for output in (args.output_json, args.output_markdown):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists: {output}; pass --overwrite")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = _dtype(args.work_dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    snapshot_manifest_path = args.snapshot_dir / "manifest.json"
    snapshot_manifest_sha256 = _sha256(snapshot_manifest_path)
    snapshot_manifest = _read_json(snapshot_manifest_path)
    if snapshot_manifest.get("format") != SNAPSHOT_FORMAT:
        raise ValueError("pairwise QR experiment requires a raw-activation snapshot")
    model = snapshot_manifest["model"]
    expected_model = {
        "num_attention_heads": NUM_QUERY_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "hidden_size": HIDDEN_SIZE,
    }
    for key, expected in expected_model.items():
        if int(model.get(key, -1)) != expected:
            raise ValueError(f"unexpected Qwen3-32B manifest field: {key}")

    command = shlex.join([sys.executable, *sys.argv])
    payload: dict[str, Any] = {
        "format": EXPERIMENT_FORMAT,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "environment": {
            "required_conda_environment": "lowrank or basis",
            "active_conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "device": str(device),
            "work_dtype": args.work_dtype,
            "tf32_enabled": False,
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "gpu_total_memory_gib": (
                torch.cuda.get_device_properties(device).total_memory / 1024**3
                if device.type == "cuda"
                else None
            ),
        },
        "configuration": {
            "layers": list(args.layers),
            "fixed_a_factor_dir": str(args.factor_dir),
            "snapshot_dir": str(args.snapshot_dir),
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "relative_damping": args.relative_damping,
            "qr_block_rows": args.qr_block_rows,
            "evaluation_row_chunk_size": args.evaluation_row_chunk_size,
            "output_column_chunk_size": args.output_column_chunk_size,
            "objective_equivalence": (
                "delta = relative_damping * trace(X^T X / N) / 8192; "
                "augmented rows are sqrt(delta) * B and sqrt(delta) * W"
            ),
        },
        "layers": [],
    }
    _write_outputs(
        payload,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    started = time.monotonic()
    try:
        for layer in args.layers:
            payload["layers"].append(
                _run_layer(
                    layer=layer,
                    args=args,
                    snapshot_manifest=snapshot_manifest,
                    snapshot_manifest_sha256=snapshot_manifest_sha256,
                    device=device,
                    dtype=dtype,
                )
            )
            _write_outputs(
                payload,
                output_json=args.output_json,
                output_markdown=args.output_markdown,
            )
    except Exception as error:
        payload["status"] = "failed"
        payload["failure"] = {"type": type(error).__name__, "message": str(error)}
        payload["wall_time_seconds"] = time.monotonic() - started
        _write_outputs(
            payload,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
        )
        raise
    payload["status"] = "complete"
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["wall_time_seconds"] = time.monotonic() - started
    _write_outputs(
        payload,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(f"wrote {args.output_json}", flush=True)
    print(f"wrote {args.output_markdown}", flush=True)


if __name__ == "__main__":
    main()
