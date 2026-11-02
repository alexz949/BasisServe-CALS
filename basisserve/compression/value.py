#!/usr/bin/env python3
"""Fit and export Value factors using explicit attention geometry."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor


from basisserve.core.gqa_routed_ov_joint import (  # noqa: E402
    covariance_with_trace_damping,
    evaluate_quadratic,
    fit_routed_ov_joint,
    initialize_group_pooled_routed_svd,
    mask_routed_covariance,
    quadratic_from_target,
)


from basisserve.compression.geometry import ValueGeometry


FORMAT = "basiskv.value_factors.v1"
LAYER_FORMAT = "basiskv.value_factors.layer.v1"
SNAPSHOT_FORMAT = "basisserve.attention_o_proj_ppl_snapshots.v1"
COVARIANCE_SNAPSHOT_FORMAT = "basisserve.attention_o_proj_covariances.v1"


def _head_to_kv_group(
    *, device: torch.device | str = "cpu", geometry: ValueGeometry
) -> Tensor:
    return torch.arange(geometry.num_heads, device=device, dtype=torch.long) // (
        geometry.num_heads // geometry.num_kv_heads
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _cg_diagnostics_payload(
    result: Any,
    *,
    mode: str,
    relative_tolerance: float,
    max_iterations: int,
) -> dict[str, Any]:
    """Serialize compact per-update CG diagnostics plus a layer summary."""
    per_sweep = []
    flat_steps = []
    for sweep in result.sweeps:
        encoder_steps = []
        for step in sweep.encoder_steps:
            cg = step.cg
            item = {
                "group_index": int(step.group_index),
                "iterations": int(cg.iterations),
                "converged": bool(cg.converged),
                "relative_residual": float(cg.relative_residual),
                "absolute_damping": float(cg.absolute_damping),
                "negative_curvature": bool(cg.negative_curvature),
                "initial_residual_norm": float(cg.initial_residual_norm),
                "final_residual_norm": float(cg.final_residual_norm),
                "fixed_iterations": bool(cg.fixed_iterations),
                "predicted_change": float(step.predicted_change),
                "realized_change": float(step.realized_change),
            }
            encoder_steps.append(item)
            flat_steps.append(item)
        per_sweep.append(
            {
                "sweep": int(sweep.sweep),
                "encoder_steps": encoder_steps,
            }
        )
    nonzero = [step for step in flat_steps if step["initial_residual_norm"] != 0.0]
    return {
        "mode": mode,
        "relative_tolerance": float(relative_tolerance),
        "max_iterations": int(max_iterations),
        "aggregate": {
            "group_solves": len(flat_steps),
            "nonzero_rhs_group_solves": len(nonzero),
            "exact_zero_rhs": len(flat_steps) - len(nonzero),
            "total_iterations": sum(step["iterations"] for step in flat_steps),
            "minimum_iterations_nonzero_rhs": min(
                (step["iterations"] for step in nonzero), default=0
            ),
            "maximum_iterations": max(
                (step["iterations"] for step in flat_steps), default=0
            ),
            "converged_at_tolerance": sum(step["converged"] for step in flat_steps),
            "negative_curvature": sum(
                step["negative_curvature"] for step in flat_steps
            ),
            "maximum_relative_residual": max(
                (step["relative_residual"] for step in flat_steps), default=0.0
            ),
            "maximum_predicted_realized_change_error": max(
                (
                    abs(step["predicted_change"] - step["realized_change"])
                    for step in flat_steps
                ),
                default=0.0,
            ),
        },
        "per_sweep": per_sweep,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, payload: Mapping[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(payload), str(temporary))
    os.replace(temporary, path)


def _attention_layers(*, geometry: ValueGeometry) -> tuple[int, ...]:
    return geometry.attention_layers


def _parse_layers(raw: str, *, geometry: ValueGeometry) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return _attention_layers(geometry=geometry)
    layers: set[int] = set()
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            first, last = map(int, piece.split("-", 1))
            assert not (last < first), f"descending layer range: {piece}"
            layers.update(range(first, last + 1))
        else:
            layers.add(int(piece))
    selected = tuple(sorted(layers))
    assert not (
        not selected or min(selected) < 0 or max(selected) >= geometry.num_layers
    ), f"selected layers are outside {geometry.model_type}"
    assert set(selected) <= set(_attention_layers(geometry=geometry))
    return selected


def _parse_checkpoint_sweeps(raw: str | None, maximum: int) -> tuple[int, ...]:
    if raw is None or not raw.strip():
        return ()
    sweeps = tuple(
        sorted({int(piece.strip()) for piece in raw.split(",") if piece.strip()})
    )
    if any(sweep < 0 or sweep > maximum for sweep in sweeps):
        print(
            f"[Error] checkpoint export sweeps must lie in [0, {maximum}]",
            file=sys.stderr,
            flush=True,
        )
        return ()
    return sweeps


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _factor_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _canonicalize_basis_signs(value: Tensor) -> Tensor:
    """Choose deterministic signs for the columns of a two-dimensional basis."""

    assert not (value.ndim != 2), "basis sign canonicalization requires a matrix"
    if value.shape[1] == 0:
        return value
    pivots = value.abs().argmax(dim=0)
    columns = torch.arange(value.shape[1], device=value.device)
    signs = torch.sign(value[pivots, columns])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return value * signs.unsqueeze(0)


@torch.no_grad()
def _weight_only_encoder_initialization(
    target: Tensor, *, rank: int, mapping: Tensor, geometry: ValueGeometry
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Return one pooled weight-only left singular subspace per KV group."""

    assert not (
        target.ndim != 3
        or tuple(target.shape[:2]) != (geometry.num_heads, geometry.head_dim)
    ), "weight-only initialization received an invalid target"
    assert not (not 0 < rank <= geometry.head_dim), (
        "weight-only initialization rank is invalid"
    )
    encoders = target.new_empty(geometry.num_kv_heads, geometry.head_dim, rank)
    diagnostics: list[dict[str, Any]] = []
    for group in range(geometry.num_kv_heads):
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        block = target.index_select(0, heads)
        gram = torch.einsum("hio,hjo->ij", block, block)
        gram = 0.5 * (gram + gram.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues.index_select(0, order).clamp_min(0)
        basis = _canonicalize_basis_signs(eigenvectors.index_select(1, order[:rank]))
        encoders[group].copy_(basis)
        singular_values = torch.sqrt(eigenvalues)
        total_energy = float(eigenvalues.sum())
        tail_energy = float(eigenvalues[rank:].sum())
        diagnostics.append(
            {
                "method": "weight-only-svd",
                "group_index": group,
                "head_indices": list(map(int, heads.tolist())),
                "rank": rank,
                "leading_singular_values": singular_values[:rank].tolist(),
                "boundary_singular_value": float(singular_values[rank - 1]),
                "next_singular_value": (
                    float(singular_values[rank]) if rank < geometry.head_dim else None
                ),
                "weight_tail_energy_fraction": (
                    tail_energy / total_energy if total_energy > 0 else 0.0
                ),
            }
        )
    return encoders, diagnostics


@torch.no_grad()
def _random_orthogonal_encoder_initialization(
    *,
    rank: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    geometry: ValueGeometry,
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Return reproducible independent Haar-subspace controls for all heads."""

    assert not (not 0 < rank <= geometry.head_dim), (
        "random initialization rank is invalid"
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch.randn(
        geometry.num_kv_heads,
        geometry.head_dim,
        rank,
        generator=generator,
        device="cpu",
        dtype=torch.float64,
    )
    encoders = torch.empty_like(random)
    diagnostics: list[dict[str, Any]] = []
    mapping = _head_to_kv_group(geometry=geometry)
    for group in range(geometry.num_kv_heads):
        basis, _ = torch.linalg.qr(random[group], mode="reduced")
        basis = _canonicalize_basis_signs(basis)
        encoders[group].copy_(basis)
        diagnostics.append(
            {
                "method": "random-orthogonal",
                "group_index": group,
                "head_indices": list(
                    map(
                        int,
                        torch.nonzero(mapping == group, as_tuple=False)
                        .flatten()
                        .tolist(),
                    )
                ),
                "rank": rank,
                "layer_seed": seed,
            }
        )
    return encoders.to(device=device, dtype=dtype), diagnostics


def _snapshot_manifest(
    snapshot_dir: Path, *, geometry: ValueGeometry
) -> dict[str, Any]:
    path = snapshot_dir / "manifest.json"
    assert not (not path.is_file()), path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert not (
        manifest.get("format") not in {SNAPSHOT_FORMAT, COVARIANCE_SNAPSHOT_FORMAT}
    ), "incompatible attention snapshot format"
    model = manifest.get("model", {})
    observed = (
        model.get("model_type"),
        model.get("attention_type"),
        int(model.get("num_hidden_layers", -1)),
        int(model.get("num_attention_heads", -1)),
        int(model.get("num_key_value_heads", -1)),
        int(model.get("head_dim", -1)),
        int(model.get("hidden_size", -1)),
    )
    expected = (
        geometry.model_type,
        geometry.attention_type,
        geometry.num_layers,
        geometry.num_heads,
        geometry.num_kv_heads,
        geometry.head_dim,
        geometry.hidden_size,
    )
    assert not (observed != expected), (
        f"unexpected {geometry.model_type} snapshot geometry: {observed}"
    )
    assert not (
        tuple(map(int, manifest.get("layers", ())))
        != _attention_layers(geometry=geometry)
    ), f"snapshot must cover every {geometry.model_type} decoder layer"
    return manifest


def _resolve_window_partition(
    *,
    snapshot_windows: int,
    fit_windows: int,
    validation_windows: int,
    validation_window_start: int | None,
) -> int:
    assert not (min(snapshot_windows, fit_windows, validation_windows) <= 0), (
        "snapshot, fit, and validation partitions must be nonempty"
    )
    if validation_window_start is None:
        assert not (fit_windows + validation_windows != snapshot_windows), (
            "fit and validation windows must exactly partition the snapshot "
            "unless --validation-window-start is explicit"
        )
        return fit_windows
    start = int(validation_window_start)
    assert not (start < fit_windows), (
        "validation windows overlap the leading fit partition"
    )
    assert not (start + validation_windows > snapshot_windows), (
        "validation windows exceed the snapshot"
    )
    return start


def _snapshot_shape(manifest: Mapping[str, Any]) -> tuple[int, int, int]:
    calibration = manifest["calibration"]
    rows = int(calibration["rows_per_layer"])
    windows = int(calibration["window_count"])
    positions = int(calibration["positions_per_window"])
    assert not (rows != windows * positions), (
        "snapshot rows do not factor into windows and positions"
    )
    return rows, windows, positions


def _fit_config(
    args: argparse.Namespace,
    snapshot_dir: Path,
    manifest: Mapping[str, Any],
    *,
    validation_snapshot_dir: Path | None = None,
    validation_manifest: Mapping[str, Any] | None = None,
    geometry: ValueGeometry,
) -> dict[str, Any]:
    cg_max_iterations = (
        args.encoder_cg_max_iterations
        if args.encoder_cg_max_iterations is not None
        else args.encoder_cg_fixed_iterations
    )
    assert not (cg_max_iterations <= 0), (
        "encoder CG maximum iterations must be positive"
    )
    assert not (args.encoder_cg_relative_tolerance <= 0), (
        "encoder CG relative tolerance must be positive"
    )
    assert not (args.encoder_initialization_seed < 0), (
        "encoder initialization seed must be non-negative"
    )
    _, windows, positions = _snapshot_shape(manifest)
    assert not (not 0 < args.fit_windows <= windows), (
        "fit windows exceed the fit snapshot"
    )
    separate_validation = validation_manifest is not None
    assert not (separate_validation != (validation_snapshot_dir is not None)), (
        "validation snapshot path and manifest must be provided together"
    )
    if separate_validation:
        assert validation_manifest is not None
        assert validation_snapshot_dir is not None
        _, validation_snapshot_windows, validation_positions = _snapshot_shape(
            validation_manifest
        )
        assert not (
            manifest["model"]["config_sha256"]
            != validation_manifest["model"]["config_sha256"]
            or manifest["model"]["path"] != validation_manifest["model"]["path"]
        ), "fit and validation snapshots use different models"
        validation_window_start = int(args.validation_window_start or 0)
        assert not (
            validation_window_start < 0
            or validation_window_start + args.validation_windows
            > validation_snapshot_windows
        ), "validation windows exceed the validation snapshot"
    else:
        validation_snapshot_windows = windows
        validation_positions = positions
        validation_window_start = _resolve_window_partition(
            snapshot_windows=windows,
            fit_windows=args.fit_windows,
            validation_windows=args.validation_windows,
            validation_window_start=args.validation_window_start,
        )
    assert not (not 0 < args.cache_rank <= geometry.head_dim), (
        "cache rank is outside the attention head width"
    )
    config = {
        "model": str(Path(manifest["model"]["path"]).expanduser().resolve()),
        "model_config_sha256": manifest["model"]["config_sha256"],
        "snapshot_dir": str(snapshot_dir),
        "snapshot_manifest_sha256": _sha256(snapshot_dir / "manifest.json"),
        "fit_windows": args.fit_windows,
        "validation_windows": args.validation_windows,
        "positions_per_window": positions,
        "validation_positions_per_window": validation_positions,
        "fit_rows": args.fit_windows * positions,
        "validation_rows": args.validation_windows * validation_positions,
        "cache_rank_per_head": args.cache_rank,
        "model_label": geometry.model_type,
        "geometry": geometry.to_dict(),
        "model_type": geometry.model_type,
        "attention_type": geometry.attention_type,
        "num_query_heads": geometry.num_heads,
        "num_physical_kv_heads": geometry.num_kv_heads,
        "head_dim": geometry.head_dim,
        "hidden_size": geometry.hidden_size,
        "num_hidden_layers": geometry.num_layers,
        "total_v_cache_rank": geometry.num_kv_heads * args.cache_rank,
        "dense_v_cache_rank": geometry.num_kv_heads * geometry.head_dim,
        "v_retained_ratio": args.cache_rank / geometry.head_dim,
        "total_kv_retained_ratio_with_dense_k": (
            geometry.num_kv_heads * geometry.head_dim
            + geometry.num_kv_heads * args.cache_rank
        )
        / (2 * geometry.num_kv_heads * geometry.head_dim),
        "work_dtype": args.work_dtype,
        "allow_tf32": args.allow_tf32,
        "factor_dtype": args.factor_dtype,
        "encoder_initialization": args.encoder_initialization,
        "encoder_initialization_seed": args.encoder_initialization_seed,
        "covariance_damping": args.covariance_damping,
        "encoder_sweeps": args.encoder_sweeps,
        "export_checkpoint_sweeps": list(
            _parse_checkpoint_sweeps(
                args.export_checkpoint_sweeps,
                args.encoder_sweeps,
            )
        ),
        "encoder_cg_mode": args.encoder_cg_mode,
        "encoder_cg_relative_tolerance": args.encoder_cg_relative_tolerance,
        "encoder_cg_max_iterations": cg_max_iterations,
        "encoder_cg_fixed_iterations": (
            cg_max_iterations if args.encoder_cg_mode == "fixed" else None
        ),
        "covariance_row_chunk_size": args.covariance_row_chunk_size,
        "stored_covariance_psd_policy": (
            "fp64-cholesky-check-then-negative-eigenvalue-clip"
            if manifest.get("format") == COVARIANCE_SNAPSHOT_FORMAT
            else None
        ),
        "decoder_objective": args.decoder_objective,
        "checkpoint_policy": (
            f"fixed decoder-refitted endpoint after encoder sweep {args.encoder_sweeps}"
        ),
        "objective": (
            "full-layer attention-output MSE with cross-head covariance"
            if args.decoder_objective == "full_layer"
            else "sum of independent per-head attention-output MSEs"
        ),
    }
    if separate_validation:
        assert validation_snapshot_dir is not None
        assert validation_manifest is not None
        config.update(
            {
                "validation_snapshot_dir": str(validation_snapshot_dir),
                "validation_snapshot_manifest_sha256": _sha256(
                    validation_snapshot_dir / "manifest.json"
                ),
                "fit_snapshot_windows": windows,
                "validation_snapshot_windows": validation_snapshot_windows,
                "validation_window_start": validation_window_start,
                "validation_row_start": (
                    validation_window_start * validation_positions
                ),
                "unused_fit_snapshot_windows": windows - args.fit_windows,
                "unused_validation_snapshot_windows": (
                    validation_snapshot_windows
                    - validation_window_start
                    - args.validation_windows
                ),
            }
        )
    elif args.validation_window_start is not None:
        config.update(
            {
                "snapshot_windows": windows,
                "validation_window_start": validation_window_start,
                "validation_row_start": validation_window_start * positions,
                "unused_snapshot_windows": (
                    windows - args.fit_windows - args.validation_windows
                ),
            }
        )
    return config


def _load_snapshot_layer(
    snapshot_dir: Path,
    manifest: Mapping[str, Any],
    layer: int,
    *,
    geometry: ValueGeometry,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    record = manifest["artifacts"][str(layer)]
    path = snapshot_dir / record["file"]
    assert not (_sha256(path) != record["sha256"]), (
        f"snapshot hash mismatch at layer {layer}"
    )
    payload = load_file(str(path), device="cpu")
    activation = payload["activation"].contiguous()
    weight = payload["weight"].contiguous()
    expected_rows = int(manifest["calibration"]["rows_per_layer"])
    query_width = geometry.num_heads * geometry.head_dim
    assert not (tuple(activation.shape) != (expected_rows, query_width)), (
        f"unexpected activation shape at layer {layer}"
    )
    assert not (tuple(weight.shape) != (geometry.hidden_size, query_width)), (
        f"unexpected o_proj shape at layer {layer}"
    )
    return activation, weight, {"path": str(path), **record}


def _load_covariance_layer(
    snapshot_dir: Path,
    manifest: Mapping[str, Any],
    layer: int,
    *,
    geometry: ValueGeometry,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
    assert not (manifest.get("format") != COVARIANCE_SNAPSHOT_FORMAT), (
        "snapshot does not contain covariance sufficient statistics"
    )
    record = manifest["artifacts"][str(layer)]
    path = snapshot_dir / record["file"]
    assert not (_sha256(path) != record["sha256"]), (
        f"covariance snapshot hash mismatch at layer {layer}"
    )
    payload = load_file(str(path), device="cpu")
    expected_width = geometry.num_heads * geometry.head_dim
    fit = payload["fit_covariance"].contiguous()
    heldout = payload["heldout_covariance"].contiguous()
    weight = payload["weight"].contiguous()
    assert not (tuple(fit.shape) != (expected_width, expected_width)), (
        f"unexpected fit covariance shape at layer {layer}"
    )
    assert not (tuple(heldout.shape) != (expected_width, expected_width)), (
        f"unexpected held-out covariance shape at layer {layer}"
    )
    assert not (tuple(weight.shape) != (geometry.hidden_size, expected_width)), (
        f"unexpected o_proj shape at layer {layer}"
    )
    return fit, heldout, weight, {"path": str(path), **record}


def _covariance_matrix_to_blocks(
    covariance: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    geometry: ValueGeometry,
) -> Tensor:
    width = geometry.num_heads * geometry.head_dim
    assert not (tuple(covariance.shape) != (width, width)), (
        "covariance matrix has incompatible attention width"
    )
    work = covariance.to(device=device, dtype=dtype)
    blocks = (
        work.reshape(
            geometry.num_heads, geometry.head_dim, geometry.num_heads, geometry.head_dim
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return 0.5 * (blocks + blocks.permute(1, 0, 3, 2))


@torch.no_grad()
def _repair_stored_covariance_psd(
    covariance: Tensor, *, device: torch.device
) -> tuple[Tensor, dict[str, Any]]:
    """Remove finite-precision negative spectrum from a stored Gram matrix."""

    work = covariance.to(device=device, dtype=torch.float64)
    work = 0.5 * (work + work.T)
    cholesky, info = torch.linalg.cholesky_ex(work, check_errors=False)
    if int(info.max()) == 0:
        del cholesky
        return work, {
            "applied": False,
            "method": "fp64_cholesky_psd_check",
            "negative_eigenvalue_count": 0,
        }
    del cholesky
    eigenvalues, eigenvectors = torch.linalg.eigh(work)
    negative = eigenvalues < 0
    clipped = eigenvalues.clamp_min(0)
    repaired = (eigenvectors * clipped.unsqueeze(0)) @ eigenvectors.T
    repaired = 0.5 * (repaired + repaired.T)
    negative_norm = torch.linalg.vector_norm(eigenvalues[negative])
    total_norm = torch.linalg.vector_norm(eigenvalues)
    diagnostics = {
        "applied": True,
        "method": "fp64_symmetric_eigendecomposition_negative_clip",
        "minimum_eigenvalue_before": float(eigenvalues[0]),
        "maximum_eigenvalue_before": float(eigenvalues[-1]),
        "negative_eigenvalue_count": int(negative.sum()),
        "relative_spectral_correction_frobenius": float(
            negative_norm / total_norm.clamp_min(torch.finfo(torch.float64).tiny)
        ),
    }
    del work, eigenvalues, eigenvectors, negative, clipped
    return repaired, diagnostics


@torch.no_grad()
def _activation_covariance_blocks(
    activation: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int,
    head_dim: int,
    row_chunk_size: int | None = None,
) -> Tensor:
    """Return E[x_h^T x_k] as [head, head, d, d] blocks."""

    hidden_size = num_heads * head_dim
    assert not (activation.ndim != 2 or int(activation.shape[1]) != hidden_size), (
        f"activation must have shape [rows, {hidden_size}]"
    )
    rows = int(activation.shape[0])
    assert not (row_chunk_size is not None and row_chunk_size <= 0), (
        "covariance row chunk size must be positive"
    )
    if row_chunk_size is None or row_chunk_size >= rows:
        work = activation.to(device=device, dtype=dtype)
        gram = work.transpose(0, 1) @ work
    else:
        gram = torch.zeros(hidden_size, hidden_size, device=device, dtype=dtype)
        for start in range(0, rows, row_chunk_size):
            stop = min(start + row_chunk_size, rows)
            work = activation[start:stop].to(device=device, dtype=dtype)
            gram.addmm_(work.transpose(0, 1), work)
            del work
    gram.div_(rows)
    blocks = (
        gram.reshape(num_heads, head_dim, num_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return 0.5 * (blocks + blocks.permute(1, 0, 3, 2))


def _dense_head_targets(
    weight: Tensor, *, device: torch.device, dtype: torch.dtype, geometry: ValueGeometry
) -> Tensor:
    assert not (
        tuple(weight.shape)
        != (geometry.hidden_size, geometry.num_heads * geometry.head_dim)
    ), "dense o_proj weight has incompatible attention geometry"
    return (
        weight.to(device=device, dtype=dtype)
        .transpose(0, 1)
        .reshape(geometry.num_heads, geometry.head_dim, geometry.hidden_size)
        .contiguous()
    )


def _relative_loss(loss: float, constant: Tensor) -> float:
    return float(loss) / max(abs(float(constant)), 1.0e-300)


class _CheckpointRecorder:
    def __init__(
        self,
        fit_objective: Any,
        validation_objective: Any,
        mapping: Tensor,
        *,
        export_sweeps: Sequence[int] = (),
        export_root: Path | None = None,
        layer: int | None = None,
        factor_dtype: torch.dtype | None = None,
    ) -> None:
        self.fit_objective = fit_objective
        self.validation_objective = validation_objective
        self.mapping = mapping
        self.export_sweeps = frozenset(map(int, export_sweeps))
        self.export_root = export_root
        self.layer = layer
        self.factor_dtype = factor_dtype
        self.records: list[dict[str, Any]] = []
        self.exports: dict[int, dict[str, Any]] = {}

    def __call__(self, checkpoint: Any, A: Tensor, D: Tensor) -> None:
        heldout = evaluate_quadratic(
            self.validation_objective,
            A,
            D,
            self.mapping,
        )
        row = {
            "boundary": checkpoint.boundary,
            "sweep": int(checkpoint.sweep),
            "fit_loss": float(checkpoint.loss),
            "validation_loss": float(heldout),
            "validation_relative_mse": _relative_loss(
                heldout, self.validation_objective.constant
            ),
        }
        self.records.append(row)
        decoder_closed = checkpoint.boundary in {"decoder_only", "after_redecoder"}
        sweep = int(checkpoint.sweep)
        if not decoder_closed or sweep not in self.export_sweeps:
            return
        if self.export_root is None or self.layer is None or self.factor_dtype is None:
            return
        export_dir = self.export_root / f"sweep_{sweep:03d}"
        export_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = export_dir / f"layer_{self.layer:03d}.safetensors"
        artifact_A = A.detach().to(device="cpu", dtype=self.factor_dtype).contiguous()
        artifact_D = D.detach().to(device="cpu", dtype=self.factor_dtype).contiguous()
        _atomic_safetensors(
            artifact_path,
            {
                "value_coordinate_encoders": artifact_A,
                "head_output_decoders": artifact_D,
            },
        )
        device = self.fit_objective.covariance.device
        work_dtype = self.fit_objective.covariance.dtype
        quantized_A = artifact_A.to(device=device, dtype=work_dtype)
        quantized_D = artifact_D.to(device=device, dtype=work_dtype)
        fit_loss = evaluate_quadratic(
            self.fit_objective,
            quantized_A,
            quantized_D,
            self.mapping,
        )
        heldout_dtype = self.validation_objective.covariance.dtype
        heldout_loss = evaluate_quadratic(
            self.validation_objective,
            artifact_A.to(device=device, dtype=heldout_dtype),
            artifact_D.to(device=device, dtype=heldout_dtype),
            self.mapping,
        )
        self.exports[sweep] = {
            "boundary": str(checkpoint.boundary),
            "sweep": sweep,
            "artifact": {
                "file": artifact_path.name,
                "sha256": _sha256(artifact_path),
                "tensors": {
                    "value_coordinate_encoders": {
                        "shape": list(artifact_A.shape),
                        "dtype": str(artifact_A.dtype),
                    },
                    "head_output_decoders": {
                        "shape": list(artifact_D.shape),
                        "dtype": str(artifact_D.dtype),
                    },
                },
            },
            "fit_factor_dtype_relative_mse": _relative_loss(
                fit_loss, self.fit_objective.constant
            ),
            "heldout_factor_dtype_relative_mse": _relative_loss(
                heldout_loss, self.validation_objective.constant
            ),
        }


def _verified_prior(
    path: Path, artifact: Path, layer: int, fit_config: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not path.is_file() or not artifact.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    assert not (
        record.get("format") != LAYER_FORMAT or int(record.get("layer", -1)) != layer
    ), f"incompatible layer checkpoint: {path}"
    assert not (record.get("fit_config") != fit_config), (
        f"layer checkpoint configuration changed: {path}"
    )
    assert not (record.get("artifact", {}).get("sha256") != _sha256(artifact)), (
        f"layer artifact hash changed: {artifact}"
    )
    return record


def _write_exact_full_rank_layer(
    *,
    layer: int,
    weight: Tensor,
    fit_source: Mapping[str, Any],
    validation_source: Mapping[str, Any],
    artifact_path: Path,
    record_path: Path,
    fit_config: Mapping[str, Any],
    args: argparse.Namespace,
    factor_dtype: torch.dtype,
    started: float,
    geometry: ValueGeometry,
) -> dict[str, Any]:
    """Materialize the analytic dense endpoint when rank equals head width."""

    assert not (args.cache_rank != geometry.head_dim), (
        "exact dense endpoint requires cache rank equal to head_dim"
    )
    artifact_A = (
        torch.eye(geometry.head_dim, dtype=factor_dtype)
        .unsqueeze(0)
        .repeat(geometry.num_kv_heads, 1, 1)
        .contiguous()
    )
    artifact_D = (
        weight.transpose(0, 1)
        .reshape(geometry.num_heads, geometry.head_dim, geometry.hidden_size)
        .to(dtype=factor_dtype)
        .contiguous()
    )
    artifact_tensors = {
        "value_coordinate_encoders": artifact_A,
        "head_output_decoders": artifact_D,
    }
    _atomic_safetensors(artifact_path, artifact_tensors)
    artifact_metadata = {
        "file": artifact_path.name,
        "sha256": _sha256(artifact_path),
        "tensors": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in artifact_tensors.items()
        },
    }
    checkpoint_exports: dict[str, Any] = {}
    if 0 in map(int, fit_config.get("export_checkpoint_sweeps", ())):
        export_dir = artifact_path.parent / "sweep_000"
        export_dir.mkdir(parents=True, exist_ok=True)
        export_artifact_path = export_dir / artifact_path.name
        _atomic_safetensors(export_artifact_path, artifact_tensors)
        checkpoint_exports["0"] = {
            "boundary": "exact_dense_endpoint",
            "sweep": 0,
            "artifact": {
                **artifact_metadata,
                "sha256": _sha256(export_artifact_path),
            },
            "fit_factor_dtype_relative_mse": 0.0,
            "heldout_factor_dtype_relative_mse": 0.0,
        }
    checkpoint = {
        "boundary": "exact_dense_endpoint",
        "sweep": 0,
        "fit_loss": 0.0,
        "validation_loss": 0.0,
        "validation_relative_mse": 0.0,
    }
    group_heads = geometry.num_heads // geometry.num_kv_heads
    empty_cg = {
        "mode": str(fit_config["encoder_cg_mode"]),
        "relative_tolerance": float(fit_config["encoder_cg_relative_tolerance"]),
        "max_iterations": int(fit_config["encoder_cg_max_iterations"]),
        "aggregate": {
            "group_solves": 0,
            "nonzero_rhs_group_solves": 0,
            "exact_zero_rhs": 0,
            "total_iterations": 0,
            "minimum_iterations_nonzero_rhs": 0,
            "maximum_iterations": 0,
            "converged_at_tolerance": 0,
            "negative_curvature": 0,
            "maximum_relative_residual": 0.0,
            "maximum_predicted_realized_change_error": 0.0,
        },
        "per_sweep": [],
    }
    record = {
        "format": LAYER_FORMAT,
        "layer": layer,
        "fit_config": dict(fit_config),
        "source_snapshot": {
            "fit": dict(fit_source),
            "validation": dict(validation_source),
        },
        "artifact": artifact_metadata,
        "checkpoint": {
            "policy": "exact dense endpoint",
            "boundary": "exact_dense_endpoint",
            "sweep": 0,
            "checkpoints": [checkpoint],
        },
        "checkpoint_exports": checkpoint_exports,
        "fit": {"relative_mse": 0.0, "factor_dtype_relative_mse": 0.0},
        "heldout": {"relative_mse": 0.0, "factor_dtype_relative_mse": 0.0},
        "covariance": {
            "fit_absolute_trace_damping": None,
            "heldout_regularized": False,
            "skipped_for_exact_full_rank_endpoint": True,
        },
        "initialization": [
            {
                "group_index": group,
                "logical_heads": list(
                    range(group * group_heads, (group + 1) * group_heads)
                ),
                "rank": geometry.head_dim,
                "method": "identity_dense_endpoint",
            }
            for group in range(geometry.num_kv_heads)
        ],
        "initialization_summary": {
            "method": "identity_dense_endpoint",
            "requested_method": args.encoder_initialization,
            "base_seed": int(args.encoder_initialization_seed),
            "effective_layer_seed": None,
        },
        "solver": {
            "method": "analytic_exact_full_rank_endpoint",
            "decoder_objective": args.decoder_objective,
            "initial_loss": 0.0,
            "decoder_only_loss": 0.0,
            "endpoint_loss": 0.0,
            "endpoint_relative_mse": 0.0,
            "sweeps": 0,
            "cg": empty_cg,
            "attribution": {
                "anchor_loss": 0.0,
                "decoder_only_loss": 0.0,
                "endpoint_loss": 0.0,
                "initial_decoder_reduction": 0.0,
                "decoder_reduction": 0.0,
                "encoder_reduction": 0.0,
                "total_reduction": 0.0,
                "decoder_fraction": 0.0,
                "encoder_fraction": 0.0,
                "identity_error": 0.0,
                "steps": [],
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(record_path, record)
    print(
        f"[{geometry.model_type} C1] layer={layer} complete exact_dense_endpoint "
        f"seconds={record['elapsed_seconds']:.2f}",
        flush=True,
    )
    return record


@torch.no_grad()
def _fit_layer(
    *,
    layer: int,
    snapshot_dir: Path,
    snapshot_manifest: Mapping[str, Any],
    validation_snapshot_dir: Path,
    validation_snapshot_manifest: Mapping[str, Any],
    output_dir: Path,
    fit_config: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    geometry: ValueGeometry,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifact_path = output_dir / f"layer_{layer:03d}.safetensors"
    record_path = output_dir / f"layer_{layer:03d}.json"
    if args.resume:
        prior = _verified_prior(record_path, artifact_path, layer, fit_config)
        if prior is not None:
            print(
                f"[{geometry.model_type} C1] layer={layer} resume=verified", flush=True
            )
            return prior
    assert not (artifact_path.exists() or record_path.exists()), (
        f"partial layer output exists: {record_path}"
    )

    work_dtype = _dtype(args.work_dtype)
    factor_dtype = _factor_dtype(args.factor_dtype)
    covariance_snapshot = snapshot_manifest.get("format") == COVARIANCE_SNAPSHOT_FORMAT
    if covariance_snapshot:
        assert not (validation_snapshot_dir != snapshot_dir), (
            "covariance snapshots contain their own held-out statistics"
        )
        calibration = snapshot_manifest["calibration"]
        assert not (
            int(calibration.get("fit_windows", -1)) != args.fit_windows
            or int(calibration.get("heldout_windows", -1)) != args.validation_windows
        ), "requested fit/held-out windows differ from covariance snapshot"
        fit_matrix, validation_matrix, weight, fit_source = _load_covariance_layer(
            snapshot_dir, snapshot_manifest, layer, geometry=geometry
        )
        validation_source = fit_source
    else:
        activation, weight, fit_source = _load_snapshot_layer(
            snapshot_dir, snapshot_manifest, layer, geometry=geometry
        )
        if validation_snapshot_dir == snapshot_dir:
            validation_source_activation = activation
            validation_source = fit_source
        else:
            validation_source_activation, validation_weight, validation_source = (
                _load_snapshot_layer(
                    validation_snapshot_dir,
                    validation_snapshot_manifest,
                    layer,
                    geometry=geometry,
                )
            )
            assert not (not torch.equal(weight, validation_weight)), (
                f"fit and validation weights differ at layer {layer}"
            )
            del validation_weight
    if args.cache_rank == geometry.head_dim:
        record = _write_exact_full_rank_layer(
            layer=layer,
            weight=weight,
            fit_source=fit_source,
            validation_source=validation_source,
            artifact_path=artifact_path,
            record_path=record_path,
            fit_config=fit_config,
            args=args,
            factor_dtype=factor_dtype,
            started=started,
            geometry=geometry,
        )
        del weight
        return record
    fit_rows = int(fit_config["fit_rows"])
    validation_rows = int(fit_config["validation_rows"])
    print(
        f"[{geometry.model_type} C1] layer={layer} covariance fit={fit_rows} "
        f"heldout={validation_rows} source="
        f"{'sufficient_statistics' if covariance_snapshot else 'activations'}",
        flush=True,
    )
    covariance_psd_repair: dict[str, Any] | None = None
    if covariance_snapshot:
        fit_matrix, fit_psd_diagnostics = _repair_stored_covariance_psd(
            fit_matrix, device=device
        )
        validation_matrix, heldout_psd_diagnostics = _repair_stored_covariance_psd(
            validation_matrix, device=device
        )
        covariance_psd_repair = {
            "fit": fit_psd_diagnostics,
            "heldout": heldout_psd_diagnostics,
        }
        if fit_psd_diagnostics["applied"] or heldout_psd_diagnostics["applied"]:
            print(
                f"[{geometry.model_type} C1] layer={layer} covariance_psd_repair="
                f"{covariance_psd_repair}",
                flush=True,
            )
        fit_raw = _covariance_matrix_to_blocks(
            fit_matrix, device=device, dtype=work_dtype, geometry=geometry
        )
        fit_evaluation_raw = _covariance_matrix_to_blocks(
            fit_matrix, device=device, dtype=torch.float64, geometry=geometry
        )
        validation_covariance = _covariance_matrix_to_blocks(
            validation_matrix, device=device, dtype=torch.float64, geometry=geometry
        )
        del fit_matrix, validation_matrix
    else:
        validation_row_start = int(fit_config.get("validation_row_start", fit_rows))
        fit_activation = activation[:fit_rows]
        validation_activation = validation_source_activation[
            validation_row_start : validation_row_start + validation_rows
        ]
        fit_raw = _activation_covariance_blocks(
            fit_activation,
            device=device,
            dtype=work_dtype,
            num_heads=geometry.num_heads,
            head_dim=geometry.head_dim,
            row_chunk_size=args.covariance_row_chunk_size,
        )
        validation_covariance = _activation_covariance_blocks(
            validation_activation,
            device=device,
            dtype=work_dtype,
            num_heads=geometry.num_heads,
            head_dim=geometry.head_dim,
            row_chunk_size=args.covariance_row_chunk_size,
        )
        fit_evaluation_raw = fit_raw.to(dtype=torch.float64)
        validation_covariance = validation_covariance.to(dtype=torch.float64)
    fit_covariance, absolute_damping = covariance_with_trace_damping(
        fit_raw, relative_damping=args.covariance_damping
    )
    del fit_raw
    fit_evaluation_covariance, _ = covariance_with_trace_damping(
        fit_evaluation_raw,
        relative_damping=args.covariance_damping,
    )
    del fit_evaluation_raw
    target = _dense_head_targets(
        weight, device=device, dtype=work_dtype, geometry=geometry
    )
    fit_objective = quadratic_from_target(
        covariance=fit_covariance,
        target=target,
        name=f"{geometry.model_type}_c1_fit_layer_{layer:03d}",
        trace_normalize=False,
    )
    fit_evaluation_objective = quadratic_from_target(
        covariance=fit_evaluation_covariance,
        target=target.to(dtype=torch.float64),
        name=f"{geometry.model_type}_c1_fit_evaluation_layer_{layer:03d}",
        trace_normalize=False,
    )
    validation_objective = quadratic_from_target(
        covariance=validation_covariance.to(dtype=torch.float64),
        target=target.to(dtype=torch.float64),
        name=f"{geometry.model_type}_c1_heldout_layer_{layer:03d}",
        trace_normalize=False,
    )
    mapping = _head_to_kv_group(device=device, geometry=geometry)
    if args.decoder_objective == "full_layer":
        solver_objective = fit_objective
        solver_evaluation_objective = fit_evaluation_objective
        decoder_coupling_mode = "full_layer"
        solver_group_ranks: tuple[int, ...] | None = (
            args.cache_rank,
        ) * geometry.num_kv_heads
    elif args.decoder_objective == "per_head":
        independent_covariance = mask_routed_covariance(
            fit_objective.covariance,
            head_to_kv_group=mapping,
            mode="diagonal",
        )
        solver_objective = quadratic_from_target(
            covariance=independent_covariance,
            target=target,
            name=f"{geometry.model_type}_a3_per_head_fit_layer_{layer:03d}",
            trace_normalize=False,
        )
        evaluation_independent_covariance = mask_routed_covariance(
            fit_evaluation_objective.covariance,
            head_to_kv_group=mapping,
            mode="diagonal",
        )
        solver_evaluation_objective = quadratic_from_target(
            covariance=evaluation_independent_covariance,
            target=target.to(dtype=torch.float64),
            name=f"{geometry.model_type}_a3_per_head_fit_evaluation_layer_{layer:03d}",
            trace_normalize=False,
        )
        decoder_coupling_mode = "diagonal"
        # Uniform ranks do not need the ragged solver, whose decoder path is
        # intentionally restricted to full-layer coupling.
        solver_group_ranks = None
    else:  # Defensive against programmatic callers that bypass argparse.
        assert False, f"unsupported decoder objective: {args.decoder_objective}"
    initialization = None
    effective_initialization_seed: int | None = None
    if args.encoder_initialization == "activation-weighted-svd":
        initialization = initialize_group_pooled_routed_svd(
            covariance=fit_objective.covariance,
            target=target,
            head_to_kv_group=mapping,
            group_ranks=(args.cache_rank,) * geometry.num_kv_heads,
            covariance_ridge=args.covariance_damping,
        )
        initial_A = initialization.A_unique
        initialization_groups = [asdict(item) for item in initialization.groups]
    elif args.encoder_initialization == "weight-only-svd":
        initial_A, initialization_groups = _weight_only_encoder_initialization(
            target, rank=args.cache_rank, mapping=mapping, geometry=geometry
        )
    elif args.encoder_initialization == "random-orthogonal":
        effective_initialization_seed = (
            int(args.encoder_initialization_seed) + 1_000_003 * layer
        )
        initial_A, initialization_groups = _random_orthogonal_encoder_initialization(
            rank=args.cache_rank,
            seed=effective_initialization_seed,
            device=device,
            dtype=work_dtype,
            geometry=geometry,
        )
    else:  # Defensive against programmatic callers that bypass argparse.
        assert False, (
            f"unsupported encoder initialization: {args.encoder_initialization}"
        )
    initial_D = torch.zeros(
        geometry.num_heads,
        args.cache_rank,
        geometry.hidden_size,
        device=device,
        dtype=work_dtype,
    )
    recorder = _CheckpointRecorder(
        fit_evaluation_objective,
        validation_objective,
        mapping,
        export_sweeps=_parse_checkpoint_sweeps(
            args.export_checkpoint_sweeps,
            args.encoder_sweeps,
        ),
        export_root=output_dir,
        layer=layer,
        factor_dtype=factor_dtype,
    )
    print(
        f"[{geometry.model_type} C1] layer={layer} "
        f"encoder_initialization={args.encoder_initialization} "
        f"decoder_objective={args.decoder_objective}",
        flush=True,
    )
    result = fit_routed_ov_joint(
        objective=solver_objective,
        evaluation_objective=solver_evaluation_objective,
        initial_A=initial_A,
        initial_D=initial_D,
        head_to_kv_group=mapping,
        coupling_mode=decoder_coupling_mode,
        maximum_sweeps=args.encoder_sweeps,
        minimum_sweeps=args.encoder_sweeps,
        relative_objective_tolerance=0.0,
        patience=1,
        decoder_relative_jitter=0.0,
        encoder_relative_damping=0.0,
        cg_relative_tolerance=float(fit_config["encoder_cg_relative_tolerance"]),
        cg_max_iterations=int(fit_config["encoder_cg_max_iterations"]),
        cg_fixed_iterations=fit_config["encoder_cg_mode"] == "fixed",
        maximum_backtracks=0,
        final_decoder_solve=True,
        verify_encoder_step_objective=work_dtype == torch.float64,
        group_ranks=solver_group_ranks,
        checkpoint_callback=recorder,
        decoder_stationarity_override=lambda *_: 0.0,
        work_dtype=work_dtype,
        work_device=device,
    )
    selected_A = result.A_unique.to(device=device, dtype=work_dtype)
    selected_D = result.D_heads.to(device=device, dtype=work_dtype)
    endpoint = result.checkpoints[-1]
    expected_boundary = (
        "decoder_only" if args.encoder_sweeps == 0 else "after_redecoder"
    )
    assert endpoint.boundary == expected_boundary
    assert endpoint.sweep == args.encoder_sweeps
    fit_loss = evaluate_quadratic(
        fit_evaluation_objective, selected_A, selected_D, mapping
    )
    validation_loss = evaluate_quadratic(
        validation_objective, selected_A, selected_D, mapping
    )
    artifact_A = selected_A.to(device="cpu", dtype=factor_dtype).contiguous()
    artifact_D = selected_D.to(device="cpu", dtype=factor_dtype).contiguous()
    artifact_fit_loss = evaluate_quadratic(
        fit_evaluation_objective,
        artifact_A.to(device=device, dtype=work_dtype),
        artifact_D.to(device=device, dtype=work_dtype),
        mapping,
    )
    artifact_validation_loss = evaluate_quadratic(
        validation_objective,
        artifact_A.to(device=device, dtype=work_dtype),
        artifact_D.to(device=device, dtype=work_dtype),
        mapping,
    )
    artifact_tensors = {
        "value_coordinate_encoders": artifact_A,
        "head_output_decoders": artifact_D,
    }
    _atomic_safetensors(artifact_path, artifact_tensors)
    record = {
        "format": LAYER_FORMAT,
        "layer": layer,
        "fit_config": dict(fit_config),
        "source_snapshot": {
            "fit": fit_source,
            "validation": validation_source,
        },
        "artifact": {
            "file": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "tensors": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in artifact_tensors.items()
            },
        },
        "checkpoint": {
            "policy": fit_config["checkpoint_policy"],
            "boundary": endpoint.boundary,
            "sweep": endpoint.sweep,
            "checkpoints": recorder.records,
        },
        "checkpoint_exports": {
            str(sweep): export for sweep, export in sorted(recorder.exports.items())
        },
        "fit": {
            "relative_mse": _relative_loss(fit_loss, fit_objective.constant),
            "factor_dtype_relative_mse": _relative_loss(
                artifact_fit_loss, fit_objective.constant
            ),
        },
        "heldout": {
            "relative_mse": _relative_loss(
                validation_loss, validation_objective.constant
            ),
            "factor_dtype_relative_mse": _relative_loss(
                artifact_validation_loss, validation_objective.constant
            ),
        },
        "covariance": {
            "fit_absolute_trace_damping": absolute_damping,
            "heldout_regularized": False,
            "stored_gram_psd_repair": covariance_psd_repair,
        },
        "initialization": initialization_groups,
        "initialization_summary": {
            "method": args.encoder_initialization,
            "base_seed": int(args.encoder_initialization_seed),
            "effective_layer_seed": effective_initialization_seed,
        },
        "solver": {
            "decoder_objective": args.decoder_objective,
            "initial_loss": result.initial_loss,
            "decoder_only_loss": result.decoder_only_loss,
            "endpoint_loss": result.final_loss,
            "endpoint_relative_mse": _relative_loss(
                result.final_loss, solver_objective.constant
            ),
            "sweeps": len(result.sweeps),
            "cg": _cg_diagnostics_payload(
                result,
                mode=str(fit_config["encoder_cg_mode"]),
                relative_tolerance=float(fit_config["encoder_cg_relative_tolerance"]),
                max_iterations=int(fit_config["encoder_cg_max_iterations"]),
            ),
            "attribution": asdict(result.attribution),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(record_path, record)
    print(
        f"[{geometry.model_type} C1] layer={layer} complete "
        f"heldout={record['heldout']['factor_dtype_relative_mse']:.8g} "
        f"seconds={record['elapsed_seconds']:.2f}",
        flush=True,
    )
    del (
        weight,
        fit_covariance,
        validation_covariance,
        target,
        fit_objective,
        solver_objective,
        validation_objective,
        initialization,
        initialization_groups,
        initial_A,
        initial_D,
        result,
        selected_A,
        selected_D,
        artifact_A,
        artifact_D,
        artifact_tensors,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def fit_shard(args: argparse.Namespace, *, geometry: ValueGeometry) -> None:
    assert not (
        args.layer_shard_count <= 0
        or not 0 <= args.layer_shard_index < args.layer_shard_count
    ), "invalid layer shard"
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    device = torch.device(args.device)
    assert device.type in ("cpu", "cuda")
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA is unavailable"
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = _snapshot_manifest(snapshot_dir, geometry=geometry)
    validation_snapshot_dir = (
        Path(args.validation_snapshot_dir).expanduser().resolve()
        if args.validation_snapshot_dir is not None
        else snapshot_dir
    )
    validation_manifest = (
        _snapshot_manifest(validation_snapshot_dir, geometry=geometry)
        if validation_snapshot_dir != snapshot_dir
        else manifest
    )
    fit_config = _fit_config(
        args,
        snapshot_dir,
        manifest,
        validation_snapshot_dir=(
            validation_snapshot_dir if validation_snapshot_dir != snapshot_dir else None
        ),
        validation_manifest=(
            validation_manifest if validation_snapshot_dir != snapshot_dir else None
        ),
        geometry=geometry,
    )
    requested = _parse_layers(args.layers, geometry=geometry)
    layers = tuple(
        layer
        for position, layer in enumerate(requested)
        if position % args.layer_shard_count == args.layer_shard_index
    )
    assert not (not layers), "layer shard is empty"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()
    for layer in layers:
        records.append(
            _fit_layer(
                layer=layer,
                snapshot_dir=snapshot_dir,
                snapshot_manifest=manifest,
                validation_snapshot_dir=validation_snapshot_dir,
                validation_snapshot_manifest=validation_manifest,
                output_dir=output_dir,
                fit_config=fit_config,
                args=args,
                device=device,
                geometry=geometry,
            )
        )
    shard = {
        "format": FORMAT,
        "status": "shard_complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "layer_shard_index": args.layer_shard_index,
        "layer_shard_count": args.layer_shard_count,
        "layers": list(layers),
        "fit_config": fit_config,
        "records": [int(row["layer"]) for row in records],
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None,
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_dir / f"shard_{args.layer_shard_index:02d}.json", shard)
    print(
        f"[{geometry.model_type} C1] shard={args.layer_shard_index}/{args.layer_shard_count} complete",
        flush=True,
    )


def _summary(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    geometry: ValueGeometry,
) -> str:
    heldout = [float(row["heldout"]["factor_dtype_relative_mse"]) for row in records]
    cache_rank = int(config["cache_rank_per_head"])
    total_retention = float(config["total_kv_retained_ratio_with_dense_k"])
    lines = [
        f"# {geometry.model_type} {geometry.attention_type.upper()} C1 V{cache_rank} joint fit",
        "",
        (
            f"- K remains dense; every one of {geometry.num_kv_heads} physical V heads retains "
            f"rank {cache_rank}/{geometry.head_dim}."
        ),
        (
            f"- Total KV-cache retention is {100 * total_retention:.6g}% "
            f"({100 * (1 - total_retention):.6g}% reduction)."
        ),
        f"- Encoder initialization: `{config.get('encoder_initialization', 'activation-weighted-svd')}`.",
        f"- Checkpoint policy: `{config['checkpoint_policy']}`.",
        (
            "- V encoders and head output decoders use a full-layer joint solve "
            "with cross-head covariance."
            if config.get("decoder_objective", "full_layer") == "full_layer"
            else "- V encoders use the same attention-output-aware initialization; "
            "each head decoder is then solved independently without cross-head covariance."
        ),
        f"- Fit contexts: {config['fit_windows']}; held-out diagnostic contexts: {config['validation_windows']}.",
        f"- Mean held-out factor-dtype relative MSE: `{sum(heldout) / len(heldout):.9g}`.",
        "",
        "| Layer | Exported boundary | Sweep | Held-out relative MSE |",
        "|---:|:---|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['layer']} | {row['checkpoint']['boundary']} | "
            f"{row['checkpoint']['sweep']} | "
            f"{row['heldout']['factor_dtype_relative_mse']:.9g} |"
        )
    lines.append("")
    return "\n".join(lines)


def _merge_exported_checkpoints(
    *,
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    fit_config: Mapping[str, Any],
    parent_result_path: Path,
    geometry: ValueGeometry,
) -> None:
    requested = tuple(map(int, fit_config.get("export_checkpoint_sweeps", ())))
    if not requested:
        return
    parent_sha256 = _sha256(parent_result_path)
    for sweep in requested:
        export_dir = output_dir / f"sweep_{sweep:03d}"
        exported_records = []
        for source in records:
            layer = int(source["layer"])
            exported = source["checkpoint_exports"][str(sweep)]
            checkpoint_config = dict(fit_config)
            checkpoint_config["encoder_sweeps"] = sweep
            checkpoint_config["checkpoint_policy"] = (
                "initialization decoder closure with zero encoder sweeps"
                if sweep == 0
                else f"fixed decoder-refitted endpoint after encoder sweep {sweep}"
            )
            record = {
                "format": LAYER_FORMAT,
                "layer": layer,
                "fit_config": checkpoint_config,
                "source_snapshot": source["source_snapshot"],
                "artifact": exported["artifact"],
                "checkpoint": {
                    "policy": checkpoint_config["checkpoint_policy"],
                    "boundary": exported["boundary"],
                    "sweep": sweep,
                },
                "fit": {
                    "factor_dtype_relative_mse": exported[
                        "fit_factor_dtype_relative_mse"
                    ]
                },
                "heldout": {
                    "factor_dtype_relative_mse": exported[
                        "heldout_factor_dtype_relative_mse"
                    ]
                },
                "initialization": source["initialization"],
                "initialization_summary": source["initialization_summary"],
                "source_run": {
                    "results": str(parent_result_path),
                    "results_sha256": parent_sha256,
                },
            }
            _atomic_json(export_dir / f"layer_{layer:03d}.json", record)
            exported_records.append(record)
        checkpoint_config = dict(exported_records[0]["fit_config"])
        payload = {
            "format": FORMAT,
            "status": "complete",
            "command": shlex.join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "layers": [int(row["layer"]) for row in exported_records],
            "fit_config": checkpoint_config,
            "artifacts": {
                str(row["layer"]): row["artifact"] for row in exported_records
            },
            "records": exported_records,
            "aggregate": {
                "mean_fit_factor_dtype_relative_mse": sum(
                    float(row["fit"]["factor_dtype_relative_mse"])
                    for row in exported_records
                )
                / len(exported_records),
                "mean_heldout_factor_dtype_relative_mse": sum(
                    float(row["heldout"]["factor_dtype_relative_mse"])
                    for row in exported_records
                )
                / len(exported_records),
            },
            "source_run": {
                "results": str(parent_result_path),
                "results_sha256": parent_sha256,
            },
        }
        result_path = export_dir / "results.json"
        _atomic_json(result_path, payload)
        (export_dir / "summary.md").write_text(
            _summary(exported_records, checkpoint_config, geometry=geometry),
            encoding="utf-8",
        )
        print(
            f"[{geometry.model_type} C1] exported decoder-closed sweep={sweep} "
            f"to {result_path}",
            flush=True,
        )


def merge(args: argparse.Namespace, *, geometry: ValueGeometry) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    assert not (not output_dir.is_dir()), output_dir
    result_path = output_dir / "results.json"
    assert not (result_path.exists()), result_path
    layers = _parse_layers(args.layers, geometry=geometry)
    records = []
    fit_config = None
    for layer in layers:
        record_path = output_dir / f"layer_{layer:03d}.json"
        artifact_path = output_dir / f"layer_{layer:03d}.safetensors"
        assert not (not record_path.is_file() or not artifact_path.is_file()), (
            f"missing layer {layer} output"
        )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert not (
            record.get("format") != LAYER_FORMAT
            or int(record.get("layer", -1)) != layer
        ), f"invalid layer record {record_path}"
        assert not (record["artifact"]["sha256"] != _sha256(artifact_path)), (
            f"artifact hash mismatch at layer {layer}"
        )
        if fit_config is None:
            fit_config = record["fit_config"]
        assert not (record["fit_config"] != fit_config), (
            "layer fit configurations differ"
        )
        records.append(record)
    assert fit_config is not None
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "layers": list(layers),
        "fit_config": fit_config,
        "artifacts": {str(row["layer"]): row["artifact"] for row in records},
        "records": records,
        "aggregate": {
            "mean_fit_factor_dtype_relative_mse": sum(
                float(row["fit"]["factor_dtype_relative_mse"]) for row in records
            )
            / len(records),
            "mean_heldout_factor_dtype_relative_mse": sum(
                float(row["heldout"]["factor_dtype_relative_mse"]) for row in records
            )
            / len(records),
        },
    }
    _atomic_json(result_path, payload)
    (output_dir / "summary.md").write_text(
        _summary(records, fit_config, geometry=geometry), encoding="utf-8"
    )
    _merge_exported_checkpoints(
        output_dir=output_dir,
        records=records,
        fit_config=fit_config,
        parent_result_path=result_path,
        geometry=geometry,
    )
    print(
        f"[{geometry.model_type} C1] merged {len(records)} layers into {result_path}",
        flush=True,
    )
