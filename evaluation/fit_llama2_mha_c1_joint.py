#!/usr/bin/env python3
"""Fit foldable Llama-2-7B MHA C1 V/Wo factors.

The dense attention ``o_proj`` input is split into 32 head blocks.  Each head
keeps a rank-96 Value coordinate encoder, while all 32 output decoders are
solved jointly against the complete attention-output MSE, including
cross-head error cancellation.  Encoders and decoders alternate under the
same deterministic routed-OV solver used by the Qwen3 C1 experiment.

The existing 64-document C4 snapshot is split by document: the first 48
windows fit factors and the final 16 windows select the earliest minimum-MSE
solver checkpoint.  The final WikiText-2 test set is never touched here.
"""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.gqa_routed_ov_joint import (  # noqa: E402
    covariance_with_trace_damping,
    evaluate_quadratic,
    fit_routed_ov_joint,
    initialize_group_pooled_routed_svd,
    mask_routed_covariance,
    quadratic_from_target,
)


FORMAT = "basisserve.llama2_7b.mha_c1_v96_joint.v1"
LAYER_FORMAT = "basisserve.llama2_7b.mha_c1_v96_joint.layer.v1"
SNAPSHOT_FORMAT = "basisserve.attention_o_proj_ppl_snapshots.v1"
NUM_LAYERS = 32
NUM_HEADS = 32
HEAD_DIM = 128
HIDDEN_SIZE = 4096


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
            "converged_at_tolerance": sum(
                step["converged"] for step in flat_steps
            ),
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


def _parse_layers(raw: str) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return tuple(range(NUM_LAYERS))
    layers: set[int] = set()
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            first, last = map(int, piece.split("-", 1))
            if last < first:
                raise ValueError(f"descending layer range: {piece}")
            layers.update(range(first, last + 1))
        else:
            layers.add(int(piece))
    selected = tuple(sorted(layers))
    if not selected or min(selected) < 0 or max(selected) >= NUM_LAYERS:
        raise ValueError("selected layers are outside Llama-2-7B")
    return selected


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

    if value.ndim != 2:
        raise ValueError("basis sign canonicalization requires a matrix")
    if value.shape[1] == 0:
        return value
    pivots = value.abs().argmax(dim=0)
    columns = torch.arange(value.shape[1], device=value.device)
    signs = torch.sign(value[pivots, columns])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return value * signs.unsqueeze(0)


@torch.no_grad()
def _weight_only_encoder_initialization(
    target: Tensor,
    *,
    rank: int,
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Return independent weight-only left singular subspaces for all heads."""

    if target.ndim != 3 or tuple(target.shape[:2]) != (NUM_HEADS, HEAD_DIM):
        raise ValueError("weight-only initialization received an invalid target")
    if not 0 < rank <= HEAD_DIM:
        raise ValueError("weight-only initialization rank is invalid")
    encoders = target.new_empty(NUM_HEADS, HEAD_DIM, rank)
    diagnostics: list[dict[str, Any]] = []
    for head in range(NUM_HEADS):
        block = target[head]
        gram = block @ block.T
        gram = 0.5 * (gram + gram.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues.index_select(0, order).clamp_min(0)
        basis = _canonicalize_basis_signs(
            eigenvectors.index_select(1, order[:rank])
        )
        encoders[head].copy_(basis)
        singular_values = torch.sqrt(eigenvalues)
        total_energy = float(eigenvalues.sum())
        tail_energy = float(eigenvalues[rank:].sum())
        diagnostics.append(
            {
                "method": "weight-only-svd",
                "group_index": head,
                "head_indices": [head],
                "rank": rank,
                "leading_singular_values": singular_values[:rank].tolist(),
                "boundary_singular_value": float(singular_values[rank - 1]),
                "next_singular_value": (
                    float(singular_values[rank]) if rank < HEAD_DIM else None
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
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Return reproducible independent Haar-subspace controls for all heads."""

    if not 0 < rank <= HEAD_DIM:
        raise ValueError("random initialization rank is invalid")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch.randn(
        NUM_HEADS,
        HEAD_DIM,
        rank,
        generator=generator,
        device="cpu",
        dtype=torch.float64,
    )
    encoders = torch.empty_like(random)
    diagnostics: list[dict[str, Any]] = []
    for head in range(NUM_HEADS):
        basis, _ = torch.linalg.qr(random[head], mode="reduced")
        basis = _canonicalize_basis_signs(basis)
        encoders[head].copy_(basis)
        diagnostics.append(
            {
                "method": "random-orthogonal",
                "group_index": head,
                "head_indices": [head],
                "rank": rank,
                "layer_seed": seed,
            }
        )
    return encoders.to(device=device, dtype=dtype), diagnostics


def _snapshot_manifest(snapshot_dir: Path) -> dict[str, Any]:
    path = snapshot_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != SNAPSHOT_FORMAT:
        raise ValueError("incompatible attention snapshot format")
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
        "llama",
        "mha",
        NUM_LAYERS,
        NUM_HEADS,
        NUM_HEADS,
        HEAD_DIM,
        HIDDEN_SIZE,
    )
    if observed != expected:
        raise ValueError(f"unexpected Llama snapshot geometry: {observed}")
    if tuple(map(int, manifest.get("layers", ()))) != tuple(range(NUM_LAYERS)):
        raise ValueError("snapshot must cover every Llama decoder layer")
    return manifest


def _resolve_window_partition(
    *,
    snapshot_windows: int,
    fit_windows: int,
    validation_windows: int,
    validation_window_start: int | None,
) -> int:
    if min(snapshot_windows, fit_windows, validation_windows) <= 0:
        raise ValueError("snapshot, fit, and validation partitions must be nonempty")
    if validation_window_start is None:
        if fit_windows + validation_windows != snapshot_windows:
            raise ValueError(
                "fit and validation windows must exactly partition the snapshot "
                "unless --validation-window-start is explicit"
            )
        return fit_windows
    start = int(validation_window_start)
    if start < fit_windows:
        raise ValueError("validation windows overlap the leading fit partition")
    if start + validation_windows > snapshot_windows:
        raise ValueError("validation windows exceed the snapshot")
    return start


def _snapshot_shape(manifest: Mapping[str, Any]) -> tuple[int, int, int]:
    calibration = manifest["calibration"]
    rows = int(calibration["rows_per_layer"])
    windows = int(calibration["window_count"])
    positions = int(calibration["positions_per_window"])
    if rows != windows * positions:
        raise ValueError("snapshot rows do not factor into windows and positions")
    return rows, windows, positions


def _fit_config(
    args: argparse.Namespace,
    snapshot_dir: Path,
    manifest: Mapping[str, Any],
    *,
    validation_snapshot_dir: Path | None = None,
    validation_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cg_max_iterations = (
        args.encoder_cg_max_iterations
        if args.encoder_cg_max_iterations is not None
        else args.encoder_cg_fixed_iterations
    )
    if cg_max_iterations <= 0:
        raise ValueError("encoder CG maximum iterations must be positive")
    if args.encoder_cg_relative_tolerance <= 0:
        raise ValueError("encoder CG relative tolerance must be positive")
    if args.encoder_initialization_seed < 0:
        raise ValueError("encoder initialization seed must be non-negative")
    _, windows, positions = _snapshot_shape(manifest)
    if not 0 < args.fit_windows <= windows:
        raise ValueError("fit windows exceed the fit snapshot")
    separate_validation = validation_manifest is not None
    if separate_validation != (validation_snapshot_dir is not None):
        raise ValueError("validation snapshot path and manifest must be provided together")
    if separate_validation:
        assert validation_manifest is not None
        assert validation_snapshot_dir is not None
        _, validation_snapshot_windows, validation_positions = _snapshot_shape(
            validation_manifest
        )
        if (
            manifest["model"]["config_sha256"]
            != validation_manifest["model"]["config_sha256"]
            or manifest["model"]["path"] != validation_manifest["model"]["path"]
        ):
            raise ValueError("fit and validation snapshots use different models")
        validation_window_start = int(args.validation_window_start or 0)
        if (
            validation_window_start < 0
            or validation_window_start + args.validation_windows
            > validation_snapshot_windows
        ):
            raise ValueError("validation windows exceed the validation snapshot")
    else:
        validation_snapshot_windows = windows
        validation_positions = positions
        validation_window_start = _resolve_window_partition(
            snapshot_windows=windows,
            fit_windows=args.fit_windows,
            validation_windows=args.validation_windows,
            validation_window_start=args.validation_window_start,
        )
    if not 0 < args.cache_rank <= HEAD_DIM:
        raise ValueError("cache rank is outside the MHA head width")
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
        "total_v_cache_rank": NUM_HEADS * args.cache_rank,
        "dense_v_cache_rank": NUM_HEADS * HEAD_DIM,
        "v_retained_ratio": args.cache_rank / HEAD_DIM,
        "total_kv_retained_ratio_with_dense_k": (
            NUM_HEADS * HEAD_DIM + NUM_HEADS * args.cache_rank
        )
        / (2 * NUM_HEADS * HEAD_DIM),
        "work_dtype": args.work_dtype,
        "factor_dtype": args.factor_dtype,
        "encoder_initialization": args.encoder_initialization,
        "encoder_initialization_seed": args.encoder_initialization_seed,
        "selection_boundaries": args.selection_boundaries,
        "covariance_damping": args.covariance_damping,
        "encoder_sweeps": args.encoder_sweeps,
        "minimum_encoder_sweeps": args.minimum_encoder_sweeps,
        "encoder_relative_tolerance": args.encoder_relative_tolerance,
        "encoder_patience": args.encoder_patience,
        "decoder_relative_jitter": args.decoder_relative_jitter,
        "encoder_relative_damping": args.encoder_relative_damping,
        "encoder_cg_mode": args.encoder_cg_mode,
        "encoder_cg_relative_tolerance": args.encoder_cg_relative_tolerance,
        "encoder_cg_max_iterations": cg_max_iterations,
        "encoder_cg_fixed_iterations": (
            cg_max_iterations if args.encoder_cg_mode == "fixed" else None
        ),
        "maximum_backtracks": args.maximum_backtracks,
        "covariance_row_chunk_size": args.covariance_row_chunk_size,
        "decoder_objective": args.decoder_objective,
        "selection": _selection_criterion(args.selection_boundaries),
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
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    record = manifest["artifacts"][str(layer)]
    path = snapshot_dir / record["file"]
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"snapshot hash mismatch at layer {layer}")
    payload = load_file(str(path), device="cpu")
    activation = payload["activation"].contiguous()
    weight = payload["weight"].contiguous()
    expected_rows = int(manifest["calibration"]["rows_per_layer"])
    if tuple(activation.shape) != (expected_rows, HIDDEN_SIZE):
        raise ValueError(f"unexpected activation shape at layer {layer}")
    if tuple(weight.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError(f"unexpected o_proj shape at layer {layer}")
    return activation, weight, {"path": str(path), **record}


@torch.no_grad()
def _activation_covariance_blocks(
    activation: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
    row_chunk_size: int | None = None,
) -> Tensor:
    """Return E[x_h^T x_k] as [head, head, d, d] blocks."""

    hidden_size = num_heads * head_dim
    if activation.ndim != 2 or int(activation.shape[1]) != hidden_size:
        raise ValueError(f"activation must have shape [rows, {hidden_size}]")
    rows = int(activation.shape[0])
    if row_chunk_size is not None and row_chunk_size <= 0:
        raise ValueError("covariance row chunk size must be positive")
    if row_chunk_size is None or row_chunk_size >= rows:
        work = activation.to(device=device, dtype=dtype)
        gram = work.transpose(0, 1) @ work
    else:
        gram = torch.zeros(
            hidden_size, hidden_size, device=device, dtype=dtype
        )
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


def _dense_head_targets(weight: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if tuple(weight.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError("dense o_proj weight must be 4096 square")
    return (
        weight.to(device=device, dtype=dtype)
        .transpose(0, 1)
        .reshape(NUM_HEADS, HEAD_DIM, HIDDEN_SIZE)
        .contiguous()
    )


def _relative_loss(loss: float, constant: Tensor) -> float:
    return float(loss) / max(abs(float(constant)), 1.0e-300)


DECODER_CLOSED_BOUNDARIES = frozenset(("decoder_only", "after_redecoder"))


def _selection_criterion(selection_boundaries: str) -> str:
    if selection_boundaries == "all":
        return "earliest minimum heldout-context aggregate-output MSE checkpoint"
    if selection_boundaries == "decoder-closed":
        return (
            "earliest minimum heldout-context aggregate-output MSE among "
            "decoder-closed checkpoints"
        )
    raise ValueError(f"unsupported selection boundary policy: {selection_boundaries}")


class _HeldoutSelector:
    def __init__(
        self,
        validation_objective: Any,
        mapping: Tensor,
        *,
        selection_boundaries: str,
    ) -> None:
        self.validation_objective = validation_objective
        self.mapping = mapping
        self.selection_boundaries = selection_boundaries
        if selection_boundaries == "all":
            self.allowed_boundaries: frozenset[str] | None = None
        elif selection_boundaries == "decoder-closed":
            self.allowed_boundaries = DECODER_CLOSED_BOUNDARIES
        else:
            raise ValueError(
                f"unsupported selection boundary policy: {selection_boundaries}"
            )
        self.records: list[dict[str, Any]] = []
        self.best_loss = float("inf")
        self.best_A: Tensor | None = None
        self.best_D: Tensor | None = None
        self.best_boundary: str | None = None
        self.best_sweep: int | None = None

    def __call__(self, checkpoint: Any, A: Tensor, D: Tensor) -> None:
        heldout = evaluate_quadratic(
            self.validation_objective,
            A,
            D,
            self.mapping,
        )
        eligible = (
            self.allowed_boundaries is None
            or str(checkpoint.boundary) in self.allowed_boundaries
        )
        row = {
            "boundary": checkpoint.boundary,
            "sweep": int(checkpoint.sweep),
            "fit_loss": float(checkpoint.loss),
            "validation_loss": float(heldout),
            "validation_relative_mse": _relative_loss(
                heldout, self.validation_objective.constant
            ),
            "selection_eligible": eligible,
        }
        self.records.append(row)
        if eligible and float(heldout) < self.best_loss:
            self.best_loss = float(heldout)
            self.best_A = A.detach().clone()
            self.best_D = D.detach().clone()
            self.best_boundary = str(checkpoint.boundary)
            self.best_sweep = int(checkpoint.sweep)

    def selected(self) -> tuple[Tensor, Tensor]:
        if self.best_A is None or self.best_D is None:
            raise RuntimeError(
                "held-out selector observed no eligible solver checkpoints"
            )
        return self.best_A, self.best_D


def _verified_prior(path: Path, artifact: Path, layer: int, fit_config: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file() or not artifact.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("format") != LAYER_FORMAT or int(record.get("layer", -1)) != layer:
        raise ValueError(f"incompatible layer checkpoint: {path}")
    if record.get("fit_config") != fit_config:
        raise ValueError(f"layer checkpoint configuration changed: {path}")
    if record.get("artifact", {}).get("sha256") != _sha256(artifact):
        raise ValueError(f"layer artifact hash changed: {artifact}")
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
) -> dict[str, Any]:
    started = time.perf_counter()
    artifact_path = output_dir / f"layer_{layer:03d}.safetensors"
    record_path = output_dir / f"layer_{layer:03d}.json"
    if args.resume:
        prior = _verified_prior(record_path, artifact_path, layer, fit_config)
        if prior is not None:
            print(f"[Llama C1] layer={layer} resume=verified", flush=True)
            return prior
    elif artifact_path.exists() or record_path.exists():
        raise FileExistsError(f"partial layer output exists: {record_path}")

    work_dtype = _dtype(args.work_dtype)
    factor_dtype = _factor_dtype(args.factor_dtype)
    activation, weight, fit_source = _load_snapshot_layer(
        snapshot_dir, snapshot_manifest, layer
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
            )
        )
        if not torch.equal(weight, validation_weight):
            raise ValueError(f"fit and validation weights differ at layer {layer}")
        del validation_weight
    fit_rows = int(fit_config["fit_rows"])
    validation_rows = int(fit_config["validation_rows"])
    validation_row_start = int(
        fit_config.get("validation_row_start", fit_rows)
    )
    fit_activation = activation[:fit_rows]
    validation_activation = validation_source_activation[
        validation_row_start : validation_row_start + validation_rows
    ]
    print(
        f"[Llama C1] layer={layer} covariance fit={len(fit_activation)} "
        f"heldout={len(validation_activation)}",
        flush=True,
    )
    fit_raw = _activation_covariance_blocks(
        fit_activation,
        device=device,
        dtype=work_dtype,
        row_chunk_size=args.covariance_row_chunk_size,
    )
    fit_covariance, absolute_damping = covariance_with_trace_damping(
        fit_raw, relative_damping=args.covariance_damping
    )
    del fit_raw
    validation_covariance = _activation_covariance_blocks(
        validation_activation,
        device=device,
        dtype=work_dtype,
        row_chunk_size=args.covariance_row_chunk_size,
    )
    target = _dense_head_targets(weight, device=device, dtype=work_dtype)
    fit_objective = quadratic_from_target(
        covariance=fit_covariance,
        target=target,
        name=f"llama2_mha_c1_v96_fit_layer_{layer:03d}",
        trace_normalize=False,
    )
    validation_objective = quadratic_from_target(
        covariance=validation_covariance,
        target=target,
        name=f"llama2_mha_c1_v96_heldout_layer_{layer:03d}",
        trace_normalize=False,
    )
    mapping = torch.arange(NUM_HEADS, device=device, dtype=torch.long)
    if args.decoder_objective == "full_layer":
        solver_objective = fit_objective
        decoder_coupling_mode = "full_layer"
        solver_group_ranks: tuple[int, ...] | None = (
            (args.cache_rank,) * NUM_HEADS
        )
    elif args.decoder_objective == "per_head":
        independent_covariance = mask_routed_covariance(
            fit_objective.covariance,
            head_to_kv_group=mapping,
            mode="diagonal",
        )
        solver_objective = quadratic_from_target(
            covariance=independent_covariance,
            target=target,
            name=f"llama2_mha_a3_per_head_v96_fit_layer_{layer:03d}",
            trace_normalize=False,
        )
        decoder_coupling_mode = "diagonal"
        # Uniform ranks do not need the ragged solver, whose decoder path is
        # intentionally restricted to full-layer coupling.
        solver_group_ranks = None
    else:  # Defensive against programmatic callers that bypass argparse.
        raise ValueError(f"unsupported decoder objective: {args.decoder_objective}")
    initialization = None
    effective_initialization_seed: int | None = None
    if args.encoder_initialization == "activation-weighted-svd":
        initialization = initialize_group_pooled_routed_svd(
            covariance=fit_objective.covariance,
            target=target,
            head_to_kv_group=mapping,
            group_ranks=(args.cache_rank,) * NUM_HEADS,
            covariance_ridge=args.covariance_damping,
        )
        initial_A = initialization.A_unique
        initialization_groups = [asdict(item) for item in initialization.groups]
    elif args.encoder_initialization == "weight-only-svd":
        initial_A, initialization_groups = _weight_only_encoder_initialization(
            target,
            rank=args.cache_rank,
        )
    elif args.encoder_initialization == "random-orthogonal":
        effective_initialization_seed = (
            int(args.encoder_initialization_seed) + 1_000_003 * layer
        )
        initial_A, initialization_groups = (
            _random_orthogonal_encoder_initialization(
                rank=args.cache_rank,
                seed=effective_initialization_seed,
                device=device,
                dtype=work_dtype,
            )
        )
    else:  # Defensive against programmatic callers that bypass argparse.
        raise ValueError(
            f"unsupported encoder initialization: {args.encoder_initialization}"
        )
    initial_D = torch.zeros(
        NUM_HEADS,
        args.cache_rank,
        HIDDEN_SIZE,
        device=device,
        dtype=work_dtype,
    )
    selector = _HeldoutSelector(
        validation_objective,
        mapping,
        selection_boundaries=args.selection_boundaries,
    )
    print(
        f"[Llama C1] layer={layer} "
        f"encoder_initialization={args.encoder_initialization} "
        f"decoder_objective={args.decoder_objective}",
        flush=True,
    )
    result = fit_routed_ov_joint(
        objective=solver_objective,
        initial_A=initial_A,
        initial_D=initial_D,
        head_to_kv_group=mapping,
        coupling_mode=decoder_coupling_mode,
        maximum_sweeps=args.encoder_sweeps,
        minimum_sweeps=args.minimum_encoder_sweeps,
        relative_objective_tolerance=args.encoder_relative_tolerance,
        patience=args.encoder_patience,
        decoder_relative_jitter=args.decoder_relative_jitter,
        encoder_relative_damping=args.encoder_relative_damping,
        cg_relative_tolerance=float(
            fit_config["encoder_cg_relative_tolerance"]
        ),
        cg_max_iterations=int(fit_config["encoder_cg_max_iterations"]),
        cg_fixed_iterations=fit_config["encoder_cg_mode"] == "fixed",
        maximum_backtracks=args.maximum_backtracks,
        final_decoder_solve=True,
        verify_encoder_step_objective=work_dtype == torch.float64,
        group_ranks=solver_group_ranks,
        checkpoint_callback=selector,
        decoder_stationarity_override=lambda *_: 0.0,
        work_dtype=work_dtype,
        work_device=device,
    )
    selected_A, selected_D = selector.selected()
    fit_loss = evaluate_quadratic(
        fit_objective, selected_A, selected_D, mapping
    )
    validation_loss = evaluate_quadratic(
        validation_objective, selected_A, selected_D, mapping
    )
    artifact_A = selected_A.to(device="cpu", dtype=factor_dtype).contiguous()
    artifact_D = selected_D.to(device="cpu", dtype=factor_dtype).contiguous()
    artifact_fit_loss = evaluate_quadratic(
        fit_objective,
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
        "selection": {
            "criterion": _selection_criterion(args.selection_boundaries),
            "boundary_policy": args.selection_boundaries,
            "boundary": selector.best_boundary,
            "sweep": selector.best_sweep,
            "checkpoints": selector.records,
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
                relative_tolerance=float(
                    fit_config["encoder_cg_relative_tolerance"]
                ),
                max_iterations=int(fit_config["encoder_cg_max_iterations"]),
            ),
            "attribution": asdict(result.attribution),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(record_path, record)
    print(
        f"[Llama C1] layer={layer} complete "
        f"heldout={record['heldout']['factor_dtype_relative_mse']:.8g} "
        f"seconds={record['elapsed_seconds']:.2f}",
        flush=True,
    )
    del (
        activation,
        validation_source_activation,
        weight,
        fit_activation,
        validation_activation,
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
    torch.cuda.empty_cache()
    return record


def _fit_shard(args: argparse.Namespace) -> None:
    if args.layer_shard_count <= 0 or not 0 <= args.layer_shard_index < args.layer_shard_count:
        raise ValueError("invalid layer shard")
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C1 joint fitting requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = _snapshot_manifest(snapshot_dir)
    validation_snapshot_dir = (
        Path(args.validation_snapshot_dir).expanduser().resolve()
        if args.validation_snapshot_dir is not None
        else snapshot_dir
    )
    validation_manifest = (
        _snapshot_manifest(validation_snapshot_dir)
        if validation_snapshot_dir != snapshot_dir
        else manifest
    )
    fit_config = _fit_config(
        args,
        snapshot_dir,
        manifest,
        validation_snapshot_dir=(
            validation_snapshot_dir
            if validation_snapshot_dir != snapshot_dir
            else None
        ),
        validation_manifest=(
            validation_manifest
            if validation_snapshot_dir != snapshot_dir
            else None
        ),
    )
    requested = _parse_layers(args.layers)
    layers = tuple(
        layer
        for position, layer in enumerate(requested)
        if position % args.layer_shard_count == args.layer_shard_index
    )
    if not layers:
        raise ValueError("layer shard is empty")
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
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_dir / f"shard_{args.layer_shard_index:02d}.json", shard)
    print(
        f"[Llama C1] shard={args.layer_shard_index}/{args.layer_shard_count} complete",
        flush=True,
    )


def _summary(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> str:
    heldout = [float(row["heldout"]["factor_dtype_relative_mse"]) for row in records]
    cache_rank = int(config["cache_rank_per_head"])
    total_retention = float(config["total_kv_retained_ratio_with_dense_k"])
    lines = [
        f"# Llama-2-7B MHA C1 V{cache_rank} joint fit",
        "",
        (
            f"- K remains dense; every one of {NUM_HEADS} V heads retains "
            f"rank {cache_rank}/{HEAD_DIM}."
        ),
        (
            f"- Total KV-cache retention is {100 * total_retention:.6g}% "
            f"({100 * (1 - total_retention):.6g}% reduction)."
        ),
        f"- Encoder initialization: `{config.get('encoder_initialization', 'activation-weighted-svd')}`.",
        f"- Held-out selection boundaries: `{config.get('selection_boundaries', 'all')}`.",
        (
            "- V encoders and head output decoders use a full-layer joint solve "
            "with cross-head covariance."
            if config.get("decoder_objective", "full_layer") == "full_layer"
            else "- V encoders use the same attention-output-aware initialization; "
            "each head decoder is then solved independently without cross-head covariance."
        ),
        f"- Fit contexts: {config['fit_windows']}; held-out selection contexts: {config['validation_windows']}.",
        f"- Mean held-out factor-dtype relative MSE: `{sum(heldout) / len(heldout):.9g}`.",
        "",
        "| Layer | Selected boundary | Sweep | Held-out relative MSE |",
        "|---:|:---|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['layer']} | {row['selection']['boundary']} | "
            f"{row['selection']['sweep']} | "
            f"{row['heldout']['factor_dtype_relative_mse']:.9g} |"
        )
    lines.append("")
    return "\n".join(lines)


def _merge(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    result_path = output_dir / "results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    layers = _parse_layers(args.layers)
    records = []
    fit_config = None
    for layer in layers:
        record_path = output_dir / f"layer_{layer:03d}.json"
        artifact_path = output_dir / f"layer_{layer:03d}.safetensors"
        if not record_path.is_file() or not artifact_path.is_file():
            raise FileNotFoundError(f"missing layer {layer} output")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("format") != LAYER_FORMAT or int(record.get("layer", -1)) != layer:
            raise ValueError(f"invalid layer record {record_path}")
        if record["artifact"]["sha256"] != _sha256(artifact_path):
            raise ValueError(f"artifact hash mismatch at layer {layer}")
        if fit_config is None:
            fit_config = record["fit_config"]
        elif record["fit_config"] != fit_config:
            raise ValueError("layer fit configurations differ")
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
        _summary(records, fit_config), encoding="utf-8"
    )
    print(f"[Llama C1] merged {len(records)} layers into {result_path}", flush=True)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument(
        "--validation-snapshot-dir",
        help="Optional separate snapshot used only for held-out evaluation",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--fit-windows", type=int, default=48)
    parser.add_argument("--validation-windows", type=int, default=16)
    parser.add_argument(
        "--validation-window-start",
        type=int,
        help=(
            "Explicit zero-based validation-window offset inside a larger "
            "snapshot; omitted means fit and validation exactly partition it"
        ),
    )
    parser.add_argument("--cache-rank", type=int, default=96)
    parser.add_argument("--work-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--factor-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--encoder-initialization",
        choices=(
            "activation-weighted-svd",
            "weight-only-svd",
            "random-orthogonal",
        ),
        default="activation-weighted-svd",
        help=(
            "Initial encoder subspace; every choice is followed by the same "
            "closed-form decoder refit"
        ),
    )
    parser.add_argument(
        "--encoder-initialization-seed",
        type=int,
        default=0,
        help=(
            "Base seed for random-orthogonal initialization; the effective "
            "seed is deterministically offset by layer"
        ),
    )
    parser.add_argument(
        "--selection-boundaries",
        choices=("all", "decoder-closed"),
        default="all",
        help=(
            "Checkpoints eligible for held-out selection. decoder-closed "
            "permits only decoder_only and after_redecoder checkpoints while "
            "retaining the complete trajectory for diagnostics."
        ),
    )
    parser.add_argument("--covariance-damping", type=float, default=1.0e-7)
    parser.add_argument(
        "--covariance-row-chunk-size",
        type=int,
        help="CPU-offloaded activation rows transferred per covariance update",
    )
    parser.add_argument("--encoder-sweeps", type=int, default=5)
    parser.add_argument("--minimum-encoder-sweeps", type=int, default=2)
    parser.add_argument("--encoder-relative-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--encoder-patience", type=int, default=2)
    parser.add_argument("--decoder-relative-jitter", type=float, default=0.0)
    parser.add_argument(
        "--decoder-objective",
        choices=("full_layer", "per_head"),
        default="full_layer",
        help=(
            "full_layer jointly fits all head decoders to aggregate attention output; "
            "per_head reproduces the A3-style sum of independent head-output losses"
        ),
    )
    parser.add_argument("--encoder-relative-damping", type=float, default=1.0e-8)
    parser.add_argument(
        "--encoder-cg-mode",
        choices=("fixed", "tolerance"),
        default="fixed",
    )
    parser.add_argument(
        "--encoder-cg-relative-tolerance", type=float, default=1.0e-8
    )
    parser.add_argument(
        "--encoder-cg-max-iterations",
        type=int,
        help="Maximum CG iterations; defaults to --encoder-cg-fixed-iterations",
    )
    parser.add_argument(
        "--encoder-cg-fixed-iterations",
        type=int,
        default=16,
        help="Legacy fixed CG budget and fallback maximum iteration count",
    )
    parser.add_argument("--maximum-backtracks", type=int, default=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    fit = subparsers.add_parser("fit-shard")
    _add_common(fit)
    fit.add_argument("--layer-shard-index", type=int, required=True)
    fit.add_argument("--layer-shard-count", type=int, required=True)
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--torch-num-threads", type=int, default=4)
    fit.add_argument("--resume", action="store_true")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--layers", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command_name == "fit-shard":
        if args.decoder_objective == "per_head" and (
            args.encoder_sweeps != 0 or args.minimum_encoder_sweeps != 0
        ):
            raise ValueError(
                "--decoder-objective per_head requires zero encoder sweeps for the "
                "strict A3 decoder ablation"
            )
        _fit_shard(args)
    else:
        _merge(args)


if __name__ == "__main__":
    main()
