#!/usr/bin/env python3
"""Fit DeepSeek-V2-Lite TP8 post-attention C1 ``W_o`` factors.

The MLA and KV-cache paths are untouched.  The 2048-wide input to ``o_proj``
is partitioned into eight contiguous 256-wide TP sources.  Each layer is fit
independently, while the eight source encoders and decoders are optimized
jointly against full-layer output MSE with cross-source covariance.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
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
    quadratic_from_target,
)
from basisserve.core.tp_source_wo_c1 import (  # noqa: E402
    TPSourceWOLayout,
    covariance_to_source_blocks,
    identity_factors,
    weight_to_source_targets,
)


FORMAT = "basisserve.deepseek_v2_lite.tp8_source_wo_c1_joint.v1"
LAYER_FORMAT = "basisserve.deepseek_v2_lite.tp8_source_wo_c1_joint.layer.v1"
SNAPSHOT_FORMAT = "basisserve.attention_o_proj_covariances.v1"
MODEL_TYPE = "deepseek_v2"
NUM_LAYERS = 27
HIDDEN_SIZE = 2048
NUM_ATTENTION_HEADS = 16
VALUE_HEAD_DIM = 128
TP_SIZE = 8
SOURCE_WIDTH = NUM_ATTENTION_HEADS * VALUE_HEAD_DIM // TP_SIZE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Tensor):
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


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: Mapping[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(tensors), str(temporary))
    os.replace(temporary, path)


def _parse_layers(raw: str) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return tuple(range(NUM_LAYERS))
    selected: set[int] = set()
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            first, last = map(int, piece.split("-", 1))
            if last < first:
                raise ValueError(f"descending layer range: {piece}")
            selected.update(range(first, last + 1))
        else:
            selected.add(int(piece))
    layers = tuple(sorted(selected))
    if not layers or min(layers) < 0 or max(layers) >= NUM_LAYERS:
        raise ValueError("selected layers are outside DeepSeek-V2-Lite")
    return layers


def _work_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _factor_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _snapshot_manifest(snapshot_dir: Path) -> dict[str, Any]:
    path = snapshot_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != SNAPSHOT_FORMAT:
        raise ValueError("DeepSeek C1 requires covariance sufficient statistics")
    model = manifest.get("model", {})
    observed = (
        str(model.get("model_type")),
        str(model.get("attention_type")),
        int(model.get("num_hidden_layers", -1)),
        int(model.get("hidden_size", -1)),
        int(model.get("num_attention_heads", -1)),
        int(model.get("head_dim", -1)),
    )
    expected = (
        MODEL_TYPE,
        "mla",
        NUM_LAYERS,
        HIDDEN_SIZE,
        NUM_ATTENTION_HEADS,
        VALUE_HEAD_DIM,
    )
    if observed != expected:
        raise ValueError(f"unexpected DeepSeek-V2-Lite snapshot geometry: {observed}")
    if tuple(map(int, manifest.get("layers", ()))) != tuple(range(NUM_LAYERS)):
        raise ValueError("snapshot must cover all DeepSeek-V2-Lite layers")
    calibration = manifest.get("calibration", {})
    if calibration.get("storage") != "normalized_covariance_sufficient_statistics":
        raise ValueError("snapshot is not normalized covariance statistics")
    return manifest


def _layout(source_rank: int) -> TPSourceWOLayout:
    return TPSourceWOLayout(
        input_width=NUM_ATTENTION_HEADS * VALUE_HEAD_DIM,
        output_width=HIDDEN_SIZE,
        tp_size=TP_SIZE,
        source_rank=source_rank,
        dtype_bytes=2,
    )


def _fit_config(
    args: argparse.Namespace,
    snapshot_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    layout = _layout(args.source_rank)
    calibration = manifest["calibration"]
    if (
        int(calibration.get("fit_windows", -1)) != args.fit_windows
        or int(calibration.get("heldout_windows", -1)) != args.validation_windows
    ):
        raise ValueError("requested fit/validation windows differ from snapshot")
    cg_max_iterations = (
        args.encoder_cg_fixed_iterations
        if args.encoder_cg_max_iterations is None
        else args.encoder_cg_max_iterations
    )
    if min(
        args.fit_windows,
        args.validation_windows,
        args.encoder_cg_fixed_iterations,
        cg_max_iterations,
    ) <= 0:
        raise ValueError("fit and CG configuration must be positive")
    if args.encoder_cg_mode == "fixed" and (
        cg_max_iterations < args.encoder_cg_fixed_iterations
    ):
        raise ValueError("CG maximum is smaller than the fixed iteration count")
    return {
        "model": str(Path(manifest["model"]["path"]).expanduser().resolve()),
        "model_config_sha256": manifest["model"]["config_sha256"],
        "snapshot_dir": str(snapshot_dir),
        "snapshot_manifest_sha256": _sha256(snapshot_dir / "manifest.json"),
        "layers": list(_parse_layers(args.layers)),
        "fit_windows": args.fit_windows,
        "validation_windows": args.validation_windows,
        "sequence_length": int(calibration["sequence_length"]),
        "fit_rows": int(calibration["fit_rows"]),
        "validation_rows": int(calibration["heldout_rows"]),
        "compression_target": "post_attention_tp_source_wo_allgather",
        "kv_cache_compression": "none",
        "tp_size": TP_SIZE,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "value_head_dim": VALUE_HEAD_DIM,
        "source_width": SOURCE_WIDTH,
        "source_rank": args.source_rank,
        "communication": layout.accounting(),
        "work_dtype": args.work_dtype,
        "factor_dtype": args.factor_dtype,
        "encoder_initialization": "activation-weighted-svd",
        "selection_boundaries": "decoder-closed",
        "covariance_damping": args.covariance_damping,
        "encoder_sweeps": args.encoder_sweeps,
        "minimum_encoder_sweeps": args.minimum_encoder_sweeps,
        "encoder_relative_tolerance": args.encoder_relative_tolerance,
        "encoder_patience": args.encoder_patience,
        "encoder_relative_damping": args.encoder_relative_damping,
        "encoder_cg_mode": args.encoder_cg_mode,
        "encoder_cg_relative_tolerance": args.encoder_cg_relative_tolerance,
        "encoder_cg_fixed_iterations": args.encoder_cg_fixed_iterations,
        "encoder_cg_max_iterations": cg_max_iterations,
        "decoder_relative_jitter": args.decoder_relative_jitter,
        "maximum_backtracks": args.maximum_backtracks,
    }


def _load_layer(
    snapshot_dir: Path,
    manifest: Mapping[str, Any],
    layer: int,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
    record = manifest["artifacts"][str(layer)]
    path = snapshot_dir / record["file"]
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"snapshot hash mismatch at layer {layer}")
    tensors = load_file(str(path), device="cpu")
    fit = tensors["fit_covariance"].contiguous()
    heldout = tensors["heldout_covariance"].contiguous()
    weight = tensors["weight"].contiguous()
    if tuple(fit.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError(f"unexpected fit covariance at layer {layer}")
    if tuple(heldout.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError(f"unexpected heldout covariance at layer {layer}")
    if tuple(weight.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError(f"unexpected o_proj weight at layer {layer}")
    return fit, heldout, weight, {"path": str(path), **record}


def _relative_loss(loss: float, constant: Tensor) -> float:
    return float(loss) / max(abs(float(constant)), 1.0e-300)


class _HeldoutSelector:
    def __init__(self, objective: Any, mapping: Tensor) -> None:
        self.objective = objective
        self.mapping = mapping
        self.records: list[dict[str, Any]] = []
        self.best_loss = float("inf")
        self.best_A: Tensor | None = None
        self.best_D: Tensor | None = None
        self.best_boundary: str | None = None
        self.best_sweep: int | None = None

    def __call__(self, checkpoint: Any, A: Tensor, D: Tensor) -> None:
        loss = evaluate_quadratic(self.objective, A, D, self.mapping)
        eligible = str(checkpoint.boundary) in {"decoder_only", "after_redecoder"}
        row = {
            "boundary": str(checkpoint.boundary),
            "sweep": int(checkpoint.sweep),
            "fit_loss": float(checkpoint.loss),
            "validation_loss": float(loss),
            "validation_relative_mse": _relative_loss(loss, self.objective.constant),
            "selection_eligible": eligible,
        }
        self.records.append(row)
        if eligible and float(loss) < self.best_loss:
            self.best_loss = float(loss)
            self.best_A = A.detach().clone()
            self.best_D = D.detach().clone()
            self.best_boundary = str(checkpoint.boundary)
            self.best_sweep = int(checkpoint.sweep)

    def selected(self) -> tuple[Tensor, Tensor]:
        if self.best_A is None or self.best_D is None:
            raise RuntimeError("heldout selector saw no decoder-closed checkpoint")
        return self.best_A, self.best_D


def _cg_payload(result: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    steps = [
        step
        for sweep in result.sweeps
        for step in sweep.encoder_steps
    ]
    return {
        "mode": config["encoder_cg_mode"],
        "fixed_iterations": config["encoder_cg_fixed_iterations"],
        "maximum_iterations": config["encoder_cg_max_iterations"],
        "group_solves": len(steps),
        "total_iterations": sum(int(step.cg.iterations) for step in steps),
        "converged_at_tolerance": sum(bool(step.cg.converged) for step in steps),
        "negative_curvature": sum(
            bool(step.cg.negative_curvature) for step in steps
        ),
        "maximum_relative_residual": max(
            (float(step.cg.relative_residual) for step in steps), default=0.0
        ),
    }


def _verified_prior(
    record_path: Path,
    artifact_path: Path,
    *,
    layer: int,
    fit_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not record_path.is_file() or not artifact_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        record.get("format") != LAYER_FORMAT
        or int(record.get("layer", -1)) != layer
        or record.get("fit_config") != fit_config
        or record.get("artifact", {}).get("sha256") != _sha256(artifact_path)
    ):
        raise ValueError(f"existing layer output is incompatible: {record_path}")
    return record


@torch.no_grad()
def _fit_layer(
    *,
    layer: int,
    snapshot_dir: Path,
    manifest: Mapping[str, Any],
    output_dir: Path,
    fit_config: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifact_path = output_dir / f"layer_{layer:03d}.safetensors"
    record_path = output_dir / f"layer_{layer:03d}.json"
    if args.resume:
        prior = _verified_prior(
            record_path,
            artifact_path,
            layer=layer,
            fit_config=fit_config,
        )
        if prior is not None:
            print(f"[DeepSeek C1] layer={layer} resume=verified", flush=True)
            return prior
    elif artifact_path.exists() or record_path.exists():
        raise FileExistsError(f"partial output exists at layer {layer}")

    fit_matrix, heldout_matrix, weight, source = _load_layer(
        snapshot_dir, manifest, layer
    )
    layout = _layout(args.source_rank)
    factor_dtype = _factor_dtype(args.factor_dtype)
    if args.source_rank == SOURCE_WIDTH:
        encoders, decoders = identity_factors(weight, layout, dtype=factor_dtype)
        artifacts = {
            "source_encoders": encoders.cpu().contiguous(),
            "source_decoders": decoders.cpu().contiguous(),
        }
        _atomic_safetensors(artifact_path, artifacts)
        record = {
            "format": LAYER_FORMAT,
            "layer": layer,
            "fit_config": dict(fit_config),
            "source_snapshot": source,
            "artifact": {
                "file": artifact_path.name,
                "sha256": _sha256(artifact_path),
                "tensors": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in artifacts.items()
                },
            },
            "selection": {"boundary": "identity", "sweep": 0, "checkpoints": []},
            "fit": {"factor_dtype_relative_mse": 0.0},
            "heldout": {"factor_dtype_relative_mse": 0.0},
            "solver": {"method": "analytic_identity", "sweeps": 0},
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(record_path, record)
        return record

    work_dtype = _work_dtype(args.work_dtype)
    fit_raw = covariance_to_source_blocks(
        fit_matrix, layout, device=device, dtype=work_dtype
    )
    heldout_covariance = covariance_to_source_blocks(
        heldout_matrix, layout, device=device, dtype=work_dtype
    )
    fit_covariance, absolute_damping = covariance_with_trace_damping(
        fit_raw, relative_damping=args.covariance_damping
    )
    target = weight_to_source_targets(
        weight, layout, device=device, dtype=work_dtype
    )
    fit_objective = quadratic_from_target(
        covariance=fit_covariance,
        target=target,
        name=f"deepseek_v2_lite_tp8_c1_fit_layer_{layer:03d}",
        trace_normalize=False,
    )
    heldout_objective = quadratic_from_target(
        covariance=heldout_covariance,
        target=target,
        name=f"deepseek_v2_lite_tp8_c1_heldout_layer_{layer:03d}",
        trace_normalize=False,
    )
    mapping = torch.arange(TP_SIZE, device=device, dtype=torch.long)
    initialization = initialize_group_pooled_routed_svd(
        covariance=fit_objective.covariance,
        target=target,
        head_to_kv_group=mapping,
        group_ranks=(args.source_rank,) * TP_SIZE,
        covariance_ridge=0.0,
    )
    initial_A = initialization.A_unique
    initial_D = torch.zeros(
        TP_SIZE,
        args.source_rank,
        HIDDEN_SIZE,
        device=device,
        dtype=work_dtype,
    )
    selector = _HeldoutSelector(heldout_objective, mapping)
    print(
        f"[DeepSeek C1] layer={layer} sources={TP_SIZE} "
        f"width={SOURCE_WIDTH} rank={args.source_rank} ALS={args.encoder_sweeps}",
        flush=True,
    )
    result = fit_routed_ov_joint(
        objective=fit_objective,
        initial_A=initial_A,
        initial_D=initial_D,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
        maximum_sweeps=args.encoder_sweeps,
        minimum_sweeps=args.minimum_encoder_sweeps,
        relative_objective_tolerance=args.encoder_relative_tolerance,
        patience=args.encoder_patience,
        decoder_relative_jitter=args.decoder_relative_jitter,
        encoder_relative_damping=args.encoder_relative_damping,
        cg_relative_tolerance=args.encoder_cg_relative_tolerance,
        cg_max_iterations=int(fit_config["encoder_cg_max_iterations"]),
        cg_fixed_iterations=args.encoder_cg_mode == "fixed",
        maximum_backtracks=args.maximum_backtracks,
        final_decoder_solve=True,
        group_ranks=(args.source_rank,) * TP_SIZE,
        checkpoint_callback=selector,
        decoder_stationarity_override=lambda *_: 0.0,
        work_dtype=work_dtype,
        work_device=device,
    )
    selected_A, selected_D = selector.selected()
    artifact_A = selected_A.to(dtype=factor_dtype).cpu().contiguous()
    artifact_D = selected_D.to(dtype=factor_dtype).cpu().contiguous()
    artifact_fit = evaluate_quadratic(
        fit_objective,
        artifact_A.to(device=device, dtype=work_dtype),
        artifact_D.to(device=device, dtype=work_dtype),
        mapping,
    )
    artifact_heldout = evaluate_quadratic(
        heldout_objective,
        artifact_A.to(device=device, dtype=work_dtype),
        artifact_D.to(device=device, dtype=work_dtype),
        mapping,
    )
    artifacts = {
        "source_encoders": artifact_A,
        "source_decoders": artifact_D,
    }
    _atomic_safetensors(artifact_path, artifacts)
    record = {
        "format": LAYER_FORMAT,
        "layer": layer,
        "fit_config": dict(fit_config),
        "source_snapshot": source,
        "artifact": {
            "file": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "tensors": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in artifacts.items()
            },
        },
        "selection": {
            "criterion": "earliest minimum heldout MSE at decoder-closed boundaries",
            "boundary": selector.best_boundary,
            "sweep": selector.best_sweep,
            "checkpoints": selector.records,
        },
        "fit": {
            "factor_dtype_relative_mse": _relative_loss(
                artifact_fit, fit_objective.constant
            )
        },
        "heldout": {
            "factor_dtype_relative_mse": _relative_loss(
                artifact_heldout, heldout_objective.constant
            )
        },
        "covariance": {
            "fit_absolute_trace_damping": absolute_damping,
            "heldout_regularized": False,
        },
        "initialization": [asdict(item) for item in initialization.groups],
        "solver": {
            "method": "full_layer_joint_source_als",
            "initial_loss": result.initial_loss,
            "decoder_only_loss": result.decoder_only_loss,
            "endpoint_loss": result.final_loss,
            "sweeps": len(result.sweeps),
            "cg": _cg_payload(result, fit_config),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(record_path, record)
    print(
        f"[DeepSeek C1] layer={layer} complete "
        f"heldout={record['heldout']['factor_dtype_relative_mse']:.9g} "
        f"seconds={record['elapsed_seconds']:.2f}",
        flush=True,
    )
    return record


def _fit_shard(args: argparse.Namespace) -> None:
    if (
        args.layer_shard_count <= 0
        or not 0 <= args.layer_shard_index < args.layer_shard_count
    ):
        raise ValueError("invalid layer shard")
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DeepSeek C1 fitting requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = _snapshot_manifest(snapshot_dir)
    fit_config = _fit_config(args, snapshot_dir, manifest)
    requested = _parse_layers(args.layers)
    layers = tuple(
        layer
        for position, layer in enumerate(requested)
        if position % args.layer_shard_count == args.layer_shard_index
    )
    if not layers:
        raise ValueError("layer shard is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = [
        _fit_layer(
            layer=layer,
            snapshot_dir=snapshot_dir,
            manifest=manifest,
            output_dir=output_dir,
            fit_config=fit_config,
            args=args,
            device=device,
        )
        for layer in layers
    ]
    shard = {
        "format": FORMAT,
        "status": "shard_complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "layer_shard_index": args.layer_shard_index,
        "layer_shard_count": args.layer_shard_count,
        "layers": list(layers),
        "fit_config": fit_config,
        "records": [int(record["layer"]) for record in records],
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_dir / f"shard_{args.layer_shard_index:02d}.json", shard)


def _merge(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    layers = _parse_layers(args.layers)
    records = []
    fit_config = None
    for layer in layers:
        record_path = output_dir / f"layer_{layer:03d}.json"
        artifact_path = output_dir / f"layer_{layer:03d}.safetensors"
        if not record_path.is_file() or not artifact_path.is_file():
            raise FileNotFoundError(f"missing layer {layer} output")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            record.get("format") != LAYER_FORMAT
            or int(record.get("layer", -1)) != layer
            or record["artifact"]["sha256"] != _sha256(artifact_path)
        ):
            raise ValueError(f"invalid layer output at layer {layer}")
        if fit_config is None:
            fit_config = record["fit_config"]
        elif record["fit_config"] != fit_config:
            raise ValueError("layer fit configurations differ")
        records.append(record)
    assert fit_config is not None
    result_path = output_dir / "results.json"
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
                float(row["fit"]["factor_dtype_relative_mse"])
                for row in records
            )
            / len(records),
            "mean_heldout_factor_dtype_relative_mse": sum(
                float(row["heldout"]["factor_dtype_relative_mse"])
                for row in records
            )
            / len(records),
        },
    }
    _atomic_json(result_path, payload)
    lines = [
        "# DeepSeek-V2-Lite TP8 post-attention C1",
        "",
        "- MLA and KV cache are unchanged.",
        f"- Source width/rank: `{SOURCE_WIDTH}/{fit_config['source_rank']}`.",
        (
            "- Communication reduction versus dense AllGather: "
            f"`{100 * fit_config['communication']['reduction_vs_dense_allgather']:.6g}%`."
        ),
        "",
        "| Layer | Boundary | Sweep | Heldout relative MSE |",
        "|---:|:---|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['layer']} | {row['selection']['boundary']} | "
            f"{row['selection']['sweep']} | "
            f"{row['heldout']['factor_dtype_relative_mse']:.9g} |"
        )
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[DeepSeek C1] merged {len(records)} layers into {result_path}")


def _add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--fit-windows", type=int, default=256)
    parser.add_argument("--validation-windows", type=int, default=64)
    parser.add_argument("--source-rank", type=int, default=192)
    parser.add_argument("--work-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--factor-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--covariance-damping", type=float, default=1.0e-5)
    parser.add_argument("--encoder-sweeps", type=int, default=5)
    parser.add_argument("--minimum-encoder-sweeps", type=int, default=2)
    parser.add_argument("--encoder-relative-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--encoder-patience", type=int, default=2)
    parser.add_argument("--encoder-relative-damping", type=float, default=0.0)
    parser.add_argument("--decoder-relative-jitter", type=float, default=0.0)
    parser.add_argument("--encoder-cg-mode", choices=("fixed", "tolerance"), default="fixed")
    parser.add_argument("--encoder-cg-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--encoder-cg-fixed-iterations", type=int, default=16)
    parser.add_argument("--encoder-cg-max-iterations", type=int)
    parser.add_argument("--maximum-backtracks", type=int, default=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command_name", required=True)
    fit = commands.add_parser("fit-shard")
    _add_fit_arguments(fit)
    fit.add_argument("--layer-shard-index", type=int, required=True)
    fit.add_argument("--layer-shard-count", type=int, required=True)
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--torch-num-threads", type=int, default=4)
    fit.add_argument("--resume", action="store_true")
    merge = commands.add_parser("merge")
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--layers", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command_name == "fit-shard":
        _fit_shard(args)
    else:
        _merge(args)


if __name__ == "__main__":
    main()
