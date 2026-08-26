#!/usr/bin/env python3
"""Refine a frozen Qwen3-32B Global-KL rank schedule with ragged C1 ALS.

The allocation must already be frozen on C4 documents disjoint from the
256-fit/64-heldout covariance snapshot.  This stage never changes a rank.  It
initializes every physical KV source with activation-aware SVD at its selected
rank, then runs the existing full-layer decoder/encoder ALS and selects a
decoder-closed checkpoint on the 64 held-out documents.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.decoder_closed_rank_candidates import tensor_sha256  # noqa: E402
from basisserve.core.gqa_routed_ov_joint import (  # noqa: E402
    covariance_with_trace_damping,
    evaluate_quadratic,
    fit_routed_ov_joint,
    initialize_group_pooled_routed_svd,
    quadratic_from_target,
)
from evaluation import allocate_qwen3_32b_c1_tp_source_global_kl as allocation  # noqa: E402
from evaluation import fit_llama2_mha_c1_joint as uniform  # noqa: E402


uniform.activate_model_profile("qwen3_32b")

FORMAT = "basisserve.qwen3_32b.gqa_c1.ragged_schedule_als.v1"
LAYER_FORMAT = "basisserve.qwen3_32b.gqa_c1.ragged_schedule_als.layer.v1"
ALLOCATION_FORMAT = "basisserve.qwen3_32b.gqa_c1.tp_source_global_kl_allocation.v2"
LAYER_ALLOCATION_FORMAT = "basisserve.qwen3_32b.gqa_c1.layer_global_kl_allocation.v1"


def _load_allocation(
    directory: Path,
    *,
    model_config_sha256: str,
) -> tuple[dict[str, Any], tuple[tuple[int, ...], ...]]:
    path = directory / "result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    allocation_format = result.get("format")
    if allocation_format not in {ALLOCATION_FORMAT, LAYER_ALLOCATION_FORMAT} or result.get("status") != "complete":
        raise ValueError("Global-KL allocation is incomplete or incompatible")
    if result.get("model_config_sha256") != model_config_sha256:
        raise ValueError("Global-KL allocation belongs to another model")
    schedule = tuple(
        tuple(int(rank) for rank in layer)
        for layer in result.get("selection", {}).get("selected_schedule", ())
    )
    if len(schedule) != allocation.NUM_LAYERS or any(
        len(layer) != allocation.NUM_KV_HEADS for layer in schedule
    ):
        raise ValueError("Global-KL allocation has an incomplete rank schedule")
    candidates = tuple(
        int(rank) for rank in result["selection"]["candidate_ranks"]
    )
    if any(rank not in candidates for layer in schedule for rank in layer):
        raise ValueError("selected schedule contains a rank outside the candidate grid")
    if allocation_format == LAYER_ALLOCATION_FORMAT and any(
        len(set(layer)) != 1 for layer in schedule
    ):
        raise ValueError("per-layer Global-KL allocation has a nonuniform layer rank")
    target = int(result["selection"]["target_source_rank_sum"])
    if sum(map(sum, schedule)) != target:
        raise ValueError("selected schedule violates its frozen rank budget")
    return result, schedule


def _fit_config(
    args: argparse.Namespace,
    *,
    snapshot_dir: Path,
    snapshot_manifest: Mapping[str, Any],
    allocation_dir: Path,
    allocation_result: Mapping[str, Any],
    schedule: Sequence[Sequence[int]],
) -> dict[str, Any]:
    calibration = snapshot_manifest["calibration"]
    if (
        int(calibration.get("fit_windows", -1)) != args.fit_windows
        or int(calibration.get("heldout_windows", -1)) != args.validation_windows
        or int(calibration.get("positions_per_window", -1)) != 2048
    ):
        raise ValueError("ragged ALS requires the 256-fit/64-heldout full-2048 snapshot")
    flat = [int(rank) for layer in schedule for rank in layer]
    return {
        "model": str(Path(snapshot_manifest["model"]["path"]).resolve()),
        "model_config_sha256": snapshot_manifest["model"]["config_sha256"],
        "snapshot_dir": str(snapshot_dir),
        "snapshot_manifest_sha256": uniform._sha256(snapshot_dir / "manifest.json"),
        "allocation_dir": str(allocation_dir),
        "allocation_result_sha256": uniform._sha256(allocation_dir / "result.json"),
        "allocation_format": allocation_result["format"],
        "allocation_selected_candidate": allocation_result["selection"][
            "selected_candidate"
        ],
        "rank_schedule": [list(map(int, layer)) for layer in schedule],
        "rank_histogram": {
            str(rank): flat.count(rank) for rank in sorted(set(flat))
        },
        "source_rank_sum": sum(flat),
        "dense_source_rank_sum": (
            allocation.NUM_LAYERS
            * allocation.NUM_KV_HEADS
            * allocation.HEAD_DIM
        ),
        "fit_windows": args.fit_windows,
        "validation_windows": args.validation_windows,
        "positions_per_window": 2048,
        "fit_rows": args.fit_windows * 2048,
        "validation_rows": args.validation_windows * 2048,
        "work_dtype": "float32",
        "factor_dtype": "bfloat16",
        "encoder_initialization": "activation-weighted-svd",
        "selection_boundaries": "decoder-closed",
        "covariance_damping": args.covariance_damping,
        "encoder_sweeps": args.encoder_sweeps,
        "minimum_encoder_sweeps": args.minimum_encoder_sweeps,
        "encoder_relative_tolerance": args.encoder_relative_tolerance,
        "encoder_patience": args.encoder_patience,
        "decoder_relative_jitter": args.decoder_relative_jitter,
        "encoder_relative_damping": args.encoder_relative_damping,
        "encoder_cg_mode": "fixed",
        "encoder_cg_fixed_iterations": args.encoder_cg_fixed_iterations,
        "maximum_backtracks": args.maximum_backtracks,
        "decoder_objective": "full_layer",
        "objective": "full-layer attention-output MSE with cross-head covariance",
    }


@torch.no_grad()
def _fit_layer(
    *,
    layer: int,
    ranks: Sequence[int],
    snapshot_dir: Path,
    snapshot_manifest: Mapping[str, Any],
    output_dir: Path,
    fit_config: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifact_path = output_dir / f"layer_{layer:03d}.safetensors"
    record_path = output_dir / f"layer_{layer:03d}.json"
    if args.resume and artifact_path.is_file() and record_path.is_file():
        prior = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            prior.get("format") != LAYER_FORMAT
            or prior.get("fit_config") != fit_config
            or prior.get("artifact", {}).get("sha256")
            != uniform._sha256(artifact_path)
        ):
            raise ValueError(f"incompatible ragged ALS checkpoint at layer {layer}")
        print(f"[Qwen3 ragged ALS] layer={layer} resume=verified", flush=True)
        return prior
    if artifact_path.exists() or record_path.exists():
        raise FileExistsError(f"partial layer output exists for layer {layer}")

    fit_matrix, heldout_matrix, weight, source = uniform._load_covariance_layer(
        snapshot_dir,
        snapshot_manifest,
        layer,
    )
    fit_raw = uniform._covariance_matrix_to_blocks(
        fit_matrix,
        device=device,
        dtype=torch.float32,
    )
    heldout_covariance = uniform._covariance_matrix_to_blocks(
        heldout_matrix,
        device=device,
        dtype=torch.float32,
    )
    fit_covariance, absolute_damping = covariance_with_trace_damping(
        fit_raw,
        relative_damping=args.covariance_damping,
    )
    target = uniform._dense_head_targets(
        weight,
        device=device,
        dtype=torch.float32,
    )
    fit_objective = quadratic_from_target(
        covariance=fit_covariance,
        target=target,
        name=f"qwen3_32b_ragged_als_fit_layer_{layer:03d}",
        trace_normalize=False,
    )
    heldout_objective = quadratic_from_target(
        covariance=heldout_covariance,
        target=target,
        name=f"qwen3_32b_ragged_als_heldout_layer_{layer:03d}",
        trace_normalize=False,
    )
    mapping = uniform._head_to_kv_group(device=device)
    initialization = initialize_group_pooled_routed_svd(
        covariance=fit_objective.covariance,
        target=target,
        head_to_kv_group=mapping,
        group_ranks=ranks,
        covariance_ridge=args.covariance_damping,
    )
    maximum_rank = max(map(int, ranks))
    initial_D = torch.zeros(
        allocation.NUM_QUERY_HEADS,
        maximum_rank,
        allocation.HIDDEN_SIZE,
        device=device,
        dtype=torch.float32,
    )
    selector = uniform._HeldoutSelector(
        heldout_objective,
        mapping,
        selection_boundaries="decoder-closed",
    )
    result = fit_routed_ov_joint(
        objective=fit_objective,
        initial_A=initialization.A_unique,
        initial_D=initial_D,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
        maximum_sweeps=args.encoder_sweeps,
        minimum_sweeps=args.minimum_encoder_sweeps,
        relative_objective_tolerance=args.encoder_relative_tolerance,
        patience=args.encoder_patience,
        decoder_relative_jitter=args.decoder_relative_jitter,
        encoder_relative_damping=args.encoder_relative_damping,
        cg_relative_tolerance=1.0e-8,
        cg_max_iterations=args.encoder_cg_fixed_iterations,
        cg_fixed_iterations=True,
        maximum_backtracks=args.maximum_backtracks,
        final_decoder_solve=True,
        group_ranks=ranks,
        checkpoint_callback=selector,
        decoder_stationarity_override=lambda *_: 0.0,
        work_dtype=torch.float32,
        work_device=device,
    )
    selected_A, selected_D = selector.selected()
    fit_loss = evaluate_quadratic(fit_objective, selected_A, selected_D, mapping)
    heldout_loss = evaluate_quadratic(
        heldout_objective,
        selected_A,
        selected_D,
        mapping,
    )
    artifact_A = selected_A.cpu().to(torch.bfloat16).contiguous()
    artifact_D = selected_D.cpu().to(torch.bfloat16).contiguous()
    artifact_ranks = torch.tensor(tuple(map(int, ranks)), dtype=torch.int32)
    artifact_fit = evaluate_quadratic(
        fit_objective,
        artifact_A.to(device=device, dtype=torch.float32),
        artifact_D.to(device=device, dtype=torch.float32),
        mapping,
    )
    artifact_heldout = evaluate_quadratic(
        heldout_objective,
        artifact_A.to(device=device, dtype=torch.float32),
        artifact_D.to(device=device, dtype=torch.float32),
        mapping,
    )
    tensors = {
        "value_coordinate_encoders": artifact_A,
        "head_output_decoders": artifact_D,
        "source_ranks": artifact_ranks,
    }
    uniform._atomic_safetensors(artifact_path, tensors)
    record = {
        "format": LAYER_FORMAT,
        "layer": layer,
        "source_ranks": list(map(int, ranks)),
        "fit_config": dict(fit_config),
        "source_snapshot": source,
        "artifact": {
            "file": artifact_path.name,
            "sha256": uniform._sha256(artifact_path),
            "encoder_sha256": tensor_sha256(artifact_A),
            "decoder_sha256": tensor_sha256(artifact_D),
            "tensors": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in tensors.items()
            },
        },
        "selection": {
            "criterion": uniform._selection_criterion("decoder-closed"),
            "boundary": selector.best_boundary,
            "sweep": selector.best_sweep,
            "checkpoints": selector.records,
        },
        "fit": {
            "relative_mse": uniform._relative_loss(
                fit_loss, fit_objective.constant
            ),
            "factor_dtype_relative_mse": uniform._relative_loss(
                artifact_fit, fit_objective.constant
            ),
        },
        "heldout": {
            "relative_mse": uniform._relative_loss(
                heldout_loss, heldout_objective.constant
            ),
            "factor_dtype_relative_mse": uniform._relative_loss(
                artifact_heldout, heldout_objective.constant
            ),
        },
        "covariance": {
            "fit_absolute_trace_damping": absolute_damping,
            "heldout_regularized": False,
        },
        "initialization": [asdict(item) for item in initialization.groups],
        "solver": {
            "initial_loss": result.initial_loss,
            "decoder_only_loss": result.decoder_only_loss,
            "endpoint_loss": result.final_loss,
            "sweeps": len(result.sweeps),
            "attribution": asdict(result.attribution),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    uniform._atomic_json(record_path, record)
    print(
        f"[Qwen3 ragged ALS] layer={layer} ranks={list(ranks)} "
        f"heldout={record['heldout']['factor_dtype_relative_mse']:.8g} "
        f"seconds={record['elapsed_seconds']:.2f}",
        flush=True,
    )
    return record


def _fit_shard(args: argparse.Namespace) -> None:
    if args.layer_shard_count <= 0 or not 0 <= args.layer_shard_index < args.layer_shard_count:
        raise ValueError("invalid layer shard")
    if args.minimum_encoder_sweeps > args.encoder_sweeps:
        raise ValueError("minimum encoder sweeps exceed maximum sweeps")
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ragged C1 ALS requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    snapshot_manifest = uniform._snapshot_manifest(snapshot_dir)
    allocation_dir = Path(args.allocation_dir).expanduser().resolve()
    allocation_result, schedule = _load_allocation(
        allocation_dir,
        model_config_sha256=snapshot_manifest["model"]["config_sha256"],
    )
    fit_config = _fit_config(
        args,
        snapshot_dir=snapshot_dir,
        snapshot_manifest=snapshot_manifest,
        allocation_dir=allocation_dir,
        allocation_result=allocation_result,
        schedule=schedule,
    )
    requested = uniform._parse_layers(args.layers)
    layers = tuple(
        layer
        for position, layer in enumerate(requested)
        if position % args.layer_shard_count == args.layer_shard_index
    )
    if not layers:
        raise ValueError("layer shard is empty")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = [
        _fit_layer(
            layer=layer,
            ranks=schedule[layer],
            snapshot_dir=snapshot_dir,
            snapshot_manifest=snapshot_manifest,
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
    uniform._atomic_json(
        output_dir / f"shard_{args.layer_shard_index:02d}.json",
        shard,
    )


def _merge(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    layers = uniform._parse_layers(args.layers)
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
        if record["artifact"]["sha256"] != uniform._sha256(artifact_path):
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
        "selection": {
            "selected_candidate": fit_config["allocation_selected_candidate"],
            "selected_schedule": fit_config["rank_schedule"],
            "source_rank_sum": fit_config["source_rank_sum"],
        },
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
    uniform._atomic_json(result_path, payload)
    print(f"[Qwen3 ragged ALS] merged {len(records)} layers into {result_path}")


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--allocation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--fit-windows", type=int, default=256)
    parser.add_argument("--validation-windows", type=int, default=64)
    parser.add_argument("--covariance-damping", type=float, default=1.0e-5)
    parser.add_argument("--encoder-sweeps", type=int, default=5)
    parser.add_argument("--minimum-encoder-sweeps", type=int, default=2)
    parser.add_argument("--encoder-relative-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--encoder-patience", type=int, default=2)
    parser.add_argument("--decoder-relative-jitter", type=float, default=0.0)
    parser.add_argument("--encoder-relative-damping", type=float, default=1.0e-6)
    parser.add_argument("--encoder-cg-fixed-iterations", type=int, default=16)
    parser.add_argument("--maximum-backtracks", type=int, default=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    fit = subparsers.add_parser("fit-shard")
    _add_fit_args(fit)
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
    if args.stage == "fit-shard":
        _fit_shard(args)
    else:
        _merge(args)


if __name__ == "__main__":
    main()
