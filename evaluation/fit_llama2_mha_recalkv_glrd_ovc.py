#!/usr/bin/env python3
"""Fit an unofficial fusion-compatible ReCalKV G-LRD Value reference.

K remains dense.  Llama-2-7B's 32 Value heads are split into contiguous
groups.  Each group is calibrated independently with the same alternating
least-squares OVC update as ReCalKV.  The matched V75 configuration uses eight
four-head groups with rank 384 per group, for total cached width 3072.

The saved factors are evaluated by explicitly reconstructing dense V as a
quality reference.  They are also algebraically fusion-compatible: split each
group decoder into per-head blocks and absorb those blocks into the matching
columns of Wo after applying each head's attention map to the shared latent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import save_file
import torch
from torch import Tensor

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.fit_llama2_mha_recalkv_global_ovc import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_HEADS,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    _atomic_json,
    _factor_dtype,
    _load_v_weight,
    _load_whitening_layer,
    _model_metadata,
    _parse_layers,
    _sha256,
    _whitening_metadata,
    _work_dtype,
    global_ovc_from_sufficient_statistics,
)


FORMAT = "basisserve.llama2_7b.recalkv_glrd_ovc_reference.v1"
LAYER_FORMAT = "basisserve.llama2_7b.recalkv_glrd_ovc_reference.layer.v1"


def _atomic_safetensors(path: Path, payload: Mapping[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(payload), str(temporary))
    os.replace(temporary, path)


def _validate_geometry(*, group_size: int, rank_per_group: int) -> tuple[int, int]:
    if group_size <= 0 or NUM_HEADS % group_size:
        raise ValueError("group size must divide the Llama attention heads")
    group_dim = group_size * HEAD_DIM
    if not 0 < rank_per_group <= group_dim:
        raise ValueError("rank per group exceeds the G-LRD group width")
    return NUM_HEADS // group_size, group_dim


def grouped_relative_covariance_error(
    weight: Tensor,
    decoders: Tensor,
    encoders: Tensor,
    covariance_factor: Tensor,
) -> float:
    if decoders.ndim != 3 or encoders.ndim != 3:
        raise ValueError("grouped OVC factors must be rank-three tensors")
    groups, group_dim, rank = map(int, decoders.shape)
    if tuple(encoders.shape) != (groups, rank, int(weight.shape[1])):
        raise ValueError("grouped OVC encoder geometry is incompatible")
    if int(weight.shape[0]) != groups * group_dim:
        raise ValueError("grouped OVC decoder geometry is incompatible")
    target = weight @ covariance_factor
    residuals = []
    for group in range(groups):
        start = group * group_dim
        stop = start + group_dim
        prediction = decoders[group] @ (encoders[group] @ covariance_factor)
        residuals.append(target[start:stop] - prediction)
    denominator = torch.linalg.vector_norm(target)
    if float(denominator) == 0.0:
        raise ValueError("zero calibration target norm")
    return float(torch.linalg.vector_norm(torch.cat(residuals)) / denominator)


@torch.no_grad()
def grouped_ovc_from_sufficient_statistics(
    weight: Tensor,
    initialization_whitening: Tensor,
    *,
    group_size: int,
    rank_per_group: int,
    iterations: int,
    ovc_whitening: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Return stacked group decoders and encoders for G-LRD OVC."""

    groups, group_dim = _validate_geometry(
        group_size=group_size, rank_per_group=rank_per_group
    )
    if tuple(weight.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError("G-LRD OVC expects the full Llama Value matrix")
    if tuple(initialization_whitening.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError("initialization whitening has incompatible geometry")
    if ovc_whitening is None:
        ovc_whitening = initialization_whitening

    decoder_list = []
    encoder_list = []
    group_diagnostics = []
    for group in range(groups):
        start = group * group_dim
        stop = start + group_dim
        decoder, encoder, diagnostics = global_ovc_from_sufficient_statistics(
            weight[start:stop],
            initialization_whitening,
            rank=rank_per_group,
            iterations=iterations,
            ovc_whitening=ovc_whitening,
        )
        decoder_list.append(decoder)
        encoder_list.append(encoder)
        group_diagnostics.append({"group": group, **diagnostics})
    decoders = torch.stack(decoder_list)
    encoders = torch.stack(encoder_list)
    diagnostics = {
        "groups": group_diagnostics,
        "initial_relative_rmse_initialization_covariance": (
            grouped_relative_covariance_error(
                weight, decoders, encoders, initialization_whitening
            )
        ),
        "final_relative_rmse_ovc_covariance": (
            grouped_relative_covariance_error(
                weight, decoders, encoders, ovc_whitening
            )
        ),
    }
    return decoders, encoders, diagnostics


def _fit_config(
    args: argparse.Namespace,
    *,
    model: Mapping[str, Any],
    whitening: Mapping[str, Any],
    ovc_whitening: Mapping[str, Any],
) -> dict[str, Any]:
    groups, group_dim = _validate_geometry(
        group_size=args.group_size, rank_per_group=args.rank_per_group
    )
    if args.ovc_iterations < 0:
        raise ValueError("OVC iterations must be nonnegative")
    total_rank = groups * args.rank_per_group
    return {
        "method": "unofficial ReCalKV OVC on contiguous PaLU G-LRD groups",
        "classification": "fusion-compatible dense-reconstruction quality reference",
        "model": model["path"],
        "model_config_sha256": model["config_sha256"],
        "model_safetensors_index_sha256": model["safetensors_index_sha256"],
        "key_projection": "dense and unchanged",
        "value_factorization_scope": "contiguous G-LRD head groups",
        "head_order": "original contiguous order",
        "group_size": args.group_size,
        "group_count": groups,
        "group_dense_width": group_dim,
        "rank_per_group": args.rank_per_group,
        "total_value_rank": total_rank,
        "dense_value_width": HIDDEN_SIZE,
        "value_retained_ratio": total_rank / HIDDEN_SIZE,
        "total_kv_retained_ratio_with_dense_k": (HIDDEN_SIZE + total_rank)
        / (2 * HIDDEN_SIZE),
        "dense_reconstruction_before_mha_for_quality_evaluation": True,
        "fusion_compatible": True,
        "fused_latent_width_per_query_head": args.rank_per_group,
        "fused_attention_output_width": NUM_HEADS * args.rank_per_group,
        "fused_attention_output_width_vs_dense": args.rank_per_group / HEAD_DIM,
        "deployable_fused_kernel_implemented": False,
        "ovc_iterations": args.ovc_iterations,
        "work_dtype": args.work_dtype,
        "factor_dtype": args.factor_dtype,
        "initialization_whitening": dict(whitening),
        "ovc_whitening": dict(ovc_whitening),
        "ovc_covariance_matches_initialization": whitening == ovc_whitening,
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "status": "unofficial extension; upstream forces Value to one global group",
            "ovc_update": "groupwise decoder least squares followed by encoder pinv refit",
        },
    }


def _resume_record(
    record_path: Path,
    artifact_path: Path,
    *,
    layer: int,
    fit_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not record_path.exists() and not artifact_path.exists():
        return None
    if not record_path.is_file() or not artifact_path.is_file():
        raise FileExistsError(f"partial output exists for layer {layer}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        record.get("format") != LAYER_FORMAT
        or int(record.get("layer", -1)) != layer
        or record.get("fit_config") != fit_config
        or record.get("artifact", {}).get("sha256") != _sha256(artifact_path)
    ):
        raise ValueError(f"incompatible resume output at layer {layer}")
    print(f"[ReCalKV G-LRD OVC] layer={layer} resume hit", flush=True)
    return record


@torch.no_grad()
def _fit_layer(
    *,
    layer: int,
    model_path: Path,
    weight_map: Mapping[str, str],
    whitening_path: Path,
    ovc_whitening_path: Path,
    output_dir: Path,
    fit_config: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    record_path = output_dir / f"layer_{layer:03d}.json"
    artifact_path = output_dir / f"layer_{layer:03d}.safetensors"
    if args.resume:
        resumed = _resume_record(
            record_path, artifact_path, layer=layer, fit_config=fit_config
        )
        if resumed is not None:
            return resumed
    elif record_path.exists() or artifact_path.exists():
        raise FileExistsError(f"layer {layer} output already exists")

    started = time.perf_counter()
    work_dtype = _work_dtype(args.work_dtype)
    factor_dtype = _factor_dtype(args.factor_dtype)
    weight_cpu, weight_source = _load_v_weight(model_path, weight_map, layer)
    whitening_cpu = _load_whitening_layer(whitening_path, layer)
    ovc_whitening_cpu = (
        whitening_cpu
        if ovc_whitening_path == whitening_path
        else _load_whitening_layer(ovc_whitening_path, layer)
    )
    weight = weight_cpu.to(device=device, dtype=work_dtype)
    whitening = whitening_cpu.to(device=device, dtype=work_dtype)
    ovc_whitening = ovc_whitening_cpu.to(device=device, dtype=work_dtype)
    decoders, encoders, diagnostics = grouped_ovc_from_sufficient_statistics(
        weight,
        whitening,
        group_size=args.group_size,
        rank_per_group=args.rank_per_group,
        iterations=args.ovc_iterations,
        ovc_whitening=ovc_whitening,
    )
    artifact_decoders = decoders.to(device="cpu", dtype=factor_dtype).contiguous()
    artifact_encoders = encoders.to(device="cpu", dtype=factor_dtype).contiguous()
    quantized_decoders = artifact_decoders.to(device=device, dtype=work_dtype)
    quantized_encoders = artifact_encoders.to(device=device, dtype=work_dtype)
    diagnostics["factor_dtype_relative_rmse_initialization_covariance"] = (
        grouped_relative_covariance_error(
            weight, quantized_decoders, quantized_encoders, whitening
        )
    )
    diagnostics["factor_dtype_relative_rmse_ovc_covariance"] = (
        grouped_relative_covariance_error(
            weight, quantized_decoders, quantized_encoders, ovc_whitening
        )
    )
    tensors = {
        "v_group_encoder_weight": artifact_encoders,
        "v_group_reconstruction_weight": artifact_decoders,
    }
    _atomic_safetensors(artifact_path, tensors)
    record = {
        "format": LAYER_FORMAT,
        "layer": layer,
        "fit_config": dict(fit_config),
        "source_weight": weight_source,
        "artifact": {
            "file": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "tensors": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in tensors.items()
            },
        },
        "diagnostics": diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(record_path, record)
    print(
        f"[ReCalKV G-LRD OVC] layer={layer} "
        f"rmse={diagnostics['factor_dtype_relative_rmse_initialization_covariance']:.8g} "
        f"seconds={record['elapsed_seconds']:.2f}",
        flush=True,
    )
    del (
        weight_cpu,
        whitening_cpu,
        ovc_whitening_cpu,
        weight,
        whitening,
        ovc_whitening,
        decoders,
        encoders,
        artifact_decoders,
        artifact_encoders,
        quantized_decoders,
        quantized_encoders,
        tensors,
    )
    torch.cuda.empty_cache()
    return record


def _fit_shard(args: argparse.Namespace) -> None:
    if args.layer_shard_count <= 0 or not 0 <= args.layer_shard_index < args.layer_shard_count:
        raise ValueError("invalid layer shard")
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("G-LRD OVC fitting requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model_path = Path(args.model).expanduser().resolve()
    whitening_dir = Path(args.whitening_dir).expanduser().resolve()
    ovc_whitening_dir = (
        Path(args.ovc_whitening_dir).expanduser().resolve()
        if args.ovc_whitening_dir is not None
        else whitening_dir
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    model, weight_map = _model_metadata(model_path)
    whitening, whitening_path = _whitening_metadata(
        whitening_dir, model_config_sha256=model["config_sha256"]
    )
    if ovc_whitening_dir == whitening_dir:
        ovc_whitening, ovc_whitening_path = whitening, whitening_path
    else:
        ovc_whitening, ovc_whitening_path = _whitening_metadata(
            ovc_whitening_dir, model_config_sha256=model["config_sha256"]
        )
    fit_config = _fit_config(
        args, model=model, whitening=whitening, ovc_whitening=ovc_whitening
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
    started = time.perf_counter()
    records = []
    for layer in layers:
        records.append(
            _fit_layer(
                layer=layer,
                model_path=model_path,
                weight_map=weight_map,
                whitening_path=whitening_path,
                ovc_whitening_path=ovc_whitening_path,
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
        "records": [int(record["layer"]) for record in records],
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
        f"[ReCalKV G-LRD OVC] shard={args.layer_shard_index}/{args.layer_shard_count} complete",
        flush=True,
    )


def _summary(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> str:
    metric = "factor_dtype_relative_rmse_initialization_covariance"
    errors = [float(record["diagnostics"][metric]) for record in records]
    lines = [
        "# Unofficial ReCalKV G-LRD OVC quality reference",
        "",
        "- K remains completely dense and unchanged.",
        (
            f"- Value uses {config['group_count']} contiguous groups of "
            f"{config['group_size']} heads, rank {config['rank_per_group']} per group."
        ),
        (
            f"- Total Value latent width: {config['total_value_rank']} "
            f"({100 * config['value_retained_ratio']:.6g}% V retention)."
        ),
        "- OVC is independently calibrated inside each G-LRD group.",
        "- Evaluation reconstructs dense V; factors are fusion-compatible, but no production fused kernel is claimed.",
        (
            f"- A fused implementation has {config['fused_attention_output_width_vs_dense']:.6g}x "
            "the dense Value-attention/output width."
        ),
        f"- OVC alternating updates: {config['ovc_iterations']}.",
        f"- Mean factor-dtype calibration relative RMSE: `{sum(errors) / len(errors):.9g}`.",
        "",
        "| Layer | Factor-dtype relative RMSE |",
        "|---:|---:|",
    ]
    for record in records:
        lines.append(f"| {record['layer']} | {record['diagnostics'][metric]:.9g} |")
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
    metric = "factor_dtype_relative_rmse_initialization_covariance"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": "Unofficial ReCalKV G4-LRD OVC V384 fusion-compatible quality reference",
        "layers": list(layers),
        "fit_config": fit_config,
        "artifacts": {str(record["layer"]): record["artifact"] for record in records},
        "records": records,
        "aggregate": {
            "mean_factor_dtype_relative_rmse": sum(
                float(record["diagnostics"][metric]) for record in records
            )
            / len(records),
        },
    }
    _atomic_json(result_path, payload)
    (output_dir / "summary.md").write_text(
        _summary(records, fit_config), encoding="utf-8"
    )
    print(f"[ReCalKV G-LRD OVC] merged {len(records)} layers into {result_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    fit = subparsers.add_parser("fit-shard")
    fit.add_argument("--model", required=True)
    fit.add_argument("--whitening-dir", required=True)
    fit.add_argument("--ovc-whitening-dir")
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--layers", default="all")
    fit.add_argument("--group-size", type=int, default=4)
    fit.add_argument("--rank-per-group", type=int, default=384)
    fit.add_argument("--ovc-iterations", type=int, default=1)
    fit.add_argument("--work-dtype", choices=("float32", "float64"), default="float32")
    fit.add_argument(
        "--factor-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    fit.add_argument("--layer-shard-index", type=int, required=True)
    fit.add_argument("--layer-shard-count", type=int, required=True)
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--torch-num-threads", type=int, default=2)
    fit.add_argument("--resume", action="store_true")
    merge = subparsers.add_parser("merge")
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
