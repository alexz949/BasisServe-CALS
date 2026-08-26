#!/usr/bin/env python3
"""Build decoder-closed ALS factors for all Qwen3.5 GDN output wires."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from safetensors import safe_open
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.qwen35_gdn_private_ag import (  # noqa: E402
    fit_qwen35_private_ag_joint_factors,
)
from basisserve.core.qwen35_gdn_private_ag_runtime import (  # noqa: E402
    FACTOR_FORMAT as GDN_FACTOR_FORMAT,
)
from scripts.collect_qwen35_gdn_wo_activations import (  # noqa: E402
    FORMAT as GDN_MOMENT_FORMAT,
)


@dataclass(frozen=True)
class Qwen35PrivateAGBuildSpec:
    description: str
    block_type: str
    moment_format: str
    factor_format: str
    tensor_name_template: str
    log_label: str
    objective: str


GDN_BUILD_SPEC = Qwen35PrivateAGBuildSpec(
    description=__doc__,
    block_type="linear_attention",
    moment_format=GDN_MOMENT_FORMAT,
    factor_format=GDN_FACTOR_FORMAT,
    tensor_name_template=(
        "model.language_model.layers.{layer_index}.linear_attn.out_proj.weight"
    ),
    log_label="GDNPrivateAGBuild",
    objective="decoder_closed_c1_als_gdn_out_proj_reconstruction",
)


def parse_args(spec: Qwen35PrivateAGBuildSpec = GDN_BUILD_SPEC) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=spec.description)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--fit-moments", required=True)
    parser.add_argument("--heldout-moments", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument(
        "--local-rank",
        type=int,
        default=256,
        help="C1 rank per TP source; 256 gives 75%% reduction for TP8/H=4096.",
    )
    parser.add_argument("--encoder-sweeps", type=int, default=5)
    parser.add_argument("--minimum-encoder-sweeps", type=int, default=1)
    parser.add_argument("--covariance-damping", type=float, default=1.0e-5)
    parser.add_argument("--decoder-relative-jitter", type=float, default=1.0e-8)
    parser.add_argument("--encoder-relative-damping", type=float, default=1.0e-5)
    parser.add_argument(
        "--encoder-cg-relative-tolerance", type=float, default=1.0e-5
    )
    parser.add_argument("--encoder-cg-max-iterations", type=int, default=32)
    parser.add_argument("--encoder-cg-fixed-iterations", action="store_true")
    parser.add_argument("--encoder-relative-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--encoder-patience", type=int, default=1)
    parser.add_argument("--maximum-backtracks", type=int, default=8)
    parser.add_argument(
        "--factor-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _parse_layers(raw: str, available: tuple[int, ...]) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return available
    values = tuple(sorted({int(piece.strip()) for piece in raw.split(",") if piece.strip()}))
    missing = sorted(set(values) - set(available))
    if not values or missing:
        raise ValueError(f"invalid layer selection; missing={missing}")
    return values


def _weight_map(model_path: Path) -> dict[str, str]:
    payload = json.loads(
        (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    return dict(payload["weight_map"])


def _load_moments(
    path: Path,
    spec: Qwen35PrivateAGBuildSpec = GDN_BUILD_SPEC,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != spec.moment_format
        or int(payload.get("schema_version", -1)) != 1
    ):
        raise ValueError(f"unsupported Qwen3.5 {spec.block_type} moments: {path}")
    return payload


def _record_sample_indices(payload: dict[str, Any]) -> set[int]:
    return {
        int(record["sample_index"])
        for record in payload["collection"].get("records", ())
    }


def _validate_moment_pair(
    fit: dict[str, Any],
    heldout: dict[str, Any],
) -> None:
    for key in ("geometry", "model"):
        if fit[key] != heldout[key]:
            raise ValueError(f"fit and held-out moments differ in {key}")
    fit_layers = tuple(int(layer["layer_index"]) for layer in fit["layers"])
    heldout_layers = tuple(int(layer["layer_index"]) for layer in heldout["layers"])
    if fit_layers != heldout_layers:
        raise ValueError("fit and held-out moments cover different layers")
    overlap = _record_sample_indices(fit) & _record_sample_indices(heldout)
    if overlap:
        raise ValueError(
            f"fit and held-out moments reuse calibration windows: {sorted(overlap)[:8]}"
        )


def main(spec: Qwen35PrivateAGBuildSpec = GDN_BUILD_SPEC) -> None:
    args = parse_args(spec)
    torch.set_num_threads(args.torch_num_threads)
    if args.tp <= 1 or args.local_rank <= 0:
        raise ValueError("TP and local rank must be positive")
    model_path = Path(args.model_path).expanduser().resolve()
    fit_moments_path = Path(args.fit_moments).expanduser().resolve()
    heldout_moments_path = Path(args.heldout_moments).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Private AG factors: {output_path}")
    fit_moments = _load_moments(fit_moments_path, spec)
    heldout_moments = _load_moments(heldout_moments_path, spec)
    _validate_moment_pair(fit_moments, heldout_moments)
    fit_by_layer = {
        int(layer["layer_index"]): layer for layer in fit_moments["layers"]
    }
    heldout_by_layer = {
        int(layer["layer_index"]): layer for layer in heldout_moments["layers"]
    }
    by_layer = fit_by_layer
    available = tuple(sorted(by_layer))
    layers = _parse_layers(args.layers, available)
    geometry = fit_moments["geometry"]
    input_width = int(geometry["wire_input_width"])
    if input_width % args.tp:
        raise ValueError(f"{spec.block_type} wire does not divide across TP")
    local_rank = args.local_rank
    total_private_rank = args.tp * local_rank
    local_width = input_width // args.tp
    if local_rank > local_width:
        raise ValueError("Private AG local rank exceeds local wire width")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("all-layer Private AG factor fitting requires CUDA")
    cuda_index = device.index if device.index is not None else 0
    torch.cuda.set_device(cuda_index)
    weight_map = _weight_map(model_path)
    factor_dtype = _dtype(args.factor_dtype)
    started = time.perf_counter()
    timestamp_started = datetime.now(timezone.utc).isoformat()
    output_layers: list[dict[str, Any]] = []
    peak_cuda = 0

    for ordinal, layer_index in enumerate(layers, start=1):
        layer_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(cuda_index)
        tensor_name = spec.tensor_name_template.format(layer_index=layer_index)
        shard_name = weight_map.get(tensor_name)
        if shard_name is None:
            raise KeyError(tensor_name)
        with safe_open(
            str(model_path / shard_name),
            framework="pt",
            device="cpu",
        ) as handle:
            weight = handle.get_tensor(tensor_name).to(device)
        fit_payload = fit_by_layer[layer_index]["moments"]
        heldout_payload = heldout_by_layer[layer_index]["moments"]
        fit_rows = int(fit_payload["rows"])
        heldout_rows = int(heldout_payload["rows"])
        fit_second_moment = fit_payload["gram"].to(device) / float(fit_rows)
        heldout_second_moment = (
            heldout_payload["gram"].to(device) / float(heldout_rows)
        )
        print(
            f"[{spec.log_label}] layer={layer_index} "
            f"{ordinal}/{len(layers)} phase=fit",
            flush=True,
        )
        fitted = fit_qwen35_private_ag_joint_factors(
            weight,
            fit_second_moment,
            heldout_second_moment,
            tp_size=args.tp,
            local_rank=local_rank,
            encoder_sweeps=args.encoder_sweeps,
            minimum_encoder_sweeps=args.minimum_encoder_sweeps,
            covariance_damping=args.covariance_damping,
            decoder_relative_jitter=args.decoder_relative_jitter,
            encoder_relative_damping=args.encoder_relative_damping,
            encoder_cg_relative_tolerance=args.encoder_cg_relative_tolerance,
            encoder_cg_max_iterations=args.encoder_cg_max_iterations,
            encoder_cg_fixed_iterations=args.encoder_cg_fixed_iterations,
            encoder_relative_tolerance=args.encoder_relative_tolerance,
            encoder_patience=args.encoder_patience,
            maximum_backtracks=args.maximum_backtracks,
            factor_dtype=factor_dtype,
        )
        layer_peak = int(torch.cuda.max_memory_allocated(cuda_index))
        peak_cuda = max(peak_cuda, layer_peak)
        output_layers.append(
            {
                "layer_index": layer_index,
                "tensor_name": tensor_name,
                "fit_rows": fit_rows,
                "heldout_rows": heldout_rows,
                "private_encoders": fitted.private_encoders,
                "joint_decoder_weight": fitted.joint_decoder_weight,
                "metrics": fitted.metrics,
                "elapsed_seconds": time.perf_counter() - layer_started,
                "peak_cuda_allocated_bytes": layer_peak,
            }
        )
        print(
            f"[{spec.log_label}] layer={layer_index} complete "
            f"fit={fitted.metrics['fit_relative_output_mse']:.8g} "
            f"heldout={fitted.metrics['heldout_relative_output_mse']:.8g} "
            f"quantized_heldout="
            f"{fitted.metrics['quantized_heldout_relative_output_mse']:.8g} "
            f"selected={fitted.metrics['selected_boundary']}:"
            f"{fitted.metrics['selected_sweep']}",
            flush=True,
        )
        del weight, fit_second_moment, heldout_second_moment, fitted
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    result = {
        "format": spec.factor_format,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "timestamp_started_utc": timestamp_started,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "model_path": str(model_path),
        "fit_moments": str(fit_moments_path),
        "heldout_moments": str(heldout_moments_path),
        "geometry": geometry,
        "fit_collection": fit_moments["collection"],
        "heldout_collection": heldout_moments["collection"],
        "layers_requested": list(layers),
        "tp_size": args.tp,
        "local_rank": local_rank,
        "total_private_rank": total_private_rank,
        "als": {
            "encoder_sweeps": args.encoder_sweeps,
            "minimum_encoder_sweeps": args.minimum_encoder_sweeps,
            "covariance_damping": args.covariance_damping,
            "decoder_relative_jitter": args.decoder_relative_jitter,
            "encoder_relative_damping": args.encoder_relative_damping,
            "encoder_cg_relative_tolerance": (
                args.encoder_cg_relative_tolerance
            ),
            "encoder_cg_max_iterations": args.encoder_cg_max_iterations,
            "encoder_cg_fixed_iterations": args.encoder_cg_fixed_iterations,
            "encoder_relative_tolerance": args.encoder_relative_tolerance,
            "encoder_patience": args.encoder_patience,
            "maximum_backtracks": args.maximum_backtracks,
        },
        "factor_dtype": args.factor_dtype,
        "block_type": spec.block_type,
        "objective": spec.objective,
        "recurrent_state_compressed": False,
        "attention_cache_compressed": False,
        "layers": output_layers,
        "environment": {
            "conda": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": str(torch.__version__),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(cuda_index),
            "torch_num_threads": torch.get_num_threads(),
            "peak_cuda_allocated_bytes": peak_cuda,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(result, temporary)
    os.replace(temporary, output_path)
    print(
        f"[{spec.log_label}] complete layers={len(output_layers)} "
        f"output={output_path} seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
