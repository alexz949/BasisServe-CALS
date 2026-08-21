#!/usr/bin/env python3
"""Build all-layer activation-aware Qwen3.5 GDN Private AG factors."""

from __future__ import annotations

import argparse
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
    fit_gdn_private_ag_joint_factors,
)
from basisserve.core.qwen35_gdn_private_ag_runtime import (  # noqa: E402
    FACTOR_FORMAT,
)
from scripts.collect_qwen35_gdn_wo_activations import FORMAT as MOMENT_FORMAT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--moments", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--baseline-rank", type=int, default=1536)
    parser.add_argument("--decoder-relative-ridge", type=float, default=3.0)
    parser.add_argument("--relative-damping", type=float, default=1.0e-5)
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


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if args.tp <= 1 or args.baseline_rank <= 0:
        raise ValueError("TP and baseline rank must be positive")
    model_path = Path(args.model_path).expanduser().resolve()
    moments_path = Path(args.moments).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Private AG factors: {output_path}")
    moments = torch.load(moments_path, map_location="cpu", weights_only=True)
    if (
        moments.get("format") != MOMENT_FORMAT
        or int(moments.get("schema_version", -1)) != 1
    ):
        raise ValueError("unsupported Qwen3.5 GDN post-gate moments")
    by_layer = {int(layer["layer_index"]): layer for layer in moments["layers"]}
    available = tuple(sorted(by_layer))
    layers = _parse_layers(args.layers, available)
    geometry = moments["geometry"]
    input_width = int(geometry["wire_input_width"])
    if input_width % args.tp:
        raise ValueError("GDN wire does not divide across TP")
    total_private_rank = 2 * args.baseline_rank
    if total_private_rank % args.tp:
        raise ValueError("equal-ring Private AG rank does not divide across TP")
    local_rank = total_private_rank // args.tp
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
        tensor_name = (
            f"model.language_model.layers.{layer_index}.linear_attn.out_proj.weight"
        )
        shard_name = weight_map.get(tensor_name)
        if shard_name is None:
            raise KeyError(tensor_name)
        with safe_open(
            str(model_path / shard_name),
            framework="pt",
            device="cpu",
        ) as handle:
            weight = handle.get_tensor(tensor_name).to(device)
        moment_payload = by_layer[layer_index]["moments"]
        rows = int(moment_payload["rows"])
        second_moment = moment_payload["gram"].to(device) / float(rows)
        print(
            f"[GDNPrivateAGBuild] layer={layer_index} {ordinal}/{len(layers)} phase=fit",
            flush=True,
        )
        fitted = fit_gdn_private_ag_joint_factors(
            weight,
            second_moment,
            tp_size=args.tp,
            local_rank=local_rank,
            decoder_relative_ridge=args.decoder_relative_ridge,
            relative_damping=args.relative_damping,
            factor_dtype=factor_dtype,
        )
        layer_peak = int(torch.cuda.max_memory_allocated(cuda_index))
        peak_cuda = max(peak_cuda, layer_peak)
        output_layers.append(
            {
                "layer_index": layer_index,
                "tensor_name": tensor_name,
                "rows": rows,
                "private_encoders": fitted.private_encoders,
                "joint_decoder_weight": fitted.joint_decoder_weight,
                "metrics": fitted.metrics,
                "elapsed_seconds": time.perf_counter() - layer_started,
                "peak_cuda_allocated_bytes": layer_peak,
            }
        )
        print(
            f"[GDNPrivateAGBuild] layer={layer_index} complete "
            f"ind={fitted.metrics['independent_relative_output_mse']:.8g} "
            f"joint={fitted.metrics['joint_relative_output_mse']:.8g} "
            f"bf16={fitted.metrics['quantized_joint_relative_output_mse']:.8g}",
            flush=True,
        )
        del weight, second_moment, fitted
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    result = {
        "format": FACTOR_FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "timestamp_started_utc": timestamp_started,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "model_path": str(model_path),
        "source_moments": str(moments_path),
        "geometry": geometry,
        "collection": moments["collection"],
        "layers_requested": list(layers),
        "tp_size": args.tp,
        "baseline_allreduce_rank": args.baseline_rank,
        "equal_ring_units": 2 * args.baseline_rank,
        "local_rank": local_rank,
        "total_private_rank": total_private_rank,
        "decoder_relative_ridge": args.decoder_relative_ridge,
        "relative_damping": args.relative_damping,
        "factor_dtype": args.factor_dtype,
        "objective": "activation_aware_complete_gdn_out_proj_reconstruction",
        "recurrent_state_compressed": False,
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
        f"[GDNPrivateAGBuild] complete layers={len(output_layers)} "
        f"output={output_path} seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
