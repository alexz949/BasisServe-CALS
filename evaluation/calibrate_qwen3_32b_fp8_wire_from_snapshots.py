#!/usr/bin/env python3
"""Derive Qwen3-32B C1 FP8 wire scales from saved pre-o_proj activations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from basisserve.core.qwen3_32b_tp4_decode import (  # noqa: E402
    FACTOR_FORMAT,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_QUERY_HEADS,
    QUERY_HEADS_PER_KV_HEAD,
    TP_SIZE,
)
from basisserve.kernels.fp8_wire import (  # noqa: E402
    FP8_E4M3_MAX,
    quantize_e4m3_static,
)


FORMAT = "basisserve.qwen3_32b.snapshot_fp8_wire_calibration.v1"
SOURCES_PER_PROCESS = NUM_KV_HEADS // TP_SIZE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)


def _load_manifests(
    snapshot_dir: Path,
    factor_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot_path = snapshot_dir / "manifest.json"
    factor_path = factor_dir / "result.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    factors = json.loads(factor_path.read_text(encoding="utf-8"))
    if snapshot.get("format") != "basisserve.attention_o_proj_ppl_snapshots.v1":
        raise ValueError("snapshot manifest has an unsupported format")
    geometry = snapshot.get("model", {})
    expected_geometry = {
        "num_attention_heads": NUM_QUERY_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "num_hidden_layers": NUM_LAYERS,
    }
    for key, expected in expected_geometry.items():
        if int(geometry.get(key, -1)) != expected:
            raise ValueError(f"snapshot geometry mismatch for {key}")
    if factors.get("format") != FACTOR_FORMAT or factors.get("status") != "complete":
        raise ValueError("factor manifest is not a completed Qwen3-32B C1 checkpoint")
    schedule = factors.get("selection", {}).get("selected_schedule")
    if not isinstance(schedule, list) or len(schedule) != NUM_LAYERS:
        raise ValueError("factor schedule does not cover all layers")
    return snapshot, factors


def _layer_tensors(
    snapshot_dir: Path,
    factor_dir: Path,
    snapshot_manifest: dict[str, Any],
    factor_manifest: dict[str, Any],
    layer: int,
) -> tuple[Tensor, Tensor, dict[str, str]]:
    snapshot_record = snapshot_manifest["artifacts"][str(layer)]
    factor_record = factor_manifest["artifacts"][str(layer)]
    snapshot_path = snapshot_dir / snapshot_record["file"]
    factor_path = factor_dir / factor_record["file"]
    snapshot_hash = _sha256(snapshot_path)
    factor_hash = _sha256(factor_path)
    if snapshot_hash != snapshot_record["sha256"]:
        raise ValueError(f"snapshot hash mismatch at layer {layer}")
    if factor_hash != factor_record["sha256"]:
        raise ValueError(f"factor hash mismatch at layer {layer}")
    snapshot_payload = load_file(str(snapshot_path), device="cpu")
    if set(snapshot_payload) != {"activation", "weight"}:
        raise ValueError(f"unexpected snapshot tensors at layer {layer}")
    with safe_open(factor_path, framework="pt", device="cpu") as handle:
        factor_keys = set(handle.keys())
        required = {
            "value_coordinate_encoders",
            "head_output_decoders",
            "source_ranks",
        }
        if factor_keys != required:
            raise ValueError(f"unexpected factor tensors at layer {layer}")
        encoders = handle.get_tensor("value_coordinate_encoders").contiguous()
        source_ranks = handle.get_tensor("source_ranks")
    activation = snapshot_payload["activation"].contiguous()
    expected_activation = (
        int(snapshot_manifest["calibration"]["rows_per_layer"]),
        NUM_QUERY_HEADS * HEAD_DIM,
    )
    if tuple(activation.shape) != expected_activation:
        raise ValueError(f"unexpected activation shape at layer {layer}")
    if tuple(encoders.shape) != (NUM_KV_HEADS, HEAD_DIM, 64):
        raise ValueError(f"uniform V64 encoder shape mismatch at layer {layer}")
    ranks = tuple(map(int, source_ranks.tolist()))
    if ranks != (64,) * NUM_KV_HEADS:
        raise ValueError(f"layer {layer} is not uniform V64: {ranks}")
    scheduled = tuple(
        map(int, factor_manifest["selection"]["selected_schedule"][layer])
    )
    if scheduled != ranks:
        raise ValueError(f"factor schedule mismatch at layer {layer}")
    return activation, encoders, {
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": snapshot_hash,
        "factor_path": str(factor_path),
        "factor_sha256": factor_hash,
    }


@torch.inference_mode()
def project_process_latents(
    activation: Tensor,
    encoders: Tensor,
) -> tuple[Tensor, ...]:
    """Project dense head outputs and pack two KV sources per TP4 process."""

    if activation.ndim != 2 or encoders.ndim != 3:
        raise ValueError("activation and encoders must be rank-2/rank-3 tensors")
    sources, head_dim, rank = map(int, encoders.shape)
    if sources <= 0 or int(activation.shape[1]) % (sources * head_dim):
        raise ValueError("activation width is incompatible with encoder geometry")
    query_heads = int(activation.shape[1]) // head_dim
    if query_heads % sources:
        raise ValueError("query heads must divide evenly across KV sources")
    heads_per_source = query_heads // sources
    if sources % TP_SIZE:
        raise ValueError("KV sources must divide evenly across TP processes")
    source_outputs: list[Tensor] = []
    heads = activation.view(int(activation.shape[0]), query_heads, head_dim)
    for source in range(sources):
        head_start = source * heads_per_source
        head_stop = head_start + heads_per_source
        source_outputs.append(
            torch.matmul(heads[:, head_start:head_stop], encoders[source])
        )
    sources_per_process = sources // TP_SIZE
    return tuple(
        torch.cat(
            source_outputs[
                process * sources_per_process : (process + 1) * sources_per_process
            ],
            dim=1,
        ).reshape(int(activation.shape[0]), sources_per_process * heads_per_source * rank)
        for process in range(TP_SIZE)
    )


@torch.inference_mode()
def _calibrate_layer(
    activation: Tensor,
    encoders: Tensor,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, list[dict[str, float | int]]]:
    activation_device = activation.to(device=device, non_blocking=False)
    encoders_device = encoders.to(
        device=device,
        dtype=activation_device.dtype,
        non_blocking=False,
    )
    local_latents = project_process_latents(activation_device, encoders_device)
    process_amax: list[Tensor] = []
    process_scales: list[Tensor] = []
    metrics: list[dict[str, float | int]] = []
    for local in local_latents:
        amax = local.float().abs().amax()
        scale = (amax / FP8_E4M3_MAX).clamp_min(torch.finfo(torch.float32).tiny)
        quantized = quantize_e4m3_static(local, scale)
        reconstructed = quantized.float() * scale
        reference = local.float()
        delta = reconstructed - reference
        energy = reference.square().sum().clamp_min(torch.finfo(torch.float32).tiny)
        clipped = int((reference.abs() > scale * FP8_E4M3_MAX).sum().item())
        process_amax.append(amax)
        process_scales.append(scale)
        metrics.append(
            {
                "amax": float(amax.item()),
                "scale": float(scale.item()),
                "relative_rmse": float((delta.square().sum() / energy).sqrt().item()),
                "maximum_absolute_error": float(delta.abs().amax().item()),
                "clipped_elements": clipped,
                "elements": local.numel(),
            }
        )
    return (
        torch.stack(process_amax).cpu(),
        torch.stack(process_scales).cpu(),
        metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--output-scales", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    args = parser.parse_args()

    if args.torch_num_threads <= 0:
        raise ValueError("torch thread count must be positive")
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("snapshot FP8 calibration requires CUDA")
    torch.cuda.set_device(device)
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    factor_dir = Path(args.factor_dir).expanduser().resolve()
    snapshot_manifest, factor_manifest = _load_manifests(
        snapshot_dir,
        factor_dir,
    )

    started = time.perf_counter()
    wire_amax: list[Tensor] = []
    wire_scales: list[Tensor] = []
    records: list[dict[str, Any]] = []
    for layer in range(NUM_LAYERS):
        activation, encoders, sources = _layer_tensors(
            snapshot_dir,
            factor_dir,
            snapshot_manifest,
            factor_manifest,
            layer,
        )
        amax, scales, metrics = _calibrate_layer(
            activation,
            encoders,
            device=device,
        )
        wire_amax.append(amax)
        wire_scales.append(scales)
        records.append(
            {
                "layer": layer,
                "processes": metrics,
                "sources": sources,
            }
        )
        del activation, encoders
        print(
            json.dumps(
                {
                    "event": "layer_complete",
                    "layer": layer,
                    "amax": amax.tolist(),
                    "relative_rmse": [item["relative_rmse"] for item in metrics],
                }
            ),
            flush=True,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    amax_tensor = torch.stack(wire_amax).float().contiguous()
    scale_tensor = torch.stack(wire_scales).float().contiguous()
    output_scales = Path(args.output_scales).expanduser().resolve()
    _atomic_safetensors(
        output_scales,
        {"wire_amax": amax_tensor, "wire_scales": scale_tensor},
    )
    relative_rmse = [
        float(process["relative_rmse"])
        for record in records
        for process in record["processes"]
    ]
    clipped = sum(
        int(process["clipped_elements"])
        for record in records
        for process in record["processes"]
    )
    elements = sum(
        int(process["elements"])
        for record in records
        for process in record["processes"]
    )
    result: dict[str, Any] = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "tp_size": TP_SIZE,
            "sources_per_process": SOURCES_PER_PROCESS,
            "query_heads_per_source": QUERY_HEADS_PER_KV_HEAD,
            "rank_per_source": 64,
            "wire_dtype": "float8_e4m3fn",
            "fp8_max": FP8_E4M3_MAX,
            "scale_rule": "per-layer/per-TP-process amax / fp8_max",
            "projection": "saved pre-o_proj head output @ C1 value encoder",
            "algebra": "attention(V @ A_s) = attention(V) @ A_s",
        },
        "inputs": {
            "snapshot_dir": str(snapshot_dir),
            "snapshot_manifest_sha256": _sha256(snapshot_dir / "manifest.json"),
            "factor_dir": str(factor_dir),
            "factor_manifest_sha256": _sha256(factor_dir / "result.json"),
            "snapshot_rows_per_layer": int(
                snapshot_manifest["calibration"]["rows_per_layer"]
            ),
            "snapshot_windows": int(
                snapshot_manifest["calibration"]["window_count"]
            ),
            "positions_per_window": int(
                snapshot_manifest["calibration"]["positions_per_window"]
            ),
        },
        "summary": {
            "mean_relative_rmse": statistics.fmean(relative_rmse),
            "maximum_relative_rmse": max(relative_rmse),
            "minimum_scale": float(scale_tensor.min().item()),
            "maximum_scale": float(scale_tensor.max().item()),
            "clipped_elements": clipped,
            "elements": elements,
            "clipped_fraction": clipped / elements,
        },
        "artifact": {
            "path": str(output_scales),
            "sha256": _sha256(output_scales),
            "shape": list(scale_tensor.shape),
            "keys": ["wire_amax", "wire_scales"],
        },
        "layers": records,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "elapsed_seconds": elapsed,
    }
    output_json = Path(args.output_json).expanduser().resolve()
    _atomic_json(output_json, result)
    print(
        json.dumps(
            {
                "event": "calibration_complete",
                "output_scales": str(output_scales),
                "output_json": str(output_json),
                "summary": result["summary"],
                "elapsed_seconds": elapsed,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
