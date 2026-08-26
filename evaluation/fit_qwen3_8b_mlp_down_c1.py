#!/usr/bin/env python3
"""Fit Qwen3-8B MLP ``down_proj`` shared-decoder C1 factors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.mlp_down_c1 import FACTOR_FORMAT, fit_mlp_down_c1  # noqa: E402
from evaluation.capture_attention_o_proj_ppl_snapshots import (  # noqa: E402
    _atomic_json,
    _git_commit,
    _installed_version,
    _sha256,
)
from evaluation.capture_mlp_down_output_covariances import (  # noqa: E402
    FORMAT as COVARIANCE_FORMAT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--covariance-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=2048)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument(
        "--factor-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        mapping = payload.get("weight_map")
        if not isinstance(mapping, dict):
            raise ValueError("model safetensors index has no weight_map")
        return {str(name): str(file) for name, file in mapping.items()}
    single = model_path / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(index_path)
    with safe_open(str(single), framework="pt", device="cpu") as handle:
        return {name: single.name for name in handle.keys()}


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != COVARIANCE_FORMAT
        or manifest.get("status") != "complete"
        or int(manifest.get("schema_version", -1)) != 1
    ):
        raise ValueError("MLP output covariance artifact is incompatible")
    return manifest


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("MLP C1 fitting requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    model_path = Path(args.model_path).expanduser().resolve()
    covariance_dir = Path(args.covariance_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    covariance_manifest = _load_manifest(covariance_dir)
    if covariance_manifest["model"]["config_sha256"] != _sha256(
        model_path / "config.json"
    ):
        raise ValueError("MLP covariance and model config differ")
    hidden_size = int(covariance_manifest["model"]["hidden_size"])
    intermediate_size = int(covariance_manifest["model"]["intermediate_size"])
    num_layers = int(covariance_manifest["model"]["num_hidden_layers"])
    if (hidden_size, intermediate_size, num_layers) != (4096, 12288, 36):
        raise ValueError("unexpected Qwen3-8B MLP geometry")
    if not 0 < args.rank <= hidden_size:
        raise ValueError("rank must be within the hidden width")
    if args.tp_size <= 1 or intermediate_size % args.tp_size:
        raise ValueError("TP size must divide the MLP intermediate width")
    layers = tuple(map(int, covariance_manifest["layers"]))
    if layers != tuple(range(num_layers)):
        raise ValueError("MLP covariance does not cover all model layers")

    factor_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.factor_dtype]
    weight_map = _weight_map(model_path)
    partial_dir.mkdir(parents=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    artifacts: dict[str, Any] = {}
    metrics: list[dict[str, Any]] = []
    for ordinal, layer in enumerate(layers, start=1):
        layer_started = time.perf_counter()
        covariance_record = covariance_manifest["artifacts"][str(layer)]
        covariance_path = covariance_dir / covariance_record["file"]
        if _sha256(covariance_path) != covariance_record["sha256"]:
            raise ValueError(f"MLP covariance hash mismatch at layer {layer}")
        moments = load_file(str(covariance_path), device=str(device))
        tensor_name = f"model.layers.{layer}.mlp.down_proj.weight"
        shard_name = weight_map.get(tensor_name)
        if shard_name is None:
            raise KeyError(tensor_name)
        with safe_open(
            str(model_path / shard_name), framework="pt", device="cpu"
        ) as handle:
            weight = handle.get_tensor(tensor_name).to(device)
        if tuple(weight.shape) != (hidden_size, intermediate_size):
            raise ValueError(f"layer {layer} has unexpected down_proj shape")
        print(
            f"[MLP C1 fit] layer={layer} {ordinal}/{num_layers} rank={args.rank}",
            flush=True,
        )
        factors = fit_mlp_down_c1(
            weight,
            moments["fit_output_second_moment"],
            moments["heldout_output_second_moment"],
            rank=args.rank,
            factor_dtype=factor_dtype,
        )
        factor_path = partial_dir / f"layer_{layer:03d}.safetensors"
        save_file(
            {
                "input_factor": factors.input_factor,
                "output_basis": factors.output_basis,
            },
            str(factor_path),
        )
        record = {
            "layer": layer,
            **dict(factors.metrics),
            "elapsed_seconds": time.perf_counter() - layer_started,
        }
        metrics.append(record)
        artifacts[str(layer)] = {
            "file": factor_path.name,
            "sha256": _sha256(factor_path),
            "input_factor_shape": list(factors.input_factor.shape),
            "output_basis_shape": list(factors.output_basis.shape),
            "factor_dtype": str(factors.input_factor.dtype),
            "metrics": record,
        }
        print(
            f"[MLP C1 fit] layer={layer} fit_mse="
            f"{record['fit_relative_output_mse']:.8g} heldout_mse="
            f"{record['heldout_relative_output_mse']:.8g}",
            flush=True,
        )
        del moments, weight, factors
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    mean_fit = sum(row["fit_relative_output_mse"] for row in metrics) / len(metrics)
    mean_heldout = sum(
        row["heldout_relative_output_mse"] for row in metrics
    ) / len(metrics)
    manifest: Mapping[str, Any] = {
        "format": FACTOR_FORMAT,
        "schema_version": 1,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "timestamp_started_utc": started_utc,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            "model_type": "qwen3",
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_layers,
        },
        "covariance": {
            "path": str(covariance_dir),
            "manifest_sha256": _sha256(covariance_dir / "manifest.json"),
            "format": covariance_manifest["format"],
            "calibration": covariance_manifest["calibration"],
        },
        "fit_config": {
            "rank": args.rank,
            "tp_size": args.tp_size,
            "factor_dtype": args.factor_dtype,
            "objective": "post_swiglu_complete_mlp_output_mse_after_tp_sum",
            "algorithm": "teacher_output_pca_exact_shared_decoder",
            "damping": 0.0,
            "als_sweeps": 0,
        },
        "communication": {
            "collective": "latent_allreduce",
            "dense_width": hidden_size,
            "latent_width": args.rank,
            "fraction_of_dense_allreduce": args.rank / hidden_size,
            "reduction_fraction": 1.0 - args.rank / hidden_size,
            "ring_elements_per_token_per_rank": (
                2.0 * (args.tp_size - 1) / args.tp_size * args.rank
            ),
        },
        "summary": {
            "mean_fit_relative_output_mse": mean_fit,
            "mean_heldout_relative_output_mse": mean_heldout,
        },
        "layers": list(layers),
        "artifacts": artifacts,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "safetensors": _installed_version("safetensors"),
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(partial_dir / "manifest.json", manifest)
    os.replace(partial_dir, output_dir)
    print(
        f"[MLP C1 fit] complete output={output_dir} mean_fit={mean_fit:.8g} "
        f"mean_heldout={mean_heldout:.8g} seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
