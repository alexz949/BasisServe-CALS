#!/usr/bin/env python3
"""Finalize a complete attention snapshot left after a capture timeout.

This recovery path never rewrites activation artifacts.  It validates the
existing safetensors files, recomputes their hashes, writes the missing
manifest, and atomically promotes ``OUTPUT.partial`` to ``OUTPUT``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file
import torch
from transformers import AutoConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.analyze_qwen35_gated_attention_sparsity import (  # noqa: E402
    _git_commit,
    _installed_version,
)
from evaluation.analyze_qwen3_o_proj_collective_endpoints import (  # noqa: E402
    _sample_positions,
)
from evaluation.capture_attention_o_proj_ppl_snapshots import (  # noqa: E402
    FORMAT,
    _atomic_json,
    _limit_windows,
    _load_windows,
    _parse_layers,
    _position_sampling_indices,
    _sha256,
    _validate_attention,
)


_SAFETENSORS_TO_TORCH_DTYPE = {
    "BF16": "torch.bfloat16",
    "F16": "torch.float16",
    "F32": "torch.float32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-attention", choices=("gqa", "mha"), required=True
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--window-split", choices=("all", "fit", "heldout"), default="all"
    )
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--positions-per-window", type=int, default=128)
    parser.add_argument("--position-seed", type=int, default=20260901)
    parser.add_argument(
        "--position-record-field", choices=("source_window_id",)
    )
    parser.add_argument("--capture-command", required=True)
    parser.add_argument("--capture-job-id", required=True)
    parser.add_argument("--capture-started-utc", required=True)
    parser.add_argument("--capture-finished-utc", required=True)
    parser.add_argument("--capture-elapsed-seconds", type=float, required=True)
    parser.add_argument("--capture-conda-environment", default="lowrankarena")
    parser.add_argument("--capture-cuda-device", required=True)
    parser.add_argument("--capture-torch-num-threads", type=int, default=4)
    parser.add_argument("--hash-workers", type=int, default=4)
    return parser.parse_args()


def _artifact_metadata(path: Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"activation", "weight"}:
            raise ValueError(f"unexpected safetensors keys in {path}")
        activation = handle.get_slice("activation")
        weight = handle.get_slice("weight")
        activation_dtype = str(activation.get_dtype())
        weight_dtype = str(weight.get_dtype())
        if activation_dtype not in _SAFETENSORS_TO_TORCH_DTYPE:
            raise ValueError(f"unsupported activation dtype in {path}")
        if weight_dtype not in _SAFETENSORS_TO_TORCH_DTYPE:
            raise ValueError(f"unsupported weight dtype in {path}")
        return {
            "activation_shape": list(map(int, activation.get_shape())),
            "weight_shape": list(map(int, weight.get_shape())),
            "activation_dtype": _SAFETENSORS_TO_TORCH_DTYPE[activation_dtype],
            "weight_dtype": _SAFETENSORS_TO_TORCH_DTYPE[weight_dtype],
        }


def _hash_artifacts(paths: dict[int, Path], workers: int) -> dict[int, str]:
    if workers <= 0:
        raise ValueError("hash workers must be positive")
    hashes: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(_sha256, path): layer for layer, path in paths.items()}
        for future in as_completed(pending):
            layer = pending[future]
            hashes[layer] = future.result()
            print(f"[Finalize] hashed layer={layer}", flush=True)
    return hashes


def main() -> None:
    args = parse_args()
    if args.positions_per_window <= 0:
        raise ValueError("positions per window must be positive")

    model_path = Path(args.model_path).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    if not partial_dir.is_dir():
        raise FileNotFoundError(partial_dir)
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    geometry = _validate_attention(config, args.expected_attention)
    layers = _parse_layers(args.layers, int(geometry["num_hidden_layers"]))

    windows, window_manifest, window_indices = _load_windows(
        windows_path, args.window_split
    )
    windows, window_indices = _limit_windows(
        windows, window_indices, args.max_windows
    )
    if args.positions_per_window > int(windows.shape[1]):
        raise ValueError("positions per window exceed the sequence length")
    position_sampling_indices = _position_sampling_indices(
        window_manifest=window_manifest,
        window_indices=window_indices,
        record_field=args.position_record_field,
    )
    position_pool = _sample_positions(
        windows=max(position_sampling_indices) + 1,
        sequence_length=int(windows.shape[1]),
        positions_per_window=args.positions_per_window,
        seed=args.position_seed,
    )
    expected_positions = position_pool.index_select(
        0, torch.tensor(position_sampling_indices, dtype=torch.long)
    ).contiguous()
    positions_path = partial_dir / "sampled_positions.safetensors"
    observed_positions = load_file(str(positions_path), device="cpu").get("positions")
    if observed_positions is None or not torch.equal(
        observed_positions, expected_positions
    ):
        raise ValueError("saved sampled positions differ from deterministic positions")

    rows = int(windows.shape[0]) * args.positions_per_window
    artifact_paths = {
        layer: partial_dir / f"layer_{layer:03d}.safetensors" for layer in layers
    }
    metadata: dict[int, dict[str, Any]] = {}
    for layer, path in artifact_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        record = _artifact_metadata(path)
        if tuple(record["activation_shape"]) != (rows, int(geometry["hidden_size"])):
            raise ValueError(f"unexpected activation shape in {path}")
        if tuple(record["weight_shape"]) != (
            int(geometry["hidden_size"]),
            int(geometry["hidden_size"]),
        ):
            raise ValueError(f"unexpected weight shape in {path}")
        metadata[layer] = record
    temporary_files = sorted(path.name for path in partial_dir.glob(".tmp*"))
    if temporary_files:
        raise ValueError(f"stale temporary files remain: {temporary_files}")

    hashes = _hash_artifacts(artifact_paths, args.hash_workers)
    artifacts = {
        str(layer): {
            "file": artifact_paths[layer].name,
            "sha256": hashes[layer],
            **metadata[layer],
        }
        for layer in layers
    }
    manifest = {
        "format": FORMAT,
        "schema_version": 1,
        "command": args.capture_command,
        "git_commit": _git_commit(),
        "timestamp_started_utc": args.capture_started_utc,
        "timestamp_finished_utc": args.capture_finished_utc,
        "elapsed_seconds": args.capture_elapsed_seconds,
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            **geometry,
        },
        "layers": list(layers),
        "calibration": {
            "activation_aware": True,
            "objective": "attention_o_proj_output_pod",
            "error_evaluation": False,
            "windows_file": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "windows_manifest_format": window_manifest.get("format"),
            "window_split": args.window_split,
            "max_windows": args.max_windows,
            "window_indices": window_indices,
            "window_count": int(windows.shape[0]),
            "sequence_length": int(windows.shape[1]),
            "positions_per_window": args.positions_per_window,
            "position_seed": args.position_seed,
            "position_record_field": args.position_record_field,
            "position_sampling_indices": position_sampling_indices,
            "rows_per_layer": rows,
            "sampled_positions_file": positions_path.name,
            "sampled_positions_sha256": _sha256(positions_path),
        },
        "artifacts": artifacts,
        "environment": {
            "conda_environment": args.capture_conda_environment,
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "cuda_device": args.capture_cuda_device,
            "peak_cuda_allocated_bytes": None,
            "torch_num_threads": args.capture_torch_num_threads,
        },
        "recovery": {
            "capture_job_id": args.capture_job_id,
            "capture_terminal_state": "TIMEOUT",
            "capture_artifacts_rewritten": False,
            "finalizer_command": shlex.join(sys.argv),
            "finalized_utc": datetime.now(timezone.utc).isoformat(),
            "hash_workers": args.hash_workers,
            "note": (
                "All layer artifacts were complete after the capture job timed out "
                "while finalizing the last layer hash."
            ),
        },
    }
    _atomic_json(partial_dir / "manifest.json", manifest)
    os.replace(partial_dir, output_dir)
    print(f"[Finalize] complete output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
