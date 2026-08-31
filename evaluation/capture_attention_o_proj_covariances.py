#!/usr/bin/env python3
"""Stream full-token attention ``o_proj`` covariance sufficient statistics.

Every token position from the ordered fit and held-out C4 windows contributes
to a per-layer ``X^T X`` matrix.  Raw activations are never retained.
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

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
from transformers import AutoModel


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.capture_attention_o_proj_ppl_snapshots import (  # noqa: E402
    _atomic_json,
    _decoder_layers,
    _git_commit,
    _installed_version,
    _parse_layers,
    _sha256,
    _validate_attention as _validate_standard_attention,
)


FORMAT = "basisserve.attention_o_proj_covariances.v1"


def _cuda_device_indices() -> tuple[int, ...]:
    indices = []
    for index in range(torch.cuda.device_count()):
        try:
            torch.cuda.get_device_properties(index)
        except RuntimeError:
            continue
        indices.append(index)
    return tuple(indices)


def _validate_attention(config: Any, expected: str) -> dict[str, int | str]:
    if expected != "mla":
        return _validate_standard_attention(config, expected)
    if str(config.model_type) != "deepseek_v2":
        raise ValueError("MLA covariance capture currently supports DeepSeek-V2 only")
    hidden_size = int(config.hidden_size)
    num_heads = int(config.num_attention_heads)
    value_head_dim = int(config.v_head_dim)
    if min(hidden_size, num_heads, value_head_dim) <= 0:
        raise ValueError("checkpoint MLA geometry is invalid")
    return {
        "model_type": str(config.model_type),
        "attention_type": "mla",
        "hidden_size": hidden_size,
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,
        "head_dim": value_head_dim,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-attention", choices=("gqa", "mha", "mla"), required=True
    )
    parser.add_argument("--fit-windows", type=int, required=True)
    parser.add_argument("--heldout-windows", type=int, required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow checkpoints such as DeepSeek-V2-Lite to load custom model code",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-map",
        choices=("none", "balanced", "balanced_low_0"),
        default="none",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _load_ordered_windows(path: Path, *, expected: int) -> tuple[Tensor, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = load_file(str(path), device="cpu")
    if set(payload) != {"input_ids"}:
        raise ValueError("ordered covariance windows must contain only input_ids")
    windows = payload["input_ids"].to(torch.long).contiguous()
    if windows.ndim != 2 or len(windows) != expected:
        raise ValueError(
            f"expected {expected} ordered windows, found {tuple(windows.shape)}"
        )
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return windows, manifest


class _StreamingCovarianceCapture:
    def __init__(self, modules: Mapping[int, nn.Linear], input_width: int) -> None:
        self.modules = dict(modules)
        self.input_width = int(input_width)
        self.sums: dict[str, dict[int, Tensor]] = {"fit": {}, "heldout": {}}
        self.rows = {"fit": 0, "heldout": 0}
        self.active_split: str | None = None
        self.expected_rows = 0
        self.seen: set[int] = set()
        self.handles = [
            module.register_forward_pre_hook(self._hook(layer))
            for layer, module in self.modules.items()
        ]

    def _hook(self, layer: int):
        def capture(module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            del module
            if self.active_split is None or not inputs:
                raise RuntimeError("covariance hook fired outside an active batch")
            activation = inputs[0].detach()
            if activation.ndim != 3 or int(activation.shape[-1]) != self.input_width:
                raise ValueError(
                    f"layer {layer} has unexpected o_proj input {tuple(activation.shape)}"
                )
            flat = activation.reshape(-1, self.input_width).float()
            if len(flat) != self.expected_rows:
                raise ValueError("o_proj hook row count differs from the input batch")
            destination = self.sums[self.active_split].get(layer)
            if destination is None:
                destination = torch.zeros(
                    self.input_width,
                    self.input_width,
                    dtype=torch.float32,
                    device=activation.device,
                )
                self.sums[self.active_split][layer] = destination
            destination.addmm_(flat.transpose(0, 1), flat)
            self.seen.add(layer)
            del flat

        return capture

    def begin(self, split: str, *, rows: int) -> None:
        if self.active_split is not None or split not in self.sums or rows <= 0:
            raise RuntimeError("invalid covariance capture batch transition")
        self.active_split = split
        self.expected_rows = int(rows)
        self.seen.clear()

    def finish(self) -> None:
        if self.active_split is None or self.seen != set(self.modules):
            missing = sorted(set(self.modules) - self.seen)
            raise RuntimeError(f"covariance capture missed layers: {missing}")
        self.rows[self.active_split] += self.expected_rows
        self.active_split = None
        self.expected_rows = 0
        self.seen.clear()

    def normalize_and_offload(self, split: str, *, expected_rows: int) -> None:
        """Normalize one complete split and move it off every model GPU."""

        if self.active_split is not None:
            raise RuntimeError("cannot offload covariance during an active batch")
        if self.rows.get(split) != expected_rows:
            raise RuntimeError(f"cannot offload incomplete {split} covariance")
        if set(self.sums[split]) != set(self.modules):
            raise RuntimeError(f"cannot offload {split} covariance with missing layers")
        for layer in self.modules:
            matrix = self.sums[split][layer]
            self.sums[split][layer] = (
                matrix.div_(float(expected_rows)).cpu().contiguous()
            )
            del matrix
        torch.cuda.empty_cache()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if (
        args.expected_attention == "mla"
        and args.attn_implementation == "sdpa"
        and args.trust_remote_code
    ):
        raise ValueError("DeepSeek-V2 remote code does not implement SDPA attention")
    torch.set_num_threads(args.torch_num_threads)
    if min(args.fit_windows, args.heldout_windows, args.batch_size) <= 0:
        raise ValueError("window and batch counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("streaming covariance capture requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be CUDA")
    torch.cuda.set_device(device)
    cuda_indices = _cuda_device_indices()
    for index in cuda_indices:
        torch.cuda.reset_peak_memory_stats(index)

    model_path = Path(args.model_path).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    windows, windows_manifest = _load_ordered_windows(
        windows_path, expected=args.fit_windows + args.heldout_windows
    )
    sequence_length = int(windows.shape[1])
    if sequence_length <= 0:
        raise ValueError("window sequence length must be positive")
    partial_dir.mkdir(parents=True)

    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.model_dtype]
    model_kwargs: dict[str, Any] = {
        "dtype": model_dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
        "local_files_only": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map == "none":
        model_kwargs["device_map"] = {"": str(device)}
    else:
        model_kwargs["device_map"] = args.device_map
        model_kwargs["max_memory"] = {
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in cuda_indices
        }
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    print(
        f"[Covariance] loading model={model_path} device_map={args.device_map}",
        flush=True,
    )
    model = AutoModel.from_pretrained(str(model_path), **model_kwargs).eval()
    model.config.use_cache = False
    placement = {
        str(key): str(value)
        for key, value in getattr(model, "hf_device_map", {}).items()
    }
    if any(value in {"cpu", "disk"} for value in placement.values()):
        raise RuntimeError(f"covariance capture offloaded parameters: {placement}")
    geometry = _validate_attention(model.config, args.expected_attention)
    decoder_layers = _decoder_layers(model)
    layers = _parse_layers(args.layers, len(decoder_layers))
    modules = {layer: decoder_layers[layer].self_attn.o_proj for layer in layers}
    query_width = int(geometry["num_attention_heads"]) * int(geometry["head_dim"])
    for layer, module in modules.items():
        if not isinstance(module, nn.Linear) or tuple(module.weight.shape) != (
            int(geometry["hidden_size"]), query_width
        ):
            raise TypeError(f"layer {layer} has unsupported o_proj")

    capture = _StreamingCovarianceCapture(modules, query_width)
    input_device = model.get_input_embeddings().weight.device
    if input_device.type != "cuda":
        raise RuntimeError("input embeddings are not on CUDA")
    try:
        total = len(windows)
        start = 0
        batch_index = 0
        while start < total:
            boundary = args.fit_windows if start < args.fit_windows else total
            stop = min(start + args.batch_size, boundary)
            split = "fit" if start < args.fit_windows else "heldout"
            input_ids = windows[start:stop].to(input_device)
            capture.begin(split, rows=int(input_ids.numel()))
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            )
            del outputs, input_ids
            capture.finish()
            batch_index += 1
            print(
                f"[Covariance] batch={batch_index} split={split} windows={stop}/{total}",
                flush=True,
            )
            if stop == args.fit_windows:
                capture.normalize_and_offload(
                    "fit", expected_rows=args.fit_windows * sequence_length
                )
                print("[Covariance] offloaded normalized fit statistics", flush=True)
            start = stop
    finally:
        capture.close()

    expected_rows = {
        "fit": args.fit_windows * sequence_length,
        "heldout": args.heldout_windows * sequence_length,
    }
    if capture.rows != expected_rows:
        raise RuntimeError(f"captured row counts differ: {capture.rows} != {expected_rows}")
    capture.normalize_and_offload(
        "heldout", expected_rows=expected_rows["heldout"]
    )
    print("[Covariance] offloaded normalized held-out statistics", flush=True)

    artifacts: dict[str, Any] = {}
    for layer in layers:
        path = partial_dir / f"layer_{layer:03d}.safetensors"
        fit_covariance = capture.sums["fit"].pop(layer)
        heldout_covariance = capture.sums["heldout"].pop(layer)
        weight = modules[layer].weight.detach().cpu().contiguous()
        tensors = {
            "fit_covariance": fit_covariance,
            "heldout_covariance": heldout_covariance,
            "weight": weight,
        }
        save_file(tensors, str(path))
        artifacts[str(layer)] = {
            "file": path.name,
            "sha256": _sha256(path),
            "fit_covariance_shape": list(fit_covariance.shape),
            "heldout_covariance_shape": list(heldout_covariance.shape),
            "weight_shape": list(weight.shape),
            "covariance_dtype": str(fit_covariance.dtype),
            "weight_dtype": str(weight.dtype),
        }
        del fit_covariance, heldout_covariance, weight, tensors
        torch.cuda.empty_cache()
        print(f"[Covariance] wrote layer={layer} file={path.name}", flush=True)

    elapsed = time.perf_counter() - started
    manifest = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "timestamp_started_utc": started_utc,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            **geometry,
        },
        "layers": list(layers),
        "calibration": {
            "activation_aware": True,
            "objective": "attention_o_proj_output_mse",
            "storage": "normalized_covariance_sufficient_statistics",
            "windows_file": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "windows_manifest_format": windows_manifest.get("format"),
            "window_count": len(windows),
            "fit_windows": args.fit_windows,
            "heldout_windows": args.heldout_windows,
            "sequence_length": sequence_length,
            "positions_per_window": sequence_length,
            "fit_rows": expected_rows["fit"],
            "heldout_rows": expected_rows["heldout"],
            "rows_per_layer": sum(expected_rows.values()),
        },
        "artifacts": artifacts,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in cuda_indices
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in cuda_indices
            },
            "device_map_strategy": args.device_map,
            "device_map": placement,
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(partial_dir / "manifest.json", manifest)
    os.replace(partial_dir, output_dir)
    print(f"[Covariance] complete output={output_dir} seconds={elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
