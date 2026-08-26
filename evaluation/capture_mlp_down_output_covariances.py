#!/usr/bin/env python3
"""Stream complete post-``down_proj`` MLP output second moments.

Every token in the ordered fit and held-out windows contributes to
``E[Y.T @ Y]`` for each decoder layer.  These moments are sufficient for the
globally optimal shared-decoder C1 fit; raw activations are never stored.
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
from typing import Any, Mapping

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
)


FORMAT = "basisserve.mlp_down_output_covariances.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-windows", type=int, required=True)
    parser.add_argument("--heldout-windows", type=int, required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-map", choices=("none", "balanced", "balanced_low_0"), default="none"
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=72)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _cuda_device_indices() -> tuple[int, ...]:
    result = []
    for index in range(torch.cuda.device_count()):
        try:
            torch.cuda.get_device_properties(index)
        except RuntimeError:
            continue
        result.append(index)
    return tuple(result)


def _load_windows(path: Path, *, expected: int) -> tuple[Tensor, dict[str, Any]]:
    payload = load_file(str(path), device="cpu")
    if set(payload) != {"input_ids"}:
        raise ValueError("MLP covariance windows must contain only input_ids")
    windows = payload["input_ids"].long().contiguous()
    if windows.ndim != 2 or len(windows) != expected:
        raise ValueError(f"expected {expected} windows, got {tuple(windows.shape)}")
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return windows, json.loads(manifest_path.read_text(encoding="utf-8"))


class _OutputMomentCapture:
    def __init__(self, modules: Mapping[int, nn.Linear], hidden_size: int) -> None:
        self.modules = dict(modules)
        self.hidden_size = int(hidden_size)
        self.sums: dict[str, dict[int, Tensor]] = {"fit": {}, "heldout": {}}
        self.rows = {"fit": 0, "heldout": 0}
        self.active_split: str | None = None
        self.expected_rows = 0
        self.seen: set[int] = set()
        self.handles = [
            module.register_forward_hook(self._hook(layer))
            for layer, module in self.modules.items()
        ]

    def _hook(self, layer: int):
        def capture(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
            del module, inputs
            if self.active_split is None or not isinstance(output, Tensor):
                raise RuntimeError("MLP output hook fired outside an active batch")
            if output.ndim != 3 or int(output.shape[-1]) != self.hidden_size:
                raise ValueError(f"layer {layer} has unexpected MLP output {output.shape}")
            flat = output.detach().reshape(-1, self.hidden_size).float()
            if len(flat) != self.expected_rows:
                raise ValueError("MLP output hook row count differs from input batch")
            destination = self.sums[self.active_split].get(layer)
            if destination is None:
                destination = torch.zeros(
                    self.hidden_size,
                    self.hidden_size,
                    dtype=torch.float32,
                    device=output.device,
                )
                self.sums[self.active_split][layer] = destination
            destination.addmm_(flat.T, flat)
            self.seen.add(layer)

        return capture

    def begin(self, split: str, *, rows: int) -> None:
        if self.active_split is not None or split not in self.sums or rows <= 0:
            raise RuntimeError("invalid MLP covariance batch transition")
        self.active_split = split
        self.expected_rows = int(rows)
        self.seen.clear()

    def finish(self) -> None:
        if self.active_split is None or self.seen != set(self.modules):
            missing = sorted(set(self.modules) - self.seen)
            raise RuntimeError(f"MLP covariance capture missed layers: {missing}")
        self.rows[self.active_split] += self.expected_rows
        self.active_split = None
        self.expected_rows = 0
        self.seen.clear()

    def normalize_and_offload(self, split: str, *, expected_rows: int) -> None:
        if self.active_split is not None or self.rows[split] != expected_rows:
            raise RuntimeError(f"cannot offload incomplete {split} covariance")
        if set(self.sums[split]) != set(self.modules):
            raise RuntimeError(f"{split} covariance has missing layers")
        for layer in self.modules:
            matrix = self.sums[split][layer]
            self.sums[split][layer] = (
                matrix.div_(float(expected_rows)).cpu().contiguous()
            )
        torch.cuda.empty_cache()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if min(args.fit_windows, args.heldout_windows, args.batch_size) <= 0:
        raise ValueError("window and batch counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("MLP covariance capture requires CUDA")
    torch.set_num_threads(args.torch_num_threads)
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
    windows, windows_manifest = _load_windows(
        windows_path, expected=args.fit_windows + args.heldout_windows
    )
    sequence_length = int(windows.shape[1])
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
    }
    if args.device_map == "none":
        model_kwargs["device_map"] = {"": str(device)}
    else:
        model_kwargs["device_map"] = args.device_map
        model_kwargs["max_memory"] = {
            index: f"{args.max_memory_per_gpu_gib}GiB" for index in cuda_indices
        }

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    print(f"[MLP covariance] loading model={model_path}", flush=True)
    model = AutoModel.from_pretrained(str(model_path), **model_kwargs).eval()
    model.config.use_cache = False
    placement = {
        str(key): str(value) for key, value in getattr(model, "hf_device_map", {}).items()
    }
    if any(value in {"cpu", "disk"} for value in placement.values()):
        raise RuntimeError(f"MLP covariance capture offloaded parameters: {placement}")
    if str(model.config.model_type) != "qwen3":
        raise ValueError("initial MLP C1 capture supports dense Qwen3 models")
    hidden_size = int(model.config.hidden_size)
    intermediate_size = int(model.config.intermediate_size)
    decoder_layers = _decoder_layers(model)
    layers = _parse_layers(args.layers, len(decoder_layers))
    modules = {layer: decoder_layers[layer].mlp.down_proj for layer in layers}
    for layer, module in modules.items():
        if not isinstance(module, nn.Linear) or tuple(module.weight.shape) != (
            hidden_size,
            intermediate_size,
        ):
            raise TypeError(f"layer {layer} has unsupported MLP down_proj")

    capture = _OutputMomentCapture(modules, hidden_size)
    input_device = model.get_input_embeddings().weight.device
    expected_rows = {
        "fit": args.fit_windows * sequence_length,
        "heldout": args.heldout_windows * sequence_length,
    }
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
                f"[MLP covariance] batch={batch_index} split={split} "
                f"windows={stop}/{total}",
                flush=True,
            )
            if stop == args.fit_windows:
                capture.normalize_and_offload("fit", expected_rows=expected_rows["fit"])
                print("[MLP covariance] offloaded fit moments", flush=True)
            start = stop
    finally:
        capture.close()
    if capture.rows != expected_rows:
        raise RuntimeError(f"captured rows differ: {capture.rows} != {expected_rows}")
    capture.normalize_and_offload("heldout", expected_rows=expected_rows["heldout"])

    artifacts: dict[str, Any] = {}
    for layer in layers:
        path = partial_dir / f"layer_{layer:03d}.safetensors"
        fit_moment = capture.sums["fit"].pop(layer)
        heldout_moment = capture.sums["heldout"].pop(layer)
        save_file(
            {
                "fit_output_second_moment": fit_moment,
                "heldout_output_second_moment": heldout_moment,
            },
            str(path),
        )
        artifacts[str(layer)] = {
            "file": path.name,
            "sha256": _sha256(path),
            "fit_shape": list(fit_moment.shape),
            "heldout_shape": list(heldout_moment.shape),
            "dtype": str(fit_moment.dtype),
        }
        print(f"[MLP covariance] wrote layer={layer}", flush=True)

    elapsed = time.perf_counter() - started
    manifest = {
        "format": FORMAT,
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
            "model_type": str(model.config.model_type),
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": int(model.config.num_hidden_layers),
        },
        "layers": list(layers),
        "calibration": {
            "objective": "post_swiglu_complete_mlp_output_mse_after_tp_sum",
            "storage": "normalized_teacher_output_second_moment",
            "windows_file": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "windows_manifest_format": windows_manifest.get("format"),
            "fit_windows": args.fit_windows,
            "heldout_windows": args.heldout_windows,
            "sequence_length": sequence_length,
            "positions_per_window": sequence_length,
            "fit_rows": expected_rows["fit"],
            "heldout_rows": expected_rows["heldout"],
        },
        "artifacts": artifacts,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "cuda_devices": [torch.cuda.get_device_name(index) for index in cuda_indices],
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
    print(f"[MLP covariance] complete output={output_dir} seconds={elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
