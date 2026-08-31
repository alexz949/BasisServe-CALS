#!/usr/bin/env python3
"""Capture all-layer attention ``o_proj`` inputs for collective PPL fitting.

The input windows must already be tokenized with a tokenizer compatible with
the requested checkpoint. A bounded set of deterministic positions is kept
for every decoder layer, together with the corresponding dense ``o_proj``
weight. No reconstruction-error split is captured: WikiText-2 PPL is the
downstream quality endpoint.
"""

from __future__ import annotations

import argparse
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
from torch import Tensor, nn
from transformers import AutoModel


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.analyze_qwen35_gated_attention_sparsity import (  # noqa: E402
    _git_commit,
    _installed_version,
)
from evaluation.analyze_qwen3_o_proj_collective_endpoints import (  # noqa: E402
    _OProjCapture,
    _sample_positions,
)


FORMAT = "basisserve.attention_o_proj_ppl_snapshots.v1"


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
    parser.add_argument(
        "--max-windows",
        type=int,
        help="Optional prefix limit after applying --window-split",
    )
    parser.add_argument("--positions-per-window", type=int, default=128)
    parser.add_argument("--position-seed", type=int, default=20260901)
    parser.add_argument(
        "--position-record-field",
        choices=("source_window_id",),
        help=(
            "Optional window-manifest record field used as the deterministic "
            "position-sampling index; useful when a fixed held-out block was reordered"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-map",
        choices=("none", "balanced", "balanced_low_0"),
        default="none",
        help="Shard models that do not fit on one GPU across visible GPUs",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    current = model
    for name in ("model", "language_model"):
        child = getattr(current, name, None)
        if child is not None:
            current = child
    layers = getattr(current, "layers", None)
    if layers is None:
        raise AttributeError("could not locate decoder layers")
    return layers


def _parse_layers(raw: str, count: int) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return tuple(range(count))
    layers = tuple(
        sorted(set(int(piece) for piece in raw.split(",") if piece.strip()))
    )
    if not layers or min(layers) < 0 or max(layers) >= count:
        raise ValueError("requested decoder layer is out of range")
    return layers


def _load_windows(
    path: Path, split: str
) -> tuple[Tensor, dict[str, Any], list[int]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = load_file(str(path), device="cpu")
    if "input_ids" not in payload:
        raise KeyError(f"input_ids is absent from {path}")
    windows = payload["input_ids"].to(dtype=torch.long)
    if windows.ndim != 2 or not int(windows.shape[0]):
        raise ValueError("window tensor must be a nonempty matrix")
    indices = torch.arange(int(windows.shape[0]), dtype=torch.long)
    if split != "all":
        if "split_codes" not in payload:
            raise ValueError("window split requested but split_codes is absent")
        split_code = {"fit": 0, "heldout": 1}[split]
        indices = torch.nonzero(
            payload["split_codes"].to(torch.long) == split_code,
            as_tuple=False,
        ).flatten()
        if not int(indices.numel()):
            raise ValueError(f"window split is empty: {split}")
        windows = windows.index_select(0, indices)
    manifest_path = path.parent / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    return windows.contiguous(), manifest, list(map(int, indices.tolist()))


def _validate_attention(config: Any, expected: str) -> dict[str, int | str]:
    hidden_size = int(config.hidden_size)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    num_layers = int(config.num_hidden_layers)
    head_dim = int(getattr(config, "head_dim", 0) or hidden_size // num_heads)
    if head_dim <= 0 or num_heads % num_kv_heads:
        raise ValueError("checkpoint attention geometry is invalid")
    actual = "mha" if num_heads == num_kv_heads else "gqa"
    if actual != expected:
        raise ValueError(f"expected {expected}, found {actual}")
    return {
        "model_type": str(config.model_type),
        "attention_type": actual,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def _limit_windows(
    windows: Tensor,
    window_indices: Sequence[int],
    maximum: int | None,
) -> tuple[Tensor, list[int]]:
    if maximum is None:
        return windows, list(map(int, window_indices))
    if maximum <= 0 or maximum > int(windows.shape[0]):
        raise ValueError("maximum windows must lie within the selected window count")
    return (
        windows[:maximum].contiguous(),
        list(map(int, window_indices[:maximum])),
    )


def _position_sampling_indices(
    *,
    window_manifest: Mapping[str, Any],
    window_indices: Sequence[int],
    record_field: str | None,
) -> list[int]:
    if record_field is None:
        return list(range(len(window_indices)))
    records = window_manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("position record field requires window-manifest records")
    selected: list[int] = []
    for window_index in window_indices:
        if not 0 <= int(window_index) < len(records):
            raise ValueError("window index exceeds window-manifest records")
        value = records[int(window_index)].get(record_field)
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"window record has invalid nonnegative integer {record_field}"
            )
        selected.append(value)
    if len(set(selected)) != len(selected):
        raise ValueError("position-sampling record indices must be unique")
    return selected


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if args.positions_per_window <= 0 or args.batch_size <= 0:
        raise ValueError("sampling and batch sizes must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("snapshot capture requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    model_path = Path(args.model_path).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path)
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
    positions = position_pool.index_select(
        0, torch.tensor(position_sampling_indices, dtype=torch.long)
    ).contiguous()

    partial_dir.mkdir(parents=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.model_dtype]
    print(
        f"[Capture] loading model={model_path} device={device} "
        f"device_map={args.device_map}",
        flush=True,
    )
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
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        }
    model = AutoModel.from_pretrained(str(model_path), **model_kwargs).eval()
    placement = {
        str(key): str(value)
        for key, value in getattr(model, "hf_device_map", {}).items()
    }
    if any(value in {"cpu", "disk"} for value in placement.values()):
        raise RuntimeError(f"snapshot capture offloaded model parameters: {placement}")
    geometry = _validate_attention(model.config, args.expected_attention)
    decoder_layers = _decoder_layers(model)
    if len(decoder_layers) != int(geometry["num_hidden_layers"]):
        raise ValueError("decoder layer count differs from config")
    layers = _parse_layers(args.layers, len(decoder_layers))
    modules = {layer: decoder_layers[layer].self_attn.o_proj for layer in layers}
    for layer, module in modules.items():
        if not isinstance(module, nn.Linear):
            raise TypeError(f"layer {layer} o_proj is not nn.Linear")
        if module.bias is not None or tuple(module.weight.shape) != (
            int(geometry["hidden_size"]),
            int(geometry["num_attention_heads"]) * int(geometry["head_dim"]),
        ):
            raise ValueError(f"layer {layer} o_proj geometry is unsupported")

    rows = int(windows.shape[0]) * args.positions_per_window
    capture = _OProjCapture(
        modules=modules,
        rows=rows,
        input_width=(
            int(geometry["num_attention_heads"]) * int(geometry["head_dim"])
        ),
    )
    input_device = model.get_input_embeddings().weight.device
    if input_device.type != "cuda":
        raise RuntimeError("model input embeddings were not placed on CUDA")
    try:
        batches = (
            int(windows.shape[0]) + args.batch_size - 1
        ) // args.batch_size
        for batch_index, start in enumerate(
            range(0, int(windows.shape[0]), args.batch_size), start=1
        ):
            stop = min(start + args.batch_size, int(windows.shape[0]))
            capture.begin(
                positions[start:stop], start * args.positions_per_window
            )
            input_ids = windows[start:stop].to(input_device)
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            )
            capture.finish()
            print(
                f"[Capture] batch={batch_index}/{batches} "
                f"windows={stop}/{len(windows)}",
                flush=True,
            )
    finally:
        capture.close()

    save_file(
        {"positions": positions},
        str(partial_dir / "sampled_positions.safetensors"),
    )
    records: dict[str, Any] = {}
    for layer in layers:
        path = partial_dir / f"layer_{layer:03d}.safetensors"
        activation = capture.rows[layer]
        weight = modules[layer].weight.detach().to(device="cpu").contiguous()
        save_file({"activation": activation, "weight": weight}, str(path))
        records[str(layer)] = {
            "file": path.name,
            "sha256": _sha256(path),
            "activation_shape": list(activation.shape),
            "weight_shape": list(weight.shape),
            "activation_dtype": str(activation.dtype),
            "weight_dtype": str(weight.dtype),
        }
        print(f"[Capture] wrote layer={layer} file={path.name}", flush=True)

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
            "sampled_positions_file": "sampled_positions.safetensors",
            "sampled_positions_sha256": _sha256(
                partial_dir / "sampled_positions.safetensors"
            ),
        },
        "artifacts": records,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in range(torch.cuda.device_count())
            },
            "device_map_strategy": args.device_map,
            "device_map": placement,
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(partial_dir / "manifest.json", manifest)
    os.replace(partial_dir, output_dir)
    print(
        f"[Capture] complete output={output_dir} seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
