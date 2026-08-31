#!/usr/bin/env python3
"""Capture paired Qwen3 routing operands for the S80 BF update."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from safetensors.torch import load_file
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM


FORMAT = "basisserve.qwen3.gqa_joint_routing_payload_s80_direct.v2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--fit-start", type=int, default=0)
    parser.add_argument("--routing-fit-windows", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=42)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _parse_layers(spec: str) -> tuple[int, ...]:
    selected = tuple(sorted({int(item) for item in spec.split(",") if item.strip()}))
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(32 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _allocate(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    values = 1
    for size in shape:
        values *= int(size)
    with path.open("wb") as handle:
        handle.truncate(values * torch.bfloat16.itemsize)
    return torch.from_file(
        str(path),
        shared=True,
        size=values,
        dtype=torch.bfloat16,
    ).reshape(shape)


class _LayerFiles:
    def __init__(
        self,
        root: Path,
        *,
        layer: int,
        routing_documents: int,
        sequence_length: int,
        query_heads: int,
        kv_heads: int,
        key_dim: int,
        joint_dim: int,
    ) -> None:
        self.paths = {
            "routing_queries": root / f"layer_{layer:03d}_routing_queries.bf16",
            "routing_joint_rows": root / f"layer_{layer:03d}_routing_joint_rows.bf16",
        }
        self.shapes = {
            "routing_queries": (routing_documents, query_heads, key_dim),
            "routing_joint_rows": (
                routing_documents,
                sequence_length,
                kv_heads,
                joint_dim,
            ),
        }
        self.tensors = {
            name: _allocate(self.paths[name], shape)
            for name, shape in self.shapes.items()
        }

    def record(self) -> dict[str, Any]:
        return {
            name: {
                "file": self.paths[name].name,
                "shape": list(self.shapes[name]),
                "dtype": "bfloat16",
                "bytes": self.paths[name].stat().st_size,
                "sha256": _sha256(self.paths[name]),
            }
            for name in self.paths
        }

    def close(self) -> None:
        self.tensors.clear()


class _DirectCollector:
    def __init__(
        self,
        model: nn.Module,
        *,
        output_dir: Path,
        selected_layers: tuple[int, ...],
        routing_fit_windows: int,
        sequence_length: int,
    ) -> None:
        config = model.config
        self.query_heads = int(config.num_attention_heads)
        self.kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(
            getattr(config, "head_dim", config.hidden_size // self.query_heads)
        )
        self.routing_fit_windows = int(routing_fit_windows)
        self.sequence_length = int(sequence_length)
        self.files = {
            layer: _LayerFiles(
                output_dir,
                layer=layer,
                routing_documents=routing_fit_windows,
                sequence_length=sequence_length,
                query_heads=self.query_heads,
                kv_heads=self.kv_heads,
                key_dim=self.head_dim,
                joint_dim=2 * self.head_dim,
            )
            for layer in selected_layers
        }
        self.active_slots: tuple[int, ...] = ()

    def attention_pre_hook(self, layer_index: int):
        def collect(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get("position_embeddings")
            batch, sequence, _ = hidden_states.shape
            query_shape = (batch, sequence, self.query_heads, self.head_dim)
            kv_shape = (batch, sequence, self.kv_heads, self.head_dim)
            query = module.q_norm(module.q_proj(hidden_states).view(query_shape)).transpose(1, 2)
            key = module.k_norm(module.k_proj(hidden_states).view(kv_shape)).transpose(1, 2)
            value = module.v_proj(hidden_states).view(kv_shape).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
            layer_files = self.files[layer_index].tensors
            for local, slot in enumerate(self.active_slots):
                if slot >= self.routing_fit_windows:
                    continue
                layer_files["routing_queries"][slot].copy_(
                    query[local, :, -1].detach().to(device="cpu", dtype=torch.bfloat16)
                )
                joint = torch.cat(
                    (
                        value[local].transpose(0, 1),
                        key[local].transpose(0, 1),
                    ),
                    dim=-1,
                )
                layer_files["routing_joint_rows"][slot].copy_(
                    joint.detach().to(device="cpu", dtype=torch.bfloat16)
                )

        return collect

    def close(self) -> None:
        for files in self.files.values():
            files.close()


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    model_path = Path(args.model).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    selected_layers = _parse_layers(args.layers)
    stored = load_file(str(windows_path), device="cpu")["input_ids"]
    stop = args.fit_start + args.routing_fit_windows
    windows = stored[args.fit_start:stop]
    partial_dir.mkdir(parents=True)
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    device_map: str | dict[str, int] = args.device_map
    if args.device_map == "single":
        device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map=device_map,
        max_memory={
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        },
    ).eval()
    model.config.use_cache = False
    collector = _DirectCollector(
        model,
        output_dir=partial_dir,
        selected_layers=selected_layers,
        routing_fit_windows=args.routing_fit_windows,
        sequence_length=args.sequence_length,
    )
    handles = []
    for layer_index in selected_layers:
        layer = model.model.layers[layer_index]
        handles.append(
            layer.self_attn.register_forward_pre_hook(
                collector.attention_pre_hook(layer_index),
                with_kwargs=True,
            )
        )
    input_device = model.model.embed_tokens.weight.device
    started = time.monotonic()
    for start in range(0, args.routing_fit_windows, args.batch_size):
        end = min(start + args.batch_size, args.routing_fit_windows)
        collector.active_slots = tuple(range(start, end))
        input_ids = windows[start:end].to(input_device)
        output = model.model(input_ids=input_ids, use_cache=False)
        del output, input_ids
        print(f"[S80 direct] windows={end}/{args.routing_fit_windows}", flush=True)
    for handle in handles:
        handle.remove()
    records = {str(layer): files.record() for layer, files in collector.files.items()}
    collector.close()
    del collector, model
    gc.collect()
    manifest = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "elapsed_seconds": time.monotonic() - started,
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "geometry": {
            "layers": int(config.num_hidden_layers),
            "layer_coverage": list(selected_layers),
            "query_heads": int(config.num_attention_heads),
            "kv_heads": int(config.num_key_value_heads),
            "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        },
        "calibration": {
            "windows": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "fit_start": args.fit_start,
            "routing_fit_windows": args.routing_fit_windows,
            "sequence_length": args.sequence_length,
            "routing_query_policy": "last_token_full_prefix",
        },
        "storage": {
            "dtype": "bfloat16",
            "layout": "raw_memory_mapped_row_major",
            "normal_equations_formed": False,
        },
        "artifacts": records,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    (partial_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial_dir, output_dir)
    print(f"[S80 direct] complete output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
