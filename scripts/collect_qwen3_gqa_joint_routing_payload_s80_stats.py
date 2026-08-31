#!/usr/bin/env python3
"""Collect paired payload/routing sufficient statistics for Qwen3 S80."""

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
from typing import Any

from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.calibration.gqa_joint_routing_payload_s80_stats import (  # noqa: E402
    S80PayloadAccumulator,
    S80RoutingAccumulator,
)


FORMAT = "basisserve.qwen3.gqa_joint_routing_payload_s80_stats.v2"
DIRECT_FORMAT = "basisserve.qwen3.gqa_joint_routing_payload_s80_direct.v2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--direct-output-dir")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--fit-start", type=int, default=0)
    parser.add_argument("--fit-windows", type=int, default=256)
    parser.add_argument("--validation-start", type=int, default=256)
    parser.add_argument("--validation-windows", type=int, default=64)
    parser.add_argument("--routing-fit-windows", type=int, default=128)
    parser.add_argument("--routing-validation-windows", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--statistics-dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=42)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _parse_layers(spec: str, total: int) -> tuple[int, ...]:
    if spec == "all":
        return tuple(range(total))
    selected: set[int] = set()
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            selected.update(range(int(left), int(right) + 1))
        else:
            selected.add(int(item))
    return tuple(sorted(selected))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)


def _allocate_bfloat16(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
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


class _DirectLayerFiles:
    def __init__(
        self,
        root: Path,
        *,
        layer: int,
        routing_windows: int,
        sequence_length: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> None:
        self.paths = {
            "routing_queries": root / f"layer_{layer:03d}_routing_queries.bf16",
            "routing_joint_rows": root
            / f"layer_{layer:03d}_routing_joint_rows.bf16",
        }
        self.shapes = {
            "routing_queries": (routing_windows, query_heads, head_dim),
            "routing_joint_rows": (
                routing_windows,
                sequence_length,
                kv_heads,
                2 * head_dim,
            ),
        }
        self.tensors = {
            name: _allocate_bfloat16(self.paths[name], shape)
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


class _S80Collector:
    def __init__(
        self,
        model: nn.Module,
        *,
        statistics_dtype: torch.dtype,
        selected_layers: tuple[int, ...],
        direct_output_dir: Path | None,
        fit_start: int,
        routing_fit_windows: int,
        validation_start: int,
        routing_validation_windows: int,
        sequence_length: int,
    ) -> None:
        config = model.config
        self.layers = len(model.model.layers)
        self.layer_indices = selected_layers
        self.query_heads = int(config.num_attention_heads)
        self.kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(
            getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            )
        )
        self.mapping = torch.arange(self.query_heads) // (
            self.query_heads // self.kv_heads
        )
        self.payload = {split: {} for split in ("fit", "validation")}
        self.routing = {split: {} for split in ("fit", "validation")}
        for split in ("fit", "validation"):
            for layer_index in self.layer_indices:
                layer = model.model.layers[layer_index]
                device = layer.self_attn.q_proj.weight.device
                self.payload[split][layer_index] = S80PayloadAccumulator(
                    num_query_heads=self.query_heads,
                    value_dim=self.head_dim,
                    key_dim=self.head_dim,
                    accumulation_dtype=statistics_dtype,
                    device=device,
                )
                self.routing[split][layer_index] = S80RoutingAccumulator(
                    head_to_kv_group=self.mapping,
                    value_dim=self.head_dim,
                    key_dim=self.head_dim,
                    storage_dtype=statistics_dtype,
                    storage_device="cpu",
                )
        self.active_split: str | None = None
        self.active_document_indices: tuple[int, ...] = ()
        self.active_routing_documents: set[int] = set()
        self.pending_routed_key: dict[int, torch.Tensor] = {}
        self.direct_starts = {
            "fit": int(fit_start),
            "validation": int(validation_start),
        }
        self.direct_counts = {
            "fit": int(routing_fit_windows),
            "validation": int(routing_validation_windows),
        }
        self.direct: dict[str, dict[int, _DirectLayerFiles]] = {}
        if direct_output_dir is not None:
            for split in ("fit", "validation"):
                split_root = direct_output_dir / split
                split_root.mkdir(parents=True)
                self.direct[split] = {
                    layer_index: _DirectLayerFiles(
                        split_root,
                        layer=layer_index,
                        routing_windows=self.direct_counts[split],
                        sequence_length=sequence_length,
                        query_heads=self.query_heads,
                        kv_heads=self.kv_heads,
                        head_dim=self.head_dim,
                    )
                    for layer_index in self.layer_indices
                }

    def attention_pre_hook(self, layer_index: int):
        def collect(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

            split = self.active_split
            if split is None:
                return
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                hidden_states = args[0]
            position_embeddings = kwargs.get("position_embeddings")
            batch, sequence, _ = hidden_states.shape
            query_shape = (batch, sequence, self.query_heads, self.head_dim)
            kv_shape = (batch, sequence, self.kv_heads, self.head_dim)
            query = module.q_norm(
                module.q_proj(hidden_states).view(query_shape)
            ).transpose(1, 2)
            key = module.k_norm(module.k_proj(hidden_states).view(kv_shape)).transpose(
                1, 2
            )
            value = module.v_proj(hidden_states).view(kv_shape).transpose(1, 2)
            query, key = apply_rotary_pos_emb(
                query,
                key,
                *position_embeddings,
            )
            attention_mask = kwargs.get("attention_mask")
            routed_key = F.scaled_dot_product_attention(
                query,
                key,
                key,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=bool(
                    module.is_causal and attention_mask is None and sequence > 1
                ),
                scale=float(module.scaling),
                enable_gqa=self.query_heads != self.kv_heads,
            )
            self.pending_routed_key[layer_index] = routed_key.detach()
            routing_accumulator = self.routing[split][layer_index]
            for local_index, document_index in enumerate(self.active_document_indices):
                if document_index not in self.active_routing_documents:
                    continue
                query_row = query[local_index, :, -1]
                value_rows = value[local_index].permute(1, 0, 2)
                key_rows = key[local_index].permute(1, 0, 2)
                routing_accumulator.update_shard(
                    query_row.unsqueeze(0),
                    value_rows,
                    key_rows,
                    metadata={
                        "document_index": int(document_index),
                        "query_position": int(sequence - 1),
                        "visible_prefix_tokens": int(sequence),
                        "query_rows": 1,
                    },
                )
                if split in self.direct:
                    slot = document_index - self.direct_starts[split]
                    tensors = self.direct[split][layer_index].tensors
                    tensors["routing_queries"][slot].copy_(
                        query_row.detach().to(device="cpu", dtype=torch.bfloat16)
                    )
                    tensors["routing_joint_rows"][slot].copy_(
                        torch.cat((value_rows, key_rows), dim=-1)
                        .detach()
                        .to(device="cpu", dtype=torch.bfloat16)
                    )

        return collect

    def direct_records(self) -> dict[str, dict[str, Any]]:
        return {
            split: {
                str(layer): files.record()
                for layer, files in layer_files.items()
            }
            for split, layer_files in self.direct.items()
        }

    def close_direct(self) -> None:
        for layer_files in self.direct.values():
            for files in layer_files.values():
                files.close()

    def o_proj_hook(self, layer_index: int):
        def collect(
            module: nn.Module,
            args: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            split = self.active_split
            if split is None:
                return
            routed_value_flat = args[0]
            batch, sequence, _ = routed_value_flat.shape
            routed_value = routed_value_flat.view(
                batch,
                sequence,
                self.query_heads,
                self.head_dim,
            )
            routed_key = self.pending_routed_key.pop(layer_index).transpose(1, 2)
            self.payload[split][layer_index].update(
                routed_value,
                routed_key,
                output,
            )

        return collect


def _run_split(
    model: nn.Module,
    windows: torch.Tensor,
    *,
    split: str,
    document_start: int,
    routing_windows: int,
    batch_size: int,
    collector: _S80Collector,
) -> None:
    collector.active_split = split
    collector.active_routing_documents = set(
        range(document_start, document_start + routing_windows)
    )
    input_device = model.model.embed_tokens.weight.device
    for offset in range(0, len(windows), batch_size):
        stop = min(offset + batch_size, len(windows))
        collector.active_document_indices = tuple(
            range(document_start + offset, document_start + stop)
        )
        input_ids = windows[offset:stop].to(
            device=input_device,
            dtype=torch.long,
            non_blocking=True,
        )
        output = model.model(input_ids=input_ids, use_cache=False)
        del output, input_ids
        print(
            f"[S80 stats] {split}: {stop}/{len(windows)} batch_size={batch_size}",
            flush=True,
        )
    collector.active_split = None
    collector.active_document_indices = ()
    collector.active_routing_documents.clear()
    collector.pending_routed_key.clear()


def _write_statistics(
    output_dir: Path,
    *,
    collector: _S80Collector,
    output_dtype: torch.dtype,
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for split in ("fit", "validation"):
        artifacts[split] = {}
        for layer_index in collector.layer_indices:
            payload = collector.payload[split][layer_index].finalize(
                output_dtype=output_dtype,
                output_device="cpu",
            )
            payload_name = f"{split}_layer_{layer_index:03d}_payload.safetensors"
            payload_path = output_dir / payload_name
            _atomic_safetensors(
                payload_path,
                {
                    "covariance_blocks": payload.covariance_blocks.contiguous(),
                    "row_count": torch.tensor(payload.row_count, dtype=torch.int64),
                    "dense_output_energy": torch.tensor(
                        payload.dense_output_energy, dtype=torch.float64
                    ),
                    "value_dim": torch.tensor(payload.value_dim, dtype=torch.int64),
                    "key_dim": torch.tensor(payload.key_dim, dtype=torch.int64),
                },
            )
            routing = collector.routing[split][layer_index].finalize()
            routing_name = f"{split}_layer_{layer_index:03d}_routing.safetensors"
            routing_path = output_dir / routing_name
            _atomic_safetensors(
                routing_path,
                {
                    "query_grams": torch.stack(
                        [item.query_grams for item in routing.shards]
                    ).contiguous(),
                    "joint_grams": torch.stack(
                        [item.joint_grams for item in routing.shards]
                    ).contiguous(),
                    "query_sums": torch.stack(
                        [item.query_sums for item in routing.shards]
                    ).contiguous(),
                    "joint_sums": torch.stack(
                        [item.joint_sums for item in routing.shards]
                    ).contiguous(),
                    "query_row_counts": torch.stack(
                        [item.query_row_counts for item in routing.shards]
                    ).contiguous(),
                    "key_row_counts": torch.stack(
                        [item.key_row_counts for item in routing.shards]
                    ).contiguous(),
                    "target_score_energy": torch.tensor(
                        [item.target_score_energy for item in routing.shards],
                        dtype=torch.float64,
                    ),
                    "head_to_kv_group": routing.head_to_kv_group.long().contiguous(),
                    "value_dim": torch.tensor(routing.value_dim, dtype=torch.int64),
                    "key_dim": torch.tensor(routing.key_dim, dtype=torch.int64),
                },
            )
            artifacts[split][str(layer_index)] = {
                "payload": {
                    "file": payload_name,
                    "sha256": _sha256(payload_path),
                    "dtype": str(output_dtype).removeprefix("torch."),
                    "shape": list(payload.covariance_blocks.shape),
                    "row_count": payload.row_count,
                    "dense_output_energy": payload.dense_output_energy,
                },
                "routing": {
                    "file": routing_name,
                    "sha256": _sha256(routing_path),
                    "dtype": str(output_dtype).removeprefix("torch."),
                    "shards": len(routing.shards),
                    "target_score_energy": routing.target_score_energy,
                    "shard_metadata": [dict(item.metadata) for item in routing.shards],
                    "paired_shards": True,
                    "score_scale_included": False,
                },
            }
            print(
                f"[S80 stats] wrote {split} layer {layer_index}/{collector.layers - 1}",
                flush=True,
            )
    return artifacts


def _write_direct_manifests(
    direct_output_dir: Path,
    *,
    records: dict[str, dict[str, Any]],
    model_path: Path,
    windows_path: Path,
    config: Any,
    collector: _S80Collector,
    args: argparse.Namespace,
    elapsed_seconds: float,
) -> None:
    split_starts = {
        "fit": args.fit_start,
        "validation": args.validation_start,
    }
    for split, artifacts in records.items():
        split_root = direct_output_dir / split
        manifest = {
            "format": DIRECT_FORMAT,
            "status": "complete",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "elapsed_seconds": elapsed_seconds,
            "model": {
                "path": str(model_path),
                "config_sha256": _sha256(model_path / "config.json"),
            },
            "geometry": {
                "layers": int(config.num_hidden_layers),
                "layer_coverage": list(collector.layer_indices),
                "query_heads": collector.query_heads,
                "kv_heads": collector.kv_heads,
                "head_dim": collector.head_dim,
                "joint_feature_dim": 2 * collector.head_dim,
                "key_convention": "post_rope",
            },
            "calibration": {
                "windows": str(windows_path),
                "windows_sha256": _sha256(windows_path),
                "fit_start": split_starts[split],
                "routing_fit_windows": collector.direct_counts[split],
                "sequence_length": args.sequence_length,
                "routing_query_policy": "last_token_full_prefix",
                "split": split,
            },
            "storage": {
                "dtype": "bfloat16",
                "layout": "raw_memory_mapped_row_major",
                "saved_operands": "terminal Q and per-token [V,K_post]",
                "payload_activations_saved": False,
                "normal_equations_formed": False,
            },
            "artifacts": artifacts,
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
        _atomic_json(split_root / "manifest.json", manifest)
        print(f"[S80 direct] wrote {split_root / 'manifest.json'}", flush=True)


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    started = time.monotonic()
    model_path = Path(args.model).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    direct_output_dir = (
        None
        if args.direct_output_dir is None
        else Path(args.direct_output_dir).expanduser().resolve()
    )
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    windows_manifest_path = windows_path.parent / "manifest.json"
    stored = load_file(str(windows_path), device="cpu")["input_ids"]
    splits = {
        "fit": stored[args.fit_start : args.fit_start + args.fit_windows],
        "validation": stored[
            args.validation_start : args.validation_start + args.validation_windows
        ],
    }
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    statistics_dtype = (
        torch.float32 if args.statistics_dtype == "float32" else torch.float64
    )
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
    selected_layers = _parse_layers(args.layers, int(config.num_hidden_layers))
    collector = _S80Collector(
        model,
        statistics_dtype=statistics_dtype,
        selected_layers=selected_layers,
        direct_output_dir=direct_output_dir,
        fit_start=args.fit_start,
        routing_fit_windows=args.routing_fit_windows,
        validation_start=args.validation_start,
        routing_validation_windows=args.routing_validation_windows,
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
        handles.append(
            layer.self_attn.o_proj.register_forward_hook(
                collector.o_proj_hook(layer_index)
            )
        )
    _run_split(
        model,
        splits["fit"],
        split="fit",
        document_start=args.fit_start,
        routing_windows=args.routing_fit_windows,
        batch_size=args.batch_size,
        collector=collector,
    )
    _run_split(
        model,
        splits["validation"],
        split="validation",
        document_start=args.validation_start,
        routing_windows=args.routing_validation_windows,
        batch_size=args.batch_size,
        collector=collector,
    )
    for handle in handles:
        handle.remove()
    output_dir.mkdir(parents=True)
    artifacts = _write_statistics(
        output_dir,
        collector=collector,
        output_dtype=statistics_dtype,
    )
    direct_records = collector.direct_records()
    if direct_output_dir is not None:
        _write_direct_manifests(
            direct_output_dir,
            records=direct_records,
            model_path=model_path,
            windows_path=windows_path,
            config=config,
            collector=collector,
            args=args,
            elapsed_seconds=time.monotonic() - started,
        )
        collector.close_direct()
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
            "layers": collector.layers,
            "layer_coverage": list(selected_layers),
            "hidden_size": int(config.hidden_size),
            "query_heads": collector.query_heads,
            "physical_kv_heads": collector.kv_heads,
            "head_dim": collector.head_dim,
            "joint_feature_dim": 2 * collector.head_dim,
            "key_convention": "post_rope",
        },
        "calibration": {
            "dataset": "pretokenized full documents",
            "windows": str(windows_path),
            "windows_sha256": _sha256(windows_path),
            "windows_manifest": str(windows_manifest_path),
            "windows_manifest_sha256": _sha256(windows_manifest_path),
            "sequence_length": args.sequence_length,
            "fit_start": args.fit_start,
            "fit_windows": args.fit_windows,
            "validation_start": args.validation_start,
            "validation_windows": args.validation_windows,
            "routing_fit_windows": args.routing_fit_windows,
            "routing_validation_windows": args.routing_validation_windows,
            "routing_query_policy": "last token of each selected full document",
            "routing_shards_preserve_document_prefix_pairing": True,
            "payload_streaming_sufficient_statistics": True,
            "direct_routing_output": (
                None if direct_output_dir is None else str(direct_output_dir)
            ),
        },
        "numerics": {
            "model_dtype": args.model_dtype,
            "statistics_dtype": args.statistics_dtype,
            "payload_covariance": "full cross-head [P V, P K_post]",
            "routing_scores": "unscaled raw QK Frobenius",
        },
        "artifacts": artifacts,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    print(f"[S80 stats] wrote {output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
