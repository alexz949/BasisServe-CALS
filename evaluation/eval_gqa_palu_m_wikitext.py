#!/usr/bin/env python3
"""Evaluate a compact K- or V-only GQA PaLU factor checkpoint on WikiText-2.

The evaluator replaces exactly one projection and proves that the other remains
unchanged. It reconstructs the dense projection output immediately, which is a
quality-equivalent reference path rather than a compressed-cache performance
measurement. For K, reconstruction happens before Qwen/Llama applies RoPE.
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

from safetensors.torch import load_file
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _eval_ppl_fp32_loss,
)
from palu.model.modules.svd_linear import HeadwiseLowRankModule  # noqa: E402


FORMAT = "basisserve.gqa.palu_m_wikitext2_ppl.v1"


def _cuda_device_indices() -> tuple[int, ...]:
    indices = []
    for index in range(torch.cuda.device_count()):
        try:
            torch.cuda.get_device_properties(index)
        except RuntimeError:
            continue
        indices.append(index)
    return tuple(indices)
SUPPORTED_CHECKPOINT_FORMATS = {
    "basisserve.llama2_7b.iclr_v_factors.v1",
    "basisserve.llama31_8b.iclr_v_factors.v1",
    "basisserve.qwen3_8b.iclr_v_factors.v1",
    "basisserve.llama31_8b.palu_m_v_only.v1",
    "basisserve.llama31_8b.palu_m_v_only_fisher.v1",
    "basisserve.llama31_70b.palu_m_v_only.v1",
    "basisserve.llama31_70b.palu_m_v_only_fisher.v1",
    "basisserve.qwen3_8b.palu_m_v_only.v1",
    "basisserve.qwen3_8b.palu_m_v_only_fisher.v1",
    "basisserve.qwen3_32b.palu_m_v_only.v1",
    "basisserve.qwen3_32b.palu_m_v_only_fisher.v1",
    "basisserve.llama31_8b.palu_m_k_only_fisher.v1",
    "basisserve.llama31_70b.palu_m_k_only_fisher.v1",
    "basisserve.qwen3_8b.palu_m_k_only_fisher.v1",
    "basisserve.qwen3_32b.palu_m_k_only_fisher.v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError(f"expected a Llama/Qwen-style decoder, found {type(model).__name__}")
    return layers


def _load_checkpoint(
    checkpoint_dir: Path,
    model_path: Path,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") not in SUPPORTED_CHECKPOINT_FORMATS:
        raise ValueError(f"unsupported PaLU checkpoint format: {manifest.get('format')!r}")
    if manifest.get("status") != "complete":
        raise ValueError("PaLU checkpoint is not marked complete")
    if manifest["model"]["config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("PaLU checkpoint belongs to another model config")
    artifact_path = checkpoint_dir / manifest["artifact"]["file"]
    if _sha256(artifact_path) != manifest["artifact"]["sha256"]:
        raise ValueError("PaLU factor artifact hash changed")
    payload = load_file(str(artifact_path), device="cpu")
    if len(payload) != int(manifest["artifact"]["tensor_count"]):
        raise ValueError("PaLU tensor count differs from the manifest")
    return manifest, payload


@torch.no_grad()
def _install_projection_factors(
    model: nn.Module,
    payload: Mapping[str, Tensor],
    *,
    target: str,
    layer_ranks: Sequence[Sequence[int]],
    head_dim: int,
    require_cuda_resident: bool = False,
) -> list[dict[str, Any]]:
    """Install one projection's factors and prove the other stays untouched."""

    if target not in {"k", "v"}:
        raise ValueError(f"unsupported projection target: {target!r}")

    layers = _decoder_layers(model)
    if len(layer_ranks) != len(layers):
        raise ValueError("rank schedule must cover every decoder layer")
    records: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        started = time.perf_counter()
        ranks = [int(rank) for rank in layer_ranks[layer_index]]
        if not ranks or min(ranks) <= 0:
            raise ValueError(f"invalid ranks at layer {layer_index}: {ranks}")
        rank_sum = sum(ranks)
        dense_v = layer.self_attn.v_proj
        dense_k = layer.self_attn.k_proj
        target_module = dense_k if target == "k" else dense_v
        preserved_module = dense_v if target == "k" else dense_k
        if not isinstance(target_module, nn.Linear) or target_module.bias is not None:
            raise TypeError(f"unsupported {target}_proj at layer {layer_index}")
        if require_cuda_resident and target_module.weight.device.type != "cuda":
            raise RuntimeError(
                f"{target}_proj at layer {layer_index} is on "
                f"{target_module.weight.device}; "
                "installing PaLU factors after CPU/disk offload is unsupported. "
                "Use enough GPU memory to keep every decoder layer resident."
            )
        writer = payload[f"layers.{layer_index}.{target}_writer.weight"]
        decoder = payload[f"layers.{layer_index}.{target}_decoder.weight"]
        group_width = target_module.out_features // len(ranks)
        expected_writer = (rank_sum, target_module.in_features)
        expected_decoder = (len(ranks), group_width, ranks[0])
        if tuple(writer.shape) != expected_writer:
            raise ValueError(f"invalid writer shape at layer {layer_index}: {writer.shape}")
        if len(set(ranks)) != 1 or tuple(decoder.shape) != expected_decoder:
            raise ValueError(f"invalid decoder shape at layer {layer_index}: {decoder.shape}")

        replacement = HeadwiseLowRankModule(
            list(ranks),
            target_module.in_features,
            target_module.out_features,
            bias=False,
        ).to(device=target_module.weight.device, dtype=target_module.weight.dtype)
        replacement.VT.weight.copy_(
            writer.to(
                device=target_module.weight.device,
                dtype=target_module.weight.dtype,
            )
        )
        for group_index, up in enumerate(replacement.U):
            up.weight.copy_(
                decoder[group_index].to(
                    device=target_module.weight.device,
                    dtype=target_module.weight.dtype,
                )
            )
        setattr(layer.self_attn, f"{target}_proj", replacement)
        other_target = "v" if target == "k" else "k"
        if getattr(layer.self_attn, f"{other_target}_proj") is not preserved_module:
            raise RuntimeError(
                f"PaLU installation changed dense {other_target.upper()} "
                f"at layer {layer_index}"
            )
        records.append(
            {
                "layer": layer_index,
                "ranks": ranks,
                "writer_shape": list(writer.shape),
                "decoder_shape": list(decoder.shape),
                "projection": target,
                "runtime": (
                    f"latent {target.upper()} writer plus explicit per-group "
                    "reconstruction"
                ),
                "heads_per_group": group_width // head_dim,
                f"{other_target}_projection": "original dense module unchanged",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    return records


def install_palu_m_factors(
    model: nn.Module,
    payload: Mapping[str, Tensor],
    *,
    layer_ranks: Sequence[Sequence[int]],
    head_dim: int,
    require_cuda_resident: bool = False,
) -> list[dict[str, Any]]:
    """Install V-only PaLU-M factors while preserving dense K."""

    return _install_projection_factors(
        model,
        payload,
        target="v",
        layer_ranks=layer_ranks,
        head_dim=head_dim,
        require_cuda_resident=require_cuda_resident,
    )


def install_palu_k_factors(
    model: nn.Module,
    payload: Mapping[str, Tensor],
    *,
    layer_ranks: Sequence[Sequence[int]],
    head_dim: int,
    require_cuda_resident: bool = False,
) -> list[dict[str, Any]]:
    """Install K-only PaLU factors while preserving dense V."""

    return _install_projection_factors(
        model,
        payload,
        target="k",
        layer_ranks=layer_ranks,
        head_dim=head_dim,
        require_cuda_resident=require_cuda_resident,
    )


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PaLU PPL evaluation requires CUDA")
    torch.cuda.set_device(device)
    cuda_indices = _cuda_device_indices()
    for device_index in cuda_indices:
        torch.cuda.reset_peak_memory_stats(device_index)
    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dir: Path | None = None
    manifest: dict[str, Any] | None = None
    payload: dict[str, Tensor] | None = None
    if args.arm != "dense":
        if args.checkpoint_dir is None:
            raise ValueError(f"--checkpoint-dir is required for --arm {args.arm}")
        checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
        manifest, payload = _load_checkpoint(checkpoint_dir, model_path)
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
        model_kwargs["max_memory"] = {
            device_index: f"{args.max_memory_per_gpu_gib}GiB"
            for device_index in cuda_indices
        }
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), **model_kwargs
    )
    if args.device_map == "none":
        model.to(device)
    model.eval()
    config_geometry: dict[str, int] = {
        "num_query_heads": int(model.config.num_attention_heads),
        "num_physical_kv_heads": int(model.config.num_key_value_heads),
        "head_dim": int(
            getattr(model.config, "head_dim", 0)
            or model.config.hidden_size // model.config.num_attention_heads
        ),
    }
    model.config.use_cache = False

    install_started = time.perf_counter()
    if args.arm != "dense":
        assert manifest is not None and payload is not None
        compression = manifest["compression"]
        expected_projection = "k" if args.arm == "palu_k" else "v"
        projection = str(compression.get("projection", "v"))
        if projection != expected_projection:
            raise ValueError(
                f"--arm {args.arm} requires {expected_projection}_proj factors, "
                f"checkpoint contains {projection}_proj factors"
            )
        if "layer_ranks" in compression:
            layer_ranks = [
                [int(rank) for rank in ranks]
                for ranks in compression["layer_ranks"]
            ]
        else:
            uniform_ranks = [int(rank) for rank in compression["ranks"]]
            layer_ranks = [uniform_ranks] * len(_decoder_layers(model))
        head_dim = int(compression["head_dim"])
        for key, value in config_geometry.items():
            if value != int(compression[key]):
                raise ValueError(f"model/checkpoint geometry mismatch for {key}")
        if len(_decoder_layers(model)) != len(manifest["layers"]):
            raise ValueError("model/checkpoint layer count mismatch")
        installer = (
            install_palu_k_factors
            if expected_projection == "k"
            else install_palu_m_factors
        )
        installation = installer(
            model,
            payload,
            layer_ranks=layer_ranks,
            head_dim=head_dim,
            require_cuda_resident=args.device_map != "none",
        )
        checkpoint_record: dict[str, Any] | None = {
            "directory": str(checkpoint_dir),
            "manifest_sha256": _sha256(checkpoint_dir / "manifest.json"),
            "artifact_sha256": manifest["artifact"]["sha256"],
            "format": manifest["format"],
        }
        model_record: Mapping[str, Any] = manifest["model"]
        if expected_projection == "k":
            runtime_description = (
                "stored latent K factors reconstructed before RoPE; mathematically "
                "equivalent quality path, not a cache-performance benchmark"
            )
        else:
            runtime_description = (
                "stored latent V factors reconstructed immediately before attention; "
                "mathematically equivalent quality path, not a cache-performance benchmark"
            )
    else:
        kv_width = config_geometry["num_physical_kv_heads"] * config_geometry["head_dim"]
        compression = {
            **config_geometry,
            "target": "dense_kv_cache",
            "key_cache": "dense",
            "value_cache": "dense",
            "rank_sum": kv_width,
            "retained_v_ratio": 1.0,
            "v_cache_compression_ratio": 0.0,
        }
        installation = []
        checkpoint_record = None
        model_record = {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            "safetensors_index_sha256": _sha256(
                model_path / "model.safetensors.index.json"
            ),
        }
        runtime_description = "unaltered dense Hugging Face reference model"
    install_seconds = time.perf_counter() - install_started
    del payload
    ppl = _eval_ppl_fp32_loss(
        model,
        tokenizer,
        dataset=args.dataset,
        split=args.split,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
    )
    result = {
        "format": FORMAT,
        "status": "complete",
        "arm": args.arm,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_record,
        "model_dtype": str(dtype),
        "attention_implementation": args.attn_implementation,
        "device_map_strategy": args.device_map,
        "checkpoint": checkpoint_record,
        "compression": compression,
        "quality_reference_runtime": {
            "description": runtime_description,
            "installation_seconds": install_seconds,
            "layers": installation,
        },
        "ppl": ppl,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(device_index)
                for device_index in cuda_indices
            ],
            "peak_cuda_allocated_bytes": {
                str(device_index): int(torch.cuda.max_memory_allocated(device_index))
                for device_index in cuda_indices
            },
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_path, result)
    print(f"[Result] ppl={ppl['ppl']:.9f} output={output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dense", "palu_m", "palu_k"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-map",
        choices=("none", "auto", "balanced", "balanced_low_0"),
        default="none",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
