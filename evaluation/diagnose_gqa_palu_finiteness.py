#!/usr/bin/env python3
"""Locate the first non-finite activation in a GQA PaLU checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.eval_gqa_palu_m_wikitext import (
    _decoder_layers,
    _load_checkpoint,
    install_palu_m_factors,
)
from scripts.eval_svdllm_safetensors_ppl_accelerate import _input_device, _token_ids


def _tensor_stats(value: Tensor) -> dict[str, Any]:
    detached = value.detach()
    nan_count = int(torch.isnan(detached).sum().cpu())
    inf_count = int(torch.isinf(detached).sum().cpu())
    is_finite = nan_count == 0 and inf_count == 0
    max_abs = float(detached.abs().max().float().cpu()) if is_finite else None
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite": is_finite,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "max_abs_finite": max_abs,
    }


def _first_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], Tensor):
        return value[0]
    raise TypeError(f"cannot extract tensor from {type(value).__name__}")


@torch.inference_mode()
def diagnose(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    manifest, payload = _load_checkpoint(checkpoint_dir, model_path)
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_memory={
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        },
    )
    model.config.use_cache = False
    compression = manifest["compression"]
    if "layer_ranks" in compression:
        layer_ranks = compression["layer_ranks"]
    else:
        layer_ranks = [compression["ranks"]] * len(_decoder_layers(model))
    install_palu_m_factors(
        model,
        payload,
        layer_ranks=layer_ranks,
        head_dim=int(compression["head_dim"]),
        require_cuda_resident=True,
    )
    del payload

    token_ids = _token_ids(tokenizer, "wikitext2", "test", None)
    total_chunks = int(token_ids.numel() // args.seqlen)
    input_device = _input_device(model)
    scan_records: list[dict[str, Any]] = []
    first_bad_chunk: int | None = None
    trace_chunk = 0
    if args.scan_all_chunks:
        for chunk_index in range(total_chunks):
            start = chunk_index * args.seqlen
            batch = token_ids[:, start : start + args.seqlen].to(input_device)
            logits = model(input_ids=batch, use_cache=False).logits
            finite = bool(torch.isfinite(logits).all().cpu())
            max_abs = (
                float(logits.abs().max().float().cpu()) if finite else None
            )
            scan_records.append(
                {"chunk": chunk_index, "logits_finite": finite, "max_abs": max_abs}
            )
            print(
                f"[Scan] chunk={chunk_index}/{total_chunks} "
                f"logits_finite={finite} max_abs={max_abs}",
                flush=True,
            )
            del logits
            if not finite:
                first_bad_chunk = chunk_index
                trace_chunk = chunk_index
                break

    records: list[dict[str, Any]] = []
    handles: list[Any] = []

    def record(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            stats = _tensor_stats(_first_tensor(output))
            records.append({"name": name, **stats})
            print(
                f"[Finite] {name} finite={stats['finite']} "
                f"max_abs={stats['max_abs_finite']} nan={stats['nan_count']} "
                f"inf={stats['inf_count']}",
                flush=True,
            )

        return hook

    should_trace = not args.scan_all_chunks or first_bad_chunk is not None
    if should_trace:
        for layer_index, layer in enumerate(_decoder_layers(model)):
            handles.append(
                layer.self_attn.v_proj.VT.register_forward_hook(
                    record(f"layers.{layer_index}.v_latent")
                )
            )
            handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    record(f"layers.{layer_index}.v_reconstructed")
                )
            )
            handles.append(
                layer.register_forward_hook(record(f"layers.{layer_index}.output"))
            )

        start = trace_chunk * args.seqlen
        input_ids = token_ids[:, start : start + args.seqlen].to(input_device)
        logits = model(input_ids=input_ids, use_cache=False).logits
        logits_stats = _tensor_stats(logits)
        for handle in handles:
            handle.remove()
    else:
        logits_stats = {"finite": True}

    first_nonfinite = next((row["name"] for row in records if not row["finite"]), None)
    result = {
        "status": "complete",
        "purpose": "diagnostic_only",
        "model": str(model_path),
        "checkpoint": str(checkpoint_dir),
        "seqlen": args.seqlen,
        "model_dtype": str(dtype),
        "scan_all_chunks": args.scan_all_chunks,
        "total_chunks": total_chunks,
        "chunks_scanned": len(scan_records),
        "first_bad_chunk": first_bad_chunk,
        "trace_chunk": trace_chunk if should_trace else None,
        "scan_records": scan_records,
        "first_nonfinite": first_nonfinite,
        "records": records,
        "logits": logits_stats,
        "loss_would_be_finite": bool(logits_stats["finite"]),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(
        f"[Result] first_nonfinite={first_nonfinite} "
        f"logits_finite={logits_stats['finite']} output={output_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument("--scan-all-chunks", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    diagnose(parse_args())
