#!/usr/bin/env python3
"""Preflight Qwen3.5-9B output-wire capture on one CUDA device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor, nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--use-kernels", action="store_true")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    return parser.parse_args()


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidates = (
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model, "language_model", None), "layers", None),
    )
    for candidate in candidates:
        if isinstance(candidate, nn.ModuleList):
            return candidate
    raise ValueError("could not locate Qwen3.5 language-model decoder layers")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if args.sequence_length <= 0 or args.batch_size <= 0:
        raise ValueError("sequence length and batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 V100 preflight requires CUDA")
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite preflight result: {output_path}")

    from transformers import AutoConfig, AutoModelForMultimodalLM, AutoTokenizer

    model_path = Path(args.model_path).expanduser().resolve()
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    text_config = config.text_config
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    load_started = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_path),
        dtype=torch.float16,
        device_map={"": args.device},
        local_files_only=True,
        attn_implementation="sdpa",
        use_kernels=args.use_kernels,
    ).eval()
    load_seconds = time.perf_counter() - load_started

    calls: dict[int, dict[str, Any]] = {}
    handles: list[Any] = []

    def register(layer_index: int, block_type: str, projection: nn.Module) -> None:
        weight = getattr(projection, "weight", None)
        if not isinstance(weight, Tensor) or weight.ndim != 2:
            raise TypeError(f"layer {layer_index} {block_type} projection has no matrix weight")
        entry = {
            "layer_index": layer_index,
            "block_type": block_type,
            "projection_shape": list(map(int, weight.shape)),
            "calls": 0,
            "input_shape": None,
            "finite": True,
        }
        calls[layer_index] = entry

        def hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            del module
            if len(inputs) != 1 or not isinstance(inputs[0], Tensor):
                raise TypeError(f"layer {layer_index} projection input is not one tensor")
            value = inputs[0]
            if int(value.shape[-1]) != int(weight.shape[1]):
                raise ValueError(f"layer {layer_index} projection input width differs from weight")
            entry["calls"] = int(entry["calls"]) + 1
            entry["input_shape"] = list(map(int, value.shape))
            entry["finite"] = bool(torch.isfinite(value).all().item())

        handles.append(projection.register_forward_pre_hook(hook))

    layers = _decoder_layers(model)
    for layer_index, layer in enumerate(layers):
        linear_attention = getattr(layer, "linear_attn", None)
        full_attention = getattr(layer, "self_attn", None)
        if linear_attention is not None:
            register(layer_index, "linear_attention", linear_attention.out_proj)
        elif full_attention is not None:
            register(layer_index, "full_attention", full_attention.o_proj)
        else:
            raise ValueError(f"layer {layer_index} has neither Qwen3.5 attention block")

    prompt_ids = tokenizer(
        "BasisServe Qwen3.5 C1 calibration preflight. ",
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]
    repeats = (args.sequence_length + int(prompt_ids.numel()) - 1) // int(prompt_ids.numel())
    input_ids = prompt_ids.repeat(repeats)[: args.sequence_length]
    input_ids = input_ids.unsqueeze(0).expand(args.batch_size, -1).contiguous().to(args.device)
    torch.cuda.reset_peak_memory_stats()
    forward_started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=1,
            )
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    forward_seconds = time.perf_counter() - forward_started

    records = [calls[index] for index in sorted(calls)]
    expected_linear = sum(kind == "linear_attention" for kind in text_config.layer_types)
    expected_full = sum(kind == "full_attention" for kind in text_config.layer_types)
    observed_linear = sum(row["block_type"] == "linear_attention" for row in records)
    observed_full = sum(row["block_type"] == "full_attention" for row in records)
    if (observed_linear, observed_full) != (expected_linear, expected_full):
        raise RuntimeError("Qwen3.5 attention-layer census differs from config")
    if any(row["calls"] != 1 or not row["finite"] for row in records):
        raise RuntimeError("not every Qwen3.5 output wire produced one finite activation")
    if not torch.isfinite(output.logits).all():
        raise RuntimeError("Qwen3.5 preflight logits contain non-finite values")

    device_index = torch.cuda.current_device()
    result = {
        "format": "basisserve.qwen35_9b.c1_v100_preflight.v1",
        "model_path": str(model_path),
        "model_commit": getattr(config, "_commit_hash", None),
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "dtype": "float16",
        "attention_implementation": "sdpa",
        "use_kernels": args.use_kernels,
        "load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "linear_attention_layers": observed_linear,
        "full_attention_layers": observed_full,
        "layers": records,
        "cuda": {
            "name": torch.cuda.get_device_name(device_index),
            "capability": list(torch.cuda.get_device_capability(device_index)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
