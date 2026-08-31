#!/usr/bin/env python3
"""Locate the first non-finite activation in a folded Qwen3-32B C1 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from evaluation.diagnose_gqa_palu_finiteness import _first_tensor, _tensor_stats
from evaluation.eval_qwen3_32b_c1_wikitext import (
    _decoder_layers,
    _load_layer_allocation_results,
    _load_ragged_results,
    install_layer_allocation_factors,
    install_ragged_c1_factors,
)
from scripts.eval_svdllm_safetensors_ppl_accelerate import _input_device, _token_ids


@torch.inference_mode()
def diagnose(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model).expanduser().resolve()
    allocation = args.allocation_dir is not None
    factor_dir = Path(
        args.allocation_dir if allocation else args.ragged_factor_dir
    ).expanduser().resolve()
    result = (
        _load_layer_allocation_results(factor_dir, model_path)
        if allocation
        else _load_ragged_results(factor_dir, model_path)
    )
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
    ).eval()
    model.config.use_cache = False
    if allocation:
        install_layer_allocation_factors(model, factor_dir, result)
    else:
        install_ragged_c1_factors(model, factor_dir, result)

    token_ids = _token_ids(tokenizer, "wikitext2", "test", None)
    input_ids = token_ids[:, : args.seqlen].to(_input_device(model))

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

    for layer_index, layer in enumerate(_decoder_layers(model)):
        handles.append(
            layer.self_attn.v_proj.register_forward_hook(
                record(f"layers.{layer_index}.v_proj")
            )
        )
        handles.append(
            layer.self_attn.register_forward_hook(
                record(f"layers.{layer_index}.self_attn")
            )
        )
        handles.append(
            layer.register_forward_hook(record(f"layers.{layer_index}.output"))
        )

    logits = model(input_ids=input_ids, use_cache=False).logits
    logits_stats = _tensor_stats(logits)
    for handle in handles:
        handle.remove()

    first_nonfinite = next((row["name"] for row in records if not row["finite"]), None)
    payload = {
        "status": "complete",
        "purpose": "diagnostic_only",
        "model": str(model_path),
        "factor_dir": str(factor_dir),
        "factor_result": str(factor_dir / "result.json"),
        "seqlen": args.seqlen,
        "model_dtype": str(dtype),
        "first_nonfinite": first_nonfinite,
        "records": records,
        "logits": logits_stats,
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
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
    factors = parser.add_mutually_exclusive_group(required=True)
    factors.add_argument("--allocation-dir")
    factors.add_argument("--ragged-factor-dir")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=72)
    return parser.parse_args()


if __name__ == "__main__":
    diagnose(parse_args())
