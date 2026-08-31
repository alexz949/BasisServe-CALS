#!/usr/bin/env python3
"""Compare dense BF16 with a selected dense C1-V96 decode kernel."""

from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import (  # noqa: E402
    install_qwen3_gqa_vo_als_export,
    transition_qwen3_dense_prefill_cache_to_c1,
)
from evaluation.eval_c1_block_scheduled_ppl import _dtype  # noqa: E402
from evaluation.eval_qwen3_8b_c1_kq_routing_ppl import (  # noqa: E402
    _attention_modules,
)
from evaluation.eval_qwen3_8b_c1_kq_routing_ruler import (  # noqa: E402
    _chunked_exact_prefill,
    _set_backend,
)
from evaluation.eval_qwen3_c1_quest_ruler import (  # noqa: E402
    _atomic_json,
    _atomic_text,
    _build_work,
    _load_dataset_manifest,
)
from evaluation.eval_qwen3_dense_ruler import _qwen3_config  # noqa: E402
from evaluation.ruler_v1 import parse_tasks, ruler_prompt  # noqa: E402


FORMAT = "basisserve.qwen3_8b.c1_v96_dense.decode_benchmark.v3"
DENSE_ARM = "dense_bf16_full_kv_sdpa"
C1_ARMS = {
    "triton": "dense_exact_k_c1_v96_triton",
    "cuda_dense": "dense_exact_k_c1_v96_cuda_shared_gqa",
}


def _pipeline_device_map(layers: int, devices: int) -> dict[str, int]:
    if layers <= 0 or devices <= 0 or devices > layers:
        raise ValueError("invalid pipeline layer/device geometry")
    mapping = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.norm": devices - 1,
        "lm_head": devices - 1,
    }
    for layer in range(layers):
        mapping[f"model.layers.{layer}"] = min(
            layer * devices // layers,
            devices - 1,
        )
    return mapping


def _synchronize(devices: tuple[torch.device, ...]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def _maximum_cuda_memory(
    devices: tuple[torch.device, ...],
) -> dict[str, int]:
    return {str(device): torch.cuda.max_memory_allocated(device) for device in devices}


def _cache_bytes(cache: Any) -> dict[str, int]:
    key_bytes = 0
    value_bytes = 0
    for layer in cache.layers:
        for name, kind in (("keys", "key"), ("values", "value")):
            tensor = getattr(layer, name, None)
            if tensor is None:
                continue
            size = tensor.numel() * tensor.element_size()
            if kind == "key":
                key_bytes += size
            else:
                value_bytes += size
    return {
        "key_bytes": key_bytes,
        "value_bytes": value_bytes,
        "total_bytes": key_bytes + value_bytes,
    }


def _dynamic_restore(cache: Any, prompt_tokens: int) -> None:
    appended = int(cache.get_seq_length()) - prompt_tokens
    if appended < 0:
        raise RuntimeError("decode shortened the dynamic prompt cache")
    if appended:
        cache.crop(-appended)
    if int(cache.get_seq_length()) != prompt_tokens:
        raise RuntimeError("dynamic cache transaction did not rollback")


def _static_restore(cache: StaticCache, prompt_tokens: int) -> None:
    for layer in cache.layers:
        if not layer.is_initialized:
            raise RuntimeError("static cache contains an uninitialized layer")
        if int(layer.cumulative_length) < prompt_tokens:
            raise RuntimeError("decode shortened the static prompt cache")
        layer.cumulative_length.fill_(prompt_tokens)
    if int(cache.get_seq_length()) != prompt_tokens:
        raise RuntimeError("static cache transaction did not rollback")


def _to_static_cache(
    dynamic_cache: Any,
    *,
    config: Any,
    capacity: int,
) -> StaticCache:
    prompt_tokens = int(dynamic_cache.get_seq_length())
    if prompt_tokens <= 0 or prompt_tokens > capacity:
        raise ValueError("prompt length is incompatible with static capacity")
    static_cache = StaticCache(config, max_cache_len=capacity)
    if len(static_cache.layers) != len(dynamic_cache.layers):
        raise RuntimeError("dynamic and static caches have different layer counts")
    for source, target in zip(
        dynamic_cache.layers,
        static_cache.layers,
        strict=True,
    ):
        if not source.is_initialized:
            raise RuntimeError("dynamic cache contains an uninitialized layer")
        target.lazy_initialization(
            source.keys[..., :1, :],
            source.values[..., :1, :],
        )
        target.keys[..., :prompt_tokens, :].copy_(source.keys)
        target.values[..., :prompt_tokens, :].copy_(source.values)
        target.cumulative_length.fill_(prompt_tokens)
    return static_cache


@torch.inference_mode()
def _greedy_continuation(
    model: torch.nn.Module,
    cache: Any,
    first_token: int,
    *,
    tokens: int,
    input_device: torch.device,
) -> list[int]:
    generated = [int(first_token)]
    while len(generated) < tokens:
        output = model(
            input_ids=torch.tensor(
                [[generated[-1]]],
                dtype=torch.long,
                device=input_device,
            ),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        generated.append(int(output.logits[0, -1].argmax().item()))
        del output
    return generated


@torch.inference_mode()
def _warmup(
    model: torch.nn.Module,
    cache: Any,
    first_token: int,
    *,
    tokens: int,
    prompt_tokens: int,
    input_device: torch.device,
    devices: tuple[torch.device, ...],
    restore: Callable[[Any, int], None],
) -> None:
    _greedy_continuation(
        model,
        cache,
        first_token,
        tokens=tokens,
        input_device=input_device,
    )
    _synchronize(devices)
    restore(cache, prompt_tokens)


@torch.inference_mode()
def _time_decode(
    model: torch.nn.Module,
    cache: Any,
    first_token: int,
    *,
    tokens: int,
    prompt_tokens: int,
    input_device: torch.device,
    devices: tuple[torch.device, ...],
    restore: Callable[[Any, int], None],
) -> dict[str, Any]:
    forward_steps = tokens - 1
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(devices)
    started = time.perf_counter()
    generated = _greedy_continuation(
        model,
        cache,
        first_token,
        tokens=tokens,
        input_device=input_device,
    )
    _synchronize(devices)
    elapsed = time.perf_counter() - started
    result = {
        "elapsed_seconds": elapsed,
        "milliseconds_per_token": 1000.0 * elapsed / forward_steps,
        "tokens_per_second": forward_steps / elapsed,
        "generated_token_ids": generated,
        "maximum_cuda_memory_bytes": _maximum_cuda_memory(devices),
    }
    restore(cache, prompt_tokens)
    return result


@torch.inference_mode()
def _one_step_logits(
    model: torch.nn.Module,
    cache: Any,
    first_token: int,
    *,
    prompt_tokens: int,
    input_device: torch.device,
    devices: tuple[torch.device, ...],
) -> torch.Tensor:
    output = model(
        input_ids=torch.tensor(
            [[first_token]],
            dtype=torch.long,
            device=input_device,
        ),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    _synchronize(devices)
    logits = output.logits[0, -1].float().cpu()
    del output
    _static_restore(cache, prompt_tokens)
    return logits


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--c1-export", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=131072)
    parser.add_argument("--task", default="niah_single_1")
    parser.add_argument("--sample-ordinal", type=int, default=0)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--pipeline-devices", type=int, default=2)
    parser.add_argument(
        "--c1-decode-backend",
        choices=tuple(C1_ARMS),
        default="triton",
    )
    parser.add_argument("--yarn-factor", type=float)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


@torch.inference_mode()
def main() -> None:
    args = _parser().parse_args()
    c1_arm = C1_ARMS[args.c1_decode_backend]
    arms_in_order = (DENSE_ARM, c1_arm)
    sizes = (
        args.sequence_length,
        args.decode_tokens,
        args.warmup_tokens,
        args.prefill_chunk_size,
        args.pipeline_devices,
    )
    if min(sizes) <= 0 or args.decode_tokens < 2:
        raise ValueError(
            "benchmark sizes must be positive and decode must exceed one token"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("C1-V96 dense decode benchmark requires CUDA")
    if torch.cuda.device_count() < args.pipeline_devices:
        raise RuntimeError("fewer visible CUDA devices than requested")

    model_path = args.model_path.expanduser().resolve()
    c1_export = args.c1_export.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_config = json.loads((model_path / "config.json").read_text())
    layers = int(raw_config["num_hidden_layers"])
    device_map = _pipeline_device_map(layers, args.pipeline_devices)
    devices = tuple(
        torch.device("cuda", index) for index in range(args.pipeline_devices)
    )
    tasks = parse_tasks(args.task)
    if len(tasks) != 1:
        raise ValueError("benchmark requires exactly one RULER task")
    _load_dataset_manifest(
        data_dir,
        sequence_length=args.sequence_length,
        samples=args.sample_ordinal + 1,
        tasks=tasks,
        tokenizer_path=model_path,
    )
    work = _build_work(data_dir, tasks, args.sample_ordinal + 1)
    _, task, ordinal, source = next(
        row for row in work if int(row[2]) == args.sample_ordinal
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    host_input_ids = tokenizer(
        ruler_prompt(source),
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"]
    prompt_tokens = int(host_input_ids.shape[1])
    if prompt_tokens + args.decode_tokens > args.sequence_length:
        raise ValueError("prompt plus decode exceeds configured context")

    arms: dict[str, dict[str, Any]] = {}
    prefills: dict[str, dict[str, Any]] = {}

    dense_config, rope_scaling = _qwen3_config(
        model_path,
        sequence_length=args.sequence_length,
        yarn_factor=args.yarn_factor,
    )
    dense_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=dense_config,
        dtype=_dtype(args.dtype),
        attn_implementation="sdpa",
        local_files_only=True,
        device_map=device_map,
    ).eval()
    dense_input_device = dense_model.get_input_embeddings().weight.device
    dense_input_ids = host_input_ids.to(dense_input_device)
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(devices)
    started = time.perf_counter()
    dense_first_token, dense_cache = _chunked_exact_prefill(
        dense_model,
        dense_input_ids,
        chunk_size=args.prefill_chunk_size,
        empty_cuda_cache_between_chunks=True,
    )
    _synchronize(devices)
    dense_prefill_seconds = time.perf_counter() - started
    prefills[DENSE_ARM] = {
        "backend": "transformers_sdpa_chunked_exact",
        "elapsed_seconds": dense_prefill_seconds,
        "first_token": dense_first_token,
        "cache_bytes": _cache_bytes(dense_cache),
        "maximum_cuda_memory_bytes": _maximum_cuda_memory(devices),
    }
    _warmup(
        dense_model,
        dense_cache,
        dense_first_token,
        tokens=args.warmup_tokens,
        prompt_tokens=prompt_tokens,
        input_device=dense_input_device,
        devices=devices,
        restore=_dynamic_restore,
    )
    arms[DENSE_ARM] = {
        **_time_decode(
            dense_model,
            dense_cache,
            dense_first_token,
            tokens=args.decode_tokens,
            prompt_tokens=prompt_tokens,
            input_device=dense_input_device,
            devices=devices,
            restore=_dynamic_restore,
        ),
        "attention_backend": "transformers_sdpa",
        "cache": "dynamic_exact_k_v_bfloat16",
    }
    del dense_input_ids, dense_cache, dense_model, dense_config
    gc.collect()
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    memory_after_dense_release = {
        str(device): torch.cuda.memory_allocated(device) for device in devices
    }

    c1_config, c1_rope_scaling = _qwen3_config(
        model_path,
        sequence_length=args.sequence_length,
        yarn_factor=args.yarn_factor,
    )
    if c1_rope_scaling != rope_scaling:
        raise RuntimeError("dense and C1 arms received different RoPE scaling")
    c1_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=c1_config,
        dtype=_dtype(args.dtype),
        attn_implementation="sdpa",
        local_files_only=True,
        device_map=device_map,
    ).eval()
    install_qwen3_gqa_vo_als_export(
        c1_model,
        c1_export,
        attention_backend="dense_prefill",
    )
    modules = _attention_modules(c1_model)
    value_ranks = {int(module.value_head_dim) for module in modules}
    if value_ranks != {96}:
        raise ValueError(f"benchmark requires uniform C1-V96, got {value_ranks}")
    c1_input_device = c1_model.get_input_embeddings().weight.device
    c1_input_ids = host_input_ids.to(c1_input_device)
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(devices)
    started = time.perf_counter()
    c1_first_token, dynamic_c1_cache = _chunked_exact_prefill(
        c1_model,
        c1_input_ids,
        chunk_size=args.prefill_chunk_size,
        empty_cuda_cache_between_chunks=True,
    )
    _synchronize(devices)
    c1_prefill_seconds = time.perf_counter() - started
    dense_dynamic_cache_bytes = _cache_bytes(dynamic_c1_cache)
    c1_prefill_peak = _maximum_cuda_memory(devices)

    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(devices)
    projection_started = time.perf_counter()
    transition_qwen3_dense_prefill_cache_to_c1(
        c1_model,
        dynamic_c1_cache,
        attention_backend="sdpa",
    )
    _synchronize(devices)
    value_cache_projection_seconds = time.perf_counter() - projection_started
    projected_dynamic_cache_bytes = _cache_bytes(dynamic_c1_cache)
    value_cache_projection_peak = _maximum_cuda_memory(devices)

    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(devices)
    conversion_started = time.perf_counter()
    static_c1_cache = _to_static_cache(
        dynamic_c1_cache,
        config=c1_model.config,
        capacity=args.sequence_length,
    )
    _synchronize(devices)
    static_conversion_seconds = time.perf_counter() - conversion_started
    static_conversion_peak = _maximum_cuda_memory(devices)
    del dynamic_c1_cache, c1_input_ids, host_input_ids
    gc.collect()
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    prefills[c1_arm] = {
        "backend": "transformers_sdpa_dense_v128_chunked_exact",
        "elapsed_seconds": c1_prefill_seconds,
        "first_token": c1_first_token,
        "dense_dynamic_cache_bytes": dense_dynamic_cache_bytes,
        "maximum_cuda_memory_bytes": c1_prefill_peak,
        "value_cache_projection": "dense_v128_times_c1_encoder_to_v96",
        "value_cache_projection_seconds": value_cache_projection_seconds,
        "projected_dynamic_cache_bytes": projected_dynamic_cache_bytes,
        "value_cache_projection_peak_cuda_memory_bytes": (
            value_cache_projection_peak
        ),
        "static_cache_conversion_seconds": static_conversion_seconds,
        "static_cache_capacity_bytes": _cache_bytes(static_c1_cache),
        "static_cache_conversion_peak_cuda_memory_bytes": static_conversion_peak,
    }

    reference_logits = _one_step_logits(
        c1_model,
        static_c1_cache,
        c1_first_token,
        prompt_tokens=prompt_tokens,
        input_device=c1_input_device,
        devices=devices,
    )
    _set_backend(modules, args.c1_decode_backend)
    backend_logits = _one_step_logits(
        c1_model,
        static_c1_cache,
        c1_first_token,
        prompt_tokens=prompt_tokens,
        input_device=c1_input_device,
        devices=devices,
    )
    logits_difference = backend_logits - reference_logits
    correctness = {
        "backend": args.c1_decode_backend,
        "sdpa_argmax": int(reference_logits.argmax()),
        "backend_argmax": int(backend_logits.argmax()),
        "argmax_equal": bool(reference_logits.argmax() == backend_logits.argmax()),
        "backend_logits_finite": bool(torch.isfinite(backend_logits).all()),
        "maximum_absolute_logit_error": float(logits_difference.abs().max()),
        "relative_l2_logit_error": float(
            torch.linalg.vector_norm(logits_difference)
            / torch.linalg.vector_norm(reference_logits).clamp_min(1e-30)
        ),
    }
    del reference_logits, backend_logits, logits_difference
    _warmup(
        c1_model,
        static_c1_cache,
        c1_first_token,
        tokens=args.warmup_tokens,
        prompt_tokens=prompt_tokens,
        input_device=c1_input_device,
        devices=devices,
        restore=_static_restore,
    )
    arms[c1_arm] = {
        **_time_decode(
            c1_model,
            static_c1_cache,
            c1_first_token,
            tokens=args.decode_tokens,
            prompt_tokens=prompt_tokens,
            input_device=c1_input_device,
            devices=devices,
            restore=_static_restore,
        ),
        "attention_backend": (
            "c1_dense_gqa_v96_decode_attention_cuda"
            if args.c1_decode_backend == "cuda_dense"
            else "compressed_v_decode_attention_triton"
        ),
        "cache": "static_exact_k128_c1_v96_bfloat16",
        "static_cache_capacity": args.sequence_length,
    }

    dense = arms[DENSE_ARM]
    c1 = arms[c1_arm]
    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "metadata": {
            "model_path": str(model_path),
            "c1_export": str(c1_export),
            "data_dir": str(data_dir),
            "task": task.name,
            "sample_ordinal": ordinal,
            "sequence_length": args.sequence_length,
            "prompt_tokens": prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "timed_decode_forward_steps": args.decode_tokens - 1,
            "warmup_tokens": args.warmup_tokens,
            "prefill_chunk_size": args.prefill_chunk_size,
            "pipeline_devices": args.pipeline_devices,
            "device_map": device_map,
            "devices": [str(device) for device in devices],
            "cuda_device_names": {
                str(device): torch.cuda.get_device_name(device) for device in devices
            },
            "dtype": args.dtype,
            "value_rank": 96,
            "c1_decode_backend": args.c1_decode_backend,
            "rope_scaling": rope_scaling,
            "memory_after_dense_model_release_bytes": memory_after_dense_release,
            "physical_pcie_offload": False,
        },
        "prefill": prefills,
        "arms": arms,
        "comparison": {
            "c1_speedup_vs_dense": (dense["elapsed_seconds"] / c1["elapsed_seconds"]),
            "c1_time_reduction_fraction": (
                1.0 - c1["elapsed_seconds"] / dense["elapsed_seconds"]
            ),
            "generated_token_ids_equal": (
                dense["generated_token_ids"] == c1["generated_token_ids"]
            ),
            "dense_and_c1_prefill_first_token_equal": (
                dense_first_token == c1_first_token
            ),
        },
        "c1_sdpa_vs_decode_backend_correctness": correctness,
    }
    _atomic_json(output_dir / "result.json", payload)
    lines = [
        f"# Qwen3-8B C1-V96 {args.c1_decode_backend} "
        f"{args.sequence_length // 1024}K decode",
        "",
        f"Prompt: `{prompt_tokens}`; generated output: `{args.decode_tokens}` "
        f"(`{args.decode_tokens - 1}` timed forwards); devices: "
        f"`{args.pipeline_devices}`.",
        "",
        "| Arm | Seconds | ms/token | tokens/s |",
        "|:---|---:|---:|---:|",
    ]
    for arm in arms_in_order:
        row = arms[arm]
        lines.append(
            f"| {arm} | {row['elapsed_seconds']:.3f} | "
            f"{row['milliseconds_per_token']:.3f} | "
            f"{row['tokens_per_second']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"C1-V96 {args.c1_decode_backend} speedup versus dense BF16 SDPA: "
            f"`{payload['comparison']['c1_speedup_vs_dense']:.3f}x`.",
            "",
            f"Dense prefill: `{dense_prefill_seconds:.3f}` seconds; C1 prefill: "
            f"`{c1_prefill_seconds:.3f}` seconds; dense-V128 to C1-V96 cache "
            f"projection: `{value_cache_projection_seconds:.3f}` seconds; "
            f"dynamic-to-static cache conversion: "
            f"`{static_conversion_seconds:.3f}` seconds.",
            "",
            "Both arms use exact dense prefill. The C1 arm then projects only the "
            "resident V cache to rank 96, preserves exact K, and uses a "
            "fixed-capacity decode cache with a device-resident valid length.",
            "",
            f"One-step C1 SDPA/{args.c1_decode_backend} check: "
            f"argmax `{correctness['sdpa_argmax']}` / "
            f"`{correctness['backend_argmax']}`; relative logits L2 error "
            f"`{correctness['relative_l2_logit_error']:.6g}`; finite "
            f"`{str(correctness['backend_logits_finite']).lower()}`.",
        ]
    )
    _atomic_text(output_dir / "summary.md", "\n".join(lines) + "\n")
    print(
        f"[C1-V96 {args.c1_decode_backend} decode] result={output_dir / 'result.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
