#!/usr/bin/env python3
"""Compare dense BF16 decode with C1 dense and KQ-routed sparse decode."""

from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import (  # noqa: E402
    GQATiedVOQwen3Attention,
    install_qwen3_gqa_vo_als_export,
    transition_qwen3_dense_prefill_cache_to_c1,
)
from basisserve.core.c1_k_routing_sidecar import (  # noqa: E402
    ROUTING_PROXY_IMPLEMENTATION,
    RoutingDynamicCache,
)
from evaluation.eval_c1_block_scheduled_ppl import _dtype  # noqa: E402
from evaluation.eval_qwen3_8b_c1_kq_routing_ppl import (  # noqa: E402
    _attention_modules,
    _load_routing_factors,
    _set_routing_rank,
    _set_sparse_policy,
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


FORMAT = "basisserve.qwen3_8b.c1_kq_dense_sparse.decode_benchmark.v3"
DENSE_BF16_ARM = "dense_bf16_full_kv"
DENSE_C1_ARM = "dense_exact_k_c1_v96"
SPARSE_C1_ARM = "r32_sparse_exact_k_c1_v96"
ARMS = (DENSE_BF16_ARM, DENSE_C1_ARM, SPARSE_C1_ARM)


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


def _module_devices(
    modules: list[GQATiedVOQwen3Attention],
) -> tuple[torch.device, ...]:
    return tuple(
        sorted(
            {module.q_proj.weight.device for module in modules},
            key=lambda device: device.index if device.index is not None else -1,
        )
    )


def _synchronize(devices: tuple[torch.device, ...]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def _set_cached_sidecar(
    modules: list[GQATiedVOQwen3Attention], *, enabled: bool
) -> None:
    for module in modules:
        module.set_cached_routing_sidecar_enabled(enabled)


def _cache_bytes(cache: Any) -> dict[str, int]:
    """Count persistent K/V tensors without counting routing metadata."""

    key_bytes = 0
    value_bytes = 0
    for layer in cache.layers:
        for name, destination in (("keys", "key"), ("values", "value")):
            tensor = getattr(layer, name, None)
            if tensor is None:
                continue
            size = tensor.numel() * tensor.element_size()
            if destination == "key":
                key_bytes += size
            else:
                value_bytes += size
    return {
        "key_bytes": key_bytes,
        "value_bytes": value_bytes,
        "total_bytes": key_bytes + value_bytes,
    }


def _to_static_cache(
    dynamic_cache: Any,
    *,
    config: Any,
    capacity: int,
) -> StaticCache:
    prompt_tokens = int(dynamic_cache.get_seq_length())
    static_cache = StaticCache(config, max_cache_len=capacity)
    for source, target in zip(
        dynamic_cache.layers,
        static_cache.layers,
        strict=True,
    ):
        target.lazy_initialization(
            source.keys[..., :1, :],
            source.values[..., :1, :],
        )
        target.keys[..., :prompt_tokens, :].copy_(source.keys)
        target.values[..., :prompt_tokens, :].copy_(source.values)
        target.cumulative_length.fill_(prompt_tokens)
    return static_cache


def _maximum_cuda_memory(devices: tuple[torch.device, ...]) -> dict[str, int]:
    return {
        str(device): torch.cuda.max_memory_allocated(device)
        for device in devices
    }


@torch.inference_mode()
def _forced_greedy_continuation(
    model: torch.nn.Module,
    cache: Any,
    first_token: int,
    *,
    tokens: int,
    input_device: torch.device,
) -> list[int]:
    if tokens <= 0:
        raise ValueError("decode token count must be positive")
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


def _restore_prompt_cache(
    cache: Any,
    *,
    prompt_tokens: int,
) -> None:
    if isinstance(cache, StaticCache):
        for layer in cache.layers:
            layer.cumulative_length.fill_(prompt_tokens)
        return
    appended = int(cache.get_seq_length()) - prompt_tokens
    if appended < 0:
        raise RuntimeError("decode shortened the prompt cache")
    if appended:
        cache.crop(-appended)
    if int(cache.get_seq_length()) != prompt_tokens:
        raise RuntimeError("decode transaction did not restore prompt cache")
    if isinstance(cache, RoutingDynamicCache):
        for layer_idx in range(len(cache.layers)):
            sidecar = cache.routing_sidecar(layer_idx)
            if sidecar is not None and int(sidecar.shape[-2]) != prompt_tokens:
                raise RuntimeError("routing sidecar transaction did not rollback")


def _cuda_sparse_last_statistics(
    modules: list[GQATiedVOQwen3Attention],
) -> dict[str, float]:
    rows = [module.cuda_sparse_last_statistics() for module in modules]
    return {
        name: sum(float(row[name]) for row in rows)
        for name in (
            "selected_pages",
            "staging_page_slots",
            "staging_exact_key_bytes",
        )
    }


def _sidecar_bytes(cache: RoutingDynamicCache) -> int:
    return sum(
        sidecar.numel() * sidecar.element_size()
        for layer_idx in range(len(cache.layers))
        if (sidecar := cache.routing_sidecar(layer_idx)) is not None
    )


@torch.inference_mode()
def _warmup_decode(
    model: torch.nn.Module,
    cache: Any,
    first_token: int,
    *,
    tokens: int,
    prompt_tokens: int,
    input_device: torch.device,
    devices: tuple[torch.device, ...],
) -> None:
    _forced_greedy_continuation(
        model,
        cache,
        first_token,
        tokens=tokens,
        input_device=input_device,
    )
    _synchronize(devices)
    _restore_prompt_cache(cache, prompt_tokens=prompt_tokens)


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
) -> dict[str, Any]:
    forward_steps = tokens - 1
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(devices)
    started = time.perf_counter()
    generated = _forced_greedy_continuation(
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
        "tokens_per_second": forward_steps / elapsed,
        "milliseconds_per_token": 1000.0 * elapsed / forward_steps,
        "generated_token_ids": generated,
        "maximum_cuda_memory_bytes": _maximum_cuda_memory(devices),
    }
    _restore_prompt_cache(cache, prompt_tokens=prompt_tokens)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--c1-export", type=Path, required=True)
    parser.add_argument("--routing-factors", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=131072)
    parser.add_argument("--task", default="niah_single_1")
    parser.add_argument("--sample-ordinal", type=int, default=0)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--routing-rank", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--exact-token-budget", type=int, default=2048)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--pipeline-devices", type=int, default=2)
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
    if not torch.cuda.is_available():
        raise RuntimeError("sidecar decode benchmark requires CUDA")
    sizes = (
        args.sequence_length,
        args.decode_tokens,
        args.warmup_tokens,
        args.routing_rank,
        args.page_size,
        args.exact_token_budget,
        args.prefill_chunk_size,
        args.pipeline_devices,
    )
    if min(sizes) <= 0:
        raise ValueError("benchmark sizes must be positive")
    if args.decode_tokens < 2:
        raise ValueError(
            "decode-tokens must be at least two because the first output token "
            "is produced by prefill"
        )
    if args.exact_token_budget % args.page_size:
        raise ValueError("exact-token budget must align to complete pages")
    if torch.cuda.device_count() < args.pipeline_devices:
        raise RuntimeError("fewer visible CUDA devices than requested")

    model_path = args.model_path.expanduser().resolve()
    c1_export = args.c1_export.expanduser().resolve()
    factor_dir = args.routing_factors.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_config = json.loads((model_path / "config.json").read_text())
    layers = int(raw_config["num_hidden_layers"])
    kv_heads = int(raw_config["num_key_value_heads"])
    query_heads = int(raw_config["num_attention_heads"])
    head_dim = int(
        raw_config.get("head_dim", raw_config["hidden_size"] // query_heads)
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
    device_map = _pipeline_device_map(layers, args.pipeline_devices)
    devices = tuple(
        torch.device("cuda", device_idx)
        for device_idx in range(args.pipeline_devices)
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
        raise ValueError("prompt plus forced decode exceeds configured context")

    arms: dict[str, dict[str, Any]] = {}
    prefills: dict[str, dict[str, Any]] = {}
    timed_decode_forward_steps = args.decode_tokens - 1

    # Arm 1: the unmodified model with exact BF16 K and exact BF16 V.
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
    dense_prefill_started = time.perf_counter()
    dense_first_token, dense_cache = _chunked_exact_prefill(
        dense_model,
        dense_input_ids,
        chunk_size=args.prefill_chunk_size,
        empty_cuda_cache_between_chunks=True,
    )
    _synchronize(devices)
    dense_prefill_seconds = time.perf_counter() - dense_prefill_started
    if isinstance(dense_cache, RoutingDynamicCache):
        raise RuntimeError("dense BF16 prefill unexpectedly returned a routing cache")
    prefills[DENSE_BF16_ARM] = {
        "backend": "transformers_sdpa_chunked_exact",
        "elapsed_seconds": dense_prefill_seconds,
        "first_token": dense_first_token,
        "cache_bytes": _cache_bytes(dense_cache),
        "routing_sidecar_bytes": 0,
        "maximum_cuda_memory_bytes": _maximum_cuda_memory(devices),
    }
    _warmup_decode(
        dense_model,
        dense_cache,
        dense_first_token,
        tokens=args.warmup_tokens,
        prompt_tokens=prompt_tokens,
        input_device=dense_input_device,
        devices=devices,
    )
    arms[DENSE_BF16_ARM] = {
        **_time_decode(
            dense_model,
            dense_cache,
            dense_first_token,
            tokens=args.decode_tokens,
            prompt_tokens=prompt_tokens,
            input_device=dense_input_device,
            devices=devices,
        ),
        "key_cache": "exact_bfloat16",
        "value_cache": "exact_bfloat16",
        "attention": "full_dense_sdpa",
        "runtime_logical": None,
    }
    del dense_input_ids, dense_cache, dense_model, dense_config
    gc.collect()
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    memory_after_dense_release = {
        str(device): torch.cuda.memory_allocated(device) for device in devices
    }

    # Arms 2 and 3 share one exact-K/C1-V96 prompt cache. The only decode
    # difference is full dense attention versus R32 routing and sparse exact QK.
    c1_config, c1_rope_scaling = _qwen3_config(
        model_path,
        sequence_length=args.sequence_length,
        yarn_factor=args.yarn_factor,
    )
    if c1_rope_scaling != rope_scaling:
        raise RuntimeError("dense and C1 arms received different RoPE scaling")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=c1_config,
        dtype=_dtype(args.dtype),
        attn_implementation="sdpa",
        local_files_only=True,
        device_map=device_map,
    ).eval()
    install_qwen3_gqa_vo_als_export(
        model,
        c1_export,
        attention_backend="dense_prefill",
    )
    modules = _attention_modules(model)
    value_ranks = {int(module.value_head_dim) for module in modules}
    if value_ranks != {96}:
        raise ValueError(f"benchmark requires uniform C1-V96, got {value_ranks}")
    key_factors, query_factors, _, _, _ = _load_routing_factors(
        factor_dir,
        model_path=model_path,
        layers=layers,
        kv_heads=kv_heads,
        query_heads=query_heads,
        head_dim=head_dim,
    )
    if args.routing_rank > int(key_factors.shape[-1]):
        raise ValueError("routing rank exceeds calibrated factor rank")
    _set_routing_rank(
        modules,
        key_factors,
        query_factors,
        args.routing_rank,
    )
    observed_devices = _module_devices(modules)
    if observed_devices != devices:
        raise RuntimeError(
            f"model uses {observed_devices}, expected pipeline devices {devices}"
        )
    input_device = model.get_input_embeddings().weight.device
    input_ids = host_input_ids.to(input_device)
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(devices)
    c1_prefill_started = time.perf_counter()
    first_token, cache = _chunked_exact_prefill(
        model,
        input_ids,
        chunk_size=args.prefill_chunk_size,
        empty_cuda_cache_between_chunks=True,
        incremental_routing_sidecar=True,
    )
    _synchronize(devices)
    c1_prefill_seconds = time.perf_counter() - c1_prefill_started
    if not isinstance(cache, RoutingDynamicCache):
        raise RuntimeError("C1 prefill did not return a routing-aware cache")
    dense_prefill_cache_bytes = _cache_bytes(cache)
    c1_prefill_peak = _maximum_cuda_memory(devices)
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(devices)
    projection_started = time.perf_counter()
    transition_qwen3_dense_prefill_cache_to_c1(
        model,
        cache,
        attention_backend="sdpa",
    )
    _synchronize(devices)
    value_cache_projection_seconds = time.perf_counter() - projection_started
    value_cache_projection_peak = _maximum_cuda_memory(devices)
    prefills["c1_shared"] = {
        "shared_by": [DENSE_C1_ARM, SPARSE_C1_ARM],
        "backend": "transformers_sdpa_dense_v128_chunked_exact",
        "elapsed_seconds": c1_prefill_seconds,
        "first_token": first_token,
        "dense_prefill_cache_bytes": dense_prefill_cache_bytes,
        "value_cache_projection_seconds": value_cache_projection_seconds,
        "projected_cache_bytes": _cache_bytes(cache),
        "routing_sidecar_bytes": _sidecar_bytes(cache),
        "maximum_cuda_memory_bytes": c1_prefill_peak,
        "value_cache_projection_peak_cuda_memory_bytes": (
            value_cache_projection_peak
        ),
    }
    del input_ids, host_input_ids

    static_cache_started = time.perf_counter()
    static_cache = _to_static_cache(
        cache,
        config=model.config,
        capacity=args.sequence_length,
    )
    _synchronize(devices)
    static_cache_seconds = time.perf_counter() - static_cache_started
    _set_sparse_policy(
        modules,
        budget=None,
        page_size=args.page_size,
        force_last_page=False,
    )
    _set_backend(modules, "cuda_dense")
    _set_cached_sidecar(modules, enabled=True)
    _warmup_decode(
        model,
        static_cache,
        first_token,
        tokens=args.warmup_tokens,
        prompt_tokens=prompt_tokens,
        input_device=input_device,
        devices=devices,
    )
    arms[DENSE_C1_ARM] = {
        **_time_decode(
            model,
            static_cache,
            first_token,
            tokens=args.decode_tokens,
            prompt_tokens=prompt_tokens,
            input_device=input_device,
            devices=devices,
        ),
        "key_cache": "exact_bfloat16",
        "value_cache": "c1_bfloat16_rank96",
        "attention": "full_dense_cuda_shared_gqa",
        "cache": "static",
        "static_cache_conversion_seconds": static_cache_seconds,
        "runtime_logical": None,
    }
    del static_cache
    gc.collect()
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    _set_backend(modules, "cuda_sparse")
    _set_sparse_policy(
        modules,
        budget=args.exact_token_budget,
        page_size=args.page_size,
        force_last_page=False,
    )
    _set_cached_sidecar(modules, enabled=True)
    _warmup_decode(
        model,
        cache,
        first_token,
        tokens=args.warmup_tokens,
        prompt_tokens=prompt_tokens,
        input_device=input_device,
        devices=devices,
    )
    _set_sparse_policy(
        modules,
        budget=args.exact_token_budget,
        page_size=args.page_size,
        force_last_page=False,
    )
    arms[SPARSE_C1_ARM] = {
        **_time_decode(
            model,
            cache,
            first_token,
            tokens=args.decode_tokens,
            prompt_tokens=prompt_tokens,
            input_device=input_device,
            devices=devices,
        ),
        "key_cache": "exact_bfloat16_gpu_resident",
        "value_cache": "c1_bfloat16_rank96",
        "attention": "r32_route_pack_exact_k_then_cuda_sparse_c1_v96",
        "cached_incremental_sidecar": True,
        "runtime_last_decode": _cuda_sparse_last_statistics(modules),
    }

    dense = arms[DENSE_BF16_ARM]
    dense_c1 = arms[DENSE_C1_ARM]
    sparse = arms[SPARSE_C1_ARM]
    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "metadata": {
            "model_path": str(model_path),
            "c1_export": str(c1_export),
            "routing_factors": str(factor_dir),
            "data_dir": str(data_dir),
            "task": task.name,
            "sample_ordinal": ordinal,
            "sequence_length": args.sequence_length,
            "prompt_tokens": prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "timed_decode_forward_steps": timed_decode_forward_steps,
            "timed_warmup_forward_steps": max(args.warmup_tokens - 1, 0),
            "warmup_tokens": args.warmup_tokens,
            "routing_rank": args.routing_rank,
            "page_size": args.page_size,
            "exact_token_budget": args.exact_token_budget,
            "prefill_chunk_size": args.prefill_chunk_size,
            "pipeline_devices": args.pipeline_devices,
            "device_map": device_map,
            "devices": [str(device) for device in devices],
            "cuda_device_names": {
                str(device): torch.cuda.get_device_name(device)
                for device in devices
            },
            "rope_scaling": rope_scaling,
            "dtype": args.dtype,
            "routing_proxy_implementation": ROUTING_PROXY_IMPLEMENTATION,
            "exact_k_gpu_resident": True,
            "physical_pcie_offload": False,
            "value_rank": 96,
            "memory_after_dense_model_release_bytes": memory_after_dense_release,
        },
        "prefill": prefills,
        "arms": arms,
        "comparison": {
            "sparse_speedup_vs_dense_bf16": (
                dense["elapsed_seconds"] / sparse["elapsed_seconds"]
            ),
            "sparse_time_reduction_vs_dense_bf16_fraction": (
                1.0 - sparse["elapsed_seconds"] / dense["elapsed_seconds"]
            ),
            "sparse_speedup_vs_dense_c1": (
                dense_c1["elapsed_seconds"] / sparse["elapsed_seconds"]
            ),
            "dense_c1_speedup_vs_dense_bf16": (
                dense["elapsed_seconds"] / dense_c1["elapsed_seconds"]
            ),
            "dense_and_c1_prefill_first_token_equal": (
                dense_first_token == first_token
            ),
            "dense_bf16_vs_sparse_generated_token_ids_equal": (
                dense["generated_token_ids"] == sparse["generated_token_ids"]
            ),
            "dense_c1_vs_sparse_generated_token_ids_equal": (
                dense_c1["generated_token_ids"] == sparse["generated_token_ids"]
            ),
        },
    }
    _atomic_json(output_dir / "result.json", payload)
    lines = [
        f"# Qwen3-8B dense versus C1 sparse {args.sequence_length // 1024}K decode",
        "",
        f"Prompt: `{prompt_tokens}` tokens; generated output: `{args.decode_tokens}` "
        f"tokens (`{timed_decode_forward_steps}` timed decode forwards); "
        f"pipeline devices: `{args.pipeline_devices}`.",
        "",
        "| Arm | Seconds | ms/token | tokens/s |",
        "|:---|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = arms[arm]
        lines.append(
            f"| {arm} | {row['elapsed_seconds']:.3f} | "
            f"{row['milliseconds_per_token']:.3f} | "
            f"{row['tokens_per_second']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Sparse speedup versus dense BF16 full-K/V: "
            f"`{payload['comparison']['sparse_speedup_vs_dense_bf16']:.3f}x`.",
            "Sparse speedup versus dense exact-K/C1-V96: "
            f"`{payload['comparison']['sparse_speedup_vs_dense_c1']:.3f}x`.",
            "Dense C1-V96 speedup versus dense BF16 full-K/V: "
            f"`{payload['comparison']['dense_c1_speedup_vs_dense_bf16']:.3f}x`.",
            "",
            f"Dense BF16 prefill: `{dense_prefill_seconds:.3f}` seconds; "
            f"C1 dense prefill: `{c1_prefill_seconds:.3f}` seconds; V128-to-V96 "
            f"cache projection: `{value_cache_projection_seconds:.3f}` seconds; "
            f"resident R32 sidecar: "
            f"`{_sidecar_bytes(cache) / (1 << 30):.3f} GiB`.",
            "",
            "The sparse arm includes R32 selection, resident exact-K page packing, "
            "and CUDA sparse C1-V96 attention. Exact K remains in GPU HBM, so this "
            "does not include physical PCIe fetches or any AllGather collective.",
        ]
    )
    _atomic_text(output_dir / "summary.md", "\n".join(lines) + "\n")
    print(f"[dense versus sparse decode] result={output_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
