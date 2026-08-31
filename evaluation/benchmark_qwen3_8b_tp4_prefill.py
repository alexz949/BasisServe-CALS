#!/usr/bin/env python3
"""Measure Qwen3-8B TP4 prefill-heavy inference on four GPUs.

Each configuration performs one full causal prompt prefill and then generates
``output_tokens`` tokens greedily. The prompt forward produces the first output
token, so the decode phase contains ``output_tokens - 1`` one-token forwards.
Timing includes the transformer, TP collectives, sharded LM head, and
distributed greedy selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from basisserve.core.qwen3_8b_tp4_decode import (  # noqa: E402
    HIDDEN_SIZE,
    Qwen3TP4C1DecodeAttention,
    Qwen3TP4DenseDecodeAttention,
    TP_SIZE,
    close_qwen3_tp4_packed_communicator,
    configure_qwen3_tp4_caches,
    install_qwen3_tp4_decode_attention,
)


FORMAT = "basisserve.qwen3_8b.tp4_prefill_heavy_benchmark.v3"
ARMS = ("dense", "c1_uniform_r64", "c1_mean_dp")


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"expected positive comma-separated integers, got {value!r}")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"duplicate values are not allowed, got {parsed}")
    return parsed


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered:
        raise ValueError("cannot summarize empty timings")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _timing_summary(values: Sequence[float]) -> dict[str, float]:
    selected = tuple(map(float, values))
    if not selected:
        raise ValueError("cannot summarize empty timings")
    return {
        "mean_ms": statistics.fmean(selected),
        "minimum_ms": min(selected),
        "p50_ms": statistics.median(selected),
        "p95_ms": _quantile(selected, 0.95),
        "maximum_ms": max(selected),
    }


def _global_max_float(value: float, *, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _global_max_int(value: int, *, device: torch.device) -> int:
    tensor = torch.tensor(int(value), dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def _distributed_greedy(local_logits: Tensor) -> Tensor:
    if local_logits.ndim != 2:
        raise ValueError("local logits must be [batch, local_vocab]")
    world_size = dist.get_world_size()
    process_rank = dist.get_rank()
    batch, local_vocab = map(int, local_logits.shape)
    values, local_ids = torch.max(local_logits.float(), dim=-1)
    global_ids = local_ids.to(torch.int64) + process_rank * local_vocab
    gathered_values = torch.empty(
        world_size * batch,
        device=values.device,
        dtype=values.dtype,
    )
    gathered_ids = torch.empty(
        world_size * batch,
        device=values.device,
        dtype=torch.int64,
    )
    dist.all_gather_into_tensor(gathered_values, values.contiguous())
    dist.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
    candidates = gathered_values.view(world_size, batch)
    winning_rank = torch.argmax(candidates, dim=0, keepdim=True)
    return torch.gather(
        gathered_ids.view(world_size, batch),
        0,
        winning_rank,
    ).squeeze(0)


def _configure_caches(
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
    *,
    batch: int,
    capacity: int,
    prompt_length: int,
) -> int:
    for module in modules:
        module.clear_cache()
    torch.cuda.empty_cache()
    return configure_qwen3_tp4_caches(
        modules,
        batch_size=batch,
        capacity=capacity,
        max_forward_tokens=prompt_length,
    )


def _reset_caches(
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
) -> None:
    for module in modules:
        module.reset_cache()


def _prompt_ids(
    *,
    batch: int,
    prompt_length: int,
    vocab_size: int,
    device: torch.device,
) -> Tensor:
    token_positions = torch.arange(prompt_length, dtype=torch.int64, device=device)
    sequence_offsets = 997 * torch.arange(batch, dtype=torch.int64, device=device)
    return ((sequence_offsets[:, None] + token_positions[None, :]) % (vocab_size - 1)) + 1


def _prefill_step(model: torch.nn.Module, prompt_ids: Tensor) -> Tensor:
    prompt_length = int(prompt_ids.shape[1])
    position_ids = torch.arange(
        prompt_length,
        dtype=torch.int64,
        device=prompt_ids.device,
    ).unsqueeze(0)
    output = model.model(
        input_ids=prompt_ids,
        position_ids=position_ids,
        use_cache=False,
    )
    local_logits = F.linear(output.last_hidden_state[:, -1, :], model.lm_head.weight)
    return _distributed_greedy(local_logits)


def _decode_step(
    model: torch.nn.Module,
    token_ids: Tensor,
    *,
    position: int,
) -> Tensor:
    batch = int(token_ids.shape[0])
    position_ids = torch.full(
        (1, 1),
        int(position),
        dtype=torch.int64,
        device=token_ids.device,
    )
    output = model.model(
        input_ids=token_ids.reshape(batch, 1),
        position_ids=position_ids,
        use_cache=False,
    )
    local_logits = F.linear(output.last_hidden_state[:, -1, :], model.lm_head.weight)
    return _distributed_greedy(local_logits)


def _execute_request(
    model: torch.nn.Module,
    prompt_ids: Tensor,
    *,
    output_tokens: int,
) -> Tensor:
    prompt_length = int(prompt_ids.shape[1])
    token = _prefill_step(model, prompt_ids)
    for decode_index in range(output_tokens - 1):
        token = _decode_step(
            model,
            token,
            position=prompt_length + decode_index,
        )
    return token


@torch.inference_mode()
def _run_configuration(
    model: torch.nn.Module,
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
    *,
    batch: int,
    prompt_length: int,
    output_tokens: int,
    warmup_runs: int,
    repeat_runs: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, Any]:
    decode_steps = output_tokens - 1
    capacity = prompt_length + decode_steps
    cache_bytes = _configure_caches(
        modules,
        batch=batch,
        capacity=capacity,
        prompt_length=prompt_length,
    )
    prompt_ids = _prompt_ids(
        batch=batch,
        prompt_length=prompt_length,
        vocab_size=vocab_size,
        device=device,
    )

    for _ in range(warmup_runs):
        _reset_caches(modules)
        dist.barrier()
        _execute_request(model, prompt_ids, output_tokens=output_tokens)
        torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    prefill_timings: list[float] = []
    decode_timings: list[float] = []
    total_timings: list[float] = []
    token: Tensor | None = None
    for _ in range(repeat_runs):
        _reset_caches(modules)
        dist.barrier()
        total_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
        token = _prefill_step(model, prompt_ids)
        prefill_end.record()
        for decode_index in range(decode_steps):
            token = _decode_step(
                model,
                token,
                position=prompt_length + decode_index,
            )
        decode_end.record()
        decode_end.synchronize()
        prefill_timings.append(
            _global_max_float(total_start.elapsed_time(prefill_end), device=device)
        )
        decode_timings.append(
            _global_max_float(prefill_end.elapsed_time(decode_end), device=device)
        )
        total_timings.append(
            _global_max_float(total_start.elapsed_time(decode_end), device=device)
        )

    if token is None:
        raise AssertionError("benchmark completed no repetitions")
    peak_allocated = _global_max_int(torch.cuda.max_memory_allocated(device), device=device)
    peak_reserved = _global_max_int(torch.cuda.max_memory_reserved(device), device=device)
    cache_bytes = _global_max_int(cache_bytes, device=device)
    final_tokens = torch.empty(TP_SIZE * batch, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(final_tokens, token.contiguous())
    by_rank = final_tokens.view(TP_SIZE, batch)
    if not torch.equal(by_rank, by_rank[0].expand_as(by_rank)):
        raise AssertionError("distributed greedy token differs across TP ranks")

    prefill = _timing_summary(prefill_timings)
    decode = _timing_summary(decode_timings)
    end_to_end = _timing_summary(total_timings)
    prompt_token_count = batch * prompt_length
    decoded_token_count = batch * decode_steps
    return {
        "batch_size": batch,
        "prompt_length": prompt_length,
        "output_tokens": output_tokens,
        "decode_steps": decode_steps,
        "repeat_runs": repeat_runs,
        "time_to_first_token": prefill,
        "decode_after_first_token": decode,
        "end_to_end": end_to_end,
        "prefill_tokens_per_second": prompt_token_count * 1000.0 / prefill["mean_ms"],
        "decode_tokens_per_second": decoded_token_count * 1000.0 / decode["mean_ms"],
        "output_tokens_per_second_end_to_end": (
            batch * output_tokens * 1000.0 / end_to_end["mean_ms"]
        ),
        "processed_tokens_per_second_end_to_end": (
            batch * (prompt_length + output_tokens) * 1000.0
            / end_to_end["mean_ms"]
        ),
        "memory": {
            "static_kv_cache_bytes_per_rank": cache_bytes,
            "peak_allocated_bytes_per_rank": peak_allocated,
            "peak_reserved_bytes_per_rank": peak_reserved,
        },
        "final_token_ids_prefix": by_rank[0, : min(batch, 8)].cpu().tolist(),
    }


def _communication_metadata(
    arm: str,
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
) -> dict[str, Any]:
    dense_element_bytes = modules[0].q_proj.weight.element_size()
    if arm == "dense":
        elements_per_layer = math.ceil(2 * (TP_SIZE - 1) * HIDDEN_SIZE / TP_SIZE)
        return {
            "collective": "Transformers rowwise o_proj AllReduce",
            "theoretical_ring_elements_per_token_per_rank_per_layer": elements_per_layer,
            "theoretical_ring_bytes_per_token_per_rank_all_layers": (
                elements_per_layer * dense_element_bytes * len(modules)
            ),
        }
    c1_modules = tuple(
        module for module in modules if isinstance(module, Qwen3TP4C1DecodeAttention)
    )
    widths = tuple(module.local_wire_width for module in c1_modules)
    sent_elements = (TP_SIZE - 1) * sum(widths)
    wire_element_bytes = (
        1 if c1_modules[0].wire_dtype == "float8_e4m3fn" else dense_element_bytes
    )
    return {
        "collective": (
            "compiled in-place NCCL feature-major packed AllGather"
        ),
        "output_projection": (
            "CUDA local pack into [wire,tokens] receive arena + one "
            + (
                "E4M3-to-BF16 cast + BF16 cuBLAS GEMM"
                if c1_modules[0].wire_dtype == "float8_e4m3fn"
                else "BF16 cuBLAS GEMM"
            )
        ),
        "local_wire_widths_by_layer": list(widths),
        "mean_local_wire_width": statistics.fmean(widths),
        "direct_send_elements_per_token_per_rank_all_layers": sent_elements,
        "wire_dtype": (
            "float8_e4m3fn"
            if wire_element_bytes == 1
            else str(c1_modules[0].q_proj.weight.dtype).removeprefix("torch.")
        ),
        "wire_element_bytes": wire_element_bytes,
        "direct_send_bytes_per_token_per_rank_all_layers": (
            sent_elements * wire_element_bytes
        ),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--factor-dir")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--prompt-lengths", default="128,256,1024,2048,4096")
    parser.add_argument("--max-batch-prompt-tokens", type=int, default=32768)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--repeat-runs", type=int, default=10)
    parser.add_argument("--c1-decode-attention", choices=("cuda", "triton"))
    parser.add_argument(
        "--c1-wire-dtype",
        choices=("bfloat16", "float8_e4m3fn"),
        default="bfloat16",
    )
    parser.add_argument("--c1-fp8-wire-scales")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    args = parser.parse_args()

    if args.arm == "dense" and args.factor_dir is not None:
        raise ValueError("dense arm must not receive --factor-dir")
    if args.arm != "dense" and args.factor_dir is None:
        raise ValueError("C1 arms require --factor-dir")
    if args.arm == "dense" and args.c1_decode_attention is not None:
        raise ValueError("dense arm must not select --c1-decode-attention")
    if args.arm == "dense" and args.c1_wire_dtype != "bfloat16":
        raise ValueError("dense arm must use --c1-wire-dtype bfloat16")
    if args.arm == "dense" and args.c1_fp8_wire_scales is not None:
        raise ValueError("dense arm must not receive --c1-fp8-wire-scales")
    if args.arm != "dense" and args.c1_decode_attention is None:
        raise ValueError("C1 arms require explicit --c1-decode-attention cuda|triton")
    if (
        args.arm != "dense"
        and args.c1_wire_dtype == "float8_e4m3fn"
        and args.c1_fp8_wire_scales is None
    ):
        raise ValueError("FP8 C1 requires --c1-fp8-wire-scales")
    if args.c1_wire_dtype == "bfloat16" and args.c1_fp8_wire_scales is not None:
        raise ValueError("BF16 C1 must not receive --c1-fp8-wire-scales")
    positive_scalars = (
        args.max_batch_prompt_tokens,
        args.output_tokens,
        args.warmup_runs,
        args.repeat_runs,
        args.torch_num_threads,
    )
    if any(value <= 0 for value in positive_scalars):
        raise ValueError("token limits, run counts, and thread count must be positive")
    if args.output_tokens < 2:
        raise ValueError("prefill-heavy protocol requires at least two output tokens")
    batches = _parse_positive_ints(args.batch_sizes)
    prompt_lengths = _parse_positive_ints(args.prompt_lengths)
    configurations = tuple(
        (batch, prompt_length)
        for batch in batches
        for prompt_length in prompt_lengths
        if batch * prompt_length <= args.max_batch_prompt_tokens
    )
    if not configurations:
        raise ValueError("token cap excludes every batch/prompt configuration")
    maximum_position = max(prompt + args.output_tokens - 1 for _, prompt in configurations)
    if maximum_position > 32768:
        raise ValueError("prompt plus output exceeds Qwen3-8B positional capacity")

    torch.set_num_threads(args.torch_num_threads)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from transformers import AutoModelForCausalLM
    from transformers.distributed import DistributedConfig

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        distributed_config=DistributedConfig(tp_size=TP_SIZE),
        local_files_only=True,
    ).eval()
    if not dist.is_initialized() or dist.get_world_size() != TP_SIZE:
        raise RuntimeError(f"benchmark requires TP{TP_SIZE}")
    modules = install_qwen3_tp4_decode_attention(
        model,
        factor_dir=args.factor_dir,
        c1_decode_attention_backend=args.c1_decode_attention,
        c1_wire_dtype=args.c1_wire_dtype,
        c1_fp8_wire_scales=args.c1_fp8_wire_scales,
    )
    torch.cuda.synchronize(device)
    load_seconds = _global_max_float(time.perf_counter() - load_started, device=device)

    local_vocab = int(model.lm_head.weight.shape[0])
    vocab_size = int(model.config.vocab_size)
    if local_vocab * TP_SIZE != vocab_size:
        raise ValueError("LM head is not an equal TP4 vocabulary shard")
    model_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    ) + sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
        if buffer is not None
    )
    model_bytes = _global_max_int(model_bytes, device=device)
    factor_records = [
        {
            "layer": module.layer_idx,
            "rank": module.source_rank,
            "path": module.factor_path,
            "sha256": module.factor_sha256,
        }
        for module in modules
        if isinstance(module, Qwen3TP4C1DecodeAttention)
    ]

    records: list[dict[str, Any]] = []
    for batch, prompt_length in configurations:
        record = _run_configuration(
            model,
            modules,
            batch=batch,
            prompt_length=prompt_length,
            output_tokens=args.output_tokens,
            warmup_runs=args.warmup_runs,
            repeat_runs=args.repeat_runs,
            vocab_size=vocab_size,
            device=device,
        )
        records.append(record)
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "event": "configuration_complete",
                        "arm": args.arm,
                        "batch": batch,
                        "prompt_length": prompt_length,
                        "ttft_ms": record["time_to_first_token"]["mean_ms"],
                        "prefill_tokens_per_second": record["prefill_tokens_per_second"],
                        "decode_tokens_per_second": record["decode_tokens_per_second"],
                        "end_to_end_ms": record["end_to_end"]["mean_ms"],
                        "peak_gib": record["memory"]["peak_allocated_bytes_per_rank"] / 2**30,
                    }
                ),
                flush=True,
            )

    if dist.get_rank() == 0:
        output = Path(args.output_json).expanduser().resolve()
        payload = {
            "format": FORMAT,
            "status": "complete",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "arm": args.arm,
            "model": str(Path(args.model).expanduser().resolve()),
            "factor_dir": (
                None
                if args.factor_dir is None
                else str(Path(args.factor_dir).expanduser().resolve())
            ),
            "protocol": {
                "tp_size": TP_SIZE,
                "dtype": "bfloat16",
                "prompt_lengths": list(prompt_lengths),
                "batch_sizes": list(batches),
                "max_batch_prompt_tokens": args.max_batch_prompt_tokens,
                "evaluated_configurations": [list(item) for item in configurations],
                "output_tokens": args.output_tokens,
                "decode_steps_after_first_token": args.output_tokens - 1,
                "warmup_runs_per_configuration": args.warmup_runs,
                "repeat_runs_per_configuration": args.repeat_runs,
                "timed_scope": (
                    "transformer + TP collectives + sharded LM head + "
                    "distributed greedy selection"
                ),
                "attention": (
                    "dense Flash-SDPA prefill and decode"
                    if args.arm == "dense"
                    else (
                        "custom Triton compressed-V prefill; handwritten CUDA decode "
                        "without Triton fallback"
                        if args.c1_decode_attention == "cuda"
                        else "custom Triton compressed-V prefill and decode"
                    )
                ),
                "c1_decode_attention": args.c1_decode_attention,
                "c1_encoder_compute_dtype": (
                    None if args.arm == "dense" else "bfloat16"
                ),
                "c1_decoder_compute_dtype": (
                    None if args.arm == "dense" else "bfloat16"
                ),
                "c1_wire_dtype": (
                    None if args.arm == "dense" else args.c1_wire_dtype
                ),
                "c1_fp8_wire_scales": (
                    None
                    if args.c1_fp8_wire_scales is None
                    else str(Path(args.c1_fp8_wire_scales).expanduser().resolve())
                ),
                "c1_output_projection": (
                    None
                    if args.arm == "dense"
                    else (
                        "packed E4M3 NCCL wire cast to BF16 and decoded by one "
                        "BF16 cuBLAS GEMM"
                        if args.c1_wire_dtype == "float8_e4m3fn"
                        else "packed feature-major NCCL arena decoded by one cuBLAS GEMM"
                    )
                ),
                "early_stopping": False,
            },
            "environment": {
                "world_size": dist.get_world_size(),
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "model_load_seconds": load_seconds,
                "model_and_factor_bytes_per_rank": model_bytes,
            },
            "communication": _communication_metadata(args.arm, modules),
            "factors": factor_records,
            "records": records,
        }
        _atomic_json(output, payload)
        print(json.dumps({"event": "result_written", "path": str(output)}), flush=True)
    dist.barrier()
    close_qwen3_tp4_packed_communicator(modules)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
