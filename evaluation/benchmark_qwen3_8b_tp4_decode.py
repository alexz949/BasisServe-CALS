#!/usr/bin/env python3
"""Measure full Qwen3-8B TP4 autoregressive decode on four GPUs.

Each configuration starts with one prompt token, excludes that prompt forward
from timing, then executes exactly ``decode_length`` one-token steps.  The
timed path includes the transformer, TP collectives, the sharded LM head, and
distributed greedy selection. Dense uses static-cache SDPA. C1 uses a packed
feature-major AllGather followed by one decoder GEMM. In FP8 mode the
collective moves raw E4M3 bytes through NCCL ``uint8``; the gathered codes are
cast to BF16 immediately before the BF16 decoder GEMM.
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
FORMAT = "basisserve.qwen3_8b.tp4_autoregressive_decode_benchmark.v4"
ARMS = ("dense", "c1_uniform_r64", "c1_mean_dp")
SEGMENT_STOPS = (128, 256, 1024, 2048, 4096)


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"expected positive comma-separated integers, got {value!r}")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"duplicate values are not allowed, got {parsed}")
    return parsed


def _parse_configurations(value: str) -> tuple[tuple[int, int], ...]:
    configurations: list[tuple[int, int]] = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        fields = item.split("x")
        if len(fields) != 2:
            raise ValueError(f"invalid batch×decode configuration {raw_item!r}")
        configuration = (int(fields[0]), int(fields[1]))
        if any(field <= 0 for field in configuration):
            raise ValueError("batch and decode length must be positive")
        configurations.append(configuration)
    parsed = tuple(configurations)
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError("configurations must be nonempty and unique")
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


def _segments(step_timings_ms: Sequence[float]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    start = 1
    total = len(step_timings_ms)
    for configured_stop in SEGMENT_STOPS:
        stop = min(total, configured_stop)
        if stop < start:
            continue
        values = step_timings_ms[start - 1 : stop]
        output.append(
            {
                "decode_step_start": start,
                "decode_step_stop": stop,
                "context_length_start": start + 1,
                "context_length_stop": stop + 1,
                "steps": len(values),
                **_timing_summary(values),
            }
        )
        start = stop + 1
        if start > total:
            break
    return output


def _configure_caches(
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
    *,
    batch: int,
    capacity: int,
) -> int:
    for module in modules:
        module.clear_cache()
    torch.cuda.empty_cache()
    return configure_qwen3_tp4_caches(
        modules,
        batch_size=batch,
        capacity=capacity,
    )


def _reset_caches(
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
) -> None:
    for module in modules:
        module.reset_cache()


def _distributed_greedy(local_logits: Tensor) -> Tensor:
    """Select global-vocabulary argmax without gathering full logits."""

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
    return torch.gather(gathered_ids.view(world_size, batch), 0, winning_rank).squeeze(0)


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
    hidden = output.last_hidden_state[:, -1, :]
    local_logits = F.linear(hidden, model.lm_head.weight)
    return _distributed_greedy(local_logits)


def _global_max_float(value: float, *, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _global_max_int(value: int, *, device: torch.device) -> int:
    tensor = torch.tensor(int(value), dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


@torch.inference_mode()
def _run_configuration(
    model: torch.nn.Module,
    modules: Sequence[Qwen3TP4DenseDecodeAttention | Qwen3TP4C1DecodeAttention],
    *,
    batch: int,
    decode_length: int,
    prompt_token_id: int,
    warmup_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    capacity = decode_length + 1
    cache_bytes = _configure_caches(
        modules,
        batch=batch,
        capacity=capacity,
    )
    configured_allocated = _global_max_int(
        torch.cuda.memory_allocated(device),
        device=device,
    )
    configured_reserved = _global_max_int(
        torch.cuda.memory_reserved(device),
        device=device,
    )
    torch.cuda.reset_peak_memory_stats(device)

    warmup_steps = min(int(warmup_tokens), decode_length)
    warm_token = torch.full(
        (batch,),
        int(prompt_token_id),
        dtype=torch.int64,
        device=device,
    )
    warm_token = _decode_step(model, warm_token, position=0)
    for step in range(warmup_steps):
        warm_token = _decode_step(model, warm_token, position=step + 1)
    torch.cuda.synchronize(device)
    post_warmup_allocated = _global_max_int(
        torch.cuda.memory_allocated(device),
        device=device,
    )
    post_warmup_reserved = _global_max_int(
        torch.cuda.memory_reserved(device),
        device=device,
    )
    _reset_caches(modules)

    token = torch.full(
        (batch,),
        int(prompt_token_id),
        dtype=torch.int64,
        device=device,
    )
    token = _decode_step(model, token, position=0)
    torch.cuda.synchronize(device)
    dist.barrier()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(decode_length)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(decode_length)]
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    for step, (start, end) in enumerate(zip(starts, ends, strict=True), start=1):
        start.record()
        token = _decode_step(model, token, position=step)
        end.record()
    total_end.record()
    total_end.synchronize()

    local_step_ms = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(local_step_ms, op=dist.ReduceOp.MAX)
    step_ms = local_step_ms.cpu().tolist()
    total_ms = _global_max_float(total_start.elapsed_time(total_end), device=device)
    peak_allocated = _global_max_int(torch.cuda.max_memory_allocated(device), device=device)
    peak_reserved = _global_max_int(torch.cuda.max_memory_reserved(device), device=device)
    post_timing_allocated = _global_max_int(
        torch.cuda.memory_allocated(device),
        device=device,
    )
    post_timing_reserved = _global_max_int(
        torch.cuda.memory_reserved(device),
        device=device,
    )
    cache_bytes = _global_max_int(cache_bytes, device=device)

    final_tokens = torch.empty(
        TP_SIZE * batch,
        dtype=torch.int64,
        device=device,
    )
    dist.all_gather_into_tensor(final_tokens, token.contiguous())
    by_rank = final_tokens.view(TP_SIZE, batch)
    if not torch.equal(by_rank, by_rank[0].expand_as(by_rank)):
        raise AssertionError("distributed greedy token differs across TP ranks")
    generated_tokens = batch * decode_length
    return {
        "batch_size": batch,
        "prompt_tokens": 1,
        "decode_length": decode_length,
        "final_context_length": decode_length + 1,
        "generated_tokens": generated_tokens,
        "critical_path_total_ms": total_ms,
        "sequences_per_second": batch * 1000.0 / total_ms,
        "generated_tokens_per_second": generated_tokens * 1000.0 / total_ms,
        "mean_critical_step_ms_from_total": total_ms / decode_length,
        "step_latency": _timing_summary(step_ms),
        "segments": _segments(step_ms),
        "memory": {
            "static_kv_cache_bytes_per_rank": cache_bytes,
            "configured_allocated_bytes_per_rank": configured_allocated,
            "configured_reserved_bytes_per_rank": configured_reserved,
            "post_warmup_allocated_bytes_per_rank": post_warmup_allocated,
            "post_warmup_reserved_bytes_per_rank": post_warmup_reserved,
            "post_timing_allocated_bytes_per_rank": post_timing_allocated,
            "post_timing_reserved_bytes_per_rank": post_timing_reserved,
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
        elements_per_layer = math.ceil(
            2 * (TP_SIZE - 1) * HIDDEN_SIZE / TP_SIZE
        )
        return {
            "collective": "Transformers rowwise o_proj AllReduce",
            "theoretical_ring_elements_per_sequence_step_per_rank_per_layer": elements_per_layer,
            "theoretical_ring_bytes_per_sequence_step_per_rank_all_layers": (
                elements_per_layer * dense_element_bytes * len(modules)
            ),
        }
    c1_modules = tuple(module for module in modules if isinstance(module, Qwen3TP4C1DecodeAttention))
    widths = tuple(module.local_wire_width for module in c1_modules)
    sent_elements = (TP_SIZE - 1) * sum(widths)
    wire_element_bytes = (
        1 if c1_modules[0].wire_dtype == "float8_e4m3fn" else dense_element_bytes
    )
    direct_attention_output = all(
        module.decode_attention_backend == "cuda"
        and module.wire_dtype == "bfloat16"
        for module in c1_modules
    )
    allgather_backend = c1_modules[0].allgather_backend
    collective = {
        "feature_direct": "dynamic in-place NCCL AllGather into a feature-major arena",
        "uniform_nccl": "prepared in-place NCCL AllGather into a feature-major arena",
        "uniform_ipc": "experimental fixed-TP CUDA-IPC AllGather into a feature-major arena",
    }[allgather_backend]
    return {
        "collective": collective,
        "allgather_backend": allgather_backend,
        "ipc_algorithm": c1_modules[0].ipc_algorithm,
        "ipc_channels": c1_modules[0].ipc_channels,
        "local_pack": (
            "none: CUDA attention writes directly into the local collective slot"
            if direct_attention_output
            else "CUDA tiled token-major to feature-major transpose"
        ),
        "output_projection": (
            "E4M3-to-BF16 arena cast followed by one BF16 cuBLAS GEMM"
            if c1_modules[0].wire_dtype == "float8_e4m3fn"
            else "single BF16 cuBLAS GEMM from the feature-major arena"
        ),
        "local_wire_widths_by_layer": list(widths),
        "mean_local_wire_width": statistics.fmean(widths),
        "direct_send_elements_per_sequence_step_per_rank_all_layers": sent_elements,
        "wire_dtype": (
            "float8_e4m3fn"
            if wire_element_bytes == 1
            else str(c1_modules[0].q_proj.weight.dtype).removeprefix("torch.")
        ),
        "wire_element_bytes": wire_element_bytes,
        "direct_send_bytes_per_sequence_step_per_rank_all_layers": (
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
    parser.add_argument("--decode-lengths", default="128,256,1024,2048,4096")
    parser.add_argument(
        "--configurations",
        help="explicit comma-separated batch×decode pairs such as 1x8192,32x8192",
    )
    parser.add_argument("--c1-decode-attention", choices=("cuda", "triton"))
    parser.add_argument(
        "--c1-wire-dtype",
        choices=("bfloat16", "float8_e4m3fn"),
        default="bfloat16",
    )
    parser.add_argument("--c1-fp8-wire-scales")
    parser.add_argument(
        "--c1-allgather-backend",
        choices=("feature_direct", "uniform_nccl", "uniform_ipc"),
        help="C1 default is uniform_nccl; dense must omit this option",
    )
    parser.add_argument(
        "--c1-ipc-algorithm",
        choices=("auto", "fanout", "fanout_warp", "recursive_doubling", "ring"),
        default="auto",
    )
    parser.add_argument(
        "--c1-ipc-channels",
        type=int,
        choices=(0, 1, 2, 4, 8),
        default=0,
        help="0 selects algorithm-specific defaults; >1 requires fanout or ring",
    )
    parser.add_argument("--prompt-token-id", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=8)
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
    if args.arm == "dense" and args.c1_allgather_backend is not None:
        raise ValueError("dense arm must not receive --c1-allgather-backend")
    if args.arm == "dense" and args.c1_ipc_algorithm != "auto":
        raise ValueError("dense arm must not receive --c1-ipc-algorithm")
    if args.arm == "dense" and args.c1_ipc_channels != 0:
        raise ValueError("dense arm must not receive --c1-ipc-channels")
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
    if args.arm != "dense" and args.c1_allgather_backend is None:
        args.c1_allgather_backend = "uniform_nccl"
    if (
        args.arm != "dense"
        and args.c1_allgather_backend != "uniform_ipc"
        and args.c1_ipc_algorithm != "auto"
    ):
        raise ValueError("explicit IPC algorithms require --c1-allgather-backend uniform_ipc")
    if (
        args.arm != "dense"
        and args.c1_allgather_backend != "uniform_ipc"
        and args.c1_ipc_channels != 0
    ):
        raise ValueError("explicit IPC channels require --c1-allgather-backend uniform_ipc")
    if args.c1_ipc_channels > 1 and args.c1_ipc_algorithm not in ("fanout", "ring"):
        raise ValueError("multiple IPC channels require --c1-ipc-algorithm fanout|ring")
    if args.warmup_tokens <= 0 or args.torch_num_threads <= 0:
        raise ValueError("warmup and torch thread counts must be positive")
    batches = _parse_positive_ints(args.batch_sizes)
    decode_lengths = _parse_positive_ints(args.decode_lengths)
    configurations = (
        _parse_configurations(args.configurations)
        if args.configurations is not None
        else tuple(
            (batch, decode_length)
            for batch in batches
            for decode_length in decode_lengths
        )
    )
    if max(decode_length for _, decode_length in configurations) + 1 > 32768:
        raise ValueError("decode length exceeds Qwen3-8B positional capacity")
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
        c1_allgather_backend=(
            "feature_direct"
            if args.arm == "dense"
            else args.c1_allgather_backend
        ),
        c1_ipc_algorithm=args.c1_ipc_algorithm,
        c1_ipc_channels=args.c1_ipc_channels,
    )
    torch.cuda.synchronize(device)
    load_seconds = _global_max_float(time.perf_counter() - load_started, device=device)

    local_vocab = int(model.lm_head.weight.shape[0])
    if local_vocab * TP_SIZE != int(model.config.vocab_size):
        raise ValueError("LM head is not an equal TP4 vocabulary shard")
    model_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
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
    for batch, decode_length in configurations:
        record = _run_configuration(
            model,
            modules,
            batch=batch,
            decode_length=decode_length,
            prompt_token_id=args.prompt_token_id,
            warmup_tokens=args.warmup_tokens,
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
                        "decode_length": decode_length,
                        "tokens_per_second": record["generated_tokens_per_second"],
                        "mean_step_ms": record["mean_critical_step_ms_from_total"],
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
                "attention": (
                    "dense torch SDPA"
                    if args.arm == "dense"
                    else (
                        "C1 handwritten architecture-specific CUDA decode without "
                        "Triton fallback"
                        if args.c1_decode_attention == "cuda"
                        else "C1 Triton decode without handwritten-CUDA fallback"
                    )
                ),
                "prompt_tokens": 1,
                "prompt_token_id": args.prompt_token_id,
                "decode_lengths": sorted(
                    {decode_length for _, decode_length in configurations}
                ),
                "batch_sizes": sorted({batch for batch, _ in configurations}),
                "configurations": [
                    {"batch_size": batch, "decode_length": decode_length}
                    for batch, decode_length in configurations
                ],
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
                "c1_allgather_backend": (
                    None if args.arm == "dense" else args.c1_allgather_backend
                ),
                "c1_ipc_algorithm": (
                    None if args.arm == "dense" else args.c1_ipc_algorithm
                ),
                "c1_ipc_channels": (
                    None if args.arm == "dense" else args.c1_ipc_channels
                ),
                "c1_fp8_wire_scales": (
                    None
                    if args.c1_fp8_wire_scales is None
                    else str(Path(args.c1_fp8_wire_scales).expanduser().resolve())
                ),
                "warmup_tokens_per_configuration": args.warmup_tokens,
                "timed_scope": (
                    "transformer + TP collectives + sharded LM head + "
                    "distributed greedy selection"
                ),
                "prompt_forward_timed": False,
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
