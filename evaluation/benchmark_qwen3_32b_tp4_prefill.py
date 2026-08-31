#!/usr/bin/env python3
"""Benchmark real Qwen3-32B TP4 prefill plus greedy decode on four GPUs."""

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

from basisserve.core.qwen3_32b_tp4_decode import (  # noqa: E402
    HIDDEN_SIZE,
    Qwen3_32BTP4RaggedC1Attention,
    TP_SIZE,
    close_qwen3_32b_tp4_communicator,
    configure_qwen3_32b_tp4_caches,
    install_qwen3_32b_tp4_attention,
)
from basisserve.kernels.feature_ragged_allgather import (  # noqa: E402
    prepare_feature_ragged_extension,
)


FORMAT = "basisserve.qwen3_32b.tp4_prefill_decode_benchmark.v2"


class _SequenceChunkedMLP(torch.nn.Module):
    """Bound prefill intermediates without changing the tokenwise MLP."""

    def __init__(self, inner: torch.nn.Module, chunk_length: int) -> None:
        super().__init__()
        if int(chunk_length) <= 0:
            raise ValueError("MLP chunk length must be positive")
        self.inner = inner
        self.chunk_length = int(chunk_length)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3:
            return self.inner(hidden_states)
        tokens = int(hidden_states.shape[1])
        if tokens <= self.chunk_length:
            return self.inner(hidden_states)
        return torch.cat(
            tuple(
                self.inner(hidden_states[:, start : start + self.chunk_length])
                for start in range(0, tokens, self.chunk_length)
            ),
            dim=1,
        )


class _SequenceChunkedRMSNorm(torch.nn.Module):
    """Apply tokenwise RMSNorm without a full-sequence FP32 temporary."""

    def __init__(self, inner: torch.nn.Module, chunk_length: int) -> None:
        super().__init__()
        if int(chunk_length) <= 0:
            raise ValueError("RMSNorm chunk length must be positive")
        self.inner = inner
        self.chunk_length = int(chunk_length)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3:
            return self.inner(hidden_states)
        tokens = int(hidden_states.shape[1])
        if tokens <= self.chunk_length:
            return self.inner(hidden_states)
        first_stop = min(self.chunk_length, tokens)
        first = self.inner(hidden_states[:, :first_stop])
        if first.shape != hidden_states[:, :first_stop].shape:
            raise ValueError("RMSNorm must preserve the hidden-state shape")
        output = torch.empty(
            hidden_states.shape,
            dtype=first.dtype,
            device=first.device,
        )
        output[:, :first_stop].copy_(first)
        for start in range(first_stop, tokens, self.chunk_length):
            stop = min(start + self.chunk_length, tokens)
            output[:, start:stop].copy_(self.inner(hidden_states[:, start:stop]))
        return output


def _install_sequence_chunked_mlps(
    model: torch.nn.Module,
    chunk_length: int,
) -> None:
    for layer in model.model.layers:
        layer.mlp = _SequenceChunkedMLP(layer.mlp, chunk_length)


def _install_sequence_chunked_rmsnorms(
    model: torch.nn.Module,
    chunk_length: int,
) -> None:
    for layer in model.model.layers:
        layer.input_layernorm = _SequenceChunkedRMSNorm(
            layer.input_layernorm,
            chunk_length,
        )
        layer.post_attention_layernorm = _SequenceChunkedRMSNorm(
            layer.post_attention_layernorm,
            chunk_length,
        )
    model.model.norm = _SequenceChunkedRMSNorm(model.model.norm, chunk_length)


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(map(float, values))
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _summary(values: Sequence[float]) -> dict[str, float]:
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


def _global_max_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _global_max_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor(int(value), dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def _distributed_greedy(local_logits: Tensor) -> Tensor:
    batch, local_vocab = map(int, local_logits.shape)
    values, local_ids = torch.max(local_logits.float(), dim=-1)
    global_ids = local_ids.to(torch.int64) + dist.get_rank() * local_vocab
    gathered_values = torch.empty(
        TP_SIZE * batch, dtype=values.dtype, device=values.device
    )
    gathered_ids = torch.empty(
        TP_SIZE * batch, dtype=torch.int64, device=values.device
    )
    dist.all_gather_into_tensor(gathered_values, values.contiguous())
    dist.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
    winning_rank = torch.argmax(
        gathered_values.view(TP_SIZE, batch), dim=0, keepdim=True
    )
    return torch.gather(
        gathered_ids.view(TP_SIZE, batch), 0, winning_rank
    ).squeeze(0)


def _prompt_ids(
    batch: int,
    prompt_length: int,
    vocab_size: int,
    device: torch.device,
) -> Tensor:
    positions = torch.arange(prompt_length, dtype=torch.int64, device=device)
    offsets = 997 * torch.arange(batch, dtype=torch.int64, device=device)
    return ((offsets[:, None] + positions[None, :]) % (vocab_size - 1)) + 1


def _prefill(model: torch.nn.Module, prompt_ids: Tensor) -> Tensor:
    position_ids = torch.arange(
        int(prompt_ids.shape[1]), dtype=torch.int64, device=prompt_ids.device
    ).unsqueeze(0)
    output = model.model(
        input_ids=prompt_ids,
        position_ids=position_ids,
        use_cache=False,
    )
    return _distributed_greedy(
        F.linear(output.last_hidden_state[:, -1], model.lm_head.weight)
    )


def _decode(
    model: torch.nn.Module,
    token_ids: Tensor,
    position: int | Tensor,
) -> Tensor:
    batch = int(token_ids.shape[0])
    if isinstance(position, Tensor):
        if (
            position.ndim != 0
            or position.dtype != torch.int64
            or position.device != token_ids.device
        ):
            raise ValueError(
                "decode position must be an int64 scalar on the token device"
            )
        position_ids = position.view(1, 1)
    else:
        position_ids = torch.full(
            (1, 1), int(position), dtype=torch.int64, device=token_ids.device
        )
    output = model.model(
        input_ids=token_ids.reshape(batch, 1),
        position_ids=position_ids,
        # The benchmark owns an unpadded static KV cache. Supplying the already
        # resolved mapping prevents Transformers from materializing a trivial
        # [batch, 1, 1, 1] causal mask only while CUDA-stream capture is active.
        attention_mask={"full_attention": None},
        use_cache=False,
    )
    return _distributed_greedy(
        F.linear(output.last_hidden_state[:, -1], model.lm_head.weight)
    )


def _reset_caches(modules: Sequence[torch.nn.Module]) -> None:
    for module in modules:
        module.reset_cache()


def _set_cache_lengths(
    modules: Sequence[torch.nn.Module],
    length: int,
) -> None:
    for module in modules:
        module.set_cache_length(length)


def _configure_graph_decode(
    modules: Sequence[torch.nn.Module],
    position: Tensor,
) -> None:
    for module in modules:
        module.configure_graph_decode(position)


def _execute(
    model: torch.nn.Module,
    modules: Sequence[torch.nn.Module],
    prompt_ids: Tensor,
    output_tokens: int,
) -> Tensor:
    _reset_caches(modules)
    prompt_length = int(prompt_ids.shape[1])
    token = _prefill(model, prompt_ids)
    for step in range(output_tokens - 1):
        token = _decode(model, token, prompt_length + step)
    return token


@torch.inference_mode()
def _benchmark(
    model: torch.nn.Module,
    modules: Sequence[torch.nn.Module],
    *,
    batch_size: int,
    prompt_length: int,
    output_tokens: int,
    warmup_runs: int,
    repeat_runs: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, Any]:
    decode_steps = output_tokens - 1
    capacity = prompt_length + decode_steps
    for module in modules:
        module.clear_cache()
    torch.cuda.empty_cache()
    cache_bytes = configure_qwen3_32b_tp4_caches(
        modules,
        batch_size=batch_size,
        capacity=capacity,
        max_forward_tokens=prompt_length,
    )
    prompt_ids = _prompt_ids(batch_size, prompt_length, vocab_size, device)

    for warmup in range(warmup_runs):
        dist.barrier()
        _execute(model, modules, prompt_ids, output_tokens)
        torch.cuda.synchronize(device)
        if dist.get_rank() == 0:
            print(json.dumps({"event": "warmup_complete", "run": warmup + 1}), flush=True)

    torch.cuda.reset_peak_memory_stats(device)
    prefill_times: list[float] = []
    decode_times: list[float] = []
    total_times: list[float] = []
    final_token: Tensor | None = None
    for repeat in range(repeat_runs):
        _reset_caches(modules)
        dist.barrier()
        total_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
        final_token = _prefill(model, prompt_ids)
        prefill_end.record()
        for step in range(decode_steps):
            final_token = _decode(model, final_token, prompt_length + step)
        decode_end.record()
        decode_end.synchronize()
        prefill_times.append(
            _global_max_float(total_start.elapsed_time(prefill_end), device)
        )
        decode_times.append(
            _global_max_float(prefill_end.elapsed_time(decode_end), device)
        )
        total_times.append(
            _global_max_float(total_start.elapsed_time(decode_end), device)
        )
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "event": "repeat_complete",
                        "run": repeat + 1,
                        "prefill_ms": prefill_times[-1],
                        "decode_ms": decode_times[-1],
                        "end_to_end_ms": total_times[-1],
                    }
                ),
                flush=True,
            )

    assert final_token is not None
    gathered = torch.empty(
        TP_SIZE * batch_size, dtype=torch.int64, device=device
    )
    dist.all_gather_into_tensor(gathered, final_token.contiguous())
    by_rank = gathered.view(TP_SIZE, batch_size)
    if not torch.equal(by_rank, by_rank[0].expand_as(by_rank)):
        raise AssertionError("greedy tokens differ across TP ranks")

    prefill = _summary(prefill_times)
    decode = _summary(decode_times)
    end_to_end = _summary(total_times)
    return {
        "batch_size": batch_size,
        "prompt_length": prompt_length,
        "output_tokens": output_tokens,
        "incremental_decode_steps": decode_steps,
        "time_to_first_token": prefill,
        "decode_after_first_token": decode,
        "end_to_end": end_to_end,
        "prefill_tokens_per_second": (
            batch_size * prompt_length * 1000.0 / prefill["mean_ms"]
        ),
        "incremental_decode_tokens_per_second": (
            batch_size * decode_steps * 1000.0 / decode["mean_ms"]
        ),
        "generated_tokens_per_second_end_to_end": (
            batch_size * output_tokens * 1000.0 / end_to_end["mean_ms"]
        ),
        "processed_tokens_per_second_end_to_end": (
            batch_size * (prompt_length + output_tokens) * 1000.0
            / end_to_end["mean_ms"]
        ),
        "memory": {
            "static_kv_cache_bytes_per_rank": _global_max_int(cache_bytes, device),
            "peak_allocated_bytes_per_rank": _global_max_int(
                torch.cuda.max_memory_allocated(device), device
            ),
            "peak_reserved_bytes_per_rank": _global_max_int(
                torch.cuda.max_memory_reserved(device), device
            ),
        },
        "final_token_ids_prefix": by_rank[0, :8].cpu().tolist(),
    }


@torch.inference_mode()
def _benchmark_cuda_graph_launch_ab(
    model: torch.nn.Module,
    modules: Sequence[torch.nn.Module],
    *,
    batch_size: int,
    prompt_length: int,
    repeats: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Measure one complete fixed-context decode step with and without a graph.

    Both arms reuse the same prompt KV prefix and overwrite its first decode
    slot. This deliberately isolates launch/scheduling overhead; it is not an
    autoregressive multi-token correctness benchmark because a single captured
    graph has a fixed KV-cache address range and RoPE position.
    """

    for module in modules:
        module.clear_cache()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    cache_bytes = configure_qwen3_32b_tp4_caches(
        modules,
        batch_size=batch_size,
        capacity=prompt_length + 1,
        max_forward_tokens=prompt_length,
    )
    prompt_ids = _prompt_ids(batch_size, prompt_length, vocab_size, device)
    _reset_caches(modules)
    dist.barrier()
    initial_token = _prefill(model, prompt_ids)
    torch.cuda.synchronize(device)
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "event": "cuda_graph_prefill_complete",
                    "context_length": prompt_length,
                }
            ),
            flush=True,
        )

    serving_stream = torch.cuda.current_stream(device)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(serving_stream)
    c1_modules = tuple(
        module
        for module in modules
        if isinstance(module, Qwen3_32BTP4RaggedC1Attention)
    )
    with torch.cuda.stream(capture_stream):
        for module in c1_modules:
            wire_dtype = (
                torch.uint8
                if module.wire_dtype == "float8_e4m3fn"
                else module.q_proj.weight.dtype
            )
            module.configure_uniform_allgather(
                tokens=batch_size,
                dtype=wire_dtype,
            )
        static_token = initial_token.clone()

        # Populate lazy library state and allocator blocks on the capture stream.
        for _ in range(2):
            _set_cache_lengths(modules, prompt_length)
            eager_output = _decode(
                model,
                static_token,
                prompt_length,
            )
    capture_stream.synchronize()

    _set_cache_lengths(modules, prompt_length)
    dist.barrier()
    torch.cuda.synchronize(device)
    allocated_before_capture = torch.cuda.memory_allocated(device)
    reserved_before_capture = torch.cuda.memory_reserved(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_output = _decode(
            model,
            static_token,
            prompt_length,
        )
    capture_stream.synchronize()
    allocated_after_capture = torch.cuda.memory_allocated(device)
    reserved_after_capture = torch.cuda.memory_reserved(device)
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "event": "cuda_graph_capture_complete",
                    "allocated_delta_bytes": (
                        allocated_after_capture - allocated_before_capture
                    ),
                    "reserved_delta_bytes": (
                        reserved_after_capture - reserved_before_capture
                    ),
                }
            ),
            flush=True,
        )

    eager_times: list[float] = []
    eager_reference: Tensor | None = None
    dist.barrier()
    for _ in range(repeats):
        _set_cache_lengths(modules, prompt_length)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(capture_stream):
            start.record()
            eager_output = _decode(
                model,
                static_token,
                prompt_length,
            )
            end.record()
        end.synchronize()
        eager_times.append(_global_max_float(start.elapsed_time(end), device))
        if eager_reference is None:
            eager_reference = eager_output.detach().clone()
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "event": "cuda_graph_eager_arm_complete",
                    "mean_ms": statistics.fmean(eager_times),
                }
            ),
            flush=True,
        )

    graph_times: list[float] = []
    dist.barrier()
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(capture_stream):
            start.record()
            graph.replay()
            end.record()
        end.synchronize()
        graph_times.append(_global_max_float(start.elapsed_time(end), device))
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "event": "cuda_graph_replay_arm_complete",
                    "mean_ms": statistics.fmean(graph_times),
                }
            ),
            flush=True,
        )

    assert eager_reference is not None
    if not torch.equal(eager_reference, graph_output):
        raise AssertionError("CUDA Graph and eager greedy tokens differ")
    gathered = torch.empty(
        TP_SIZE * batch_size,
        dtype=torch.int64,
        device=device,
    )
    dist.all_gather_into_tensor(gathered, graph_output.contiguous())
    by_rank = gathered.view(TP_SIZE, batch_size)
    if not torch.equal(by_rank, by_rank[0].expand_as(by_rank)):
        raise AssertionError("CUDA Graph greedy tokens differ across TP ranks")

    eager = _summary(eager_times)
    cuda_graph = _summary(graph_times)
    return {
        "scope": "one complete fixed-context decode step",
        "includes": (
            "64 transformer layers + TP collectives + sharded LM head + "
            "distributed greedy selection"
        ),
        "excludes": "prefill and host-side scheduler",
        "fixed_context_length": prompt_length,
        "batch_size": batch_size,
        "repeats": repeats,
        "eager": eager,
        "cuda_graph": cuda_graph,
        "speedup": eager["mean_ms"] / cuda_graph["mean_ms"],
        "latency_reduction": 1.0 - cuda_graph["mean_ms"] / eager["mean_ms"],
        "graph_memory": {
            "allocated_before_capture_bytes_per_rank": _global_max_int(
                allocated_before_capture, device
            ),
            "allocated_after_capture_bytes_per_rank": _global_max_int(
                allocated_after_capture, device
            ),
            "reserved_before_capture_bytes_per_rank": _global_max_int(
                reserved_before_capture, device
            ),
            "reserved_after_capture_bytes_per_rank": _global_max_int(
                reserved_after_capture, device
            ),
        },
        "memory": {
            "static_kv_cache_bytes_per_rank": _global_max_int(cache_bytes, device),
            "peak_allocated_bytes_per_rank": _global_max_int(
                torch.cuda.max_memory_allocated(device), device
            ),
            "peak_reserved_bytes_per_rank": _global_max_int(
                torch.cuda.max_memory_reserved(device), device
            ),
        },
        "final_token_ids_prefix": by_rank[0, :8].cpu().tolist(),
    }


@torch.inference_mode()
def _benchmark_cuda_graph_full_decode(
    model: torch.nn.Module,
    modules: Sequence[torch.nn.Module],
    *,
    batch_size: int,
    prompt_length: int,
    output_tokens: int,
    warmup_runs: int,
    repeat_runs: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Capture one decode step and replay it as an autoregressive sequence."""

    decode_steps = output_tokens - 1
    capacity = prompt_length + decode_steps
    for module in modules:
        module.clear_cache()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    cache_bytes = configure_qwen3_32b_tp4_caches(
        modules,
        batch_size=batch_size,
        capacity=capacity,
        max_forward_tokens=prompt_length,
    )
    prompt_ids = _prompt_ids(batch_size, prompt_length, vocab_size, device)
    # Static cache allocation can leave several GiB of fragmented reserve on
    # a near-capacity L40S. Release only unused allocator blocks before the
    # 2.5-GiB full-prefill RMSNorm temporary; model, factors, and caches remain.
    torch.cuda.empty_cache()
    serving_stream = torch.cuda.current_stream(device)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(serving_stream)
    c1_modules = tuple(
        module
        for module in modules
        if isinstance(module, Qwen3_32BTP4RaggedC1Attention)
    )
    with torch.cuda.stream(capture_stream):
        for module in c1_modules:
            wire_dtype = (
                torch.uint8
                if module.wire_dtype == "float8_e4m3fn"
                else module.q_proj.weight.dtype
            )
            module.configure_uniform_allgather(
                tokens=batch_size,
                dtype=wire_dtype,
            )
        # First establish a variable-shape eager reference. This runs before
        # graph mode is enabled, so each step attends exactly its valid prefix.
        eager_reference: Tensor | None = None
        for warmup in range(warmup_runs):
            _reset_caches(modules)
            token = _prefill(model, prompt_ids)
            for step in range(decode_steps):
                token = _decode(model, token, prompt_length + step)
            eager_reference = token
            if dist.get_rank() == 0:
                print(
                    json.dumps(
                        {
                            "event": "step_cuda_graph_eager_reference_enqueued",
                            "run": warmup + 1,
                        }
                    ),
                    flush=True,
                )
    capture_stream.synchronize()
    if eager_reference is None:
        raise ValueError("step CUDA Graph capture requires at least one warmup run")
    eager_reference = eager_reference.detach().clone()

    # Time prefill before graph capture. On a nearly full L40S the graph's
    # private pool and the 2.5-GiB prefill RMSNorm temporary cannot coexist.
    # The final timed prefill leaves exactly the prefix cache and first token
    # used by every decode replay below, so no model work is omitted.
    local_prefill_times: list[float] = []
    prefill_times: list[float] = []
    prefill_token: Tensor | None = None
    for repeat in range(repeat_runs):
        _reset_caches(modules)
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(serving_stream)
        prefill_token = _prefill(model, prompt_ids)
        end.record(serving_stream)
        end.synchronize()
        local_ms = float(start.elapsed_time(end))
        local_prefill_times.append(local_ms)
        prefill_times.append(_global_max_float(local_ms, device))
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "event": "step_cuda_graph_prefill_complete",
                        "run": repeat + 1,
                        "context_length": prompt_length,
                        "prefill_ms": prefill_times[-1],
                    }
                ),
                flush=True,
            )
    if prefill_token is None:
        raise AssertionError("prefill timing produced no first token")
    prefill_token = prefill_token.detach().clone()

    with torch.cuda.stream(capture_stream):
        capture_stream.wait_stream(serving_stream)
        static_token = prefill_token.clone()
        static_position = torch.full(
            (), prompt_length, dtype=torch.int64, device=device
        )
        _configure_graph_decode(modules, static_position)

        # Validate the exact fixed-cache/device-position backend that the graph
        # will capture. The native reference above deliberately retains the
        # production eager sliced-cache path as a separate fidelity check.
        token = static_token
        for _ in range(decode_steps):
            token = _decode(model, token, static_position)
            static_position.add_(1)
        graph_backend_reference = token.detach().clone()

        # Populate fixed-shape attention, GEMM, collective, and allocator state
        # once on the capture stream. These writes only touch cache slot P,
        # which the measured decode overwrites on its first replay.
        static_token.copy_(prefill_token)
        static_position.fill_(prompt_length)
        for _ in range(2):
            _decode(model, static_token, static_position)
        static_token.copy_(prefill_token)
        static_position.fill_(prompt_length)
    capture_stream.synchronize()
    dist.barrier()
    torch.cuda.synchronize(device)
    peak_allocated_before_graph = torch.cuda.max_memory_allocated(device)
    peak_reserved_before_graph = torch.cuda.max_memory_reserved(device)
    allocated_before_capture = torch.cuda.memory_allocated(device)
    reserved_before_capture = torch.cuda.memory_reserved(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_output = _decode(model, static_token, static_position)
        static_token.copy_(graph_output)
        static_position.add_(1)
    capture_stream.synchronize()
    allocated_after_capture = torch.cuda.memory_allocated(device)
    reserved_after_capture = torch.cuda.memory_reserved(device)
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "event": "step_cuda_graph_capture_complete",
                    "captured_decode_steps": 1,
                    "sequence_replays": decode_steps,
                    "allocated_delta_bytes": (
                        allocated_after_capture - allocated_before_capture
                    ),
                    "reserved_delta_bytes": (
                        reserved_after_capture - reserved_before_capture
                    ),
                }
            ),
            flush=True,
        )

    torch.cuda.reset_peak_memory_stats(device)
    decode_times: list[float] = []
    total_times: list[float] = []
    final_token: Tensor | None = None
    for repeat in range(repeat_runs):
        dist.barrier()
        decode_start = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(capture_stream):
            decode_start.record()
            static_token.copy_(prefill_token)
            static_position.fill_(prompt_length)
            for _ in range(decode_steps):
                graph.replay()
            decode_end.record()
        decode_end.synchronize()
        final_token = static_token
        local_decode_ms = float(decode_start.elapsed_time(decode_end))
        decode_times.append(_global_max_float(local_decode_ms, device))
        total_times.append(
            _global_max_float(
                local_prefill_times[repeat] + local_decode_ms,
                device,
            )
        )
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "event": "step_cuda_graph_repeat_complete",
                        "run": repeat + 1,
                        "prefill_ms": prefill_times[repeat],
                        "decode_ms": decode_times[-1],
                        "end_to_end_ms": total_times[-1],
                    }
                ),
                flush=True,
            )

    assert final_token is not None
    if not torch.equal(graph_backend_reference, final_token):
        raise AssertionError(
            "step CUDA Graph replay and fixed-cache eager tokens differ"
        )
    native_eager_match_fraction = float(
        (eager_reference == graph_backend_reference).float().mean().item()
    )
    gathered = torch.empty(
        TP_SIZE * batch_size,
        dtype=torch.int64,
        device=device,
    )
    dist.all_gather_into_tensor(gathered, final_token.contiguous())
    by_rank = gathered.view(TP_SIZE, batch_size)
    if not torch.equal(by_rank, by_rank[0].expand_as(by_rank)):
        raise AssertionError("step CUDA Graph tokens differ across TP ranks")

    prefill = _summary(prefill_times)
    decode = _summary(decode_times)
    end_to_end = _summary(total_times)
    return {
        "batch_size": batch_size,
        "prompt_length": prompt_length,
        "output_tokens": output_tokens,
        "incremental_decode_steps": decode_steps,
        "decode_execution": (
            "one captured fixed-shape decode step replayed autoregressively"
        ),
        "end_to_end_measurement": (
            "sum of pre-capture eager prefill and post-capture graph decode; "
            "both use the same retained prompt KV prefix and first token"
        ),
        "correctness": {
            "graph_matches_fixed_cache_eager": True,
            "native_sliced_cache_final_token_match_fraction": (
                native_eager_match_fraction
            ),
            "native_sliced_cache_final_token_ids_prefix": (
                eager_reference[:8].cpu().tolist()
            ),
            "fixed_cache_final_token_ids_prefix": (
                graph_backend_reference[:8].cpu().tolist()
            ),
        },
        "time_to_first_token": prefill,
        "decode_after_first_token": decode,
        "end_to_end": end_to_end,
        "prefill_tokens_per_second": (
            batch_size * prompt_length * 1000.0 / prefill["mean_ms"]
        ),
        "incremental_decode_tokens_per_second": (
            batch_size * decode_steps * 1000.0 / decode["mean_ms"]
        ),
        "generated_tokens_per_second_end_to_end": (
            batch_size * output_tokens * 1000.0 / end_to_end["mean_ms"]
        ),
        "processed_tokens_per_second_end_to_end": (
            batch_size * (prompt_length + output_tokens) * 1000.0
            / end_to_end["mean_ms"]
        ),
        "graph_capture": {
            "captured_decode_steps": 1,
            "sequence_replays": decode_steps,
            "allocated_before_capture_bytes_per_rank": _global_max_int(
                allocated_before_capture, device
            ),
            "allocated_after_capture_bytes_per_rank": _global_max_int(
                allocated_after_capture, device
            ),
            "reserved_before_capture_bytes_per_rank": _global_max_int(
                reserved_before_capture, device
            ),
            "reserved_after_capture_bytes_per_rank": _global_max_int(
                reserved_after_capture, device
            ),
        },
        "memory": {
            "static_kv_cache_bytes_per_rank": _global_max_int(cache_bytes, device),
            "peak_allocated_bytes_per_rank": _global_max_int(
                max(
                    peak_allocated_before_graph,
                    torch.cuda.max_memory_allocated(device),
                ),
                device,
            ),
            "peak_reserved_bytes_per_rank": _global_max_int(
                max(
                    peak_reserved_before_graph,
                    torch.cuda.max_memory_reserved(device),
                ),
                device,
            ),
        },
        "final_token_ids_prefix": by_rank[0, :8].cpu().tolist(),
    }


def _communication(
    arm: str,
    modules: Sequence[torch.nn.Module],
    *,
    batch_size: int,
    prompt_length: int,
    output_tokens: int,
) -> dict[str, Any]:
    attention_rows = batch_size * (prompt_length + output_tokens - 1)
    if arm == "dense":
        element_bytes = 2
        elements_per_token_all_ranks = 2 * (TP_SIZE - 1) * HIDDEN_SIZE * len(modules)
        return {
            "collective": "rowwise o_proj ring AllReduce",
            "wire_dtype": "bfloat16",
            "logical_wire_bytes_all_ranks": (
                elements_per_token_all_ranks * attention_rows * element_bytes
            ),
        }
    c1_modules = tuple(
        module
        for module in modules
        if isinstance(module, Qwen3_32BTP4RaggedC1Attention)
    )
    global_widths = tuple(module.plan.total_width for module in c1_modules)
    process_widths = tuple(module.process_wire_widths for module in c1_modules)
    allgather_backends = {module.allgather_backend for module in c1_modules}
    if len(allgather_backends) != 1:
        raise RuntimeError("C1 layers use inconsistent AllGather backends")
    allgather_backend = next(iter(allgather_backends))
    wire_dtypes = {module.wire_dtype for module in c1_modules}
    if len(wire_dtypes) != 1:
        raise RuntimeError("C1 layers use inconsistent wire dtypes")
    wire_dtype = next(iter(wire_dtypes))
    element_bytes = 1 if wire_dtype == "float8_e4m3fn" else 2
    elements_per_token_all_ranks = (TP_SIZE - 1) * sum(global_widths)
    dense_reference = 2 * (TP_SIZE - 1) * HIDDEN_SIZE * len(c1_modules)
    return {
        "collective": allgather_backend,
        "wire_dtype": wire_dtype,
        "global_wire_widths_by_layer": list(global_widths),
        "process_wire_widths_by_layer": [list(widths) for widths in process_widths],
        "mean_global_wire_width": statistics.fmean(global_widths),
        "logical_wire_bytes_all_ranks": (
            elements_per_token_all_ranks * attention_rows * element_bytes
        ),
        "dense_reference_wire_bytes_all_ranks": (
            dense_reference * attention_rows * 2
        ),
        "reduction_vs_dense": 1.0
        - (elements_per_token_all_ranks * element_bytes) / (dense_reference * 2),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dense", "c1_ragged"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--factor-dir")
    parser.add_argument(
        "--c1-decode-attention",
        choices=("cuda", "triton"),
    )
    parser.add_argument(
        "--c1-wire-dtype",
        choices=("bfloat16", "float8_e4m3fn"),
        default="bfloat16",
    )
    parser.add_argument("--c1-fp8-wire-scales")
    parser.add_argument(
        "--c1-allgather-backend",
        choices=("feature_direct", "uniform_nccl", "uniform_ipc"),
        default="feature_direct",
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
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prompt-length", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat-runs", type=int, default=3)
    graph_mode = parser.add_mutually_exclusive_group()
    graph_mode.add_argument(
        "--cuda-graph-launch-ab",
        action="store_true",
        help=(
            "replace the serving run with a fixed-context eager/CUDA-Graph "
            "single-decode-step launch-overhead comparison"
        ),
    )
    graph_mode.add_argument(
        "--cuda-graph-full-decode",
        action="store_true",
        help=(
            "capture one fixed-shape incremental decode step and replay it "
            "autoregressively after each eager prefill"
        ),
    )
    parser.add_argument("--cuda-graph-repeats", type=int, default=20)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--prefill-mlp-chunk-length", type=int, default=1024)
    parser.add_argument("--prefill-rmsnorm-chunk-length", type=int, default=128)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    if args.arm == "dense" and args.factor_dir is not None:
        raise ValueError("dense arm must not receive --factor-dir")
    if args.arm == "c1_ragged" and args.factor_dir is None:
        raise ValueError("C1 arm requires --factor-dir")
    if args.arm == "dense" and args.c1_decode_attention is not None:
        raise ValueError("dense arm must not receive --c1-decode-attention")
    if args.arm == "c1_ragged" and args.c1_decode_attention is None:
        raise ValueError("C1 arm requires --c1-decode-attention")
    if args.arm == "dense" and args.c1_wire_dtype != "bfloat16":
        raise ValueError("dense arm must use a BF16 wire")
    if args.arm == "dense" and args.c1_fp8_wire_scales is not None:
        raise ValueError("dense arm must not receive FP8 wire scales")
    if (
        args.arm == "c1_ragged"
        and args.c1_wire_dtype == "float8_e4m3fn"
        and args.c1_fp8_wire_scales is None
    ):
        raise ValueError("FP8 C1 requires --c1-fp8-wire-scales")
    if args.c1_wire_dtype == "bfloat16" and args.c1_fp8_wire_scales is not None:
        raise ValueError("BF16 C1 must not receive FP8 wire scales")
    if args.arm == "dense" and args.c1_allgather_backend != "feature_direct":
        raise ValueError("dense arm must not select a C1 AllGather backend")
    if args.c1_allgather_backend != "uniform_ipc" and args.c1_ipc_algorithm != "auto":
        raise ValueError("explicit IPC algorithms require uniform_ipc")
    if args.c1_allgather_backend != "uniform_ipc" and args.c1_ipc_channels != 0:
        raise ValueError("explicit IPC channels require uniform_ipc")
    if args.c1_ipc_channels > 1 and args.c1_ipc_algorithm not in ("fanout", "ring"):
        raise ValueError("multiple IPC channels require fanout or ring")
    if args.c1_allgather_backend == "uniform_ipc" and args.prompt_length != 1:
        raise ValueError(
            "uniform_ipc v1 is decode-only; use uniform_nccl for prefill+decode"
        )
    if min(
        args.batch_size,
        args.prompt_length,
        args.output_tokens,
        args.repeat_runs,
        args.cuda_graph_repeats,
        args.torch_num_threads,
        args.prefill_mlp_chunk_length,
        args.prefill_rmsnorm_chunk_length,
    ) <= 0 or args.warmup_runs < 0:
        raise ValueError("benchmark dimensions and run counts are invalid")
    if args.output_tokens < 2:
        raise ValueError("output token count must include at least one decode step")
    if args.cuda_graph_full_decode and args.warmup_runs < 1:
        raise ValueError("full CUDA Graph capture requires at least one warmup run")
    if (
        args.cuda_graph_full_decode
        and args.arm == "c1_ragged"
        and args.c1_decode_attention != "triton"
    ):
        raise ValueError(
            "step-replay CUDA Graph C1 requires Triton attention so the valid "
            "cache prefix can be read from device state"
        )
    if args.prompt_length + args.output_tokens - 1 > 32768:
        raise ValueError("request exceeds Qwen3 positional capacity")

    torch.set_num_threads(args.torch_num_threads)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if args.arm == "c1_ragged":
        # Loading the extension before the 32B shard avoids concurrent dlopen
        # under near-capacity device memory. A cross-process file lock in the
        # loader serializes first use on this single-node TP job.
        prepare_feature_ragged_extension()

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
        raise RuntimeError("benchmark requires physical TP4")
    _install_sequence_chunked_mlps(model, args.prefill_mlp_chunk_length)
    _install_sequence_chunked_rmsnorms(model, args.prefill_rmsnorm_chunk_length)
    modules = install_qwen3_32b_tp4_attention(
        model,
        factor_dir=args.factor_dir,
        c1_decode_attention_backend=args.c1_decode_attention,
        c1_wire_dtype=args.c1_wire_dtype,
        c1_fp8_wire_scales=args.c1_fp8_wire_scales,
        c1_allgather_backend=args.c1_allgather_backend,
        c1_ipc_algorithm=args.c1_ipc_algorithm,
        c1_ipc_channels=args.c1_ipc_channels,
    )
    torch.cuda.synchronize(device)
    load_seconds = _global_max_float(time.perf_counter() - load_started, device)

    local_vocab = int(model.lm_head.weight.shape[0])
    vocab_size = int(model.config.vocab_size)
    if local_vocab * TP_SIZE != vocab_size:
        raise ValueError("LM head is not an equal TP4 vocabulary shard")
    model_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    ) + sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
        if buffer is not None
    )
    model_bytes = _global_max_int(model_bytes, device)

    if args.cuda_graph_launch_ab:
        record = _benchmark_cuda_graph_launch_ab(
            model,
            modules,
            batch_size=args.batch_size,
            prompt_length=args.prompt_length,
            repeats=args.cuda_graph_repeats,
            vocab_size=vocab_size,
            device=device,
        )
    elif args.cuda_graph_full_decode:
        record = _benchmark_cuda_graph_full_decode(
            model,
            modules,
            batch_size=args.batch_size,
            prompt_length=args.prompt_length,
            output_tokens=args.output_tokens,
            warmup_runs=args.warmup_runs,
            repeat_runs=args.repeat_runs,
            vocab_size=vocab_size,
            device=device,
        )
    else:
        record = _benchmark(
            model,
            modules,
            batch_size=args.batch_size,
            prompt_length=args.prompt_length,
            output_tokens=args.output_tokens,
            warmup_runs=args.warmup_runs,
            repeat_runs=args.repeat_runs,
            vocab_size=vocab_size,
            device=device,
        )
    if dist.get_rank() == 0:
        communication = _communication(
            args.arm,
            modules,
            batch_size=args.batch_size,
            prompt_length=(0 if args.cuda_graph_launch_ab else args.prompt_length),
            output_tokens=(2 if args.cuda_graph_launch_ab else args.output_tokens),
        )
        if args.cuda_graph_launch_ab:
            communication["scope"] = "one fixed-context decode step"
        factors = [
            {
                "layer": module.layer_idx,
                "source_ranks": list(module.source_ranks),
                "path": module.factor_path,
                "sha256": module.factor_sha256,
            }
            for module in modules
            if isinstance(module, Qwen3_32BTP4RaggedC1Attention)
        ]
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
                "batch_size": args.batch_size,
                "prompt_length": args.prompt_length,
                "output_tokens": (
                    None if args.cuda_graph_launch_ab else args.output_tokens
                ),
                "incremental_decode_steps": (
                    1 if args.cuda_graph_launch_ab else args.output_tokens - 1
                ),
                "warmup_runs": (
                    None if args.cuda_graph_launch_ab else args.warmup_runs
                ),
                "repeat_runs": (
                    None if args.cuda_graph_launch_ab else args.repeat_runs
                ),
                "benchmark_kind": (
                    "cuda_graph_fixed_context_launch_ab"
                    if args.cuda_graph_launch_ab
                    else (
                        "cuda_graph_step_replay_decode"
                        if args.cuda_graph_full_decode
                        else "serving"
                    )
                ),
                "cuda_graph_repeats": (
                    args.cuda_graph_repeats if args.cuda_graph_launch_ab else None
                ),
                "prefill_mlp_chunk_length": args.prefill_mlp_chunk_length,
                "prefill_rmsnorm_chunk_length": args.prefill_rmsnorm_chunk_length,
                "timed_scope": (
                    (
                        "one fixed-context decode step: transformer + TP "
                        "collectives + sharded LM head + distributed greedy selection"
                    )
                    if args.cuda_graph_launch_ab
                    else (
                        (
                            "eager prefill + one captured decode step replayed "
                            "for the complete autoregressive sequence"
                        )
                        if args.cuda_graph_full_decode
                        else (
                            "transformer + TP collectives + sharded LM head + "
                            "distributed greedy selection"
                        )
                    )
                ),
                "c1_attention": (
                    None
                    if args.arm == "dense"
                    else (
                        "fused two-local-source Triton prefill + "
                        f"{args.c1_decode_attention} decode"
                    )
                ),
                "c1_allgather_backend": (
                    None if args.arm == "dense" else args.c1_allgather_backend
                ),
                "c1_wire_dtype": (
                    None if args.arm == "dense" else args.c1_wire_dtype
                ),
                "c1_fp8_wire_scales": (
                    None
                    if args.c1_fp8_wire_scales is None
                    else str(Path(args.c1_fp8_wire_scales).expanduser().resolve())
                ),
                "c1_ipc_algorithm": (
                    None if args.arm == "dense" else args.c1_ipc_algorithm
                ),
                "c1_ipc_channels": (
                    None if args.arm == "dense" else args.c1_ipc_channels
                ),
            },
            "environment": {
                "world_size": dist.get_world_size(),
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "model_load_seconds": load_seconds,
                "model_and_factor_bytes_per_rank": model_bytes,
            },
            "communication": communication,
            "factors": factors,
            "record": record,
        }
        output = Path(args.output_json).expanduser().resolve()
        _write_json(output, payload)
        print(json.dumps({"event": "result_written", "path": str(output)}), flush=True)
    dist.barrier()
    close_qwen3_32b_tp4_communicator(modules)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
