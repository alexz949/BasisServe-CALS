#!/usr/bin/env python3
"""Benchmark DeepSeek-V2-Lite dense and C1 TP4 prefill plus decode.

The timed scope is the complete transformer, all tensor-parallel collectives,
the sharded LM head, and distributed greedy token selection.  C1 changes only
the post-attention output collective: MLA latent KV and the KV cache are dense
and identical between arms.
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
from transformers.cache_utils import Cache, CacheLayerMixin  # noqa: E402

from basisserve.core.deepseek_v2_lite_tp4_c1 import (  # noqa: E402
    HIDDEN_SIZE,
    LOGICAL_SOURCES,
    NUM_LAYERS,
    SOURCE_WIDTH,
    TP_SIZE,
    DeepseekV2LiteTP4C1OutputProjection,
    close_deepseek_v2_lite_tp4_c1_communicator,
    configure_deepseek_v2_lite_tp4_c1_workspace,
    install_deepseek_v2_lite_tp4_c1_output_projections,
)


FORMAT = "basisserve.deepseek_v2_lite.tp4_prefill_heavy_benchmark.v1"
ARMS = ("dense", "c1_mean_dp")


class _PrefixStaticLayer(CacheLayerMixin):
    """Fixed-capacity cache returning only the initialized prefix.

    Transformers' generic ``StaticLayer`` returns its entire capacity and thus
    asks causal-mask construction to cover all future positions.  DeepSeek MLA
    stores asymmetric latent tensors, so this small inference-only layer keeps
    the same allocation stable while exposing only the written prefix.
    """

    is_compileable = False
    is_sliding = False

    def __init__(self, max_cache_len: int) -> None:
        super().__init__()
        self.max_cache_len = int(max_cache_len)
        self.current_length = 0

    def lazy_initialization(self, key_states: Tensor, value_states: Tensor) -> None:
        if key_states.shape[:-1] != value_states.shape[:-1]:
            raise ValueError("DeepSeek MLA latent cache prefix shapes differ")
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.batch_size = int(key_states.shape[0])
        self.num_heads = int(key_states.shape[1])
        self.keys = torch.empty(
            self.batch_size,
            self.num_heads,
            self.max_cache_len,
            int(key_states.shape[-1]),
            dtype=key_states.dtype,
            device=key_states.device,
        )
        self.values = torch.empty(
            self.batch_size,
            self.num_heads,
            self.max_cache_len,
            int(value_states.shape[-1]),
            dtype=value_states.dtype,
            device=value_states.device,
        )
        self.is_initialized = True

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        *args,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        del args, kwargs
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        width = int(key_states.shape[-2])
        start = self.current_length
        end = start + width
        if end > self.max_cache_len:
            raise RuntimeError(
                f"DeepSeek cache capacity exceeded: {end} > {self.max_cache_len}"
            )
        assert self.keys is not None and self.values is not None
        self.keys[:, :, start:end, :].copy_(key_states)
        self.values[:, :, start:end, :].copy_(value_states)
        self.current_length = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.current_length + int(query_length), 0

    def get_seq_length(self) -> int:
        return self.current_length

    def get_max_length(self) -> int:
        return self.max_cache_len

    def reset(self) -> None:
        self.current_length = 0


def _make_cache(capacity: int) -> Cache:
    return Cache(
        layers=[_PrefixStaticLayer(capacity) for _ in range(NUM_LAYERS)],
    )


def _cache_bytes(cache: Cache) -> int:
    total = 0
    for layer in cache.layers:
        for tensor in (layer.keys, layer.values):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


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
    process_rank = dist.get_rank()
    batch, local_vocab = map(int, local_logits.shape)
    values, local_ids = torch.max(local_logits.float(), dim=-1)
    global_ids = local_ids.to(torch.int64) + process_rank * local_vocab
    gathered_values = torch.empty(
        TP_SIZE * batch,
        device=values.device,
        dtype=values.dtype,
    )
    gathered_ids = torch.empty(
        TP_SIZE * batch,
        device=values.device,
        dtype=torch.int64,
    )
    dist.all_gather_into_tensor(gathered_values, values.contiguous())
    dist.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
    candidates = gathered_values.view(TP_SIZE, batch)
    winning_rank = torch.argmax(candidates, dim=0, keepdim=True)
    return torch.gather(
        gathered_ids.view(TP_SIZE, batch),
        0,
        winning_rank,
    ).squeeze(0)


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


def _prefill_step(
    model: torch.nn.Module,
    prompt_ids: Tensor,
    cache: Cache,
    *,
    chunk_size: int,
) -> Tensor:
    prompt_length = int(prompt_ids.shape[1])
    output = None
    for start in range(0, prompt_length, int(chunk_size)):
        end = min(start + int(chunk_size), prompt_length)
        position_ids = torch.arange(
            start,
            end,
            dtype=torch.int64,
            device=prompt_ids.device,
        ).unsqueeze(0)
        output = model.model(
            input_ids=prompt_ids[:, start:end],
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
    if output is None:
        raise AssertionError("prefill received an empty prompt")
    local_logits = F.linear(output.last_hidden_state[:, -1, :], model.lm_head.weight)
    return _distributed_greedy(local_logits)


def _decode_step(
    model: torch.nn.Module,
    token_ids: Tensor,
    cache: Cache,
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
        past_key_values=cache,
        use_cache=True,
    )
    local_logits = F.linear(output.last_hidden_state[:, -1, :], model.lm_head.weight)
    return _distributed_greedy(local_logits)


def _execute_request(
    model: torch.nn.Module,
    prompt_ids: Tensor,
    cache: Cache,
    *,
    output_tokens: int,
    prefill_chunk_size: int,
) -> Tensor:
    prompt_length = int(prompt_ids.shape[1])
    token = _prefill_step(
        model,
        prompt_ids,
        cache,
        chunk_size=prefill_chunk_size,
    )
    for decode_index in range(output_tokens - 1):
        token = _decode_step(
            model,
            token,
            cache,
            position=prompt_length + decode_index,
        )
    return token


@torch.inference_mode()
def _run_configuration(
    model: torch.nn.Module,
    *,
    batch: int,
    prompt_length: int,
    output_tokens: int,
    prefill_chunk_size: int,
    warmup_runs: int,
    repeat_runs: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, Any]:
    decode_steps = output_tokens - 1
    capacity = prompt_length + decode_steps
    cache = _make_cache(capacity)
    prompt_ids = _prompt_ids(
        batch=batch,
        prompt_length=prompt_length,
        vocab_size=vocab_size,
        device=device,
    )

    # Initialize the asymmetric MLA cache before all warmup and timed runs.
    dry_ids = prompt_ids[:, :1]
    _prefill_step(model, dry_ids, cache, chunk_size=1)
    torch.cuda.synchronize(device)
    cache.reset()
    cache_bytes = _global_max_int(_cache_bytes(cache), device=device)

    for _ in range(warmup_runs):
        cache.reset()
        dist.barrier()
        _execute_request(
            model,
            prompt_ids,
            cache,
            output_tokens=output_tokens,
            prefill_chunk_size=prefill_chunk_size,
        )
        torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    prefill_timings: list[float] = []
    decode_timings: list[float] = []
    total_timings: list[float] = []
    token: Tensor | None = None
    for _ in range(repeat_runs):
        cache.reset()
        dist.barrier()
        total_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
        token = _prefill_step(
            model,
            prompt_ids,
            cache,
            chunk_size=prefill_chunk_size,
        )
        prefill_end.record()
        for decode_index in range(decode_steps):
            token = _decode_step(
                model,
                token,
                cache,
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
        "prefill_chunk_size": prefill_chunk_size,
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
            "mla_latent_kv_cache_bytes_per_rank": cache_bytes,
            "peak_allocated_bytes_per_rank": peak_allocated,
            "peak_reserved_bytes_per_rank": peak_reserved,
        },
        "final_token_ids_prefix": by_rank[0, : min(batch, 8)].cpu().tolist(),
    }


def _communication_metadata(
    arm: str,
    modules: Sequence[DeepseekV2LiteTP4C1OutputProjection],
) -> dict[str, Any]:
    element_bytes = 2
    if arm == "dense":
        elements_per_layer = math.ceil(2 * (TP_SIZE - 1) * HIDDEN_SIZE / TP_SIZE)
        return {
            "collective": "Transformers rowwise o_proj AllReduce",
            "theoretical_ring_elements_per_token_per_rank_per_layer": elements_per_layer,
            "theoretical_ring_bytes_per_token_per_rank_all_layers": (
                elements_per_layer * element_bytes * NUM_LAYERS
            ),
        }
    local_widths = tuple(module.local_wire_width for module in modules)
    sent_elements = (TP_SIZE - 1) * sum(local_widths)
    dense_allgather_elements = (
        (TP_SIZE - 1) * (LOGICAL_SOURCES // TP_SIZE) * SOURCE_WIDTH * NUM_LAYERS
    )
    dense_allreduce_elements = math.ceil(
        2 * (TP_SIZE - 1) * HIDDEN_SIZE * NUM_LAYERS / TP_SIZE
    )
    return {
        "collective": "compiled in-place NCCL feature-major packed AllGather",
        "logical_checkpoint_sources": LOGICAL_SOURCES,
        "physical_tp_size": TP_SIZE,
        "logical_sources_per_process": LOGICAL_SOURCES // TP_SIZE,
        "local_wire_widths_by_layer": list(local_widths),
        "mean_local_wire_width": statistics.fmean(local_widths),
        "direct_send_elements_per_token_per_rank_all_layers": sent_elements,
        "direct_send_bytes_per_token_per_rank_all_layers": sent_elements * element_bytes,
        "reduction_vs_dense_attention_output_allgather": (
            1.0 - sent_elements / dense_allgather_elements
        ),
        "reduction_vs_standard_dense_o_proj_allreduce": (
            1.0 - sent_elements / dense_allreduce_elements
        ),
        "mla_kv_cache_compression": "none",
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--repeat-runs", type=int, default=10)
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flex_attention"),
        default="sdpa",
    )
    parser.add_argument(
        "--experts-implementation",
        choices=("eager", "grouped_mm"),
        default="grouped_mm",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    args = parser.parse_args()

    if args.arm == "dense" and args.factor_dir is not None:
        raise ValueError("dense arm must not receive --factor-dir")
    if args.arm == "c1_mean_dp" and args.factor_dir is None:
        raise ValueError("C1 arm requires --factor-dir")
    positive = (
        args.batch_size,
        args.prompt_length,
        args.prefill_chunk_size,
        args.output_tokens,
        args.warmup_runs,
        args.repeat_runs,
        args.torch_num_threads,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("benchmark dimensions and run counts must be positive")
    if args.output_tokens < 2:
        raise ValueError("benchmark requires at least two output tokens")

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
        attn_implementation=args.attn_implementation,
        experts_implementation=args.experts_implementation,
        distributed_config=DistributedConfig(tp_size=TP_SIZE),
        local_files_only=True,
        trust_remote_code=False,
    ).eval()
    if not dist.is_initialized() or dist.get_world_size() != TP_SIZE:
        raise RuntimeError(f"benchmark requires physical TP{TP_SIZE}")
    if int(model.config.num_hidden_layers) != NUM_LAYERS:
        raise ValueError("unexpected DeepSeek-V2-Lite layer count")
    if int(model.config.hidden_size) != HIDDEN_SIZE:
        raise ValueError("unexpected DeepSeek-V2-Lite hidden size")
    modules = (
        ()
        if args.arm == "dense"
        else install_deepseek_v2_lite_tp4_c1_output_projections(
            model,
            factor_dir=args.factor_dir,
        )
    )
    configure_deepseek_v2_lite_tp4_c1_workspace(
        modules,
        max_tokens=args.batch_size * min(args.prompt_length, args.prefill_chunk_size),
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

    record = _run_configuration(
        model,
        batch=args.batch_size,
        prompt_length=args.prompt_length,
        output_tokens=args.output_tokens,
        prefill_chunk_size=args.prefill_chunk_size,
        warmup_runs=args.warmup_runs,
        repeat_runs=args.repeat_runs,
        vocab_size=vocab_size,
        device=device,
    )
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "event": "configuration_complete",
                    "arm": args.arm,
                    "batch": args.batch_size,
                    "prompt_length": args.prompt_length,
                    "ttft_ms": record["time_to_first_token"]["mean_ms"],
                    "prefill_tokens_per_second": record["prefill_tokens_per_second"],
                    "decode_tokens_per_second": record["decode_tokens_per_second"],
                    "end_to_end_ms": record["end_to_end"]["mean_ms"],
                    "peak_gib": record["memory"]["peak_allocated_bytes_per_rank"] / 2**30,
                }
            ),
            flush=True,
        )

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
                "physical_tp_size": TP_SIZE,
                "checkpoint_logical_sources": LOGICAL_SOURCES,
                "dtype": "bfloat16",
                "batch_size": args.batch_size,
                "prompt_length": args.prompt_length,
                "prefill_chunk_size": args.prefill_chunk_size,
                "output_tokens": args.output_tokens,
                "decode_steps_after_first_token": args.output_tokens - 1,
                "warmup_runs": args.warmup_runs,
                "repeat_runs": args.repeat_runs,
                "attention_implementation": args.attn_implementation,
                "experts_implementation": args.experts_implementation,
                "timed_scope": (
                    "transformer + TP collectives + sharded LM head + "
                    "distributed greedy selection"
                ),
                "c1_scope": (
                    None
                    if args.arm == "dense"
                    else (
                        "post-attention source encoding + packed BF16 AllGather + "
                        "replicated decoder; MLA latent KV/cache unchanged"
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
            "factors": [
                {
                    "layer": module.layer_index,
                    "rank": module.source_rank,
                    "local_sources": list(module.local_source_indices),
                    "path": module.factor_path,
                    "sha256": module.factor_sha256,
                }
                for module in modules
            ],
            "record": record,
        }
        _atomic_json(output, payload)
        print(json.dumps({"event": "result_written", "path": str(output)}), flush=True)
    dist.barrier()
    close_deepseek_v2_lite_tp4_c1_communicator(modules)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
