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


FORMAT = "basisserve.qwen3_32b.tp4_prefill_decode_benchmark.v1"


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


def _install_sequence_chunked_mlps(
    model: torch.nn.Module,
    chunk_length: int,
) -> None:
    for layer in model.model.layers:
        layer.mlp = _SequenceChunkedMLP(layer.mlp, chunk_length)


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
    position: int,
) -> Tensor:
    batch = int(token_ids.shape[0])
    position_ids = torch.full(
        (1, 1), position, dtype=torch.int64, device=token_ids.device
    )
    output = model.model(
        input_ids=token_ids.reshape(batch, 1),
        position_ids=position_ids,
        use_cache=False,
    )
    return _distributed_greedy(
        F.linear(output.last_hidden_state[:, -1], model.lm_head.weight)
    )


def _reset_caches(modules: Sequence[torch.nn.Module]) -> None:
    for module in modules:
        module.reset_cache()


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


def _communication(
    arm: str,
    modules: Sequence[torch.nn.Module],
    *,
    batch_size: int,
    prompt_length: int,
    output_tokens: int,
) -> dict[str, Any]:
    element_bytes = 2
    attention_rows = batch_size * (prompt_length + output_tokens - 1)
    if arm == "dense":
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
    elements_per_token_all_ranks = (TP_SIZE - 1) * sum(global_widths)
    dense_reference = 2 * (TP_SIZE - 1) * HIDDEN_SIZE * len(c1_modules)
    return {
        "collective": "NCCL feature-major ragged AllGather",
        "wire_dtype": "bfloat16",
        "global_wire_widths_by_layer": list(global_widths),
        "process_wire_widths_by_layer": [list(widths) for widths in process_widths],
        "mean_global_wire_width": statistics.fmean(global_widths),
        "logical_wire_bytes_all_ranks": (
            elements_per_token_all_ranks * attention_rows * element_bytes
        ),
        "dense_reference_wire_bytes_all_ranks": (
            dense_reference * attention_rows * element_bytes
        ),
        "reduction_vs_dense": 1.0 - elements_per_token_all_ranks / dense_reference,
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prompt-length", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--prefill-mlp-chunk-length", type=int, default=1024)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    if args.arm == "dense" and args.factor_dir is not None:
        raise ValueError("dense arm must not receive --factor-dir")
    if args.arm == "c1_ragged" and args.factor_dir is None:
        raise ValueError("C1 arm requires --factor-dir")
    if min(
        args.batch_size,
        args.prompt_length,
        args.output_tokens,
        args.repeat_runs,
        args.torch_num_threads,
        args.prefill_mlp_chunk_length,
    ) <= 0 or args.warmup_runs < 0:
        raise ValueError("benchmark dimensions and run counts are invalid")
    if args.output_tokens < 2:
        raise ValueError("output token count must include at least one decode step")
    if args.prompt_length + args.output_tokens - 1 > 32768:
        raise ValueError("request exceeds Qwen3 positional capacity")

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
        raise RuntimeError("benchmark requires physical TP4")
    _install_sequence_chunked_mlps(model, args.prefill_mlp_chunk_length)
    modules = install_qwen3_32b_tp4_attention(
        model,
        factor_dir=args.factor_dir,
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
                "output_tokens": args.output_tokens,
                "incremental_decode_steps": args.output_tokens - 1,
                "warmup_runs": args.warmup_runs,
                "repeat_runs": args.repeat_runs,
                "prefill_mlp_chunk_length": args.prefill_mlp_chunk_length,
                "timed_scope": (
                    "transformer + TP collectives + sharded LM head + "
                    "distributed greedy selection"
                ),
                "c1_attention": (
                    None
                    if args.arm == "dense"
                    else "variable-width Triton prefill and decode"
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
            "communication": _communication(
                args.arm,
                modules,
                batch_size=args.batch_size,
                prompt_length=args.prompt_length,
                output_tokens=args.output_tokens,
            ),
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
