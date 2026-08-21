#!/usr/bin/env python3
"""Benchmark dense vs low-rank tensor-parallel output communication.

Run on GPUs with, for example:

    torchrun --standalone --nproc-per-node=8 \
      evaluation/run_low_rank_allreduce.py \
      --hidden-size 8192 --tokens 256 --rank-ratio 0.30 --dtype bfloat16

The default synthetic weight is exactly rank ``r``.  This isolates the systems
question from the quality/rank-selection question: dense and low-rank paths are
numerically equivalent, while the communicated payload changes from ``d`` to
``r``.  Use ``--mode communication`` to benchmark collectives only, or
``--mode end-to-end`` to include both projection GEMMs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import statistics
import time
from typing import Callable

import torch
import torch.distributed as dist

from basisserve.core import DenseRowParallelOutput, LowRankAllReduceOutput, resolve_rank


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _init_distributed(backend: str | None) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        selected = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(backend=selected, init_method="env://")
    rank = dist.get_rank() if dist.is_initialized() else 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_across_ranks(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _benchmark(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _barrier()
    _synchronize(device)

    samples_ms: list[float] = []
    if device.type == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for start, end in zip(starts, ends):
            start.record()
            fn()
            end.record()
        _synchronize(device)
        samples_ms = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    else:
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            samples_ms.append((time.perf_counter() - start) * 1e3)

    # A distributed step is gated by the slowest rank.
    mean_ms = _max_across_ranks(statistics.fmean(samples_ms), device)
    p50_ms = _max_across_ranks(statistics.median(samples_ms), device)
    p95_local = sorted(samples_ms)[max(0, min(len(samples_ms) - 1, int(0.95 * len(samples_ms)) - 1))]
    p95_ms = _max_across_ranks(p95_local, device)
    return {"mean_ms": mean_ms, "p50_ms": p50_ms, "p95_ms": p95_ms}


def _all_reduce_in_place(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _make_exact_low_rank_problem(
    *,
    hidden_size: int,
    rank: int,
    world_size: int,
    global_rank: int,
    tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden_size % world_size != 0:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by world_size={world_size}"
        )
    local_width = hidden_size // world_size

    # Every rank constructs the same replicated output basis.
    basis_generator = torch.Generator(device=device)
    basis_generator.manual_seed(1234)
    output_basis = torch.randn(
        hidden_size,
        rank,
        generator=basis_generator,
        device=device,
        dtype=torch.float32,
    ) / (rank**0.5)

    # Each rank owns a distinct input-side factor and local activation shard.
    local_generator = torch.Generator(device=device)
    local_generator.manual_seed(4321 + global_rank)
    local_input_factor = torch.randn(
        local_width,
        rank,
        generator=local_generator,
        device=device,
        dtype=torch.float32,
    ) / (local_width**0.5)
    local_hidden = torch.randn(
        tokens,
        local_width,
        generator=local_generator,
        device=device,
        dtype=torch.float32,
    )

    output_basis = output_basis.to(dtype)
    local_input_factor = local_input_factor.to(dtype)
    local_hidden = local_hidden.to(dtype)
    local_dense_weight = output_basis @ local_input_factor.transpose(0, 1)
    return local_hidden, local_dense_weight.contiguous(), local_input_factor, output_basis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument(
        "--tokens",
        type=int,
        default=256,
        help="Active decode tokens (normally continuous-batching size).",
    )
    rank_group = parser.add_mutually_exclusive_group(required=False)
    rank_group.add_argument("--rank", type=int)
    rank_group.add_argument("--rank-ratio", type=float, default=0.30)
    parser.add_argument("--rank-multiple", type=int, default=64)
    parser.add_argument("--dtype", choices=sorted(_DTYPE_MAP), default="bfloat16")
    parser.add_argument(
        "--communication-dtype",
        choices=["same", *_DTYPE_MAP.keys()],
        default="same",
    )
    parser.add_argument(
        "--mode",
        choices=["communication", "end-to-end", "both"],
        default="both",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default=None)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    global_rank, world_size, device = _init_distributed(args.backend)
    dtype = _DTYPE_MAP[args.dtype]
    if device.type == "cpu" and dtype == torch.float16:
        # CPU linear kernels commonly do not support fp16 efficiently.
        dtype = torch.float32
    communication_dtype = None
    if args.communication_dtype != "same":
        communication_dtype = _DTYPE_MAP[args.communication_dtype]

    rank = resolve_rank(
        args.hidden_size,
        args.hidden_size,
        rank=args.rank,
        rank_ratio=None if args.rank is not None else args.rank_ratio,
        multiple=args.rank_multiple,
    )

    local_hidden, local_dense_weight, local_input_factor, output_basis = (
        _make_exact_low_rank_problem(
            hidden_size=args.hidden_size,
            rank=rank,
            world_size=world_size,
            global_rank=global_rank,
            tokens=args.tokens,
            dtype=dtype,
            device=device,
        )
    )

    dense = DenseRowParallelOutput(
        local_dense_weight,
        communication_dtype=communication_dtype,
    ).to(device)
    low_rank = LowRankAllReduceOutput(
        local_input_factor,
        output_basis,
        communication_dtype=communication_dtype,
    ).to(device)

    with torch.inference_mode():
        dense_output = dense(local_hidden)
        low_rank_output = low_rank(local_hidden)
        denominator = torch.linalg.vector_norm(dense_output.float()).clamp_min(1e-12)
        relative_error = float(
            (torch.linalg.vector_norm(dense_output.float() - low_rank_output.float()) / denominator).item()
        )
        relative_error = _max_across_ranks(relative_error, device)

    results: dict[str, object] = {
        "world_size": world_size,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "communication_dtype": (
            str(communication_dtype).replace("torch.", "")
            if communication_dtype is not None
            else "same"
        ),
        "hidden_size": args.hidden_size,
        "tokens": args.tokens,
        "rank": rank,
        "rank_ratio": rank / args.hidden_size,
        "relative_output_error": relative_error,
    }

    estimate = low_rank.communication_estimate(
        args.tokens,
        dtype=communication_dtype or dtype,
    )
    results["communication"] = {
        "dense_payload_bytes": estimate.full_payload_bytes,
        "low_rank_payload_bytes": estimate.low_rank_payload_bytes,
        "payload_reduction_x": estimate.payload_reduction,
        "estimated_dense_ring_bytes_per_rank": estimate.estimated_full_ring_bytes_per_rank,
        "estimated_low_rank_ring_bytes_per_rank": estimate.estimated_low_rank_ring_bytes_per_rank,
    }

    with torch.inference_mode():
        if args.mode in {"communication", "both"}:
            dense_comm_tensor = torch.zeros(
                args.tokens,
                args.hidden_size,
                device=device,
                dtype=communication_dtype or dtype,
            )
            low_rank_comm_tensor = torch.zeros(
                args.tokens,
                rank,
                device=device,
                dtype=communication_dtype or dtype,
            )
            dense_comm = _benchmark(
                lambda: _all_reduce_in_place(dense_comm_tensor),
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            low_rank_comm = _benchmark(
                lambda: _all_reduce_in_place(low_rank_comm_tensor),
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            results["communication_only_timing"] = {
                "dense": dense_comm,
                "low_rank": low_rank_comm,
                "mean_speedup_x": dense_comm["mean_ms"] / max(low_rank_comm["mean_ms"], 1e-12),
            }

        if args.mode in {"end-to-end", "both"}:
            dense_timing = _benchmark(
                lambda: dense(local_hidden),
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            low_rank_timing = _benchmark(
                lambda: low_rank(local_hidden),
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            results["end_to_end_timing"] = {
                "dense": dense_timing,
                "low_rank": low_rank_timing,
                "mean_speedup_x": dense_timing["mean_ms"] / max(low_rank_timing["mean_ms"], 1e-12),
            }

    if global_rank == 0:
        serialized = json.dumps(results, indent=2, sort_keys=True)
        print(serialized)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(serialized + "\n", encoding="utf-8")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
