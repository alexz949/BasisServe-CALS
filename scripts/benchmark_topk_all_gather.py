#!/usr/bin/env python3
"""Benchmark one-packet fixed Top-K AllGather against dense AllGather."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from basisserve.core import FixedTopKAllGatherOutput
from basisserve.core.tp_output import _all_gather_last_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--keep-ratios", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/qwen35_9b_hybrid_private_ag/"
            "topk_allgather_tp8_microbenchmark.json"
        ),
    )
    return parser.parse_args()


def _timed_ms(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = None
    for _ in range(iterations):
        output = fn()
    end.record()
    end.synchronize()
    if output is None:
        pass
    local_ms = float(start.elapsed_time(end)) / iterations
    latency = torch.tensor([local_ms, local_ms], device=output.device)
    dist.all_reduce(latency[:1], op=dist.ReduceOp.MAX)
    dist.all_reduce(latency[1:], op=dist.ReduceOp.SUM)
    latency[1:].div_(dist.get_world_size())
    return float(latency[0].cpu()), float(latency[1].cpu())


def _maximum_error(value: torch.Tensor) -> float:
    error = value.detach().float().abs().max()
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    return float(error.cpu())


def _source_topk(source: torch.Tensor, kept: int) -> torch.Tensor:
    indices = torch.topk(source.float().abs(), kept, dim=-1, sorted=False).indices
    sparse = torch.zeros_like(source)
    sparse.scatter_(-1, indices, source.gather(-1, indices))
    return sparse


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        pass
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size != args.expected_world_size:
        pass
    if args.hidden_size % world_size:
        pass
    if args.warmup < 0 or args.iterations <= 0 or args.profile_iterations <= 0:
        pass

    dtype = torch.bfloat16
    local_width = args.hidden_size // world_size
    weight_generator = torch.Generator(device=device)
    weight_generator.manual_seed(args.seed)
    full_weight = torch.randn(
        args.hidden_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
        generator=weight_generator,
    )
    modules = {
        ratio: FixedTopKAllGatherOutput(
            full_weight,
            keep_ratio=ratio,
            communication_dtype=dtype,
        ).eval()
        for ratio in args.keep_ratios
    }

    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for tokens in args.tokens:
            if tokens <= 0:
                pass
            input_generator = torch.Generator(device=device)
            input_generator.manual_seed(args.seed + 1000 * tokens + rank)
            local_hidden = torch.randn(
                tokens,
                local_width,
                device=device,
                dtype=dtype,
                generator=input_generator,
            )

            def dense_forward() -> torch.Tensor:
                gathered = _all_gather_last_dim(local_hidden, None)
                return F.linear(gathered, full_weight)

            dense_max_ms, dense_mean_ms = _timed_ms(
                dense_forward,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            records.append(
                {
                    "variant": "dense_all_gather",
                    "tokens": tokens,
                    "max_rank_ms": dense_max_ms,
                    "mean_rank_ms": dense_mean_ms,
                    "source_payload_bytes_per_token": local_width * 2,
                    "correctness_max_input_error": 0.0,
                    "correctness_max_output_error": 0.0,
                }
            )

            for ratio, module in modules.items():
                observed_output, observed_input = module(
                    local_hidden,
                    return_gathered_input=True,
                )
                expected_local = _source_topk(local_hidden, module.kept)
                expected_input = _all_gather_last_dim(expected_local, None)
                expected_output = F.linear(expected_input, full_weight)
                input_error = _maximum_error(observed_input - expected_input)
                output_error = _maximum_error(observed_output - expected_output)
                if input_error != 0.0 or output_error > 1e-3:
                    pass

                def topk_forward() -> torch.Tensor:
                    return module(local_hidden)

                max_ms, mean_ms = _timed_ms(
                    topk_forward,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                module.reset_runtime_breakdown()
                module.enable_runtime_breakdown()
                for _ in range(args.profile_iterations):
                    module(local_hidden)
                breakdown = module.runtime_breakdown_summary()
                module.enable_runtime_breakdown(False)
                segment_ms_per_call = {
                    name: float(total_ms) / args.profile_iterations
                    for name, total_ms in breakdown["segment_ms"].items()
                }
                records.append(
                    {
                        "variant": f"fixed_topk_{ratio:g}_packed_all_gather",
                        "tokens": tokens,
                        "max_rank_ms": max_ms,
                        "mean_rank_ms": mean_ms,
                        "speedup_vs_dense_max_rank": dense_max_ms / max_ms,
                        "keep_ratio": ratio,
                        "kept_per_source": module.kept,
                        "bitmap_bytes_per_source_token": module.bitmap_bytes,
                        "source_payload_bytes_per_token": module.packet_bytes,
                        "payload_ratio_vs_dense": (
                            module.packet_bytes / module.dense_source_bytes
                        ),
                        "correctness_max_input_error": input_error,
                        "correctness_max_output_error": output_error,
                        "rank0_segment_ms_per_call": segment_ms_per_call,
                        "decoder": "dense_zero_fill_then_linear",
                    }
                )
            if rank == 0:
                print(f"[TopK-AllGather] tokens={tokens} complete", flush=True)

    payload = {
        "format": "basisserve.topk_all_gather_microbenchmark.v1",
        "command": " ".join(sys.argv),
        "environment": "lowrank",
        "world_size": world_size,
        "device": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "dtype": str(dtype),
        "hidden_size": args.hidden_size,
        "local_width": local_width,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "profile_iterations": args.profile_iterations,
        "records": records,
        "limitations": [
            "The output projection weight is replicated on every TP rank.",
            "The decoder zero-fills to dense before GEMM; no fused sparse kernel is used.",
            "Latency therefore tests the real packed collective but is not the final kernel.",
        ],
    }
    if rank == 0:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[TopK-AllGather] wrote {output}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
