#!/usr/bin/env python3
"""Measure the fixed-width feature-major AllGather boundary without a decoder.

Run with ``torchrun``.  The benchmark reports collective latency for the exact
wire shape used by compact-V TP serving: ``[local_width, tokens]`` per source
and ``[world_size * local_width, tokens]`` on every receiver.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from basisserve.kernels.feature_ragged_allgather import (  # noqa: E402
    FeatureRaggedCommunicator,
    PreparedUniformAllGather,
)
from basisserve.kernels.ragged_allgather import StaticRaggedPlan  # noqa: E402


_BACKENDS = (
    "feature_direct",
    "uniform_nccl",
    "uniform_nccl_graph",
    "uniform_ipc",
)
_DTYPE_BY_NAME = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "uint8": torch.uint8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-width",
        type=int,
        default=512,
        help="wire coordinates per rank; Qwen3-8B TP4 V64 uses 8*64=512",
    )
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=tuple(_DTYPE_BY_NAME),
        default="bfloat16",
    )
    parser.add_argument(
        "--backends",
        default=",".join(_BACKENDS),
        help=f"comma-separated subset of {','.join(_BACKENDS)}",
    )
    parser.add_argument(
        "--ipc-algorithm",
        choices=("auto", "fanout", "fanout_warp", "recursive_doubling", "ring"),
        default="auto",
    )
    parser.add_argument(
        "--ipc-channels",
        type=int,
        choices=(0, 1, 2, 4, 8),
        default=0,
        help="0 selects the algorithm default; >1 is valid for fanout/ring",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def make_local(
    *,
    rank: int,
    local_width: int,
    tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    values = torch.arange(
        local_width * tokens,
        dtype=torch.int64,
        device=device,
    ).view(local_width, tokens)
    values = values + rank * 17
    if dtype == torch.uint8:
        return values.remainder(251).to(torch.uint8).contiguous()
    return values.to(dtype).contiguous()


def expected_arena(
    *,
    world_size: int,
    local_width: int,
    tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.cat(
        [
            make_local(
                rank=rank,
                local_width=local_width,
                tokens=tokens,
                dtype=dtype,
                device=device,
            )
            for rank in range(world_size)
        ],
        dim=0,
    )


def capture_nccl_graph(
    prepared: PreparedUniformAllGather,
    *,
    capture_stream: torch.cuda.Stream,
    device: torch.device,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    # A prepared plan is bound to the stream on which it was constructed.
    # PyTorch captures on an internal side stream unless one is supplied, so
    # explicitly capture on the same stream instead of accidentally launching
    # NCCL outside the graph.
    with torch.cuda.stream(capture_stream):
        prepared.gather_inplace_fast()
    capture_stream.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        output = prepared.gather_inplace_fast()
    torch.cuda.synchronize(device)
    dist.barrier()
    return graph, output


def measure(
    run_once: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
    timing_stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, list[float]]:
    output: torch.Tensor | None = None
    with torch.cuda.stream(timing_stream):
        for _ in range(warmup):
            output = run_once()
    torch.cuda.synchronize(device)
    dist.barrier()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    # Graph replay launches on the current stream, not its capture stream.
    with torch.cuda.stream(timing_stream):
        for start, end in zip(starts, ends, strict=True):
            start.record()
            output = run_once()
            end.record()
    assert output is not None
    ends[-1].synchronize()
    local = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return output, local.cpu().tolist()


def summarize(
    samples_ms: list[float],
    *,
    block_bytes: int,
    world_size: int,
) -> dict[str, float]:
    p50 = statistics.median(samples_ms)
    bytes_sent_per_rank = (world_size - 1) * block_bytes
    return {
        "mean_us": statistics.fmean(samples_ms) * 1000.0,
        "minimum_us": min(samples_ms) * 1000.0,
        "p50_us": p50 * 1000.0,
        "p90_us": percentile(samples_ms, 0.90) * 1000.0,
        "p95_us": percentile(samples_ms, 0.95) * 1000.0,
        "p99_us": percentile(samples_ms, 0.99) * 1000.0,
        "maximum_us": max(samples_ms) * 1000.0,
        "effective_sent_gbytes_per_second_at_p50": (
            bytes_sent_per_rank / (p50 * 1.0e6)
        ),
    }


def main() -> None:
    args = parse_args()
    if args.local_width <= 0 or args.tokens <= 0:
        raise ValueError("local width and tokens must be positive")
    if args.warmup <= 0 or args.iters <= 0:
        raise ValueError("warmup and iteration counts must be positive")
    backends = tuple(item.strip() for item in args.backends.split(",") if item.strip())
    if not backends or len(set(backends)) != len(backends):
        raise ValueError("--backends must contain unique names")
    unknown = tuple(item for item in backends if item not in _BACKENDS)
    if unknown:
        raise ValueError(f"unknown backends {unknown}; expected a subset of {_BACKENDS}")

    if args.ipc_channels > 1 and args.ipc_algorithm not in ("fanout", "ring"):
        raise ValueError("multiple IPC channels require --ipc-algorithm fanout|ring")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = _DTYPE_BY_NAME[args.dtype]
    plan = StaticRaggedPlan.from_source_widths((args.local_width,) * world_size)
    local = make_local(
        rank=rank,
        local_width=args.local_width,
        tokens=args.tokens,
        dtype=dtype,
        device=device,
    )
    reference = expected_arena(
        world_size=world_size,
        local_width=args.local_width,
        tokens=args.tokens,
        dtype=dtype,
        device=device,
    )

    communicator = FeatureRaggedCommunicator.from_distributed(device=device)
    communicator.configure_direct_workspace(
        tokens=args.tokens,
        max_total_width=plan.total_width,
        dtype=dtype,
    )

    # Reproduce the existing serving fast case fairly: the attention producer
    # has already written the feature-direct arena's local source slot, so the
    # timed dynamic baseline should not include an extra local copy.
    direct_local = communicator.direct_local_feature_major_view(
        plan,
        tokens=args.tokens,
        dtype=dtype,
    )
    direct_local.copy_(local)

    prepared_nccl: PreparedUniformAllGather | None = None
    if "uniform_nccl" in backends or "uniform_nccl_graph" in backends:
        prepared_nccl = communicator.prepare_uniform(
            plan,
            tokens=args.tokens,
            dtype=dtype,
            backend="uniform_nccl",
        )
        prepared_nccl.local_feature_major_view().copy_(local)

    prepared_ipc: PreparedUniformAllGather | None = None
    if "uniform_ipc" in backends:
        communicator.prepare_ipc(
            tokens=args.tokens,
            max_total_width=plan.total_width,
            dtype=dtype,
        )
        prepared_ipc = communicator.prepare_uniform(
            plan,
            tokens=args.tokens,
            dtype=dtype,
            backend="uniform_ipc",
            ipc_algorithm=args.ipc_algorithm,
            ipc_channels=args.ipc_channels,
        )
        # Prime both alternating slots with producer output. The timed function
        # then contains only the custom collective kernel.
        for _ in range(2):
            prepared_ipc.local_feature_major_view().copy_(local)
            prepared_ipc.gather_inplace_fast()
        torch.cuda.synchronize(device)

    calls: dict[str, Callable[[], torch.Tensor]] = {
        "feature_direct": lambda: communicator.gather(
            direct_local,
            plan,
            backend="feature_direct",
            local_is_feature_major=True,
        ),
    }
    if prepared_nccl is not None:
        calls["uniform_nccl"] = prepared_nccl.gather_inplace_fast
    graph_error: str | None = None
    graph: torch.cuda.CUDAGraph | None = None
    graph_output: torch.Tensor | None = None
    prepared_graph: PreparedUniformAllGather | None = None
    capture_stream: torch.cuda.Stream | None = None
    if "uniform_nccl_graph" in backends:
        try:
            capture_stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(capture_stream):
                prepared_graph = communicator.prepare_uniform(
                    plan,
                    tokens=args.tokens,
                    dtype=dtype,
                    backend="uniform_nccl",
                )
                prepared_graph.local_feature_major_view().copy_(local)
            capture_stream.synchronize()
            graph, graph_output = capture_nccl_graph(
                prepared_graph,
                capture_stream=capture_stream,
                device=device,
            )

            def replay_graph() -> torch.Tensor:
                graph.replay()
                return graph_output

            calls["uniform_nccl_graph"] = replay_graph
        except Exception as error:  # environment/version-dependent capability
            graph_error = f"{type(error).__name__}: {error}"
    if prepared_ipc is not None:
        calls["uniform_ipc"] = prepared_ipc.gather_inplace_fast

    element_size = torch.empty((), dtype=dtype).element_size()
    block_bytes = args.local_width * args.tokens * element_size
    records: list[dict[str, object]] = []
    for backend in backends:
        unavailable_reason = None
        if backend == "uniform_nccl_graph" and backend not in calls:
            unavailable_reason = graph_error or "CUDA graph capture was unavailable"
        if unavailable_reason is not None:
            records.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "reason": unavailable_reason,
                }
            )
            continue
        output, samples = measure(
            calls[backend],
            warmup=args.warmup,
            iters=args.iters,
            device=device,
            timing_stream=(capture_stream if backend == "uniform_nccl_graph" else None),
        )
        torch.testing.assert_close(output, reference, rtol=0.0, atol=0.0)
        records.append(
            {
                "backend": backend,
                "status": "complete",
                **summarize(
                    samples,
                    block_bytes=block_bytes,
                    world_size=world_size,
                ),
            }
        )
        dist.barrier()

    payload = {
        "format": "basisserve.uniform_feature_major_allgather.v1",
        "world_size": world_size,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl_version": communicator.nccl_version,
        "local_width": args.local_width,
        "tokens": args.tokens,
        "dtype": args.dtype,
        "block_bytes_per_rank": block_bytes,
        "bytes_sent_per_rank": (world_size - 1) * block_bytes,
        "ipc_algorithm": args.ipc_algorithm,
        "ipc_channels": args.ipc_channels,
        "warmup": args.warmup,
        "iterations": args.iters,
        "records": records,
    }
    if rank == 0:
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text, flush=True)
        if args.output_json is not None:
            output_path = args.output_json.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text + "\n", encoding="utf-8")

    dist.barrier()
    # Destroy graph executables before the NCCL communicator they captured.
    # The replay closure in ``calls`` otherwise keeps the CUDAGraph alive until
    # function teardown, after communicator.close().
    calls.clear()
    graph_output = None
    graph = None
    prepared_graph = None
    capture_stream = None
    torch.cuda.synchronize(device)
    gc.collect()
    communicator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
