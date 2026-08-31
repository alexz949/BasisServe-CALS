#!/usr/bin/env python3
"""Benchmark big versus 2/4-wave NCCL AllGather-decoder boundaries."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch import Tensor  # noqa: E402

from basisserve.kernels.pipelined_allgather_decoder import (  # noqa: E402
    PipelinedAllGatherDecoder,
)
from basisserve.kernels.rank_major_decoder import rank_major_decoder  # noqa: E402


def _parse_ints(text: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="64,128")
    parser.add_argument("--local-width", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--waves", default="2,4")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--execution-modes",
        default="eager,cuda_graph",
        help="comma-separated subset of eager,cuda_graph",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    args.batches = _parse_ints(args.batches, name="--batches")
    args.waves = _parse_ints(args.waves, name="--waves")
    args.execution_modes = tuple(
        item.strip() for item in args.execution_modes.split(",") if item.strip()
    )
    if not args.execution_modes or any(
        item not in ("eager", "cuda_graph") for item in args.execution_modes
    ):
        raise ValueError("--execution-modes must contain eager and/or cuda_graph")
    if args.local_width <= 0 or args.hidden_size <= 0:
        raise ValueError("matrix widths must be positive")
    if args.warmup <= 0 or args.iterations <= 0:
        raise ValueError("warmup and iterations must be positive")
    return args


class _BigAllGatherDecoder:
    def __init__(
        self,
        decoder: Tensor,
        *,
        rows: int,
        local_width: int,
        group,
        materialize_token_major: bool,
    ) -> None:
        self.decoder = decoder
        self.rows = int(rows)
        self.local_width = int(local_width)
        self.group = group
        self.processes = dist.get_world_size(group)
        self.materialize_token_major = bool(materialize_token_major)
        self.gathered = torch.empty(
            self.processes * self.rows,
            self.local_width,
            device=decoder.device,
            dtype=decoder.dtype,
        )

    def __call__(self, local: Tensor) -> Tensor:
        work = dist.all_gather_into_tensor(
            self.gathered,
            local,
            group=self.group,
            async_op=True,
        )
        work.wait()
        if not self.materialize_token_major:
            return rank_major_decoder(
                self.gathered,
                self.decoder,
                processes=self.processes,
            )
        token_major = (
            self.gathered.view(
                self.processes,
                self.rows,
                self.local_width,
            )
            .permute(1, 0, 2)
            .reshape(self.rows, self.processes * self.local_width)
            .contiguous()
        )
        return torch.mm(token_major, self.decoder)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarize(samples_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(samples_ms),
        "minimum_ms": min(samples_ms),
        "p50_ms": statistics.median(samples_ms),
        "p90_ms": percentile(samples_ms, 0.90),
        "p95_ms": percentile(samples_ms, 0.95),
        "p99_ms": percentile(samples_ms, 0.99),
        "maximum_ms": max(samples_ms),
    }


def relative_error(observed: Tensor, reference: Tensor) -> dict[str, float]:
    difference = observed.float() - reference.float()
    return {
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1.0e-12)
        ),
        "maximum_absolute": float(difference.abs().amax()),
    }


def measure(
    run_once: Callable[[], Tensor],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
    timing_stream: torch.cuda.Stream | None = None,
) -> tuple[Tensor, list[float]]:
    output: Tensor | None = None
    for _ in range(warmup):
        output = run_once()
    torch.cuda.synchronize(device)
    dist.barrier()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record(timing_stream)
        output = run_once()
        end.record(timing_stream)
    assert output is not None
    ends[-1].synchronize()
    local = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return output, local.cpu().tolist()


def capture_graph(
    run_once: Callable[[], Tensor],
    *,
    device: torch.device,
) -> tuple[Callable[[], Tensor], Tensor, torch.cuda.Stream, torch.cuda.CUDAGraph]:
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            output = run_once()
    capture_stream.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        output = run_once()
    torch.cuda.synchronize(device)
    dist.barrier()

    def replay() -> Tensor:
        # Replay on the root stream used during capture.  The timing events are
        # recorded on this stream as well, so the closing event observes every
        # captured side-stream dependency instead of measuring only host-side
        # graph launch on the caller's current stream.
        with torch.cuda.stream(capture_stream):
            graph.replay()
        return output

    return replay, output, capture_stream, graph


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    if any(args.local_width % waves for waves in args.waves):
        raise ValueError("local width must be divisible by every wave count")

    torch.manual_seed(20260828)
    decoder = (
        torch.randn(
            world_size * args.local_width,
            args.hidden_size,
            device=device,
            dtype=dtype,
        )
        / (world_size * args.local_width) ** 0.5
    ).contiguous()

    records: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in args.batches:
            generator = torch.Generator(device=device).manual_seed(91 + rank)
            local = torch.randn(
                batch,
                args.local_width,
                generator=generator,
                device=device,
                dtype=dtype,
            ).contiguous()
            paths: dict[str, Callable[[], Tensor]] = {}
            owners: list[object] = []

            stock = _BigAllGatherDecoder(
                decoder,
                rows=batch,
                local_width=args.local_width,
                group=None,
                materialize_token_major=True,
            )
            rank_major = _BigAllGatherDecoder(
                decoder,
                rows=batch,
                local_width=args.local_width,
                group=None,
                materialize_token_major=False,
            )
            paths["stock_big"] = (
                lambda stock=stock, coordinates=local: stock(coordinates)
            )
            paths["rank_major_big"] = (
                lambda rank_major=rank_major, coordinates=local: rank_major(coordinates)
            )
            owners.extend((stock, rank_major))
            for waves in args.waves:
                pipeline = PipelinedAllGatherDecoder(
                    decoder,
                    rows=batch,
                    local_width=args.local_width,
                    waves=waves,
                    group=None,
                )
                paths[f"pipeline_{waves}"] = (
                    lambda pipeline=pipeline, coordinates=local: pipeline(coordinates)
                )
                owners.append(pipeline)

            reference = paths["stock_big"]()
            torch.cuda.synchronize(device)
            for path, run_once in paths.items():
                observed = run_once()
                torch.cuda.synchronize(device)
                error = relative_error(observed, reference)
                if error["relative_l2"] > 0.03:
                    raise AssertionError(
                        f"{path} failed correctness for batch {batch}: {error}"
                    )

                for mode in args.execution_modes:
                    graph = None
                    capture_stream = None
                    measured_call = run_once
                    if mode == "cuda_graph":
                        measured_call, observed, capture_stream, graph = capture_graph(
                            run_once,
                            device=device,
                        )
                    observed, samples = measure(
                        measured_call,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        device=device,
                        timing_stream=capture_stream,
                    )
                    error = relative_error(observed, reference)
                    record = {
                        "batch": batch,
                        "path": path,
                        "execution_mode": mode,
                        **summarize(samples),
                        **error,
                    }
                    records.append(record)
                    if rank == 0:
                        print(json.dumps(record, sort_keys=True), flush=True)
                    del graph, capture_stream, measured_call
                    torch.cuda.synchronize(device)
                    gc.collect()
                    dist.barrier()

            del paths, owners, stock, rank_major, local, reference
            torch.cuda.synchronize(device)
            gc.collect()

    payload = {
        "format": "basisserve.pipelined_allgather_decoder.v1",
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "world_size": world_size,
        },
        "configuration": {
            "batches": list(args.batches),
            "local_width": args.local_width,
            "global_width": world_size * args.local_width,
            "hidden_size": args.hidden_size,
            "waves": list(args.waves),
            "dtype": str(dtype),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "execution_modes": list(args.execution_modes),
        },
        "records": records,
    }
    if rank == 0:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {output}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
