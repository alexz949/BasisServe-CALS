#!/usr/bin/env python3
"""Benchmark replicated, output-sharded, restored, and FP8 TP decoders."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import sys
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch import Tensor  # noqa: E402

from basisserve.kernels.fp8_wire import (  # noqa: E402
    FP8_E4M3_DTYPE,
    FP8_E4M3_MAX,
    quantize_e4m3_static,
    quantize_e4m3_tensorwise_col_major,
    scaled_mm_e4m3_static,
)
from basisserve.kernels.output_sharded_decoder import (  # noqa: E402
    OutputShardedDecoder,
)


FORMAT = "basisserve.output_sharded_decoder_benchmark.v1"


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
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    return args


class _ReplicatedBF16Decoder:
    def __init__(
        self,
        decoder: Tensor,
        *,
        rows: int,
        local_width: int,
        group,
    ) -> None:
        self.decoder = decoder
        self.rows = int(rows)
        self.local_width = int(local_width)
        self.group = group
        self.processes = dist.get_world_size(group)
        self.gathered = torch.empty(
            self.processes * self.rows,
            self.local_width,
            device=decoder.device,
            dtype=decoder.dtype,
        )
        self.token_major = torch.empty(
            self.rows,
            self.processes * self.local_width,
            device=decoder.device,
            dtype=decoder.dtype,
        )
        self.output = torch.empty(
            self.rows,
            int(decoder.shape[1]),
            device=decoder.device,
            dtype=decoder.dtype,
        )

    def __call__(self, local_coordinates: Tensor) -> Tensor:
        work = dist.all_gather_into_tensor(
            self.gathered,
            local_coordinates,
            group=self.group,
            async_op=True,
        )
        work.wait()
        self.token_major.view(
            self.rows,
            self.processes,
            self.local_width,
        ).copy_(
            self.gathered.view(
                self.processes,
                self.rows,
                self.local_width,
            ).permute(1, 0, 2)
        )
        torch.mm(self.token_major, self.decoder, out=self.output)
        return self.output


class _ReplicatedFP8Decoder:
    def __init__(
        self,
        decoder: Tensor,
        *,
        rows: int,
        local_width: int,
        wire_scale: Tensor,
        group,
    ) -> None:
        self.rows = int(rows)
        self.local_width = int(local_width)
        self.group = group
        self.processes = dist.get_world_size(group)
        self.device = decoder.device
        self.wire_scale = wire_scale
        self.decoder, self.decoder_scale = quantize_e4m3_tensorwise_col_major(
            decoder
        )
        self.local_codes = torch.empty(
            self.rows,
            self.local_width,
            device=self.device,
            dtype=torch.uint8,
        )
        self.gathered_codes = torch.empty(
            self.processes * self.rows,
            self.local_width,
            device=self.device,
            dtype=torch.uint8,
        )
        self.token_major_codes = torch.empty(
            self.rows,
            self.processes * self.local_width,
            device=self.device,
            dtype=torch.uint8,
        )

    def __call__(self, local_coordinates: Tensor) -> Tensor:
        quantized = quantize_e4m3_static(local_coordinates, self.wire_scale)
        self.local_codes.copy_(quantized.view(torch.uint8))
        work = dist.all_gather_into_tensor(
            self.gathered_codes,
            self.local_codes,
            group=self.group,
            async_op=True,
        )
        work.wait()
        self.token_major_codes.view(
            self.rows,
            self.processes,
            self.local_width,
        ).copy_(
            self.gathered_codes.view(
                self.processes,
                self.rows,
                self.local_width,
            ).permute(1, 0, 2)
        )
        return scaled_mm_e4m3_static(
            self.token_major_codes.view(FP8_E4M3_DTYPE),
            self.decoder,
            left_scale=self.wire_scale,
            right_scale=self.decoder_scale,
            out_dtype=torch.bfloat16,
        )


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


def distributed_error(observed: Tensor, reference: Tensor) -> dict[str, float]:
    difference = observed.float() - reference.float()
    metrics = torch.stack(
        (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1.0e-12),
            difference.abs().amax(),
        )
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    return {
        "relative_l2": float(metrics[0]),
        "maximum_absolute": float(metrics[1]),
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
    samples = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(samples, op=dist.ReduceOp.MAX)
    return output, samples.cpu().tolist()


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
        with torch.cuda.stream(capture_stream):
            graph.replay()
        return output

    return replay, output, capture_stream, graph


def communication_bytes(
    *,
    batch: int,
    local_width: int,
    hidden_size: int,
    processes: int,
) -> dict[str, int]:
    bf16_bytes = 2
    replicated = (processes - 1) * batch * local_width * bf16_bytes
    reduce_scatter = (
        (processes - 1) * batch * hidden_size * bf16_bytes // processes
    )
    return {
        "replicated_bf16": replicated,
        "output_sharded_bf16": reduce_scatter,
        "restored_bf16": 2 * reduce_scatter,
        "replicated_fp8": replicated // bf16_bytes,
    }


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    processes = dist.get_world_size()
    dtype = torch.bfloat16
    global_width = processes * args.local_width
    if args.hidden_size % processes:
        raise ValueError("hidden size must be divisible by the process count")
    if global_width % 16 or args.hidden_size % 16:
        raise ValueError("FP8 decoder dimensions must be divisible by 16")

    torch.manual_seed(20260828)
    decoder = (
        torch.randn(
            global_width,
            args.hidden_size,
            device=device,
            dtype=dtype,
        )
        / global_width**0.5
    ).contiguous()
    local_decoder = decoder.narrow(
        0,
        rank * args.local_width,
        args.local_width,
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
            wire_amax = local.float().abs().amax()
            dist.all_reduce(wire_amax, op=dist.ReduceOp.MAX)
            wire_scale = (wire_amax / FP8_E4M3_MAX).clamp_min(
                torch.finfo(torch.float32).tiny
            )

            replicated = _ReplicatedBF16Decoder(
                decoder,
                rows=batch,
                local_width=args.local_width,
                group=None,
            )
            output_sharded = OutputShardedDecoder(
                local_decoder,
                rows=batch,
                reconstruct=False,
                group=None,
            )
            restored = OutputShardedDecoder(
                local_decoder,
                rows=batch,
                reconstruct=True,
                group=None,
            )
            fp8 = _ReplicatedFP8Decoder(
                decoder,
                rows=batch,
                local_width=args.local_width,
                wire_scale=wire_scale,
                group=None,
            )
            paths: dict[str, Callable[[], Tensor]] = {
                "replicated_bf16": (
                    lambda module=replicated, coordinates=local: module(coordinates)
                ),
                "output_sharded_bf16": (
                    lambda module=output_sharded, coordinates=local: module(coordinates)
                ),
                "restored_bf16": (
                    lambda module=restored, coordinates=local: module(coordinates)
                ),
                "replicated_fp8": (
                    lambda module=fp8, coordinates=local: module(coordinates)
                ),
            }

            reference = replicated(local).clone()
            torch.cuda.synchronize(device)
            local_hidden = args.hidden_size // processes
            references = {
                "replicated_bf16": reference,
                "output_sharded_bf16": reference.narrow(
                    1,
                    rank * local_hidden,
                    local_hidden,
                ).contiguous(),
                "restored_bf16": reference,
                "replicated_fp8": reference,
            }
            for path, run_once in paths.items():
                observed = run_once()
                torch.cuda.synchronize(device)
                error = distributed_error(observed, references[path])
                tolerance = 0.08 if path == "replicated_fp8" else 0.03
                if error["relative_l2"] > tolerance:
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
                    error = distributed_error(observed, references[path])
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

            del paths, references, reference, local
            del replicated, output_sharded, restored, fp8
            torch.cuda.synchronize(device)
            gc.collect()

    payload = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "world_size": processes,
        },
        "configuration": {
            "batches": list(args.batches),
            "local_width": args.local_width,
            "global_width": global_width,
            "hidden_size": args.hidden_size,
            "dtype": str(dtype),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "execution_modes": list(args.execution_modes),
            "fp8_wire_scale": "global tensorwise amax",
            "fp8_decoder_scale": "tensorwise",
        },
        "algorithmic_communication_bytes_per_rank": {
            str(batch): communication_bytes(
                batch=batch,
                local_width=args.local_width,
                hidden_size=args.hidden_size,
                processes=processes,
            )
            for batch in args.batches
        },
        "decoder_storage_bytes_per_rank": {
            "replicated_bf16": decoder.numel() * decoder.element_size(),
            "output_sharded_bf16": (
                local_decoder.numel() * local_decoder.element_size()
            ),
            "restored_bf16": local_decoder.numel() * local_decoder.element_size(),
            "replicated_fp8": decoder.numel(),
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
