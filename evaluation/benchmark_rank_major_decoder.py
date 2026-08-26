#!/usr/bin/env python3
"""Validate and benchmark fused rank-major C1 decoder GEMM on one GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402
import triton  # noqa: E402

from basisserve.kernels.rank_major_decoder import (  # noqa: E402
    _rank_major_decoder_kernel,
    rank_major_decoder,
    reference_rank_major_decoder,
)


def _parse_ints(value: str) -> tuple[int, ...]:
    selected = tuple(int(item) for item in value.split(",") if item)
    if not selected or any(item <= 0 for item in selected):
        raise ValueError(f"invalid positive integer list {value!r}")
    return selected


def _time_ms(function: Callable[[], Tensor], *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    measurements: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        stop.record()
        stop.synchronize()
        measurements.append(float(start.elapsed_time(stop)))
    return statistics.median(measurements)


def _materialized_function(
    rank_major: Tensor,
    decoder: Tensor,
    token_major: Tensor,
    *,
    processes: int,
) -> Tensor:
    rows = int(rank_major.shape[0]) // processes
    local_width = int(rank_major.shape[1])
    token_major.copy_(
        rank_major.view(processes, rows, local_width).permute(1, 0, 2)
    )
    return torch.mm(token_major.reshape(rows, -1), decoder)


def _segmented_cublas_function(
    rank_major: Tensor,
    decoder: Tensor,
    *,
    processes: int,
) -> Tensor:
    rows = int(rank_major.shape[0]) // processes
    local_width = int(rank_major.shape[1])
    blocks = rank_major.view(processes, rows, local_width)
    output = torch.mm(blocks[0], decoder[:local_width])
    for process in range(1, processes):
        start = process * local_width
        output.addmm_(blocks[process], decoder[start : start + local_width])
    return output


def _configured_triton_function(
    rank_major: Tensor,
    decoder: Tensor,
    output: Tensor,
    *,
    processes: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> Tensor:
    rows = int(rank_major.shape[0]) // processes
    local_width = int(rank_major.shape[1])
    output_width = int(decoder.shape[1])
    _rank_major_decoder_kernel[
        (triton.cdiv(rows, block_m), triton.cdiv(output_width, block_n))
    ](
        rank_major,
        decoder,
        output,
        rows,
        output_width,
        LOCAL_WIDTH=local_width,
        PROCESSES=processes,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=_parse_ints, default=(128, 4096, 32768))
    parser.add_argument("--local-widths", type=_parse_ints, default=(512, 896))
    parser.add_argument("--output-width", type=int, default=4096)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(20260825)

    records: list[dict[str, float | int]] = []
    for local_width in args.local_widths:
        decoder = (
            torch.randn(
                args.processes * local_width,
                args.output_width,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            / (args.processes * local_width) ** 0.5
        ).contiguous()

        check_rows = 129
        check_input = (
            torch.randn(
                args.processes * check_rows,
                local_width,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            / local_width**0.5
        ).contiguous()
        expected = reference_rank_major_decoder(
            check_input,
            decoder,
            processes=args.processes,
        )
        observed = rank_major_decoder(
            check_input,
            decoder,
            processes=args.processes,
        )
        segmented_observed = _segmented_cublas_function(
            check_input,
            decoder,
            processes=args.processes,
        )
        error = (observed.float() - expected.float()).abs()
        segmented_error = (segmented_observed.float() - expected.float()).abs()
        relative_l2 = float(
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(expected.float())
        )
        maximum_absolute = float(error.max())
        segmented_relative_l2 = float(
            torch.linalg.vector_norm(segmented_error)
            / torch.linalg.vector_norm(expected.float())
        )
        segmented_maximum_absolute = float(segmented_error.max())
        if (
            not torch.isfinite(observed).all()
            or not torch.isfinite(segmented_observed).all()
            or relative_l2 > 0.02
            or segmented_relative_l2 > 0.02
        ):
            raise RuntimeError(
                f"rank-major decoder numerical check failed for K_local={local_width}: "
                f"direct_relative_l2={relative_l2}, "
                f"segmented_relative_l2={segmented_relative_l2}"
            )

        for rows in args.rows:
            rank_major = (
                torch.randn(
                    args.processes * rows,
                    local_width,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                / local_width**0.5
            ).contiguous()
            token_major = torch.empty(
                rows,
                args.processes,
                local_width,
                device=device,
                dtype=dtype,
            )
            direct_ms = _time_ms(
                lambda: rank_major_decoder(
                    rank_major,
                    decoder,
                    processes=args.processes,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            materialized_ms = _time_ms(
                lambda: _materialized_function(
                    rank_major,
                    decoder,
                    token_major,
                    processes=args.processes,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            segmented_ms = _time_ms(
                lambda: _segmented_cublas_function(
                    rank_major,
                    decoder,
                    processes=args.processes,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            record = {
                "rows": rows,
                "local_width": local_width,
                "total_reduction_width": args.processes * local_width,
                "output_width": args.output_width,
                "direct_rank_major_ms": direct_ms,
                "materialize_then_cublas_ms": materialized_ms,
                "segmented_cublas_ms": segmented_ms,
                "direct_speedup": materialized_ms / direct_ms,
                "segmented_speedup": materialized_ms / segmented_ms,
                "eliminated_scratch_gib": (
                    rows * args.processes * local_width * 2 / 2**30
                ),
                "check_relative_l2": relative_l2,
                "check_maximum_absolute": maximum_absolute,
                "segmented_check_relative_l2": segmented_relative_l2,
                "segmented_check_maximum_absolute": segmented_maximum_absolute,
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

            if args.tune and rows >= 4096:
                configurations = (
                    (64, 64, 32, 4, 4),
                    (64, 128, 64, 4, 4),
                    (64, 128, 64, 8, 4),
                    (64, 256, 32, 8, 4),
                    (128, 128, 32, 8, 4),
                    (128, 128, 64, 8, 3),
                    (128, 256, 64, 8, 3),
                )
                tuning_output = torch.empty(
                    rows,
                    args.output_width,
                    device=device,
                    dtype=dtype,
                )
                for block_m, block_n, block_k, warps, stages in configurations:
                    configured_ms = _time_ms(
                        lambda bm=block_m, bn=block_n, bk=block_k, w=warps, s=stages: (
                            _configured_triton_function(
                                rank_major,
                                decoder,
                                tuning_output,
                                processes=args.processes,
                                block_m=bm,
                                block_n=bn,
                                block_k=bk,
                                num_warps=w,
                                num_stages=s,
                            )
                        ),
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    tuning_record = {
                        "rows": rows,
                        "local_width": local_width,
                        "block_m": block_m,
                        "block_n": block_n,
                        "block_k": block_k,
                        "num_warps": warps,
                        "num_stages": stages,
                        "configured_triton_ms": configured_ms,
                        "speedup_over_materialized": materialized_ms / configured_ms,
                    }
                    print(json.dumps(tuning_record, sort_keys=True), flush=True)
                    record.setdefault("tuning", []).append(tuning_record)

    payload = {
        "format": "basisserve.rank_major_decoder_benchmark.v1",
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "dtype": str(dtype).removeprefix("torch."),
        },
        "protocol": {
            "processes": args.processes,
            "rows": list(args.rows),
            "local_widths": list(args.local_widths),
            "output_width": args.output_width,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "tune": args.tune,
        },
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output_json)


if __name__ == "__main__":
    main()
