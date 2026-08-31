#!/usr/bin/env python3
"""Correctness and latency smoke test for packed feature-major AllGather.

Run on one node with one process per GPU, for example::

    torchrun --standalone --nproc-per-node=4 \
      tests/distributed_feature_ragged_allgather_smoke.py \
      --source-widths 512,512,512,512 --rows 1,128,4096

The reference path intentionally models the implementation being replaced:
pad to the maximum source width, use PyTorch/NCCL AllGather, globally unpack
into token-major order, then launch the decoder GEMM.  The packed path only
transposes the local source block and receives every peer directly into the
final feature-major decoder arena.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
)
from basisserve.kernels.ragged_allgather import StaticRaggedPlan  # noqa: E402


def _positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"expected positive comma-separated integers, got {value!r}")
    return values


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _maximum_across_ranks(value: float, device: torch.device) -> float:
    carrier = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(carrier, op=dist.ReduceOp.MAX)
    return float(carrier)


def _time_cuda(
    operation: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    dist.barrier()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        samples.append(_maximum_across_ranks(start.elapsed_time(stop), device))
    return {
        "minimum_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "maximum_ms": max(samples),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-widths", default="512,512,512,512")
    parser.add_argument("--rows", default="1,128,4096")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    if args.hidden_size <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("hidden size/repeats must be positive and warmup nonnegative")
    widths = _positive_ints(args.source_widths)
    rows_values = _positive_ints(args.rows)
    dtype = _dtype(args.dtype)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if len(widths) != world_size:
        raise ValueError(
            f"received {len(widths)} source widths for world size {world_size}"
        )

    plan = StaticRaggedPlan.from_source_widths(widths)
    maximum_width = max(widths)
    generator = torch.Generator(device=device).manual_seed(20260825)
    decoder = torch.randn(
        plan.total_width,
        args.hidden_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    communicator = FeatureRaggedCommunicator.from_distributed(device=device)
    nccl_version = communicator.nccl_version
    communicator.configure_direct_workspace(
        tokens=max(rows_values),
        max_total_width=plan.total_width,
        dtype=dtype,
    )

    records: list[dict[str, object]] = []
    try:
        for rows in rows_values:
            local = torch.randn(
                rows,
                widths[rank],
                dtype=dtype,
                device=device,
                generator=generator,
            )
            padded_local = torch.zeros(
                rows,
                maximum_width,
                dtype=dtype,
                device=device,
            )
            padded_local[:, : widths[rank]].copy_(local)
            rank_major = torch.empty(
                world_size * rows,
                maximum_width,
                dtype=dtype,
                device=device,
            )
            token_major = torch.empty(
                rows,
                plan.total_width,
                dtype=dtype,
                device=device,
            )

            def reference() -> torch.Tensor:
                padded_local[:, : widths[rank]].copy_(local)
                dist.all_gather_into_tensor(rank_major, padded_local)
                by_source = rank_major.view(world_size, rows, maximum_width)
                for source, (offset, width) in enumerate(
                    zip(plan.offsets, plan.source_widths, strict=True)
                ):
                    token_major[:, offset : offset + width].copy_(
                        by_source[source, :, :width]
                    )
                return torch.mm(token_major, decoder)

            arena = communicator.gather(local, plan, backend="feature_direct")
            reference()
            expected_feature_major = token_major.transpose(0, 1).contiguous()
            torch.testing.assert_close(arena, expected_feature_major, rtol=0, atol=0)

            packed_output = communicator.all_gather_decode(
                local,
                plan,
                decoder,
                backend="feature_direct",
            )
            reference_output = reference()
            difference = (packed_output.float() - reference_output.float()).abs()
            relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
                reference_output.float()
            ).clamp_min(1e-12)
            errors = torch.stack((difference.max(), relative_l2)).to(torch.float64)
            dist.all_reduce(errors, op=dist.ReduceOp.MAX)
            relative_tolerance = 2e-2 if dtype == torch.bfloat16 else 5e-3
            if float(errors[1]) > relative_tolerance:
                raise AssertionError(
                    f"relative decoder error {float(errors[1]):.6e} exceeds "
                    f"{relative_tolerance:.6e}"
                )

            local_feature_major = communicator.direct_local_feature_major_view(
                plan,
                tokens=rows,
                dtype=dtype,
            )
            local_feature_major.copy_(local.transpose(0, 1))

            def direct_slot_decode() -> torch.Tensor:
                return communicator.all_gather_decode(
                    local_feature_major,
                    plan,
                    decoder,
                    backend="feature_direct",
                    local_is_feature_major=True,
                )

            direct_slot_output = direct_slot_decode()
            direct_slot_difference = (
                direct_slot_output.float() - reference_output.float()
            ).abs()
            direct_slot_relative_l2 = torch.linalg.vector_norm(
                direct_slot_difference
            ) / torch.linalg.vector_norm(reference_output.float()).clamp_min(1e-12)
            direct_slot_errors = torch.stack(
                (direct_slot_difference.max(), direct_slot_relative_l2)
            ).to(torch.float64)
            dist.all_reduce(direct_slot_errors, op=dist.ReduceOp.MAX)
            if float(direct_slot_errors[1]) > relative_tolerance:
                raise AssertionError(
                    "direct-slot relative decoder error "
                    f"{float(direct_slot_errors[1]):.6e} exceeds "
                    f"{relative_tolerance:.6e}"
                )

            reference_timing = _time_cuda(
                reference,
                warmup=args.warmup,
                repeats=args.repeats,
                device=device,
            )
            packed_timing = _time_cuda(
                lambda: communicator.all_gather_decode(
                    local,
                    plan,
                    decoder,
                    backend="feature_direct",
                ),
                warmup=args.warmup,
                repeats=args.repeats,
                device=device,
            )
            direct_slot_timing = _time_cuda(
                direct_slot_decode,
                warmup=args.warmup,
                repeats=args.repeats,
                device=device,
            )
            records.append(
                {
                    "rows": rows,
                    "reference_padded_unpack_decode": reference_timing,
                    "packed_feature_major_decode": packed_timing,
                    "direct_slot_feature_major_decode": direct_slot_timing,
                    "median_speedup": (
                        reference_timing["median_ms"] / packed_timing["median_ms"]
                    ),
                    "direct_slot_over_packed_speedup": (
                        packed_timing["median_ms"]
                        / direct_slot_timing["median_ms"]
                    ),
                    "maximum_absolute_error": float(errors[0]),
                    "maximum_relative_l2_error": float(errors[1]),
                    "direct_slot_maximum_absolute_error": float(
                        direct_slot_errors[0]
                    ),
                    "direct_slot_maximum_relative_l2_error": float(
                        direct_slot_errors[1]
                    ),
                }
            )
            if rank == 0:
                print(json.dumps(records[-1]), flush=True)
    finally:
        dist.barrier()
        communicator.close()

    if rank == 0:
        payload = {
            "format": "basisserve.feature_ragged_allgather_smoke.v1",
            "status": "passed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "world_size": world_size,
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl_version": nccl_version,
            "dtype": args.dtype,
            "source_widths": list(widths),
            "packed_transport": (
                "native_in_place_nccl_allgather"
                if len(set(widths)) == 1
                else "exact_width_nccl_ring"
            ),
            "total_width": plan.total_width,
            "padded_total_width": plan.padded_total_width,
            "padding_overhead": plan.padding_overhead,
            "hidden_size": args.hidden_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "records": records,
        }
        rendered = json.dumps(payload, indent=2) + "\n"
        print(rendered, flush=True)
        if args.output_json:
            output = Path(args.output_json).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(rendered, encoding="utf-8")
            os.replace(temporary, output)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
