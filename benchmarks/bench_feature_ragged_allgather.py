#!/usr/bin/env python3
"""TP correctness and latency harness for the C1 ragged output boundary.

This benchmark measures collective plus one replicated decoder GEMM. Sequence
length and compact-V attention are deliberately outside this boundary; use the
artificial source-rank skew only as a transport readiness stress test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from basisserve.kernels.feature_ragged_allgather import (  # noqa: E402
    FeatureRaggedCommunicator,
)
from basisserve.kernels.ragged_allgather import (  # noqa: E402
    RaggedNcclCommunicator,
    StaticRaggedPlan,
)


_BACKENDS = (
    "existing_direct",
    "existing_registered",
    "feature_direct",
    "feature_rma",
    "padded_allgather",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-ranks",
        default="32,32,48,64,64,80,96,128",
        help="one C1 latent rank per TP source; wire width is heads-per-source * rank",
    )
    parser.add_argument("--heads-per-source", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument(
        "--backends",
        default=",".join(_BACKENDS),
        help=f"comma-separated subset of {','.join(_BACKENDS)}",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--artificial-skew-us-per-rank",
        type=float,
        default=0.0,
        help="host-side readiness delay: local source rank times this value",
    )
    parser.add_argument(
        "--feature-major-input",
        action="store_true",
        help="model an attention kernel writing [local_width, tokens] directly",
    )
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def parse_int_tuple(value: str, *, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer list") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{name} must contain positive integers, got {parsed}")
    return parsed


def padded_reference_parts(
    local_token_major: torch.Tensor,
    plan: StaticRaggedPlan,
) -> list[torch.Tensor]:
    max_width = max(plan.source_widths)
    padded = torch.zeros(
        local_token_major.shape[0],
        max_width,
        device=local_token_major.device,
        dtype=local_token_major.dtype,
    )
    padded[:, : local_token_major.shape[1]].copy_(local_token_major)
    gathered = [torch.empty_like(padded) for _ in plan.source_widths]
    dist.all_gather(gathered, padded)
    return [
        tensor[:, :width]
        for tensor, width in zip(gathered, plan.source_widths, strict=True)
    ]


def make_padded_baseline(
    local_token_major: torch.Tensor,
    decoder: torch.Tensor,
    plan: StaticRaggedPlan,
) -> Callable[[], torch.Tensor]:
    tokens = int(local_token_major.shape[0])
    world_size = len(plan.source_widths)
    max_width = max(plan.source_widths)
    local_padded = torch.zeros(
        tokens,
        max_width,
        dtype=local_token_major.dtype,
        device=local_token_major.device,
    )
    local_padded[:, : local_token_major.shape[1]].copy_(local_token_major)
    gathered_rank_major = torch.empty(
        world_size * tokens,
        max_width,
        dtype=local_token_major.dtype,
        device=local_token_major.device,
    )
    token_major_workspace = torch.empty(
        tokens,
        world_size,
        max_width,
        dtype=local_token_major.dtype,
        device=local_token_major.device,
    )
    padded_decoder = torch.zeros(
        world_size,
        max_width,
        decoder.shape[1],
        dtype=decoder.dtype,
        device=decoder.device,
    )
    offset = 0
    for source, width in enumerate(plan.source_widths):
        padded_decoder[source, :width].copy_(decoder[offset : offset + width])
        offset += width
    padded_decoder_2d = padded_decoder.reshape(world_size * max_width, decoder.shape[1])

    def run_once() -> torch.Tensor:
        dist.all_gather_into_tensor(gathered_rank_major, local_padded)
        rank_major_3d = gathered_rank_major.view(world_size, tokens, max_width)
        token_major_workspace.copy_(rank_major_3d.permute(1, 0, 2))
        return torch.mm(
            token_major_workspace.view(tokens, world_size * max_width),
            padded_decoder_2d,
        )

    return run_once


def global_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def benchmark_callable(
    run_once: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    readiness_delay_seconds: float,
) -> tuple[torch.Tensor, list[float]]:
    def run_with_skew() -> torch.Tensor:
        if readiness_delay_seconds > 0:
            time.sleep(readiness_delay_seconds)
        return run_once()

    for _ in range(warmup):
        run_with_skew()
    torch.cuda.synchronize(device)
    dist.barrier()

    samples: list[float] = []
    output = None
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = run_with_skew()
        end.record()
        end.synchronize()
        samples.append(global_max(start.elapsed_time(end), device))
    assert output is not None
    return output, samples


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_samples(
    samples: list[float],
    *,
    tokens: int,
    total_width: int,
    hidden: int,
) -> dict[str, float]:
    p50 = statistics.median(samples)
    p90 = percentile(samples, 0.90)
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": p50,
        "p90_ms": p90,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "tokens_per_second_at_p50": tokens * 1000.0 / p50,
        "decoder_tflops_at_p50": 2.0 * tokens * total_width * hidden / (p50 * 1e9),
    }


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    source_ranks = parse_int_tuple(args.source_ranks, name="--source-ranks")
    if len(source_ranks) != world_size:
        raise ValueError(
            f"--source-ranks has {len(source_ranks)} entries but world size is {world_size}"
        )
    plan = StaticRaggedPlan.from_head_ranks(
        source_ranks,
        heads_per_source=args.heads_per_source,
    )
    backends = tuple(item for item in args.backends.split(",") if item)
    if not backends or len(set(backends)) != len(backends):
        raise ValueError("--backends must contain unique backend names")
    unknown = tuple(item for item in backends if item not in _BACKENDS)
    if unknown:
        raise ValueError(f"unknown backends {unknown}; expected a subset of {_BACKENDS}")

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(1234 + rank)
    local_token_major = torch.randn(
        args.tokens,
        plan.source_widths[rank],
        device=device,
        dtype=dtype,
    )
    local_feature = local_token_major.transpose(0, 1).contiguous()
    feature_input = local_feature if args.feature_major_input else local_token_major

    torch.manual_seed(777)
    decoder = torch.randn(plan.total_width, args.hidden, device=device, dtype=dtype)
    reference_parts = padded_reference_parts(local_token_major, plan)
    reference = torch.cat(reference_parts, dim=1) @ decoder

    existing: RaggedNcclCommunicator | None = None
    existing_unavailable_reason: str | None = None
    if any(backend.startswith("existing_") for backend in backends):
        try:
            existing = RaggedNcclCommunicator.from_process_group(device=device)
        except Exception as error:
            detail = str(error).strip().splitlines()
            suffix = detail[-1] if detail else repr(error)
            existing_unavailable_reason = f"{type(error).__name__}: {suffix}"
    feature = FeatureRaggedCommunicator.from_distributed(device=device)
    rma_unavailable_reason: str | None = None
    if "feature_rma" in backends:
        if not feature.rma_available:
            rma_unavailable_reason = (
                f"NCCL {feature.nccl_version} does not expose 2.29+ host RMA"
            )
        else:
            try:
                feature.prepare_rma(
                    tokens=args.tokens,
                    max_total_width=plan.total_width,
                    dtype=dtype,
                )
            except Exception as error:
                detail = str(error).strip().splitlines()
                suffix = detail[-1] if detail else repr(error)
                rma_unavailable_reason = f"{type(error).__name__}: {suffix}"

    padded_run = make_padded_baseline(local_token_major, decoder, plan)
    calls: dict[str, Callable[[], torch.Tensor]] = {
        "feature_direct": lambda: feature.all_gather_decode(
            feature_input,
            plan,
            decoder,
            backend="feature_direct",
            local_is_feature_major=args.feature_major_input,
        ),
        "feature_rma": lambda: feature.all_gather_decode(
            feature_input,
            plan,
            decoder,
            backend="feature_rma",
            local_is_feature_major=args.feature_major_input,
        ),
        "padded_allgather": padded_run,
    }
    if existing is not None:
        calls["existing_direct"] = lambda: existing.all_gather_decode(
            local_token_major,
            decoder,
            plan,
            algorithm="direct",
            registered=False,
        )
        calls["existing_registered"] = lambda: existing.all_gather_decode(
            local_token_major,
            decoder,
            plan,
            algorithm="direct",
            registered=True,
        )

    element_size = torch.empty((), dtype=dtype).element_size()
    compact_received_bytes = tuple(
        args.tokens * (plan.total_width - width) * element_size
        for width in plan.source_widths
    )
    padded_received_bytes = (
        args.tokens * (world_size - 1) * max(plan.source_widths) * element_size
    )
    result: dict[str, object] = {
        "format": "basisserve.feature_ragged_tp_decode.v2",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nccl_version": feature.nccl_version,
        "world_size": world_size,
        "tokens": args.tokens,
        "hidden": args.hidden,
        "dtype": str(dtype),
        "source_ranks": source_ranks,
        "heads_per_source": args.heads_per_source,
        "source_widths": plan.source_widths,
        "total_width": plan.total_width,
        "feature_major_input": args.feature_major_input,
        "artificial_skew_us_per_rank": args.artificial_skew_us_per_rank,
        "compact_received_bytes_by_rank": compact_received_bytes,
        "padded_received_bytes_per_rank": padded_received_bytes,
        "backends": {},
    }

    try:
        for backend in backends:
            if backend.startswith("existing_") and existing is None:
                result["backends"][backend] = {
                    "status": "skipped",
                    "reason": existing_unavailable_reason,
                }
                if rank == 0:
                    print(f"SKIP {backend}: {existing_unavailable_reason}")
                continue
            if backend == "feature_rma" and rma_unavailable_reason is not None:
                result["backends"][backend] = {
                    "status": "skipped",
                    "reason": rma_unavailable_reason,
                }
                if rank == 0:
                    print(f"SKIP {backend}: {result['backends'][backend]['reason']}")
                continue

            dist.barrier()
            output, samples = benchmark_callable(
                calls[backend],
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                readiness_delay_seconds=(
                    source_ranks[rank] * args.artificial_skew_us_per_rank * 1e-6
                ),
            )
            torch.testing.assert_close(output, reference, atol=args.atol, rtol=args.rtol)
            decoder_input_width = (
                plan.padded_total_width
                if backend == "padded_allgather"
                else plan.total_width
            )
            metrics = summarize_samples(
                samples,
                tokens=args.tokens,
                total_width=decoder_input_width,
                hidden=args.hidden,
            )
            result["backends"][backend] = {
                "status": "passed",
                "decoder_input_width": decoder_input_width,
                **metrics,
            }
            if rank == 0:
                print(
                    f"{backend}: correct; p50={metrics['p50_ms']:.3f} ms "
                    f"p90={metrics['p90_ms']:.3f} ms "
                    f"throughput={metrics['tokens_per_second_at_p50']:.1f} token/s"
                )
    finally:
        dist.barrier()
        feature.close()
        if existing is not None:
            existing.close()
        dist.destroy_process_group()

    if rank == 0 and args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
