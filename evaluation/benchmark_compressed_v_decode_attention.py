#!/usr/bin/env python3
"""Autotune architecture-specialized CUDA split-K decode against Triton."""

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
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from basisserve.kernels.compressed_v_decode_attention import (  # noqa: E402
    compressed_v_decode_attention_cuda,
    compressed_v_decode_attention_triton,
)


FORMAT = "basisserve.compressed_v_decode_attention_benchmark.v5"


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"expected positive comma-separated integers, got {value!r}")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"duplicate values are not allowed, got {parsed}")
    return parsed


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(map(float, values))
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _measure(
    function: Callable[[], Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    result: Tensor | None = None
    for _ in range(warmup):
        result = function()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        result = function()
        end.record()
    torch.cuda.synchronize()
    if result is None:
        raise AssertionError("timed function produced no output")
    timings = [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    return {
        "minimum_ms": min(timings),
        "p50_ms": statistics.median(timings),
        "p95_ms": _quantile(timings, 0.95),
        "maximum_ms": max(timings),
    }


def _error_against(reference: Tensor, observed: Tensor) -> dict[str, float | int]:
    difference = observed.float() - reference.float()
    return {
        "maximum_absolute_error": float(difference.abs().max()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)
        ),
        "nonfinite": int(not torch.isfinite(observed).all()),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,8,32,128")
    parser.add_argument("--contexts", default="128,512,1024,4096,8192")
    parser.add_argument("--value-dims", default="32,48,64,80,96,112")
    parser.add_argument("--splits", default="1,2,4,8,16,32")
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--qk-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    batches = _parse_positive_ints(args.batches)
    contexts = _parse_positive_ints(args.contexts)
    value_dims = _parse_positive_ints(args.value_dims)
    splits = _parse_positive_ints(args.splits)
    if args.query_heads != 4 * args.kv_heads:
        raise ValueError("CUDA decode requires four query heads per KV head")
    if args.qk_dim != 128:
        raise ValueError("CUDA decode requires Q/K head dimension 128")
    if any(rank not in (32, 48, 64, 80, 96, 112) for rank in value_dims):
        raise ValueError("CUDA decode supports ranks {32,48,64,80,96,112}")
    if any(split not in (1, 2, 4, 8, 16, 32) for split in splits):
        raise ValueError("split count must lie in {1,2,4,8,16,32}")
    if args.warmup <= 0 or args.iterations <= 0:
        raise ValueError("warmup and iterations must be positive")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((8, 0), (8, 9), (9, 0)):
        raise RuntimeError("this autotuner requires an SM80, SM89, or SM90 GPU")
    dtype = torch.bfloat16
    scale = args.qk_dim**-0.5
    records: list[dict[str, object]] = []

    for value_dim in value_dims:
        for batch in batches:
            for context in contexts:
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + value_dim * 1_000_000 + batch * 10_000 + context
                )
                query = torch.randn(
                    batch,
                    args.query_heads,
                    1,
                    args.qk_dim,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                key = torch.randn(
                    batch,
                    args.kv_heads,
                    context,
                    args.qk_dim,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                value = torch.randn(
                    batch,
                    args.kv_heads,
                    context,
                    value_dim,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                reference = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale,
                    enable_gqa=True,
                )
                triton_output = compressed_v_decode_attention_triton(
                    query,
                    key,
                    value,
                    scale=scale,
                )
                triton_error = _error_against(reference, triton_output)
                if triton_error["nonfinite"] or triton_error["relative_l2_error"] > 2e-2:
                    raise AssertionError(
                        f"Triton correctness failed at rank={value_dim}, "
                        f"B={batch}, S={context}: {triton_error}"
                    )
                triton_timing = _measure(
                    lambda: compressed_v_decode_attention_triton(
                        query,
                        key,
                        value,
                        scale=scale,
                    ),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )

                split_records: dict[str, object] = {}
                for split in splits:
                    if context < split:
                        continue
                    workspace = torch.empty(
                        batch * args.query_heads,
                        split,
                        value_dim + 2,
                        dtype=torch.float32,
                        device=device,
                    )
                    output = torch.empty(
                        batch,
                        args.query_heads,
                        1,
                        value_dim,
                        dtype=dtype,
                        device=device,
                    )

                    def cuda_function(
                        selected_split: int = split,
                        selected_workspace: Tensor = workspace,
                        selected_output: Tensor = output,
                    ) -> Tensor:
                        return compressed_v_decode_attention_cuda(
                            query,
                            key,
                            value,
                            scale=scale,
                            splits=selected_split,
                            workspace=selected_workspace,
                            output=selected_output,
                        )

                    observed = cuda_function()
                    feature_major_output = torch.empty(
                        args.query_heads * value_dim,
                        batch,
                        dtype=dtype,
                        device=device,
                    )
                    compressed_v_decode_attention_cuda(
                        query,
                        key,
                        value,
                        scale=scale,
                        splits=split,
                        workspace=workspace,
                        feature_major_output=feature_major_output,
                    )
                    torch.cuda.synchronize()
                    error = _error_against(reference, observed)
                    feature_major_as_attention = (
                        feature_major_output.view(
                            args.query_heads,
                            value_dim,
                            batch,
                        )
                        .permute(2, 0, 1)
                        .unsqueeze(2)
                        .contiguous()
                    )
                    feature_major_error = _error_against(
                        reference,
                        feature_major_as_attention,
                    )
                    if error["nonfinite"] or error["relative_l2_error"] > 2e-2:
                        raise AssertionError(
                            f"CUDA correctness failed at rank={value_dim}, "
                            f"B={batch}, S={context}, splits={split}: {error}"
                        )
                    if (
                        feature_major_error["nonfinite"]
                        or feature_major_error["relative_l2_error"] > 2e-2
                    ):
                        raise AssertionError(
                            "feature-major CUDA correctness failed at "
                            f"rank={value_dim}, B={batch}, S={context}, "
                            f"splits={split}: {feature_major_error}"
                        )
                    timing = _measure(
                        cuda_function,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                    split_records[str(split)] = {
                        "error": error,
                        "feature_major_error": feature_major_error,
                        "timing": timing,
                        "speedup_over_triton": (
                            triton_timing["p50_ms"] / timing["p50_ms"]
                        ),
                    }

                best_split = min(
                    split_records,
                    key=lambda item: split_records[item]["timing"]["p50_ms"],
                )
                best_cuda = {
                    "split": int(best_split),
                    **split_records[best_split],
                }
                record = {
                    "value_dim": value_dim,
                    "batch": batch,
                    "context": context,
                    "triton": {
                        "error": triton_error,
                        "timing": triton_timing,
                    },
                    "splitk": split_records,
                    "best_cuda": best_cuda,
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            "event": "configuration_complete",
                            "value_dim": value_dim,
                            "batch": batch,
                            "context": context,
                            "triton_ms": triton_timing["p50_ms"],
                            "cuda_split": best_cuda["split"],
                            "cuda_ms": best_cuda["timing"]["p50_ms"],
                            "cuda_over_triton": best_cuda["speedup_over_triton"],
                            "cuda_relative_l2_error": best_cuda["error"][
                                "relative_l2_error"
                            ],
                        }
                    ),
                    flush=True,
                )
                torch.cuda.empty_cache()

    payload = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "configuration": {
            "batches": list(batches),
            "contexts": list(contexts),
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "qk_dim": args.qk_dim,
            "value_dims": list(value_dims),
            "splits": list(splits),
            "dtype": "bfloat16",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "records": records,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(json.dumps({"event": "result_written", "path": str(output_path)}), flush=True)


if __name__ == "__main__":
    main()
