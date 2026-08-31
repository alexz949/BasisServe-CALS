#!/usr/bin/env python3
"""Benchmark dense shared-GQA exact-K128/C1-V96 decode attention."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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
    c1_dense_gqa_v96_decode_attention_cuda,
    compressed_v_decode_attention_triton,
)
from evaluation.eval_qwen3_c1_quest_ruler import (  # noqa: E402
    _atomic_json,
    _atomic_text,
)


FORMAT = "basisserve.c1_dense_gqa_v96_decode_attention_benchmark.v1"
QUERY_HEADS = 32
KV_HEADS = 8
QK_DIM = 128
C1_VALUE_DIM = 96
DENSE_VALUE_DIM = 128


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


def _error(reference: Tensor, observed: Tensor) -> dict[str, float | bool]:
    difference = observed.float() - reference.float()
    return {
        "finite": bool(torch.isfinite(observed).all()),
        "maximum_absolute_error": float(difference.abs().max()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="32768,131072")
    parser.add_argument("--splits", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


@torch.inference_mode()
def main() -> None:
    args = _parser().parse_args()
    contexts = _parse_positive_ints(args.contexts)
    splits = _parse_positive_ints(args.splits)
    if any(split not in (1, 2, 4, 8, 16, 32, 64, 128, 256) for split in splits):
        raise ValueError("split count must lie in {1,2,4,8,16,32,64,128,256}")
    if args.warmup <= 0 or args.iterations <= 0:
        raise ValueError("warmup and iterations must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((8, 0), (8, 9), (9, 0)):
        raise RuntimeError("dense GQA V96 CUDA requires SM80, SM89, or SM90")
    dtype = torch.bfloat16
    scale = QK_DIM**-0.5
    records: list[dict[str, object]] = []

    for context in contexts:
        generator = torch.Generator(device=device).manual_seed(args.seed + context)
        query = torch.randn(
            1,
            QUERY_HEADS,
            1,
            QK_DIM,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        key = torch.randn(
            1,
            KV_HEADS,
            context,
            QK_DIM,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        c1_value = torch.randn(
            1,
            KV_HEADS,
            context,
            C1_VALUE_DIM,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        dense_value = torch.randn(
            1,
            KV_HEADS,
            context,
            DENSE_VALUE_DIM,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        valid_sequence_length = torch.tensor(
            context,
            dtype=torch.int64,
            device=device,
        )

        reference = compressed_v_decode_attention_triton(
            query,
            key,
            c1_value,
            scale=scale,
            valid_sequence_length=valid_sequence_length,
        )
        triton_timing = _measure(
            lambda: compressed_v_decode_attention_triton(
                query,
                key,
                c1_value,
                scale=scale,
                valid_sequence_length=valid_sequence_length,
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        dense_sdpa_timing = _measure(
            lambda: F.scaled_dot_product_attention(
                query,
                key,
                dense_value,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                enable_gqa=True,
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        )

        split_records: dict[str, object] = {}
        for split in splits:
            if context < split:
                continue
            workspace = torch.empty(
                QUERY_HEADS,
                split,
                C1_VALUE_DIM + 2,
                dtype=torch.float32,
                device=device,
            )
            output = torch.empty(
                1,
                QUERY_HEADS,
                1,
                C1_VALUE_DIM,
                dtype=dtype,
                device=device,
            )

            def dense_gqa_v96(
                selected_split: int = split,
                selected_workspace: Tensor = workspace,
                selected_output: Tensor = output,
            ) -> Tensor:
                return c1_dense_gqa_v96_decode_attention_cuda(
                    query,
                    key,
                    c1_value,
                    valid_sequence_length,
                    scale=scale,
                    splits=selected_split,
                    workspace=selected_workspace,
                    output=selected_output,
                )

            observed = dense_gqa_v96()
            torch.cuda.synchronize()
            error = _error(reference, observed)
            if not error["finite"] or error["relative_l2_error"] > 3e-2:
                raise AssertionError(
                    f"dense GQA V96 correctness failed at S={context}, "
                    f"splits={split}: {error}"
                )
            timing = _measure(
                dense_gqa_v96,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            split_records[str(split)] = {
                "error_against_triton": error,
                "timing": timing,
                "speedup_over_dense_bf16_sdpa": (
                    dense_sdpa_timing["p50_ms"] / timing["p50_ms"]
                ),
                "speedup_over_query_head_triton": (
                    triton_timing["p50_ms"] / timing["p50_ms"]
                ),
            }

        best_split = min(
            split_records,
            key=lambda item: split_records[item]["timing"]["p50_ms"],
        )
        best = {"split": int(best_split), **split_records[best_split]}
        record = {
            "context": context,
            "dense_bf16_sdpa": {"timing": dense_sdpa_timing},
            "query_head_triton_c1_v96": {"timing": triton_timing},
            "dense_shared_gqa_c1_v96": {
                "splits": split_records,
                "best": best,
            },
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "event": "context_complete",
                    "context": context,
                    "dense_bf16_sdpa_p50_ms": dense_sdpa_timing["p50_ms"],
                    "query_head_triton_c1_v96_p50_ms": triton_timing["p50_ms"],
                    "best_split": best["split"],
                    "shared_gqa_c1_v96_p50_ms": best["timing"]["p50_ms"],
                    "c1_speedup_over_dense_bf16": best["speedup_over_dense_bf16_sdpa"],
                }
            ),
            flush=True,
        )
        torch.cuda.empty_cache()

    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "capability": list(capability),
            "dtype": str(dtype),
        },
        "geometry": {
            "batch": 1,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "gqa_ratio": QUERY_HEADS // KV_HEADS,
            "qk_dim": QK_DIM,
            "c1_value_dim": C1_VALUE_DIM,
            "dense_value_dim": DENSE_VALUE_DIM,
        },
        "theory": {
            "dense_bf16_elements_per_physical_token": QK_DIM + DENSE_VALUE_DIM,
            "c1_v96_elements_per_physical_token": QK_DIM + C1_VALUE_DIM,
            "ideal_c1_attention_bandwidth_speedup": (
                (QK_DIM + DENSE_VALUE_DIM) / (QK_DIM + C1_VALUE_DIM)
            ),
        },
        "records": records,
    }
    _atomic_json(output_dir / "result.json", payload)

    lines = [
        "# Dense shared-GQA C1-V96 decode attention",
        "",
        "The CUDA arm scans the complete valid exact-K128/C1-V96 cache. It does "
        "not perform page selection, packing, routing, or sparse attention.",
        "",
        "| Context | BF16 SDPA ms | Old C1 Triton ms | Shared C1 ms | Split | "
        "C1 / BF16 speedup |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        best = record["dense_shared_gqa_c1_v96"]["best"]
        lines.append(
            f"| {record['context']} | "
            f"{record['dense_bf16_sdpa']['timing']['p50_ms']:.4f} | "
            f"{record['query_head_triton_c1_v96']['timing']['p50_ms']:.4f} | "
            f"{best['timing']['p50_ms']:.4f} | {best['split']} | "
            f"{best['speedup_over_dense_bf16_sdpa']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "The traffic-only upper-bound comparison is "
            f"`{payload['theory']['ideal_c1_attention_bandwidth_speedup']:.3f}x` "
            "for `(K128+V128)/(K128+V96)`.",
        ]
    )
    _atomic_text(output_dir / "summary.md", "\n".join(lines) + "\n")
    print(f"[dense GQA V96] result={output_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
