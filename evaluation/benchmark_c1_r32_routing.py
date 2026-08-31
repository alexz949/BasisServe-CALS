#!/usr/bin/env python3
"""Benchmark fused R32 page routing against the former eager PyTorch path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import sys
import time
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_k_routing_sidecar import select_routing_pages  # noqa: E402
from basisserve.kernels.c1_r32_routing import (  # noqa: E402
    c1_r32_page_lse_cuda,
    c1_r32_topk_gqa_union_cuda,
    reference_c1_r32_page_lse,
    reference_c1_r32_topk_gqa_union,
)


FORMAT = "basisserve.c1_r32.fused_page_routing_benchmark.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="32768")
    parser.add_argument("--pages-per-query-head", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _milliseconds(
    operation: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) / iterations


def _selected_sets_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    for left_row, right_row in zip(
        left.reshape(-1, left.shape[-1]),
        right.reshape(-1, right.shape[-1]),
        strict=True,
    ):
        if set(left_row[left_row >= 0].tolist()) != set(
            right_row[right_row >= 0].tolist()
        ):
            return False
    return True


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("fused R32 routing benchmark requires CUDA")
    contexts = [int(value) for value in args.contexts.split(",")]
    if any(context <= 0 or context > 131072 for context in contexts):
        raise ValueError("contexts must lie in [1, 131072]")
    if not 0 < args.pages_per_query_head <= 32:
        raise ValueError("pages per Query head must lie in [1, 32]")
    if min(args.warmup, args.iterations) <= 0:
        raise ValueError("warmup and iterations must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = 1
    query_heads = 32
    kv_heads = 8
    head_dim = 128
    rank = 32
    page_size = 64
    generator = torch.Generator(device=device).manual_seed(args.seed)
    projector = (
        torch.randn(
            query_heads,
            head_dim,
            rank,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * 0.05
    ).contiguous()
    records = []
    compile_seconds = None

    for context in contexts:
        query = torch.randn(
            batch,
            query_heads,
            1,
            head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        sidecar = (
            torch.randn(
                batch,
                kv_heads,
                context,
                rank,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            * 0.1
        )
        pages = math.ceil(context / page_size)
        page_slots = min(pages, args.pages_per_query_head * 4)
        query_code = torch.empty(
            batch,
            query_heads,
            rank,
            dtype=dtype,
            device=device,
        )
        page_scores = torch.empty(
            batch,
            query_heads,
            pages,
            dtype=dtype,
            device=device,
        )
        selected_ids = torch.empty(
            batch,
            kv_heads,
            page_slots,
            dtype=torch.int64,
            device=device,
        )
        selected_counts = torch.empty(
            batch,
            kv_heads,
            dtype=torch.int32,
            device=device,
        )

        def page_lse() -> torch.Tensor:
            return c1_r32_page_lse_cuda(
                query,
                sidecar,
                projector,
                query_code=query_code,
                output=page_scores,
            )

        def topk_union() -> tuple[torch.Tensor, torch.Tensor]:
            return c1_r32_topk_gqa_union_cuda(
                page_scores,
                pages_per_query_head=args.pages_per_query_head,
                selected_page_ids=selected_ids,
                selected_page_counts=selected_counts,
            )

        def fused_route() -> tuple[torch.Tensor, torch.Tensor]:
            page_lse()
            return topk_union()

        def eager_route() -> torch.Tensor:
            selection = select_routing_pages(
                query[0, :, 0],
                sidecar[0],
                projector,
                head_dim=head_dim,
                page_size=page_size,
                nominal_token_budget=args.pages_per_query_head * page_size,
            )
            page_indices = torch.arange(
                pages,
                dtype=torch.int64,
                device=device,
            ).expand(kv_heads, pages)
            padded = torch.where(selection.page_mask, page_indices, pages)
            compact = torch.sort(padded, dim=-1).values[:, :page_slots]
            return compact.masked_fill_(compact == pages, -1)

        first_launch = time.perf_counter()
        fused_route()
        torch.cuda.synchronize()
        if compile_seconds is None:
            compile_seconds = time.perf_counter() - first_launch

        fused_page_scores = page_scores.clone()
        fused_page_ids = selected_ids.clone()
        fused_counts = selected_counts.clone()
        reference_scores = reference_c1_r32_page_lse(
            query,
            sidecar,
            projector,
        )
        expected_fused_ids, expected_fused_counts = (
            reference_c1_r32_topk_gqa_union(
                fused_page_scores,
                pages_per_query_head=args.pages_per_query_head,
            )
        )
        eager_page_ids = eager_route().unsqueeze(0)
        difference = fused_page_scores.float() - reference_scores.float()
        relative_l2 = float(
            difference.norm() / reference_scores.float().norm().clamp_min(1e-12)
        )
        topk_union_kernel_exact = bool(
            torch.equal(fused_counts, expected_fused_counts)
            and torch.equal(fused_page_ids, expected_fused_ids)
        )
        combined_selection_sets_equal = _selected_sets_equal(
            fused_page_ids,
            eager_page_ids,
        )
        records.append(
            {
                "context": context,
                "pages": pages,
                "pages_per_query_head": args.pages_per_query_head,
                "maximum_union_slots": page_slots,
                "selected_pages": int(fused_counts.sum().item()),
                "page_score_max_abs": float(difference.abs().max().item()),
                "page_score_relative_l2": relative_l2,
                "topk_union_kernel_exact": topk_union_kernel_exact,
                "combined_selection_sets_equal_to_eager": (
                    combined_selection_sets_equal
                ),
                "passed": bool(
                    topk_union_kernel_exact
                    and combined_selection_sets_equal
                    and relative_l2 <= 2e-2
                ),
                "eager_ms": _milliseconds(
                    eager_route,
                    warmup=args.warmup,
                    iterations=args.iterations,
                ),
                "fused_page_lse_ms": _milliseconds(
                    page_lse,
                    warmup=args.warmup,
                    iterations=args.iterations,
                ),
                "fused_topk_union_ms": _milliseconds(
                    topk_union,
                    warmup=args.warmup,
                    iterations=args.iterations,
                ),
                "fused_total_ms": _milliseconds(
                    fused_route,
                    warmup=args.warmup,
                    iterations=args.iterations,
                ),
            }
        )

    for record in records:
        record["fused_speedup_vs_eager"] = (
            record["eager_ms"] / record["fused_total_ms"]
        )
    payload = {
        "format": FORMAT,
        "status": (
            "complete" if all(record["passed"] for record in records) else "failed"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "environment": {
            "device": torch.cuda.get_device_name(),
            "dtype": str(dtype),
            "torch": torch.__version__,
        },
        "extension_compile_and_first_launch_seconds": compile_seconds,
        "records": records,
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Fused R32 page-routing microbenchmark",
        "",
        "| Context | Pages | Selected | Eager ms | Page-LSE ms | Top-k/union ms | Fused ms | Speedup | Page rel. L2 | Same selection |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for record in records:
        lines.append(
            f"| {record['context']} | {record['pages']} | "
            f"{record['selected_pages']} | {record['eager_ms']:.4f} | "
            f"{record['fused_page_lse_ms']:.4f} | "
            f"{record['fused_topk_union_ms']:.4f} | "
            f"{record['fused_total_ms']:.4f} | "
            f"{record['fused_speedup_vs_eager']:.3f}x | "
            f"{record['page_score_relative_l2']:.6g} | "
            f"{record['combined_selection_sets_equal_to_eager']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"result": str(output_dir / "result.json")}))
    if payload["status"] != "complete":
        raise RuntimeError("fused R32 routing correctness check failed")


if __name__ == "__main__":
    main()
