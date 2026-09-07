#!/usr/bin/env python3
"""Validate and time one-token sparse routing kernels at RULER geometry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Callable

import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_conditional_page_attention import (  # noqa: E402
    c1_conditional_page_topk_attention,
)
from basisserve.core.c1_loki_attention import (  # noqa: E402
    c1_loki_pca_topk_attention,
)


def _timed(
    function: Callable[[], Tensor],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> tuple[Tensor, float]:
    output = function()
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        output = function()
    torch.cuda.synchronize(device)
    return output, 1.0e3 * (time.perf_counter() - started) / repeats


def _comparison(observed: Tensor, expected: Tensor) -> dict[str, float]:
    delta = (observed.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1.0e-5)
    return {
        "maximum_absolute_error": float(delta.max().item()),
        "mean_absolute_error": float(delta.mean().item()),
        "maximum_relative_error": float((delta / denominator).max().item()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--pinned-prefix-pages", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--output", type=Path, required=True)
    return parser


@torch.inference_mode()
def main() -> None:
    args = _parser().parse_args()
    assert torch.cuda.is_available()
    assert min(
        args.sequence_length,
        args.token_budget,
        args.page_size,
        args.warmup,
        args.repeats,
    ) > 0
    assert args.page_size == 32
    assert args.token_budget % args.page_size == 0
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    batch, query_heads, kv_heads = 1, 32, 8
    head_dim, value_rank = 128, 80
    scale = head_dim**-0.5
    query = torch.randn(
        batch,
        query_heads,
        1,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        batch,
        kv_heads,
        args.sequence_length,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        batch,
        kv_heads,
        args.sequence_length,
        value_rank,
        device=device,
        dtype=torch.bfloat16,
    )
    full_mask = torch.ones(
        batch,
        args.sequence_length,
        device=device,
        dtype=torch.bool,
    )

    loki_rank = 32
    loki_sidecar = torch.randn(
        batch,
        kv_heads,
        args.sequence_length,
        loki_rank,
        device=device,
        dtype=torch.bfloat16,
    )
    loki_key_projector = torch.randn(
        kv_heads,
        head_dim,
        loki_rank,
        device=device,
        dtype=torch.bfloat16,
    )
    loki_query_projector = torch.randn(
        query_heads,
        head_dim,
        loki_rank,
        device=device,
        dtype=torch.bfloat16,
    )

    def loki_kernel() -> Tensor:
        return c1_loki_pca_topk_attention(
            query,
            key,
            value,
            loki_key_projector,
            loki_query_projector,
            top_k=args.token_budget,
            scale=scale,
            query_block_size=1,
            routing_sidecar=loki_sidecar,
            collect_statistics=False,
        ).output

    def loki_torch() -> Tensor:
        return c1_loki_pca_topk_attention(
            query,
            key,
            value,
            loki_key_projector,
            loki_query_projector,
            top_k=args.token_budget,
            scale=scale,
            query_block_size=1,
            attention_mask=full_mask,
            routing_sidecar=loki_sidecar,
            collect_statistics=False,
        ).output

    conditional_rank = 136
    conditional_sidecar = torch.randn(
        batch,
        kv_heads,
        args.sequence_length,
        conditional_rank,
        device=device,
        dtype=torch.bfloat16,
    )
    conditional_query_projector = torch.randn(
        query_heads,
        head_dim,
        conditional_rank,
        device=device,
        dtype=torch.bfloat16,
    )

    def conditional_kernel() -> Tensor:
        return c1_conditional_page_topk_attention(
            query,
            key,
            value,
            conditional_sidecar,
            conditional_query_projector,
            page_size=args.page_size,
            exact_token_budget=args.token_budget,
            pinned_prefix_pages=args.pinned_prefix_pages,
            scale=scale,
            query_block_size=1,
            collect_statistics=False,
        ).output

    def conditional_torch() -> Tensor:
        return c1_conditional_page_topk_attention(
            query,
            key,
            value,
            conditional_sidecar,
            conditional_query_projector,
            page_size=args.page_size,
            exact_token_budget=args.token_budget,
            pinned_prefix_pages=args.pinned_prefix_pages,
            scale=scale,
            query_block_size=1,
            attention_mask=full_mask,
            collect_statistics=False,
        ).output

    results = {}
    for name, kernel, reference in (
        ("loki_r32_token_topk", loki_kernel, loki_torch),
        ("base16_r8_page32", conditional_kernel, conditional_torch),
    ):
        torch.cuda.empty_cache()
        kernel_output, kernel_ms = _timed(
            kernel,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        reference_output, reference_ms = _timed(
            reference,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        comparison = _comparison(kernel_output, reference_output)
        assert math.isfinite(comparison["maximum_absolute_error"])
        results[name] = {
            "kernel_milliseconds": kernel_ms,
            "pytorch_milliseconds": reference_ms,
            "speedup": reference_ms / kernel_ms,
            "comparison": comparison,
        }
        print(
            f"[{name}] kernel={kernel_ms:.3f}ms torch={reference_ms:.3f}ms "
            f"speedup={reference_ms / kernel_ms:.3f}x "
            f"max_abs={comparison['maximum_absolute_error']:.6f}",
            flush=True,
        )

    payload = {
        "format": "basisserve.sparse_decode_kernel_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "geometry": {
            "batch": batch,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "value_rank": value_rank,
            "sequence_length": args.sequence_length,
            "token_budget": args.token_budget,
            "page_size": args.page_size,
            "pinned_prefix_pages": args.pinned_prefix_pages,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "results": results,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(f"result={output_path}", flush=True)


if __name__ == "__main__":
    main()
