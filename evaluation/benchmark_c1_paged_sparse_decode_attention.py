#!/usr/bin/env python3
"""Benchmark shared-GQA C1-V96 attention over packed exact-Key pages."""

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
import time
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from basisserve.kernels.compressed_v_decode_attention import (  # noqa: E402
    c1_dense_gqa_v96_decode_attention_cuda,
    c1_pack_exact_key_pages_cuda,
    c1_paged_sparse_decode_attention_cuda,
    reference_c1_paged_sparse_decode_attention,
)


FORMAT = "basisserve.c1_paged_sparse_decode_attention_benchmark.v2"
PAGE_SIZE = 64
QUERY_HEADS = 32
KV_HEADS = 8
QK_DIM = 128
VALUE_DIM = 96


def _positive_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(",") if item)
    if not values or any(value <= 0 for value in values):
        raise ValueError("expected positive comma-separated integers")
    return values


def _measure(
    function: Callable[[], Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    milliseconds = [
        float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)
    ]
    ordered = sorted(milliseconds)
    return {
        "minimum_ms": min(milliseconds),
        "p50_ms": statistics.median(milliseconds),
        "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "maximum_ms": max(milliseconds),
    }


def _page_ids(context: int, page_count: int, device: torch.device) -> Tensor:
    total_pages = context // PAGE_SIZE
    base = torch.arange(page_count, device=device, dtype=torch.int64)
    base = torch.div(base * total_pages, page_count, rounding_mode="floor")
    per_head = [
        torch.sort((base + 13 * kv_head) % total_pages).values
        for kv_head in range(KV_HEADS)
    ]
    return torch.stack(per_head).unsqueeze(0).contiguous()


def _pack_key_pages(key: Tensor, page_ids: Tensor) -> Tensor:
    page_count = int(page_ids.shape[-1])
    offsets = torch.arange(PAGE_SIZE, device=key.device, dtype=torch.int64)
    token_ids = page_ids[..., None] * PAGE_SIZE + offsets
    gather_index = token_ids.reshape(1, KV_HEADS, page_count * PAGE_SIZE, 1)
    gather_index = gather_index.expand(1, KV_HEADS, page_count * PAGE_SIZE, QK_DIM)
    return (
        key.gather(2, gather_index)
        .reshape(1, KV_HEADS, page_count, PAGE_SIZE, QK_DIM)
        .contiguous()
    )


def _error(expected: Tensor, observed: Tensor) -> dict[str, float | bool]:
    difference = observed.float() - expected.float()
    relative_l2 = float(
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-30)
    )
    return {
        "finite": bool(torch.isfinite(observed).all()),
        "maximum_absolute_error": float(difference.abs().max()),
        "relative_l2_error": relative_l2,
        "passed": bool(torch.isfinite(observed).all() and relative_l2 <= 3e-2),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="32768,131072")
    parser.add_argument("--page-counts", default="16,32,64,128")
    parser.add_argument("--splits", default="1,4,8,16,32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contexts = _positive_ints(args.contexts)
    page_counts = _positive_ints(args.page_counts)
    splits = _positive_ints(args.splits)
    if any(context % PAGE_SIZE for context in contexts):
        raise ValueError("contexts must be divisible by page size 64")
    if any(split not in (1, 2, 4, 8, 16, 32) for split in splits):
        raise ValueError("splits must lie in {1,2,4,8,16,32}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    scale = QK_DIM**-0.5
    records: list[dict[str, object]] = []
    compilation_seconds: float | None = None

    for context in contexts:
        if any(page_count > context // PAGE_SIZE for page_count in page_counts):
            raise ValueError("page count exceeds the selected context")
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
        value = torch.randn(
            1,
            KV_HEADS,
            context,
            VALUE_DIM,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        dense_value = torch.randn(
            1,
            KV_HEADS,
            context,
            QK_DIM,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        valid_sequence_length = torch.tensor(
            context,
            dtype=torch.int64,
            device=device,
        )
        dense_bf16 = _measure(
            lambda: F.scaled_dot_product_attention(
                query,
                key,
                dense_value,
                scale=scale,
                enable_gqa=True,
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        dense_c1_splits = 128 if context <= 32768 else 64
        dense_c1_workspace = torch.empty(
            QUERY_HEADS,
            dense_c1_splits,
            VALUE_DIM + 2,
            dtype=torch.float32,
            device=device,
        )
        dense_c1_output = torch.empty(
            1,
            QUERY_HEADS,
            1,
            VALUE_DIM,
            dtype=dtype,
            device=device,
        )
        dense_c1 = _measure(
            lambda: c1_dense_gqa_v96_decode_attention_cuda(
                query,
                key,
                value,
                valid_sequence_length,
                scale=scale,
                splits=dense_c1_splits,
                workspace=dense_c1_workspace,
                output=dense_c1_output,
            ),
            warmup=args.warmup,
            iterations=args.iterations,
        )

        for page_count in page_counts:
            selected_page_ids = _page_ids(context, page_count, device)
            reference_key_pages = _pack_key_pages(key, selected_page_ids)
            packed_key_pages = torch.empty_like(reference_key_pages)
            first_started = time.perf_counter()
            c1_pack_exact_key_pages_cuda(
                key,
                selected_page_ids,
                output=packed_key_pages,
            )
            torch.cuda.synchronize(device)
            first_seconds = time.perf_counter() - first_started
            if compilation_seconds is None:
                compilation_seconds = first_seconds
            torch.testing.assert_close(
                packed_key_pages,
                reference_key_pages,
                rtol=0.0,
                atol=0.0,
            )
            expected = reference_c1_paged_sparse_decode_attention(
                query,
                reference_key_pages,
                value,
                selected_page_ids,
                scale=scale,
            )
            for split in splits:
                workspace = torch.empty(
                    QUERY_HEADS,
                    split,
                    VALUE_DIM + 2,
                    dtype=torch.float32,
                    device=device,
                )
                output = torch.empty(
                    1,
                    QUERY_HEADS,
                    1,
                    VALUE_DIM,
                    dtype=dtype,
                    device=device,
                )

                def sparse_call() -> Tensor:
                    return c1_paged_sparse_decode_attention_cuda(
                        query,
                        packed_key_pages,
                        value,
                        selected_page_ids,
                        scale=scale,
                        splits=split,
                        workspace=workspace,
                        output=output,
                    )

                def pack_call() -> Tensor:
                    return c1_pack_exact_key_pages_cuda(
                        key,
                        selected_page_ids,
                        output=packed_key_pages,
                    )

                def pack_and_sparse_call() -> Tensor:
                    pack_call()
                    return sparse_call()

                first_started = time.perf_counter()
                observed = sparse_call()
                torch.cuda.synchronize(device)
                first_seconds = time.perf_counter() - first_started
                if compilation_seconds is None:
                    compilation_seconds = first_seconds
                error = _error(expected, observed)
                if not error["passed"]:
                    raise AssertionError(
                        f"paged sparse correctness failed for context={context}, "
                        f"pages={page_count}, splits={split}: {error}"
                    )
                records.append(
                    {
                        "context": context,
                        "page_count": page_count,
                        "selected_tokens_per_kv_head": page_count * PAGE_SIZE,
                        "splits": split,
                        "error": error,
                        "attention_only_timing": _measure(
                            sparse_call,
                            warmup=args.warmup,
                            iterations=args.iterations,
                        ),
                        "page_pack_timing": _measure(
                            pack_call,
                            warmup=args.warmup,
                            iterations=args.iterations,
                        ),
                        "pack_and_attention_timing": _measure(
                            pack_and_sparse_call,
                            warmup=args.warmup,
                            iterations=args.iterations,
                        ),
                        "dense_bf16_sdpa_timing": dense_bf16,
                        "dense_c1_cuda_timing": dense_c1,
                        "dense_c1_splits": dense_c1_splits,
                    }
                )

    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "configuration": {
            "contexts": list(contexts),
            "page_counts": list(page_counts),
            "splits": list(splits),
            "page_size": PAGE_SIZE,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "qk_dim": QK_DIM,
            "value_dim": VALUE_DIM,
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
        "extension_compile_and_first_launch_seconds": compilation_seconds,
        "records": records,
    }
    result_path = output_dir / "result.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, result_path)

    lines = [
        "# C1 packed-page sparse decode microbenchmark",
        "",
        "| Context | Pages | Tokens/KV | Splits | Pack ms | Sparse-only ms | Pack+sparse ms | Dense C1 ms | Dense BF16 ms | Rel. L2 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['context']} | {record['page_count']} | "
            f"{record['selected_tokens_per_kv_head']} | {record['splits']} | "
            f"{record['page_pack_timing']['p50_ms']:.4f} | "
            f"{record['attention_only_timing']['p50_ms']:.4f} | "
            f"{record['pack_and_attention_timing']['p50_ms']:.4f} | "
            f"{record['dense_c1_cuda_timing']['p50_ms']:.4f} | "
            f"{record['dense_bf16_sdpa_timing']['p50_ms']:.4f} | "
            f"{record['error']['relative_l2_error']:.6g} |"
        )
    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"result": str(result_path), "records": len(records)}))


if __name__ == "__main__":
    main()
