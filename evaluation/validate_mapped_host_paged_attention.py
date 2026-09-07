#!/usr/bin/env python3
"""Validate mapped-host Page32/K128/V80 CUDA attention against FP32."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from basisserve.kernels.mapped_host_paged_attention import (  # noqa: E402
    append_mapped_host_key,
    mapped_host_bf16_empty,
    mapped_host_device_pointer,
    mapped_host_page32_v80_attention,
)


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    page_ids: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    batch, kv_heads, sequence, _ = map(int, key.shape)
    pages = math.ceil(sequence / page_size)
    padded_tokens = pages * page_size
    padded_key = torch.nn.functional.pad(
        key,
        (0, 0, 0, padded_tokens - sequence),
    ).reshape(batch, kv_heads, pages, page_size, 128)
    padded_value = torch.nn.functional.pad(
        value[:, :, :sequence],
        (0, 0, 0, padded_tokens - sequence),
    ).reshape(batch, kv_heads, pages, page_size, 80)
    slots = int(page_ids.shape[2])
    selected_key = padded_key.gather(
        2,
        page_ids[..., None, None].expand(
            batch, kv_heads, slots, page_size, 128
        ),
    ).reshape(batch, kv_heads, slots * page_size, 128)
    selected_value = padded_value.gather(
        2,
        page_ids[..., None, None].expand(
            batch, kv_heads, slots, page_size, 80
        ),
    ).reshape(batch, kv_heads, slots * page_size, 80)
    positions = (
        page_ids[..., None] * page_size
        + torch.arange(page_size, device=query.device)
    ).reshape(batch, kv_heads, -1)
    valid = positions < sequence
    expanded_key = selected_key.repeat_interleave(4, dim=1).float()
    expanded_value = selected_value.repeat_interleave(4, dim=1).float()
    scores = torch.matmul(query.float(), expanded_key.transpose(-1, -2))
    scores.mul_(1.0 / math.sqrt(128))
    scores.masked_fill_(
        ~valid.repeat_interleave(4, dim=1)[:, :, None],
        -torch.inf,
    )
    return torch.matmul(scores.softmax(dim=-1), expanded_value)


def main() -> None:
    wall_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=257)
    parser.add_argument("--page-slots", type=int, default=5)
    parser.add_argument("--splits", type=str, default="5")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    split_grid = tuple(int(value) for value in args.splits.split(","))
    assert args.sequence_length > 0 and args.page_slots > 0
    assert split_grid and len(split_grid) == len(set(split_grid))
    assert all(0 < splits <= args.page_slots for splits in split_grid)
    assert args.warmup >= 0 and args.repeat > 0
    torch.manual_seed(20260902)
    device = torch.device("cuda")
    batch = 1
    kv_heads = 2
    capacity = math.ceil(args.sequence_length / 32) * 32
    key = torch.randn(
        batch,
        kv_heads,
        args.sequence_length,
        128,
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn(
        batch,
        kv_heads,
        capacity,
        80,
        dtype=torch.bfloat16,
        device=device,
    )
    query = torch.randn(
        batch,
        4 * kv_heads,
        1,
        128,
        dtype=torch.bfloat16,
        device=device,
    )
    page_count = math.ceil(args.sequence_length / 32)
    assert args.page_slots <= page_count
    chosen = torch.linspace(
        0,
        page_count - 1,
        args.page_slots,
        dtype=torch.float64,
        device=device,
    ).round().to(torch.int64)
    page_ids = torch.stack((chosen, chosen.roll(1))).unsqueeze(0).contiguous()
    host_key = mapped_host_bf16_empty(
        batch=batch,
        kv_heads=kv_heads,
        capacity=capacity,
    )
    pointer = mapped_host_device_pointer(host_key)
    append_split = min(193, args.sequence_length)
    append_mapped_host_key(host_key, key[:, :, :append_split], start=0)
    if append_split < args.sequence_length:
        append_mapped_host_key(
            host_key,
            key[:, :, append_split:],
            start=append_split,
        )
    reference = _reference(query, key, value, page_ids, page_size=32)
    workspace = torch.empty(
        batch * 4 * kv_heads,
        max(split_grid),
        82,
        dtype=torch.float32,
        device=device,
    )
    output = torch.empty(
        batch,
        4 * kv_heads,
        1,
        80,
        dtype=torch.bfloat16,
        device=device,
    )
    properties = torch.cuda.get_device_properties(device)
    l2_bytes = int(getattr(properties, "L2_cache_size", 48 * 1024 * 1024))
    cache_flush = torch.zeros(
        max(2 * l2_bytes, 64 * 1024 * 1024) // 4,
        dtype=torch.float32,
        device=device,
    )
    measurements = []
    for splits in split_grid:
        for _ in range(args.warmup):
            mapped_host_page32_v80_attention(
                host_key,
                query,
                value,
                page_ids,
                sequence_length=args.sequence_length,
                splits=splits,
                host_key_device_pointer=pointer,
                workspace=workspace,
                output=output,
            )
        observed = mapped_host_page32_v80_attention(
            host_key,
            query,
            value,
            page_ids,
            sequence_length=args.sequence_length,
            splits=splits,
            host_key_device_pointer=pointer,
            workspace=workspace,
            output=output,
        )
        torch.cuda.synchronize()
        difference = observed.float() - reference
        maximum_absolute_error = float(difference.abs().max())
        mean_absolute_error = float(difference.abs().mean())
        assert maximum_absolute_error < 0.04
        assert mean_absolute_error < 0.005

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repeat):
            mapped_host_page32_v80_attention(
                host_key,
                query,
                value,
                page_ids,
                sequence_length=args.sequence_length,
                splits=splits,
                host_key_device_pointer=pointer,
                workspace=workspace,
                output=output,
            )
        stop.record()
        stop.synchronize()
        warm_microseconds = 1000.0 * start.elapsed_time(stop) / args.repeat

        cold_microseconds = []
        for _ in range(args.repeat):
            cache_flush.add_(1.0)
            start.record()
            mapped_host_page32_v80_attention(
                host_key,
                query,
                value,
                page_ids,
                sequence_length=args.sequence_length,
                splits=splits,
                host_key_device_pointer=pointer,
                workspace=workspace,
                output=output,
            )
            stop.record()
            stop.synchronize()
            cold_microseconds.append(1000.0 * start.elapsed_time(stop))
        measurements.append(
            {
                "splits": splits,
                "numerics": {
                    "maximum_absolute_error": maximum_absolute_error,
                    "mean_absolute_error": mean_absolute_error,
                },
                "kernel": {
                    "warm_mean_microseconds": warm_microseconds,
                    "cold_median_microseconds": statistics.median(
                        cold_microseconds
                    ),
                    "cold_mean_microseconds": statistics.fmean(
                        cold_microseconds
                    ),
                },
            }
        )
    best = min(
        measurements,
        key=lambda measurement: measurement["kernel"][
            "cold_median_microseconds"
        ],
    )
    payload = {
        "format": "basisserve.mapped_host_page32_v80_split_sweep.v1",
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "geometry": {
            "sequence_length": args.sequence_length,
            "page_slots": args.page_slots,
            "split_grid": list(split_grid),
            "logical_host_read_bytes": (
                batch * kv_heads * args.page_slots * 32 * 128 * 2
            ),
            "l2_cache_bytes": l2_bytes,
            "cache_flush_bytes": cache_flush.numel() * cache_flush.element_size(),
        },
        "benchmark": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "measurements": measurements,
            "best_cold_median_splits": best["splits"],
        },
        "wall_seconds": time.perf_counter() - wall_started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
