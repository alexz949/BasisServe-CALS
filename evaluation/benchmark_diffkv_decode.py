#!/usr/bin/env python3
"""Single-GPU correctness and launch sweep for SM89 QK128 DiffKV decode."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import triton
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)

from basisserve.kernels.diffkv_decode import _launch_config, diffkv_decode


def make_case(batch: int, context: int, heads: int, value_rank: int,
              page: int = 16):
    sequence_length = context + 1
    blocks_per_sequence = triton.cdiv(sequence_length, page)
    total_blocks = batch * blocks_per_sequence
    cache = torch.randn(
        total_blocks,
        page,
        1,
        128 + value_rank,
        device="cuda",
        dtype=torch.bfloat16,
    )
    permutation = torch.randperm(total_blocks, device="cuda", dtype=torch.int32)
    block_table = permutation.view(batch, blocks_per_sequence).contiguous()
    qkv = torch.randn(batch, heads * 128 + 128 + value_rank,
                      device="cuda", dtype=torch.bfloat16)
    query = qkv[:, : heads * 128].view(batch, heads, 128)
    output = torch.empty(batch, heads, value_rank,
                         device="cuda", dtype=torch.bfloat16)
    query_start = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    sequence_lengths = torch.full(
        (batch,), sequence_length, device="cuda", dtype=torch.int32
    )
    key, value = cache[..., :128], cache[..., 128:]
    padded_rank = triton.next_power_of_2(value_rank)
    partial = torch.empty(batch, heads, 16, padded_rank,
                          device="cuda", dtype=torch.float32)
    partial_max = torch.empty(batch, heads, 16,
                              device="cuda", dtype=torch.float32)
    partial_sum = torch.empty(batch, heads, 16,
                              device="cuda", dtype=torch.float32)
    return {
        "q": query,
        "k": key,
        "v": value,
        "out": output,
        "cu_seqlens_q": query_start,
        "seqused_k": sequence_lengths,
        "block_table": block_table,
        "softmax_scale": 128**-0.5,
        "softmax_segm_output": partial,
        "softmax_segm_max": partial_max,
        "softmax_segm_expsum": partial_sum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--heads", type=int, choices=(4, 8), default=4)
    parser.add_argument("--value-rank", type=int, choices=(64, 96), default=64)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (8, 9)
    torch.manual_seed(20260921)
    split_threshold = 128
    cases = [(1, 512), (1, 4096), (8, 4096), (32, 4096), (128, 4096), (256, 4096)]
    configurations = list(
        itertools.product((1, 2, 4, 8, 16), (16, 32, 64, 128), (4, 8), (2, 3))
    )
    if args.quick:
        cases = [(1, 4096), (32, 4096), (64, 4096), (256, 4096)]
        configurations = [
            (1, 32, 4, 2),
            (4, 32, 4, 2),
            (8, 32, 4, 2),
            (16, 16, 4, 2),
            (16, 32, 4, 2),
            (16, 64, 4, 2),
        ]

    records = []
    for batch, context in cases:
        data = make_case(batch, context, args.heads, args.value_rank)
        baseline_output = data["out"]
        feature_major_output = torch.empty(
            args.heads * args.value_rank,
            batch,
            device="cuda",
            dtype=torch.bfloat16,
        )
        candidate_output = feature_major_output.T.view(
            batch, args.heads, args.value_rank)
        default_config = _launch_config(
            batch, context + 1, split_threshold, args.heads, args.value_rank)

        def baseline():
            unified_attention_diffkv(
                q=data["q"],
                k=data["k"],
                v=data["v"],
                out=baseline_output,
                cu_seqlens_q=data["cu_seqlens_q"],
                seqused_k=data["seqused_k"],
                softmax_scale=data["softmax_scale"],
                causal=True,
                window_size=(-1, -1),
                block_table=data["block_table"],
                softcap=0.0,
                max_seqlen_q=1,
                seq_threshold_3D=split_threshold,
                num_par_softmax_segments=16,
                softmax_segm_output=data["softmax_segm_output"],
                softmax_segm_max=data["softmax_segm_max"],
                softmax_segm_expsum=data["softmax_segm_expsum"],
            )

        baseline()
        expected = baseline_output.clone()
        baseline_ms = triton.testing.do_bench_cudagraph(
            baseline, rep=args.repetitions
        )
        case_configurations = configurations
        if args.quick and default_config not in case_configurations:
            case_configurations = [*case_configurations, default_config]
        for segments, tile, warps, stages in case_configurations:
            if batch > split_threshold and segments != 1:
                continue

            def candidate():
                diffkv_decode(
                    **(data | {"out": candidate_output}),
                    split_threshold=split_threshold,
                    max_sequence_length=context + 1,
                    segments=segments,
                    tile=tile,
                    num_warps=warps,
                    num_stages=stages,
                )

            candidate()
            torch.testing.assert_close(
                candidate_output, expected, atol=0.004, rtol=0.02
            )
            error = (
                (candidate_output.float() - expected.float()).norm()
                / expected.float().norm()
            ).item()
            candidate_ms = triton.testing.do_bench_cudagraph(
                candidate, rep=args.repetitions
            )
            record = {
                "batch": batch,
                "context": context,
                "heads": args.heads,
                "value_rank": args.value_rank,
                "segments": segments,
                "tile": tile,
                "warps": warps,
                "stages": stages,
                "is_default": (segments, tile, warps, stages) == default_config,
                "output_layout": "feature_major",
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "speedup": baseline_ms / candidate_ms,
                "relative_l2": error,
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
