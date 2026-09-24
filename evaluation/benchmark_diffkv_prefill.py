"""Small, single-GPU correctness and tile sweep for SM89 paged DiffKV prefill."""

import argparse
import itertools
import json
from pathlib import Path

import torch
import triton
from vllm.v1.attention.ops.triton_unified_attention_diffkv import unified_attention_diffkv
from vllm.vllm_flash_attn import flash_attn_varlen_func

from basisserve.kernels.diffkv_prefill import diffkv_prefill


def make_case(query_lengths, context_lengths, heads=4, block_size=16,
              value_rank=64):
    lengths = [q + c for q, c in zip(query_lengths, context_lengths)]
    counts = [triton.cdiv(n, block_size) for n in lengths]
    blocks = sum(counts)
    cache = torch.randn(blocks, block_size, 1, 128 + value_rank,
                        device="cuda", dtype=torch.bfloat16)
    permutation = torch.randperm(blocks, device="cuda", dtype=torch.int32)
    table = torch.zeros(len(lengths), max(counts), device="cuda", dtype=torch.int32)
    start = 0
    for i, n in enumerate(counts):
        table[i, :n] = permutation[start:start + n]
        start += n
    # Match the noncontiguous Q view of a fused QKV projection.
    qkv = torch.randn(sum(query_lengths), heads * 128 + 128 + value_rank,
                      device="cuda", dtype=torch.bfloat16)
    q = qkv[:, :heads * 128].view(-1, heads, 128)
    out = torch.empty(q.shape[0], heads, value_rank, device="cuda", dtype=q.dtype)
    cu = torch.tensor([0, *itertools.accumulate(query_lengths)], device="cuda", dtype=torch.int32)
    seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    k, v = cache[..., :128], cache[..., 128:]
    return dict(q=q, k=k, v=v, out=out, cu_seqlens_q=cu,
                seqused_k=seq, block_table=table, softmax_scale=128 ** -0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-rank", type=int, choices=(64, 96), default=64)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=("ragged", "4k", "2x4k", "8k", "32k", "mixed"))
    parser.add_argument("--block-ms", type=int, nargs="+", default=[32, 64, 128])
    args = parser.parse_args()
    torch.manual_seed(42)
    cases = [("ragged", [17, 1, 65, 3], [15, 255, 5, 1024]),
             ("4k", [4096], [0]), ("2x4k", [4096, 4096], [0, 0]),
             ("8k", [8192], [0]), ("32k", [8192], [24448]),
             ("mixed", [8161] + [1] * 31, [24448] + [32639] * 31)]
    if args.cases:
        cases = [case for case in cases if case[0] in args.cases]
    configs = list(itertools.product(args.block_ms, (32, 64, 128), (4, 8), (2, 3)))
    if args.quick:
        configs = [(16, 32, 4, 2), (64, 64, 4, 2), (128, 64, 4, 2), (64, 128, 4, 2)]
    records = []
    for name, qs, contexts in cases:
        data = make_case(qs, contexts, args.heads, value_rank=args.value_rank)
        # FA2 requires equal Q/K/V dimensions. Zero padding V is mathematically
        # exact for the first V-rank output channels; preparation is outside timing.
        fa_k = data["k"].contiguous()
        fa_v = torch.nn.functional.pad(
            data["v"], (0, 128 - args.value_rank)).contiguous()
        fa_out = torch.empty(data["q"].shape, device="cuda", dtype=data["q"].dtype)
        def reference():
            return flash_attn_varlen_func(
                q=data["q"], k=fa_k, v=fa_v, out=fa_out,
                cu_seqlens_q=data["cu_seqlens_q"], seqused_k=data["seqused_k"],
                block_table=data["block_table"], max_seqlen_q=max(qs),
                max_seqlen_k=max(q + c for q, c in zip(qs, contexts)),
                softmax_scale=data["softmax_scale"], causal=True, fa_version=2)
        reference()
        expected = fa_out[..., :args.value_rank].clone()
        fa_ms = triton.testing.do_bench_cudagraph(reference, rep=100)
        def baseline():
            unified_attention_diffkv(**data, causal=True, window_size=(-1, -1),
                                     softcap=0.0, max_seqlen_q=max(qs))
        old_ms = triton.testing.do_bench_cudagraph(baseline, rep=100)
        for bm, bn, warps, stages in configs:
            def run():
                diffkv_prefill(**data, block_m=bm, block_n=bn,
                               num_warps=warps, num_stages=stages)
            run()
            torch.testing.assert_close(data["out"], expected, atol=0.004, rtol=0.02)
            error = (data["out"].float() - expected.float()).norm() / expected.float().norm()
            assert error.item() < 0.005
            ms = triton.testing.do_bench_cudagraph(run, rep=100)
            record = dict(case=name, heads=args.heads, value_rank=args.value_rank,
                          config=[bm, bn, warps, stages],
                          ms=ms, baseline_ms=old_ms, fa2_ms=fa_ms,
                          relative_l2=error.item())
            records.append(record)
            print(json.dumps(record), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
