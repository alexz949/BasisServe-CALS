"""Numerical and latency sweep for packed NUQ4 prefill tiles on SM89."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import triton
from basisserve.kernels.nuq4_attention import _attention, nuq4_attention
from basisserve.kernels.nuq4_cache import NUQ4Layout, NUQ4PagedCache, nuq4_value_stats
from basisserve.kernels.nuq4_decode import _decode, _reduce, decode_workspace, nuq4_decode


def fixture(width, shape):
    torch.manual_seed(17)
    if shape.startswith("decode"):
        queries = [1] * int(shape.removeprefix("decode"))
        lengths = [4160] * len(queries)
    else:
        decoding = 0 if shape == "prefill" else 126
        queries = [1] * decoding + [4096, 4096 - decoding]
        lengths = [4160] * decoding + queries[-2:]
    page_counts = [(length + 15) // 16 for length in lengths]
    pages = sum(page_counts)
    lower = torch.full((128,), -2.5, device="cuda")
    upper, lut = -lower, torch.linspace(-1, 1, 16, device="cuda")
    keys = NUQ4PagedCache(pages, NUQ4Layout(128, exceptions_per_token=16), lower, upper, lut)
    values = NUQ4PagedCache(pages, NUQ4Layout(width, exceptions_per_token=16),
                           lower[:width], upper[:width], lut, dynamic=True)
    physical = torch.randperm(pages, device="cuda", dtype=torch.int32)
    table = torch.zeros((len(lengths), max(page_counts)), device="cuda", dtype=torch.int32)
    slots, offset = [], 0
    for seq, (length, count) in enumerate(zip(lengths, page_counts)):
        table[seq, :count] = physical[offset:offset+count]
        positions = torch.arange(length, device="cuda")
        slots.append(table[seq, positions // 16].long() * 16 + positions % 16)
        offset += count
    slots = torch.cat(slots)
    full_v = torch.randn(sum(lengths), 8*width, device="cuda", dtype=torch.bfloat16)
    keys.append(torch.randn(sum(lengths), 128, device="cuda", dtype=torch.bfloat16), slots)
    values.append(full_v[:, :width], slots, nuq4_value_stats(full_v))
    keys.check()
    values.check()
    angles = torch.arange(max(lengths), device="cuda")[:, None] * (
        10000 ** (-torch.arange(64, device="cuda").float() / 64))[None]
    rope = torch.cat((angles.cos(), angles.sin()), 1).to(torch.bfloat16)
    query = torch.randn(sum(queries), 4, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, *torch.tensor(queries).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return query, keys, values, rope, cu, lens, table


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 96])
    parser.add_argument("--shapes", nargs="+", choices=("prefill", "mixed", "decode1", "decode16", "decode128"), default=["prefill", "mixed"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve completed tuning results"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    configs = [(32, 32, 4), (64, 32, 4), (64, 64, 4), (128, 32, 4),
               (128, 64, 4), (128, 64, 8), (128, 128, 8), (64, 128, 4)]
    if args.quick:
        configs = [(32, 32, 4), (128, 64, 8), (128, 128, 8)]
    rows = []
    for width in args.widths:
        for shape in args.shapes:
            q, kc, vc, rope, cu, lens, table = fixture(width, shape)
            reference = nuq4_attention(q, kc, vc, rope, cu, lens, table)
            output = torch.empty_like(reference)
            if shape.startswith("decode"):
                original_workspace = decode_workspace(q.shape[0], width, q.device)
                def original():
                    return nuq4_decode(q, kc, vc, rope, cu, lens, table, out=reference,
                                       workspace=original_workspace)
                original()
                selected_configs = [(split, bn, warps) for split in (1, 2, 4, 8, 16)
                                    for bn, warps in ((32, 4), (64, 4), (64, 8), (128, 8))]
                if args.quick:
                    selected_configs = [(1, 32, 4), (4, 32, 4), (4, 64, 8)]
            else:
                def original():
                    return nuq4_attention(q, kc, vc, rope, cu, lens, table, out=reference)
                selected_configs = configs
            original_row = dict(width=width, shape=shape, production=True,
                                ms=triton.testing.do_bench_cudagraph(original, rep=50))
            rows.append(original_row)
            with args.output.with_suffix(".jsonl").open("a") as handle:
                handle.write(json.dumps(original_row)+"\n")
            print(json.dumps(original_row), flush=True)
            for bm, bn, warps in selected_configs:
                if shape.startswith("decode"):
                    bv = triton.next_power_of_2(width)
                    part = torch.empty((q.shape[0], 4, bm, bv), device=q.device)
                    maximum = torch.empty((q.shape[0], 4, bm), device=q.device)
                    total = torch.empty_like(maximum)
                def run():
                    if shape.startswith("decode"):
                        compiled = _decode[(lens.numel(), bm)](
                            q, kc.storage, vc.storage, kc.lower, kc.upper, kc.lut,
                            vc.lower, vc.upper, vc.lut, rope, output, part, maximum, total,
                            cu, lens, table, *q.stride()[:2], *output.stride()[:2],
                            table.stride(0), tuple(kc.kernel_constants().values()),
                            tuple(vc.kernel_constants().values()), bm, bv, bn,
                            num_warps=warps, num_stages=1, enable_fp_fusion=False)
                        if bm > 1:
                            _reduce[(lens.numel(), 4)](part, maximum, total, output, cu,
                                *output.stride()[:2], bm, width, bv, num_warps=4)
                        return compiled
                    return _attention[(q.shape[0] // (bm // 4) + lens.numel(),)](
                        q, kc.storage, vc.storage, kc.lower, kc.upper, kc.lut,
                        vc.lower, vc.upper, vc.lut, rope, output, cu, lens, table,
                        128 ** -0.5, *q.stride()[:2], *output.stride()[:2],
                        table.stride(0), lens.numel(), tuple(kc.kernel_constants().values()),
                        tuple(vc.kernel_constants().values()), bm, bn, triton.next_power_of_2(width),
                        num_warps=warps, num_stages=1, enable_fp_fusion=False)
                compiled = run()
                torch.testing.assert_close(output, reference, atol=0.008, rtol=0.015)
                assert torch.isfinite(output).all()
                latency = triton.testing.do_bench_cudagraph(run, rep=50)
                row = dict(width=width, shape=shape, bm=bm, bn=bn, warps=warps, stages=1,
                           ms=latency, registers=compiled.n_regs, spills=compiled.n_spills,
                           shared=compiled.metadata.shared,
                           max_abs=float((output.float()-reference.float()).abs().max()))
                if shape.startswith("decode"):
                    row["splits"] = row.pop("bm")
                rows.append(row)
                with args.output.with_suffix(".jsonl").open("a") as handle:
                    handle.write(json.dumps(row)+"\n")
                print(json.dumps(row), flush=True)
            del q, kc, vc, rope, cu, lens, table, reference, output
    args.output.write_text(json.dumps(dict(environment="basis", rows=rows), indent=2)+"\n")


if __name__ == "__main__":
    main()
