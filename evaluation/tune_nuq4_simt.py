"""Compare GQA4 SIMT decode with the serving tensor-core kernel."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import triton
import triton.language as tl
from vllm.v1.attention.ops.triton_attention_helpers import softmax_step

from basisserve.kernels.nuq4_cache import load_nuq4_tile
from basisserve.kernels.nuq4_decode import _reduce, decode_workspace, nuq4_decode
from evaluation.tune_nuq4_prefill import fixture


@triton.jit
def _simt(Q, KC, VC, KLO, KHI, KLUT, VLO, VHI, VLUT, ROPE,
          PART, PMAX, PSUM, CU, LENS, TABLE,
          QS0: tl.constexpr, QS1: tl.constexpr, TS: tl.constexpr,
          KL: tl.constexpr, VL: tl.constexpr, SPLITS: tl.constexpr,
          BV: tl.constexpr, BN: tl.constexpr):
    seq, split = tl.program_id(0), tl.program_id(1)
    start = tl.load(CU + seq)
    if tl.load(CU + seq + 1) - start != 1:
        return
    length = tl.load(LENS + seq)
    per_split = tl.cdiv(length, SPLITS * BN)
    begin = split * per_split
    end = tl.minimum(begin + per_split, tl.cdiv(length, BN))
    h, d, dv = tl.arange(0, 4), tl.arange(0, 128), tl.arange(0, BV)
    query = tl.load(Q + start * QS0 + h[:, None] * QS1 + d[None, :]).to(tl.float32)
    maximum = tl.full((4,), -float("inf"), tl.float32)
    denominator = tl.zeros((4,), tl.float32)
    acc = tl.zeros((4, BV), tl.float32)
    for tile in range(begin, end):
        pos = tile * BN + tl.arange(0, BN)
        valid = pos < length
        physical = tl.load(TABLE + seq * TS + pos // KL[1], valid, 0).to(tl.int64)
        slots = physical * KL[1] + pos % KL[1]
        k = load_nuq4_tile(KC, KLO, KHI, KLUT, slots, valid, False,
            KL[0], KL[1], KL[2], KL[3], KL[4], KL[5], KL[6], KL[7], KL[8], KL[9]).to(tl.float32)
        mate = tl.gather(k, tl.broadcast_to(((d + 64) % 128)[None, :], (BN, 128)), 1)
        cos = tl.load(ROPE + pos[:, None] * 128 + (d % 64)[None, :], valid[:, None], 0).to(tl.float32)
        sin = tl.load(ROPE + pos[:, None] * 128 + (d % 64)[None, :] + 64, valid[:, None], 0).to(tl.float32)
        k = (k * cos + tl.where(d[None, :] < 64, -mate, mate) * sin).to(tl.bfloat16)
        v = load_nuq4_tile(VC, VLO, VHI, VLUT, slots, valid, True,
            VL[0], VL[1], VL[2], VL[3], VL[4], VL[5], VL[6], VL[7], VL[8], VL[9])
        scores = tl.sum(query[:, None, :] * k[None, :, :].to(tl.float32), 2) * 0.08838834764831845
        scores = tl.where(valid[None, :], scores, -float("inf"))
        maximum, denominator, p, alpha = softmax_step(scores, maximum, denominator)
        rounded_p = p.to(tl.bfloat16).to(tl.float32)
        acc = acc * alpha[:, None] + tl.sum(rounded_p[:, :, None] * v[None, :, :].to(tl.float32), 1)
    base = (seq * 4 + h) * SPLITS + split
    tl.store(PART + base[:, None] * BV + dv[None, :], acc)
    tl.store(PMAX + base, maximum)
    tl.store(PSUM + base, denominator)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 96])
    parser.add_argument("--shapes", nargs="+", default=["decode128"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(row):
        rows.append(row)
        with args.output.with_suffix(".jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    for width in args.widths:
        for shape in args.shapes:
            assert shape.startswith("decode")
            q, kc, vc, rope, cu, lens, table = fixture(width, shape)
            part, maximum, total = decode_workspace(lens.numel(), width, q.device)
            splits, bv = part.shape[2:]
            reference = torch.empty((q.shape[0], 4, width), device=q.device, dtype=q.dtype)
            out = torch.empty_like(reference)

            def original():
                nuq4_decode(q, kc, vc, rope, cu, lens, table, out=reference,
                            workspace=(part, maximum, total))

            original()
            record(dict(width=width, shape=shape, production=True,
                        ms=triton.testing.do_bench_cudagraph(original, rep=50)))
            for bn, warps in ((16, 4), (32, 4), (64, 4), (16, 8), (32, 8), (64, 8)):
                def run():
                    compiled = _simt[(lens.numel(), splits)](q, kc.storage, vc.storage,
                        kc.lower, kc.upper, kc.lut, vc.lower, vc.upper, vc.lut, rope,
                        part, maximum, total, cu, lens, table, *q.stride()[:2], table.stride(0),
                        tuple(kc.kernel_constants().values()), tuple(vc.kernel_constants().values()),
                        splits, bv, bn, num_warps=warps, num_stages=1, enable_fp_fusion=False)
                    _reduce[(lens.numel(), 4)](part, maximum, total, out, cu,
                        *out.stride()[:2], splits, width, bv, num_warps=4)
                    return compiled

                compiled = run()
                torch.testing.assert_close(out, reference, atol=0.008, rtol=0.015)
                assert torch.isfinite(out).all()
                record(dict(width=width, shape=shape, bn=bn, warps=warps, splits=splits,
                    ms=triton.testing.do_bench_cudagraph(run, rep=50), registers=compiled.n_regs,
                    spills=compiled.n_spills, shared=compiled.metadata.shared,
                    max_abs=float((out.float()-reference.float()).abs().max())))
    args.output.write_text(json.dumps(dict(environment="basis", rows=rows), indent=2)+"\n")


if __name__ == "__main__":
    main()
