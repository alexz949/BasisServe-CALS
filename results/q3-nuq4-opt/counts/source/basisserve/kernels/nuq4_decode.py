# SPDX-License-Identifier: Apache-2.0
# Split-K online softmax follows basisserve/kernels/diffkv_decode.py.
"""GQA4 split-K decode directly from packed NUQ4 pages."""

import torch
import triton
import triton.language as tl
from vllm.v1.attention.ops.triton_attention_helpers import softmax_step

from basisserve.kernels.nuq4_cache import load_nuq4_tile


@triton.jit
def _decode(Q, KC, VC, KLO, KHI, KLUT, VLO, VHI, VLUT, ROPE, O,
            PART, PMAX, PSUM, CU, LENS, TABLE,
            QS0: tl.constexpr, QS1: tl.constexpr, OS0: tl.constexpr,
            OS1: tl.constexpr, TS: tl.constexpr, KL: tl.constexpr,
            VL: tl.constexpr, SPLITS: tl.constexpr, BV: tl.constexpr,
            BN: tl.constexpr):
    seq, split = tl.program_id(0), tl.program_id(1)
    start = tl.load(CU + seq)
    if start == tl.load(CU + seq + 1):
        return
    length = tl.load(LENS + seq)
    per_split = tl.cdiv(length, SPLITS * BN)
    begin = split * per_split
    end = tl.minimum(begin + per_split, tl.cdiv(length, BN))
    h = tl.arange(0, 16)
    d = tl.arange(0, 128)
    dv = tl.arange(0, BV)
    query = tl.load(Q + start * QS0 + h[:, None] * QS1 + d[None, :], h[:, None] < 4, 0)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, BV), tl.float32)
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
        scores = tl.dot(query, tl.trans(k)) * 0.08838834764831845
        scores = tl.where((h[:, None] < 4) & valid[None, :], scores, -float("inf"))
        maximum, denominator, p, alpha = softmax_step(scores, maximum, denominator)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    if SPLITS == 1:
        tl.store(O + start * OS0 + h[:, None] * OS1 + dv[None, :],
                  acc / denominator[:, None], (h[:, None] < 4) & (dv[None, :] < VL[0]))
    else:
        base = (start * 4 + h) * SPLITS + split
        tl.store(PART + base[:, None] * BV + dv[None, :], acc, h[:, None] < 4)
        tl.store(PMAX + base, maximum, h < 4)
        tl.store(PSUM + base, denominator, h < 4)


@triton.jit
def _reduce(PART, PMAX, PSUM, O, CU, OS0: tl.constexpr, OS1: tl.constexpr,
            SPLITS: tl.constexpr, D: tl.constexpr, BV: tl.constexpr):
    seq, h = tl.program_id(0), tl.program_id(1)
    row = tl.load(CU + seq)
    if row == tl.load(CU + seq + 1):
        return
    s = tl.arange(0, SPLITS)
    d = tl.arange(0, BV)
    base = (row * 4 + h) * SPLITS + s
    maximum = tl.load(PMAX + base)
    total = tl.load(PSUM + base)
    weight = tl.exp(maximum - tl.max(maximum, 0))
    acc = tl.load(PART + base[:, None] * BV + d[None, :])
    value = tl.sum(acc * weight[:, None], 0) / tl.sum(total * weight, 0)
    tl.store(O + row * OS0 + h * OS1 + d, value, d < D)


def decode_workspace(rows, width, device):
    splits = min(32, triton.next_power_of_2(max(1, triton.cdiv(512, rows))))
    part = torch.empty((rows, 4, splits, triton.next_power_of_2(width)), device=device, dtype=torch.float32)
    maximum = torch.empty((rows, 4, splits), device=device, dtype=torch.float32)
    return part, maximum, torch.empty_like(maximum)


def nuq4_decode(query, keys, values, rope, cu_query, seq_lens, block_table, *, out, workspace):
    assert query.shape[1:] == (4, 128) and query.dtype == torch.bfloat16
    assert query.stride(-1) == out.stride(-1) == 1 and out.dtype == query.dtype
    assert keys.layout.width == 128 and not keys.dynamic and values.dynamic
    assert keys.layout.block == values.layout.block
    assert out.shape == (query.shape[0], 4, values.layout.width)
    part, maximum, total = workspace
    splits, bv = part.shape[2:]
    assert part.shape[:2] == (query.shape[0], 4)
    assert maximum.shape == total.shape == part.shape[:3]
    assert bv == triton.next_power_of_2(values.layout.width)
    assert all(t.is_contiguous() for t in workspace)
    assert all(t.dtype == torch.float32 and t.device == query.device for t in workspace)
    assert all(t.device == query.device for t in (keys.storage, values.storage, rope, cu_query, seq_lens, block_table, out))
    assert rope.is_contiguous() and rope.dtype == torch.bfloat16 and rope.shape[1] == 128
    assert cu_query.is_contiguous() and seq_lens.is_contiguous() and block_table.stride(-1) == 1
    _decode[(seq_lens.numel(), splits)](query, keys.storage, values.storage,
        keys.lower, keys.upper, keys.lut, values.lower, values.upper, values.lut,
        rope, out, part, maximum, total, cu_query, seq_lens, block_table,
        *query.stride()[:2], *out.stride()[:2], block_table.stride(0),
        tuple(keys.kernel_constants().values()), tuple(values.kernel_constants().values()),
        splits, bv, 32, num_warps=4, num_stages=1, enable_fp_fusion=False)
    if splits > 1:
        _reduce[(seq_lens.numel(), 4)](part, maximum, total, out, cu_query,
            *out.stride()[:2], splits, values.layout.width, bv, num_warps=4)
    return out
