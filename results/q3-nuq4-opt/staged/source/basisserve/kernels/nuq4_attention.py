# SPDX-License-Identifier: Apache-2.0
# The tiled causal loop follows basisserve/kernels/diffkv_prefill.py.
"""Full-context attention reading packed NUQ4 pages directly.

This tiled kernel handles prefill and mixed batches, with local GQA4.
Pure decode uses a separate split-K implementation. K is stored pre-RoPE;
positions start at zero, with no sliding window or position remapping.
"""

import torch
import triton
import triton.language as tl
from vllm.v1.attention.ops.triton_attention_helpers import resolve_seq_and_query_len, softmax_step

from basisserve.kernels.nuq4_cache import load_nuq4_tile


@triton.jit(do_not_specialize=["NSEQS"])
def _attention(Q, KC, VC, KLO, KHI, KLUT, VLO, VHI, VLUT, ROPE,
               O, CU, LENS, TABLE, scale,
               QS0: tl.constexpr, QS1: tl.constexpr,
               OS0: tl.constexpr, OS1: tl.constexpr,
               TS: tl.constexpr, NSEQS,
               KL: tl.constexpr, VL: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BV: tl.constexpr,
               PREFILL_ONLY: tl.constexpr = False):
    BQ: tl.constexpr = BM // 4
    seq, local, start, qlen, kvlen = resolve_seq_and_query_len(CU, LENS, tl.program_id(0), NSEQS, BQ)
    if local * BQ >= qlen:
        return
    if PREFILL_ONLY and qlen == 1:
        return
    row = tl.arange(0, BM)
    qpos = local * BQ + row // 4
    token = start + qpos
    head = row % 4
    dq = tl.arange(0, 128)
    dv = tl.arange(0, BV)
    q = tl.load(Q + token[:, None] * QS0 + head[:, None] * QS1 + dq[None, :],
                 qpos[:, None] < qlen, 0)
    context = kvlen - qlen
    prefix = tl.minimum(kvlen, context + (local + 1) * BQ)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.full((BM,), 1.0, tl.float32)
    acc = tl.zeros((BM, BV), tl.float32)
    for tile in range(tl.cdiv(prefix, BN)):
        pos = tile * BN + tl.arange(0, BN)
        valid = pos < prefix
        physical = tl.load(TABLE + seq * TS + pos // KL[1], valid, 0).to(tl.int64)
        slots = physical * KL[1] + pos % KL[1]
        k = load_nuq4_tile(KC, KLO, KHI, KLUT, slots, valid, False,
            KL[0], KL[1], KL[2], KL[3], KL[4], KL[5], KL[6], KL[7], KL[8], KL[9]).to(tl.float32)
        mate = tl.gather(k, tl.broadcast_to(((dq + 64) % 128)[None, :], (BN, 128)), 1)
        cos = tl.load(ROPE + pos[:, None] * 128 + (dq % 64)[None, :], valid[:, None], 0).to(tl.float32)
        sin = tl.load(ROPE + pos[:, None] * 128 + (dq % 64)[None, :] + 64, valid[:, None], 0).to(tl.float32)
        rotated = (k * cos + tl.where(dq[None, :] < 64, -mate, mate) * sin).to(tl.bfloat16)
        v = load_nuq4_tile(VC, VLO, VHI, VLUT, slots, valid, True,
            VL[0], VL[1], VL[2], VL[3], VL[4], VL[5], VL[6], VL[7], VL[8], VL[9])
        scores = tl.dot(q, tl.trans(rotated)) * scale
        causal = pos[None, :] <= context + qpos[:, None]
        scores = tl.where((qpos[:, None] < qlen) & valid[None, :] & causal, scores, -float("inf"))
        maximum, denom, probabilities, alpha = softmax_step(scores, maximum, denom)
        acc = acc * alpha[:, None] + tl.dot(probabilities.to(tl.bfloat16), v)
    answer = acc / denom[:, None]
    tl.store(O + token[:, None] * OS0 + head[:, None] * OS1 + dv[None, :], answer,
              (qpos[:, None] < qlen) & (dv[None, :] < VL[0]))


def nuq4_attention(query, keys, values, rope, cu_query, seq_lens, block_table, *, out=None, prefill_only=False):
    """query is post-RoPE, [tokens, 4, 128]; caches share physical slot IDs."""
    assert query.dtype == torch.bfloat16 and query.shape[1:] == (4, 128)
    assert query.stride(-1) == 1
    assert keys.layout.width == 128 and not keys.dynamic and values.dynamic
    assert keys.layout.block == values.layout.block
    assert keys.storage.shape[0] == values.storage.shape[0]
    assert rope.is_contiguous() and rope.shape[1] == 128 and rope.dtype == torch.bfloat16
    assert cu_query.numel() == seq_lens.numel() + 1 and cu_query.is_contiguous()
    assert seq_lens.is_contiguous() and block_table.stride(-1) == 1
    assert block_table.shape[0] == seq_lens.numel()
    assert all(t.device == query.device for t in (keys.storage, values.storage, rope, cu_query, seq_lens, block_table))
    if out is None:
        out = torch.empty((query.shape[0], 4, values.layout.width), device=query.device, dtype=query.dtype)
    assert out.shape == (query.shape[0], 4, values.layout.width)
    assert out.dtype == query.dtype and out.device == query.device and out.stride(-1) == 1
    bm, bn = 256, 32
    _attention[(query.shape[0] // (bm // 4) + seq_lens.numel(),)](
        query, keys.storage, values.storage, keys.lower, keys.upper, keys.lut,
        values.lower, values.upper, values.lut, rope, out, cu_query, seq_lens,
        block_table, 128 ** -0.5, *query.stride()[:2], *out.stride()[:2],
        block_table.stride(0), seq_lens.numel(), tuple(keys.kernel_constants().values()),
        tuple(values.kernel_constants().values()), bm, bn, triton.next_power_of_2(values.layout.width),
        prefill_only, num_warps=16, num_stages=1, enable_fp_fusion=False)
    return out
