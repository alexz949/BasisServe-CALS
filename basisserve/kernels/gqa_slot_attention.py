"""Shared-KV GQA split attention with independent K-slot and V-token indices."""
import triton
import triton.language as tl
from basisserve.kernels.slot_indexed_attention import _merge_kernel


@triton.jit
def _gqa_split(Q, K, V, IDS, SLOTS, PART, LSE,
               QS0: tl.constexpr, QS1: tl.constexpr,
               KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
               VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
               H: tl.constexpr, G: tl.constexpr, BG: tl.constexpr,
               D: tl.constexpr, DV: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
               N: tl.constexpr, CAP: tl.constexpr, SPLITS: tl.constexpr, PER: tl.constexpr,
               SCALE: tl.constexpr):
    row = tl.program_id(0); split = tl.program_id(1)
    batch = row // H; kv = row % H
    heads = tl.arange(0, BG)
    dims = tl.arange(0, BD); vdims = tl.arange(0, BV)
    query = tl.load(Q + batch * QS0 + (kv * G + heads[:, None]) * QS1 + dims[None, :],
                    (heads[:, None] < G) & (dims[None, :] < D), 0).to(tl.float32)
    maximum = tl.full((BG,), -float('inf'), tl.float32)
    denominator = tl.zeros((BG,), tl.float32)
    accumulator = tl.zeros((BG, BV), tl.float32)
    for start in range(split * PER, (split + 1) * PER, 32):
        offsets = start + tl.arange(0, 32)
        tokens = tl.load(IDS + row * N + offsets, offsets < N, -1)
        slots = tl.load(SLOTS + row * N + offsets, offsets < N, -1)
        valid = (offsets < N) & (tokens >= 0) & (tokens < CAP) & (slots >= 0) & (slots < N)
        keys = tl.load(K + batch * KS0 + kv * KS1 + slots[:, None] * KS2 + dims[None, :],
                       valid[:, None] & (dims[None, :] < D), 0).to(tl.float32)
        scores = tl.sum(query[:, None, :] * keys[None, :, :], 2) * SCALE
        scores = tl.where(valid[None, :], scores, -float('inf'))
        any_valid = tl.sum(valid.to(tl.int32), 0) > 0
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(any_valid, next_maximum, 0.)
        correction = tl.where(any_valid, tl.exp(maximum - safe_max), 1.)
        probabilities = tl.where(valid[None, :], tl.exp(scores - safe_max[:, None]), 0.)
        values = tl.load(V + batch * VS0 + kv * VS1 + tokens[:, None] * VS2 + vdims[None, :],
                         valid[:, None] & (vdims[None, :] < DV), 0).to(tl.float32)
        accumulator = accumulator * correction[:, None] + tl.sum(probabilities[:, :, None] * values[None, :, :], 1)
        denominator = denominator * correction + tl.sum(probabilities, 1)
        maximum = tl.where(any_valid, next_maximum, maximum)
    qrow = batch * H * G + kv * G + heads
    result = accumulator / tl.where(denominator > 0, denominator, 1.)[:, None]
    tl.store(PART + (qrow[:, None] * SPLITS + split) * DV + vdims[None, :], result,
             (heads[:, None] < G) & (vdims[None, :] < DV))
    logs = tl.where(denominator > 0, maximum + tl.log(denominator), -float('inf'))
    tl.store(LSE + qrow * SPLITS + split, logs, heads < G)


def gqa_slot_attention(q, k, v, ids, slots, workspace, *, scale, num_warps=4):
    partial, lse, out = workspace
    batch, heads, qt, dim = q.shape
    kv_heads = k.shape[1]; group = heads // kv_heads
    assert qt == 1 and heads % kv_heads == 0
    assert ids.is_contiguous() and slots.is_contiguous() and ids.shape == slots.shape
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    splits = partial.shape[2]; width = v.shape[-1]
    per = triton.cdiv(triton.cdiv(ids.shape[-1], splits), 32) * 32
    _gqa_split[(batch * kv_heads, splits)](q, k, v, ids, slots, partial, lse,
        q.stride(0), q.stride(1), *k.stride()[:3], *v.stride()[:3],
        kv_heads, group, triton.next_power_of_2(group), dim, width,
        triton.next_power_of_2(dim), triton.next_power_of_2(width),
        ids.shape[-1], v.shape[2], splits, per, scale, num_warps=num_warps)
    _merge_kernel[(batch * heads,)](partial, lse, out, R=width, BR=triton.next_power_of_2(width), SPLITS=splits, num_warps=4)
    return out
