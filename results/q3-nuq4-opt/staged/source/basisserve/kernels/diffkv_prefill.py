# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's triton_unified_attention_diffkv.py (vLLM contributors).
"""SM89 QK128 paged prefill specialization for V64 and V96.

Reuse upstream sequence lookup and online softmax, with a dedicated causal
prefill loop. BLOCK_M counts (query token, GQA head) pairs, not tokens.
No runtime autotuning: launches must remain safe inside CUDA Graph capture.
"""

import triton
import triton.language as tl
from vllm.v1.attention.ops.triton_attention_helpers import (
    resolve_seq_and_query_len,
    softmax_step,
)


@triton.jit
def kernel_diffkv_prefill_sm89(
    Q, K, V, O, CU_Q, SEQ_K, TABLE, scale,
    q_stride0: tl.int64, q_stride1: tl.int64,
    o_stride0: tl.int64, o_stride1: tl.int64,
    k_stride0: tl.int64, k_stride1: tl.int64,
    v_stride0: tl.int64, v_stride1: tl.int64,
    table_stride: tl.int64, num_seqs: tl.int32,
    GROUP: tl.constexpr, PAGE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    VALUE_DIM: tl.constexpr, BLOCK_V: tl.constexpr,
):
    BLOCK_Q: tl.constexpr = BLOCK_M // GROUP
    seq, local_block, start, q_len, kv_len = resolve_seq_and_query_len(
        CU_Q, SEQ_K, tl.program_id(0), num_seqs, BLOCK_Q)
    if local_block * BLOCK_Q >= q_len:
        return
    rows = tl.arange(0, BLOCK_M)
    qpos = local_block * BLOCK_Q + rows // GROUP
    heads = rows % GROUP
    tokens = start + qpos
    valid_q = qpos < q_len
    dq = tl.arange(0, 128)
    dv = tl.arange(0, BLOCK_V)
    valid_v = dv < VALUE_DIM
    q = tl.load(Q + tokens[:, None] * q_stride0 + heads[:, None] * q_stride1
                + dq[None, :], mask=valid_q[:, None], other=0)
    context = kv_len - q_len
    prefix = tl.minimum(kv_len, context + (local_block + 1) * BLOCK_Q)
    maximum = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    denominator = tl.full((BLOCK_M,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)
    for tile in range(tl.cdiv(prefix, BLOCK_N)):
        offsets = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_kv = offsets < prefix
        # Mask the page-table access itself, not just the subsequent KV loads.
        physical = tl.load(TABLE + seq * table_stride + offsets // PAGE,
                           mask=valid_kv, other=0).to(tl.int64)
        k = tl.load(K + physical[None, :] * k_stride0
                    + (offsets % PAGE)[None, :] * k_stride1 + dq[:, None],
                    mask=valid_kv[None, :], other=0)
        v = tl.load(V + physical[:, None] * v_stride0
                    + (offsets % PAGE)[:, None] * v_stride1 + dv[None, :],
                    mask=valid_kv[:, None] & valid_v[None, :], other=0)
        scores = scale * tl.dot(q, k)
        causal = offsets[None, :] <= context + qpos[:, None]
        scores = tl.where(valid_q[:, None] & valid_kv[None, :] & causal,
                          scores, float("-inf"))
        maximum, denominator, probabilities, alpha = softmax_step(
            scores, maximum, denominator)
        accumulator = accumulator * alpha[:, None]
        accumulator += tl.dot(probabilities.to(v.dtype), v)
    answer = accumulator / denominator[:, None]
    tl.store(O + tokens[:, None] * o_stride0 + heads[:, None] * o_stride1
             + dv[None, :], answer,
             mask=valid_q[:, None] & valid_v[None, :])


def _launch_config(
    group, value_dim, num_seqs, max_query_len, max_sequence_length
):
    assert value_dim in (64, 96)
    if value_dim == 96:
        if max_query_len is not None and max_query_len <= 128:
            return (32, 128, 8, 2)
        if (
            num_seqs == 1
            and max_sequence_length is not None
            and max_sequence_length >= 16384
        ):
            return (128, 32, 4, 3)
        return (128, 128, 8, 2)
    if max_query_len is not None and max_query_len <= 128:
        return (32, 128, 4, 3 if group == 4 else 2)
    if group == 8:
        return (128, 128, 8, 3)
    if num_seqs == 1:
        return (128, 64, 4, 3)
    return (64, 64, 4, 3)


def diffkv_prefill(q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                  softmax_scale, *, max_query_len=None,
                  max_sequence_length=None, block_m=None, block_n=None,
                  num_warps=None, num_stages=None):
    value_dim = int(v.shape[-1])
    assert q.shape[-1] == k.shape[-1] == 128 and value_dim in (64, 96)
    assert k.shape[2] == 1 and q.shape[1] in (4, 8)
    assert out.shape[1:] == (q.shape[1], value_dim)
    group = q.shape[1]
    block_v = triton.next_power_of_2(value_dim)
    if block_m is None:
        block_m, block_n, num_warps, num_stages = _launch_config(
            group, value_dim, len(seqused_k), max_query_len,
            max_sequence_length)
    block_q = block_m // group
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == out.stride(-1) == 1
    kernel_diffkv_prefill_sm89[(q.shape[0] // block_q + len(seqused_k),)](
        q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale,
        q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        block_table.stride(0), len(seqused_k),
        GROUP=group, PAGE=k.shape[1], BLOCK_M=block_m, BLOCK_N=block_n,
        VALUE_DIM=value_dim, BLOCK_V=block_v,
        num_warps=num_warps, num_stages=num_stages,
    )
