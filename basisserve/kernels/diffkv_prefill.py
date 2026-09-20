# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's triton_unified_attention_diffkv.py (vLLM contributors).
"""SM89 QK128/V64 paged prefill specialization of vLLM's attention kernel.

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
    dv = tl.arange(0, 64)
    q = tl.load(Q + tokens[:, None] * q_stride0 + heads[:, None] * q_stride1
                + dq[None, :], mask=valid_q[:, None], other=0)
    context = kv_len - q_len
    prefix = tl.minimum(kv_len, context + (local_block + 1) * BLOCK_Q)
    maximum = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    denominator = tl.full((BLOCK_M,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_M, 64), tl.float32)
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
                    mask=valid_kv[:, None], other=0)
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
             + dv[None, :], answer, mask=valid_q[:, None])


def diffkv_prefill(q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                  softmax_scale, *, block_m=64, block_n=64, num_warps=4,
                  num_stages=2):
    assert q.shape[-1] == k.shape[-1] == 128 and v.shape[-1] == 64
    assert k.shape[2] == 1 and q.shape[1] in (4, 8)
    group = q.shape[1]
    block_q = block_m // group
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == out.stride(-1) == 1
    kernel_diffkv_prefill_sm89[(q.shape[0] // block_q + len(seqused_k),)](
        q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale,
        q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        block_table.stride(0), len(seqused_k),
        GROUP=group, PAGE=k.shape[1], BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=num_warps, num_stages=num_stages,
    )
