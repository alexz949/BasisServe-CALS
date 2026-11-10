"""Replicated PaLU group latent in vLLM's fused DiffKV kernel: cache packing and attention.

Runs inside the vLLM environment on one GPU. The attention reference recomputes, per query
head, softmax(q k^T) over its own KV head's dense Keys and applies it to the single group
latent shared by all local heads.
"""

import pytest
import torch

pytest.importorskip("vllm")
from vllm.v1.attention.ops.triton_unified_attention_diffkv import unified_attention_diffkv

from basisserve.vllm.palu_group_attention import group_cache_views, pack_group_slots

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
KV_HEADS, HEADS, BLOCK = 2, 8, 32


def packed_cache(blocks, rank):
    return torch.zeros(blocks, 1, BLOCK, KV_HEADS * 128 + rank, device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("rank", [224, 416, 512])
def test_pack_writes_keys_and_one_latent_per_slot(rank):
    torch.manual_seed(0)
    tokens, blocks = 70, 5
    key = torch.randn(tokens, KV_HEADS, 128, device="cuda", dtype=torch.bfloat16)
    latent = torch.randn(tokens, rank, device="cuda", dtype=torch.bfloat16)
    slots = torch.randperm(blocks * BLOCK, device="cuda")[:tokens]
    slots[::7] = -1  # padded tokens must leave the cache untouched
    cache = packed_cache(blocks, rank)
    pack_group_slots(key, latent, cache[:, 0], slots)
    flat = cache[:, 0].reshape(blocks * BLOCK, -1)
    valid = slots >= 0
    assert torch.equal(flat[slots[valid], :KV_HEADS * 128], key[valid].reshape(-1, KV_HEADS * 128))
    assert torch.equal(flat[slots[valid], KV_HEADS * 128:], latent[valid])
    untouched = torch.ones(blocks * BLOCK, dtype=torch.bool, device="cuda")
    untouched[slots[valid]] = False
    assert not flat[untouched].any()


def reference(query, keys, latent, lengths, query_lengths, block_table):
    """Float32 attention with per-KV-head Keys and a shared latent, causal within a sequence."""
    outputs = []
    start = 0
    for seq, (length, count) in enumerate(zip(lengths.tolist(), query_lengths.tolist())):
        slots = (block_table[seq, :, None] * BLOCK + torch.arange(BLOCK, device="cuda")).flatten()[:length]
        k = keys.reshape(-1, KV_HEADS, 128)[slots].float()
        z = latent.reshape(-1, latent.shape[-1])[slots].float()
        for position in range(count):
            q = query[start + position].float()
            visible = length - count + position + 1
            rows = []
            for head in range(HEADS):
                scores = (k[:visible, head // (HEADS // KV_HEADS)] @ q[head]) * 128 ** -0.5
                rows.append(torch.softmax(scores, dim=0) @ z[:visible])
            outputs.append(torch.stack(rows))
        start += count
    return torch.stack(outputs)


@pytest.mark.parametrize("rank,query_lengths,lengths", [
    (416, [1, 1, 1], [70, 33, 160]),      # decode
    (224, [40, 1, 17], [40, 90, 100]),     # prefill and mixed
    (512, [1] * 6, [31, 32, 33, 64, 65, 300]),
])
def test_attention_shares_one_latent_across_local_heads(rank, query_lengths, lengths):
    torch.manual_seed(1)
    lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    query_lengths = torch.tensor(query_lengths, device="cuda", dtype=torch.int32)
    max_blocks = int((lengths.max().item() + BLOCK - 1) // BLOCK)
    total_blocks = max_blocks * len(lengths)
    cache = packed_cache(total_blocks, rank)
    cache.normal_()
    block_table = torch.randperm(total_blocks, device="cuda", dtype=torch.int32).view(len(lengths), max_blocks)
    flat, keys, shared = group_cache_views(cache, KV_HEADS, rank)
    assert shared.stride(2) == 0
    tokens = int(query_lengths.sum())
    query = torch.randn(tokens, HEADS, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(tokens, HEADS, rank, device="cuda", dtype=torch.bfloat16)
    starts = torch.zeros(len(lengths) + 1, device="cuda", dtype=torch.int32)
    starts[1:] = torch.cumsum(query_lengths, 0)
    unified_attention_diffkv(
        q=query, k=keys, v=shared, out=out, cu_seqlens_q=starts, seqused_k=lengths,
        softmax_scale=128 ** -0.5, causal=True, window_size=(-1, -1), block_table=block_table,
        softcap=0.0, max_seqlen_q=int(query_lengths.max()))
    expected = reference(query, flat[..., :KV_HEADS * 128], flat[..., KV_HEADS * 128:KV_HEADS * 128 + rank],
                         lengths, query_lengths, block_table)
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
