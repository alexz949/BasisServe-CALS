"""vLLM attention for a PaLU G-LRD group latent sharded across its TP ranks.

Each rank stores its dense Keys and one shard of the group's latent features. Decode gathers
attention weights inside the group: every rank computes softmax(q K^T) for its own query heads,
each applies its own latent shard to all sets of weights, and AllToAll sends partial outputs
back to their query owners. Prefill instead gathers the latent shards of the scheduled
sequences and attends over a transient full-latent copy with the fused kernel. The output per
query head is the complete rank-r latent-weighted sum, in canonical feature order.
"""

from dataclasses import replace
from functools import lru_cache

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.ops.triton_unified_attention_diffkv import unified_attention_diffkv

from basisserve.vllm.palu_group_attention import (
    KEY_DIM, PaluGroupImpl, PaluGroupMetadataBuilder, group_cache_views, pack_group_slots,
    palu_group_backend)
from basisserve.vllm.palu_collectives import assemble_latent, exchange_outputs, gather_group

WEIGHT_TILE = 64


@triton.jit
def _decode_scores(q_ptr, cache_ptr, table_ptr, lens_ptr, out_ptr, q_stride_token, q_stride_head,
                   block_stride, token_stride, table_stride, out_stride_seq, out_stride_head,
                   scale, Q_PER_KV: tl.constexpr, BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
                   TILE: tl.constexpr):
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    length = tl.load(lens_ptr + seq)
    tokens = tl.program_id(2) * TILE + tl.arange(0, TILE)
    valid = tokens < length
    block = tl.load(table_ptr + seq * table_stride + tokens // BLOCK_SIZE, mask=valid, other=0).to(tl.int64)
    rows = block * block_stride + (tokens % BLOCK_SIZE) * token_stride + kv_head * HEAD_DIM
    dims = tl.arange(0, HEAD_DIM)
    keys = tl.load(cache_ptr + rows[:, None] + dims[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    for sub in tl.static_range(Q_PER_KV):
        head = kv_head * Q_PER_KV + sub
        query = tl.load(q_ptr + seq * q_stride_token + head * q_stride_head + dims).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        tl.store(out_ptr + seq * out_stride_seq + head * out_stride_head + tokens,
                 tl.where(valid, scores, float("-inf")))


@triton.jit
def _apply_weights(p_ptr, cache_ptr, table_ptr, lens_ptr, out_ptr, p_stride_seq, p_stride_head,
                   block_stride, token_stride, table_stride, out_stride_seq, out_stride_head,
                   shard_offset, shard, HEADS: tl.constexpr, HEAD_PAD: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr, TILE: tl.constexpr, R_TILE: tl.constexpr):
    seq = tl.program_id(0)
    columns = tl.program_id(1) * R_TILE + tl.arange(0, R_TILE)
    column_mask = columns < shard
    heads = tl.arange(0, HEAD_PAD)
    head_mask = heads < HEADS
    length = tl.load(lens_ptr + seq)
    acc = tl.zeros([HEAD_PAD, R_TILE], dtype=tl.float32)
    for start in range(0, length, TILE):
        tokens = start + tl.arange(0, TILE)
        valid = tokens < length
        block = tl.load(table_ptr + seq * table_stride + tokens // BLOCK_SIZE, mask=valid, other=0).to(tl.int64)
        rows = block * block_stride + (tokens % BLOCK_SIZE) * token_stride + shard_offset
        latent = tl.load(cache_ptr + rows[:, None] + columns[None, :],
                         mask=valid[:, None] & column_mask[None, :], other=0.0)
        weights = tl.load(p_ptr + seq * p_stride_seq + heads[:, None] * p_stride_head + tokens[None, :],
                          mask=head_mask[:, None] & valid[None, :], other=0.0)
        acc += tl.dot(weights, latent)
    tl.store(out_ptr + seq * out_stride_seq + heads[:, None] * out_stride_head + columns[None, :],
             acc, mask=head_mask[:, None] & column_mask[None, :])


def decode_weights(query, flat, block_table, lengths, max_length, kv_heads, scale):
    """softmax(q K^T) for every decode query over its own KV head's paged Keys: [seqs, heads, T].
    max_length is the scheduler's host-side bound on the lengths, so no device sync is needed."""
    seqs, heads, head_dim = query.shape
    tile = 128
    width = triton.cdiv(max_length, tile) * tile
    scores = torch.empty(seqs, heads, width, device=query.device, dtype=torch.float32)
    _decode_scores[(seqs, kv_heads, width // tile)](
        query, flat, block_table, lengths, scores, query.stride(0), query.stride(1),
        flat.stride(0), flat.stride(1), block_table.stride(0), scores.stride(0), scores.stride(1),
        scale, Q_PER_KV=heads // kv_heads, BLOCK_SIZE=flat.shape[1], HEAD_DIM=head_dim, TILE=tile)
    return torch.softmax(scores, dim=-1).to(query.dtype)


def apply_weights(weights, flat, block_table, lengths, kv_heads, shard):
    """Latent-shard weighted sums for every decode query: [seqs, heads, shard] in float32."""
    seqs, heads, _ = weights.shape
    r_tile = min(64, next_power_of_2(shard))
    out = torch.empty(seqs, heads, shard, device=weights.device, dtype=torch.float32)
    _apply_weights[(seqs, triton.cdiv(shard, r_tile))](
        weights, flat, block_table, lengths, out, weights.stride(0), weights.stride(1),
        flat.stride(0), flat.stride(1), block_table.stride(0), out.stride(0), out.stride(1),
        kv_heads * KEY_DIM, shard, HEADS=heads, HEAD_PAD=max(16, next_power_of_2(heads)),
        BLOCK_SIZE=flat.shape[1], TILE=WEIGHT_TILE, R_TILE=r_tile)
    return out


class PaluShardImpl(PaluGroupImpl):
    """The cache holds one equal shard of the full value_rank latent per token."""

    def __init__(self, *args, latent_group, **kwargs):
        super().__init__(*args, **kwargs)
        self.group, self.group_rank = latent_group
        self.group_size = dist.get_world_size(self.group)
        self.shard = self.value_rank // self.group_size
        self.traffic = dict(decode_weight_bytes=0, decode_partial_bytes=0, prefill_shard_bytes=0,
                            decode_steps=0, prefill_steps=0)

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        pack_group_slots(key, value[:, 0, :self.shard], kv_cache[:, 0], slot_mapping)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return output.fill_(0)
        seqs = len(attn_metadata.seq_lens)
        starts = attn_metadata.query_start_loc[:seqs + 1]
        query_lengths = starts[1:] - starts[:-1]
        flat = kv_cache[:, 0]
        # A pure decode step is known from host-side metadata; only mixed steps inspect the
        # device tensors, so steady decode never forces a device synchronization here.
        all_decode = attn_metadata.max_query_len == 1
        decode = None if all_decode else query_lengths == 1
        if all_decode or bool(decode.any()):
            rows = slice(0, seqs) if all_decode else starts[:-1][decode]
            lengths = (attn_metadata.seq_lens if all_decode else attn_metadata.seq_lens[decode]).to(torch.int32)
            table = attn_metadata.block_table[:seqs] if all_decode else attn_metadata.block_table[:seqs][decode]
            weights = decode_weights(query[rows], flat, table, lengths, attn_metadata.max_seq_len,
                                     self.num_kv_heads, self.scale)
            gathered = gather_group(weights, self.group)
            # Apply this latent shard to every group's query head in one kernel launch.
            all_weights = gathered.permute(1, 0, 2, 3).flatten(1, 2)
            partials = apply_weights(all_weights, flat, table, lengths, self.num_kv_heads, self.shard)
            partials = partials.view(weights.shape[0], self.group_size, weights.shape[1], self.shard)
            partials = partials.permute(1, 0, 2, 3).to(query.dtype).contiguous()
            output[rows] = exchange_outputs(partials, self.group)
            peers = self.group_size - 1
            self.traffic["decode_weight_bytes"] += peers * weights.numel() * weights.element_size()
            self.traffic["decode_partial_bytes"] += peers * partials[0].numel() * partials.element_size()
            self.traffic["decode_steps"] += 1
        if not all_decode and bool((~decode).any()):
            self.prefill(query, output, flat, attn_metadata, starts, query_lengths, ~decode)
        return output

    def prefill(self, query, output, flat, attn_metadata, starts, query_lengths, chosen):
        """Attend over a transient paged copy holding Keys and the complete latent."""
        block_size = flat.shape[1]
        seqs = len(attn_metadata.seq_lens)
        lengths = attn_metadata.seq_lens[:seqs][chosen].to(torch.int32)
        tables = attn_metadata.block_table[:seqs][chosen]
        blocks = (lengths + block_size - 1) // block_size
        offsets = torch.cumsum(blocks, 0) - blocks
        device = query.device
        max_blocks = int(blocks.max())
        compact = torch.arange(max_blocks, device=device, dtype=torch.int32)[None] + offsets[:, None].to(torch.int32)
        # Slots of every token of every chosen sequence, in the persistent and transient caches.
        positions = torch.arange(int(lengths.max()), device=device)
        valid = positions[None] < lengths[:, None]
        blocks_of = torch.gather(tables.long(), 1, positions[None].expand(len(lengths), -1) // block_size)
        within = (positions[None] % block_size).expand(len(lengths), -1)
        transient = (offsets[:, None] * block_size + positions[None])[valid]
        rows = flat[blocks_of[valid], within[valid]]
        keys = rows[:, :self.num_kv_heads * KEY_DIM]
        shard = rows[:, self.num_kv_heads * KEY_DIM:self.num_kv_heads * KEY_DIM + self.shard]
        latent = assemble_latent(gather_group(shard, self.group))
        scratch = torch.zeros(int(blocks.sum()), 1, block_size, self.num_kv_heads * KEY_DIM + self.value_rank,
                              device=device, dtype=query.dtype)
        pack_group_slots(keys.reshape(-1, self.num_kv_heads, KEY_DIM), latent, scratch[:, 0], transient.to(torch.int64))
        token_rows = torch.cat([torch.arange(int(s), int(s + n), device=device)
                                for s, n in zip(starts[:-1][chosen].tolist(), query_lengths[chosen].tolist())])
        cu = torch.zeros(len(lengths) + 1, device=device, dtype=torch.int32)
        cu[1:] = torch.cumsum(query_lengths[chosen], 0)
        result = torch.empty(len(token_rows), query.shape[1], self.value_rank, device=device, dtype=query.dtype)
        _, keys_view, latent_view = group_cache_views(scratch, self.num_kv_heads, self.value_rank)
        unified_attention_diffkv(
            q=query[token_rows], k=keys_view, v=latent_view, out=result, cu_seqlens_q=cu, seqused_k=lengths,
            softmax_scale=self.scale, causal=True, alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt, window_size=self.sliding_window, block_table=compact,
            softcap=self.logits_soft_cap, sinks=self.sinks, max_seqlen_q=int(query_lengths[chosen].max()))
        output[token_rows] = result
        self.traffic["prefill_shard_bytes"] += (self.group_size - 1) * shard.numel() * shard.element_size()
        self.traffic["prefill_steps"] += 1


@lru_cache(maxsize=None)
def palu_shard_backend(rank, group_size):
    base = palu_group_backend(rank)

    class PaluShardBackend(base):
        @classmethod
        def customize_spec(cls, spec):
            return replace(spec, num_head_slots=1,
                           state_content_bytes=(spec.num_kv_heads * KEY_DIM + rank // group_size) * 2)

        @staticmethod
        def get_impl_cls():
            return PaluShardImpl

        @staticmethod
        def get_builder_cls():
            return PaluGroupMetadataBuilder

    PaluShardBackend.__name__ = f"PaluShardBackendR{rank}G{group_size}"
    PaluShardBackend.__qualname__ = PaluShardBackend.__name__
    return PaluShardBackend
