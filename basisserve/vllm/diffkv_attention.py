"""Separate Qwen3 C1 prefill and decode specializations for L40S.

Keep vLLM's DiffKV backend identity/layout and metadata builder. Only prefill
dispatch changes; decode retains its existing split-KV kernel and reduction.
"""

from vllm.v1.attention.backends.triton_attn_diffkv import (
    TritonAttentionDiffKVBackend,
    TritonAttentionDiffKVImpl,
)

from basisserve.kernels.diffkv_prefill import diffkv_prefill


class C1DiffKVBackend(TritonAttentionDiffKVBackend):
    @staticmethod
    def get_impl_cls():
        return C1DiffKVImpl


class C1DiffKVImpl(TritonAttentionDiffKVImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.head_size == 128 and self.num_heads in (4, 8)
        assert self.num_kv_heads == 1
        assert self.alibi_slopes is None and self.sinks is None
        assert self.sliding_window == (-1, -1) and self.logits_soft_cap == 0
        assert not self.use_alibi_sqrt

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        assert output_scale is None and output_block_scale is None
        if attn_metadata is None:
            return output.fill_(0)
        if attn_metadata.max_query_len <= 1:
            return super().forward(layer, query, key, value, kv_cache,
                                   attn_metadata, output)
        assert not attn_metadata.use_cascade
        n = attn_metadata.num_actual_tokens
        cache = kv_cache.transpose(1, 2)
        diffkv_prefill(
            q=query[:n], k=cache[..., :128], v=cache[..., 128:192],
            out=output[:n], cu_seqlens_q=attn_metadata.query_start_loc,
            seqused_k=attn_metadata.seq_lens,
            block_table=attn_metadata.block_table, softmax_scale=self.scale,
        )
        return output
