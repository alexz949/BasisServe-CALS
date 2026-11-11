"""C1 DiffKV prefill and decode with one or two physical KV heads per rank.

Keep vLLM's DiffKV backend identity and cache layout. Launch parameters use
the fixed policy validated on GH200; rank-specific metadata builders preserve
V64/V96 cache dimensions and reserve split-KV scratch for all batch sizes.
"""

from functools import lru_cache

import torch

from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backends.triton_attn_diffkv import (
    TritonAttentionDiffKVBackend,
    TritonAttentionDiffKVImpl,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from basisserve.kernels.diffkv_decode import diffkv_decode
from basisserve.kernels.diffkv_prefill import diffkv_prefill
from basisserve.core.tp4_step import CURRENT_OUTPUT_STEP, classify_step


class C1DiffKVBackend(TritonAttentionDiffKVBackend):
    @staticmethod
    def get_impl_cls():
        return C1DiffKVImpl


class C1DiffKVImpl(TritonAttentionDiffKVImpl):
    def __init__(self, *args, **kwargs):
        self.value_rank = int(kwargs.pop("value_rank", 64))
        super().__init__(*args, **kwargs)
        assert self.head_size == 128 and self.num_kv_heads in (1, 2)
        assert self.num_heads // self.num_kv_heads in (4, 8)
        assert self.num_heads % self.num_kv_heads == 0
        assert self.value_rank in (64, 96)
        assert self.alibi_slopes is None and self.sinks is None
        assert self.sliding_window == (-1, -1) and self.logits_soft_cap == 0
        assert not self.use_alibi_sqrt

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        assert output_scale is None and output_block_scale is None
        if attn_metadata is None:
            return output.fill_(0)
        if attn_metadata.max_query_len <= 1:
            n = attn_metadata.num_actual_tokens
            cache = kv_cache.transpose(1, 2)
            diffkv_decode(
                q=query[:n],
                k=cache[..., :128],
                v=cache[..., 128:128 + self.value_rank],
                out=output[:n],
                cu_seqlens_q=attn_metadata.query_start_loc,
                seqused_k=attn_metadata.seq_lens,
                block_table=attn_metadata.block_table,
                softmax_scale=self.scale,
                softmax_segm_output=attn_metadata.softmax_segm_output,
                softmax_segm_max=attn_metadata.softmax_segm_max,
                softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
                split_threshold=attn_metadata.seq_threshold_3D,
                max_sequence_length=attn_metadata.max_seq_len,
            )
            return output
        assert not attn_metadata.use_cascade
        n = attn_metadata.num_actual_tokens
        cache = kv_cache.transpose(1, 2)
        diffkv_prefill(
            q=query[:n], k=cache[..., :128],
            v=cache[..., 128:128 + self.value_rank],
            out=output[:n], cu_seqlens_q=attn_metadata.query_start_loc,
            seqused_k=attn_metadata.seq_lens,
            block_table=attn_metadata.block_table, softmax_scale=self.scale,
            max_query_len=attn_metadata.max_query_len,
            max_sequence_length=attn_metadata.max_seq_len,
        )
        return output


@lru_cache(maxsize=None)
def c1_diffkv_backend(value_rank: int):
    """Return a cached DiffKV backend with V-sized split-KV scratch."""

    selected_rank = int(value_rank)
    assert selected_rank in (64, 96)

    class RankedMetadataBuilder(TritonAttentionMetadataBuilder):
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
            common = common_attn_metadata
            flags = common.is_prefilling
            metadata.basisserve_step = CURRENT_OUTPUT_STEP.get()
            if metadata.basisserve_step is not None:
                return metadata
            if flags is not None:
                assert flags.device.type == 'cpu'
                lengths = common.query_start_loc_cpu.diff().tolist()
                flags = flags.tolist()
                # Padded rows can have a synthetic query token but no context.
                if common.seq_lens_cpu_upper_bound is not None:
                    lengths = [n if context > 0 else 0 for n, context in
                               zip(lengths, common.seq_lens_cpu_upper_bound.tolist())]
                metadata.basisserve_step = classify_step(lengths, flags)
            return metadata

        def build_for_cudagraph_capture(self, common_attn_metadata):
            metadata = super().build_for_cudagraph_capture(common_attn_metadata)
            assert common_attn_metadata.max_query_len == 1
            metadata.basisserve_step = classify_step(
                [1] * common_attn_metadata.num_reqs, [False] * common_attn_metadata.num_reqs)
            return metadata

        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            # Fixed split-KV decode also needs scratch for large/padded batches.
            capture_sizes = vllm_config.compilation_config.cudagraph_capture_sizes
            scratch_sequences = max(
                vllm_config.scheduler_config.max_num_seqs,
                max(capture_sizes or [0]),
            )
            self.num_par_softmax_segments = 4
            scratch_shape = (
                scratch_sequences, self.num_heads_q, self.num_par_softmax_segments
            )
            self.softmax_segm_output = torch.empty(
                (*scratch_shape, next_power_of_2(selected_rank)),
                dtype=torch.float32,
                device=device,
            )
            self.softmax_segm_max = torch.empty(
                scratch_shape, dtype=torch.float32, device=device
            )
            self.softmax_segm_expsum = torch.empty_like(self.softmax_segm_max)

    class RankedBackend(C1DiffKVBackend):
        head_size_v = selected_rank

        @staticmethod
        def get_builder_cls():
            return RankedMetadataBuilder

    RankedMetadataBuilder.__name__ = f"C1DiffKVMetadataBuilderV{selected_rank}"
    RankedMetadataBuilder.__qualname__ = RankedMetadataBuilder.__name__
    RankedBackend.__name__ = f"C1DiffKVBackendV{selected_rank}"
    RankedBackend.__qualname__ = RankedBackend.__name__
    return RankedBackend


__all__ = ["C1DiffKVBackend", "C1DiffKVImpl", "c1_diffkv_backend"]
