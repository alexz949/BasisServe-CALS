"""Byte-accurate vLLM cache specifications for packed NUQ4 full attention."""

from dataclasses import replace
from functools import lru_cache

import torch
import triton
from vllm.v1.attention.backends.triton_attn_diffkv import TritonAttentionDiffKVBackend, TritonAttentionDiffKVImpl
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder

from basisserve.kernels.nuq4_cache import NUQ4Layout
from basisserve.kernels.nuq4_prefill import plan_prefill

SERVING_EXCEPTIONS_PER_TOKEN = 16


def serving_layout(width):
    return NUQ4Layout(width, exceptions_per_token=SERVING_EXCEPTIONS_PER_TOKEN)


class NUQ4MetadataBuilder(TritonAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.prefill_capacity = triton.cdiv(max(vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.max_model_len), 16) * 16

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build=fast_build)
        metadata.nuq4_prefill_chunks = []
        if metadata.max_query_len > 1:
            upper = common_attn_metadata.seq_lens_cpu_upper_bound
            assert upper is not None, "Prefill planning requires scheduler CPU length bounds"
            count = common_attn_metadata.num_reqs
            metadata.nuq4_prefill_chunks = plan_prefill(common_attn_metadata.query_start_loc_cpu[:count+1],
                upper[:count], metadata.seq_lens, self.prefill_capacity)
        return metadata


class NUQ4Impl(TritonAttentionDiffKVImpl):
    def __init__(self, *args, **kwargs):
        kwargs.pop("value_rank")
        super().__init__(*args, **kwargs)


@lru_cache(maxsize=None)
def nuq4_backend(width):
    assert 16 <= width <= 128 and width % 16 == 0

    class NUQ4Backend(TritonAttentionDiffKVBackend):
        head_size_v = width

        @classmethod
        def customize_spec(cls, spec):
            assert spec.block_size == 16 and spec.num_kv_heads == 1
            assert spec.head_size == 128 and spec.head_size_v == width
            size = serving_layout(128).page_bytes + serving_layout(width).page_bytes
            return replace(spec, dtype=torch.uint8, tokens_per_state=16,
                           num_head_slots=1, state_content_bytes=size)

        @staticmethod
        def get_supported_kernel_block_sizes():
            return [16]

        @staticmethod
        def get_impl_cls():
            return NUQ4Impl

        @staticmethod
        def get_builder_cls():
            return NUQ4MetadataBuilder

    NUQ4Backend.__name__ = f"NUQ4BackendV{width}"
    NUQ4Backend.__qualname__ = NUQ4Backend.__name__
    return NUQ4Backend
