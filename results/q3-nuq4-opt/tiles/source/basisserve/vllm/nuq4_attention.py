"""Byte-accurate vLLM cache specifications for packed NUQ4 full attention."""

from dataclasses import replace
from functools import lru_cache

import torch
from vllm.v1.attention.backends.triton_attn_diffkv import TritonAttentionDiffKVBackend, TritonAttentionDiffKVImpl
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder

from basisserve.kernels.nuq4_cache import NUQ4Layout

SERVING_EXCEPTIONS_PER_TOKEN = 16


def serving_layout(width):
    return NUQ4Layout(width, exceptions_per_token=SERVING_EXCEPTIONS_PER_TOKEN)


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
            return TritonAttentionMetadataBuilder

    NUQ4Backend.__name__ = f"NUQ4BackendV{width}"
    NUQ4Backend.__qualname__ = NUQ4Backend.__name__
    return NUQ4Backend
