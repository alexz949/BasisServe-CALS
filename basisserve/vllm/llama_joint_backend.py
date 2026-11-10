"""vLLM 0.30 paged host-K backend for Llama Joint-ALS/B16R16."""

from dataclasses import dataclass, replace

import torch

from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.attention.backends.triton_attn_diffkv import (
    TritonAttentionDiffKVBackend,
    TritonAttentionDiffKVImpl,
    TritonAttentionDiffKVMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from basisserve.kernels.paged_joint_cache import PagedJointCache


@dataclass
class JointMetadata(TritonAttentionMetadata):
    num_decodes: int = 0
    num_decode_tokens: int = 0
    positions: torch.Tensor | None = None


class JointMetadataBuilder(TritonAttentionDiffKVMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_reorder_batch_threshold(1)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        common = common_attn_metadata
        # FULL graph metadata includes padded requests, while vLLM's default
        # model state supplies is_prefilling for real requests only. Padding
        # represents inert decode rows and must use False prefill flags.
        assert common.is_prefilling is not None
        if common.is_prefilling.numel() < common.num_reqs:
            is_prefilling = torch.zeros(common.num_reqs, dtype=torch.bool,
                                        device=common.is_prefilling.device)
            is_prefilling[:common.is_prefilling.numel()].copy_(common.is_prefilling)
            common = replace(common, is_prefilling=is_prefilling)
        base = super().build(common_prefix_len, common, fast_build)
        decodes, _, decode_tokens, _ = split_decodes_and_prefills(
            common, decode_threshold=1, treat_short_extends_as_decodes=False,
        )
        assert decodes == decode_tokens
        assert common.positions is not None
        return JointMetadata(**vars(base), num_decodes=decodes,
                             num_decode_tokens=decode_tokens, positions=common.positions)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        common = replace(common_attn_metadata, is_prefilling=torch.zeros(
            common_attn_metadata.num_reqs, dtype=torch.bool))
        metadata = self.build(0, common)
        metadata.seq_lens.fill_(1)
        return metadata


@register_backend(AttentionBackendEnum.CUSTOM)
class JointRoutingBackend(TritonAttentionDiffKVBackend):
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return JointRoutingImpl

    @staticmethod
    def get_builder_cls():
        return JointMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [32]

    @classmethod
    def customize_spec(cls, spec):
        return replace(spec, state_content_bytes=256)


class JointRoutingImpl(TritonAttentionDiffKVImpl):
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        assert output_scale is None and output_block_scale is None
        if attn_metadata is None:
            return output.fill_(0)
        meta = attn_metadata
        assert not meta.use_cascade
        assert kv_cache.shape[2:] == (32, 128)
        if layer.joint_cache is None or layer.joint_cache.gpu.data_ptr() != kv_cache.data_ptr():
            assert not torch.cuda.is_current_stream_capturing()
            host_bytes = kv_cache.numel() * kv_cache.element_size() * 32
            assert host_bytes <= layer.joint_host_budget
            layer.joint_cache = PagedJointCache(kv_cache)
        cache = layer.joint_cache
        n = meta.num_actual_tokens
        cache.append(key[:n], value[:n], meta.positions[:n], meta.slot_mapping[:n],
                     layer.base_right, layer.base_bias, layer.residual_encoder, layer.joint_rope)
        cache.update_bounds(meta.block_table, meta.seq_lens, meta.query_start_loc)
        decodes, decode_tokens = meta.num_decodes, meta.num_decode_tokens
        if decodes:
            table, lengths = meta.block_table[:decodes], meta.seq_lens[:decodes]
            support = layer.joint_router.select(
                query[:decode_tokens], cache, table, lengths,
                layer.base_right, layer.base_bias, layer.residual_query, layer.joint_rope,
            )
            layer.joint_slots.attend(query[:decode_tokens], cache, table, lengths,
                                     support, output[:decode_tokens])
        if decodes < meta.seq_lens.numel():
            table, lengths = meta.block_table[decodes:], meta.seq_lens[decodes:]
            dense, dense_table = cache.prefill_cache(table, lengths, meta.max_seq_len)
            prefill = replace(meta, num_actual_tokens=n - decode_tokens,
                              query_start_loc=meta.query_start_loc[decodes:] - decode_tokens,
                              seq_lens=lengths, block_table=dense_table)
            super().forward(layer, query[decode_tokens:n], key[decode_tokens:n],
                            value[decode_tokens:n], dense, prefill, output[decode_tokens:n])
        return output
