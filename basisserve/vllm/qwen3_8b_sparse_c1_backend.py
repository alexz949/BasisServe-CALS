"""vLLM Page32 backend for Qwen3-8B C1-V80 sparse-Key decode."""

from __future__ import annotations

import math

import torch

from basisserve.kernels.vllm_sparse_decode_attention import (
    CACHE_VALUE_DIM,
    PAGE_SIZE,
    ROUTING_AUX_DIM,
    VALUE_DIM,
    conditional_page32_log_mass,
    loki_scores,
    paged_sparse_attention,
    physical_union_count,
    select_group_shared_pages,
    selected_page_tokens,
)
from vllm.v1.attention.backends.flash_attn_diffkv import (
    FlashAttentionDiffKVBackend,
    FlashAttentionDiffKVImpl,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)


CACHE_VALUE_HEAD_SIZE = CACHE_VALUE_DIM
CACHE_PADDING_DIM = CACHE_VALUE_HEAD_SIZE - VALUE_DIM - ROUTING_AUX_DIM
assert CACHE_PADDING_DIM == 16


class BasisServeQwen3_8BSparseC1Backend(FlashAttentionDiffKVBackend):
    """K128/V128 cache with V80, R32, and zero-padded FlashAttention ABI."""

    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["BasisServeQwen3_8BSparseC1Impl"]:
        return BasisServeQwen3_8BSparseC1Impl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        assert block_size == PAGE_SIZE
        return (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size + CACHE_VALUE_HEAD_SIZE,
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4)
        return (0, 1, 2, 3)


class BasisServeQwen3_8BSparseC1Impl(FlashAttentionDiffKVImpl):
    """Use FlashAttention prefill and repository sparse routing for decode."""

    @staticmethod
    def _record(
        layer: torch.nn.Module,
        *,
        queries: int,
        physical_tokens: torch.Tensor,
        logical_tokens: int,
    ) -> None:
        layer._basisserve_routing_queries.add_(queries)
        layer._basisserve_physical_tokens.add_(physical_tokens)
        layer._basisserve_logical_tokens.add_(logical_tokens)

    def _run_sparse_decode(
        self,
        *,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        batch = int(query.shape[0])
        maximum_sequence = int(block_table.shape[1]) * PAGE_SIZE
        maximum_pages = math.ceil(maximum_sequence / PAGE_SIZE)
        selected_table = block_table[:batch, :maximum_pages]
        mode = layer._basisserve_router_mode
        if mode == "conditional":
            rope_cache = layer._basisserve_rotary_emb._match_cos_sin_cache_dtype(query)
            page_log_mass = conditional_page32_log_mass(
                query,
                kv_cache,
                selected_table,
                seq_lens[:batch],
                layer._basisserve_base_right,
                layer._basisserve_base_bias,
                layer._basisserve_residual_query,
                rope_cache,
                scale=self.scale,
            )
            selected_pages = select_group_shared_pages(
                page_log_mass,
                seq_lens[:batch],
                pages_per_kv_head=layer._basisserve_token_budget // PAGE_SIZE,
                pinned_prefix_pages=layer._basisserve_pinned_prefix_pages,
            )
            selected_tokens = selected_page_tokens(
                selected_pages,
                query_heads=self.num_heads,
            )
            physical_tokens = (
                (
                    selected_pages[..., None] * PAGE_SIZE
                    + torch.arange(PAGE_SIZE, device=query.device)
                )
                .lt(seq_lens[:batch, None, None, None])
                .sum(dtype=torch.int64)
            )
        else:
            scores = loki_scores(
                query,
                layer._basisserve_loki_query_projector,
                kv_cache,
                selected_table,
                seq_lens[:batch],
                scale=self.scale,
                maximum_sequence=maximum_sequence,
            )
            selected_tokens = torch.topk(
                scores,
                layer._basisserve_token_budget,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices.contiguous()
            physical_tokens = physical_union_count(
                selected_tokens,
                kv_heads=self.num_kv_heads,
                maximum_sequence=maximum_sequence,
            )
        coordinates = paged_sparse_attention(
            query,
            kv_cache,
            selected_table,
            seq_lens[:batch],
            selected_tokens,
            scale=self.scale,
        )
        output.zero_()
        output[..., :VALUE_DIM].copy_(coordinates)
        self._record(
            layer,
            queries=batch,
            physical_tokens=physical_tokens,
            logical_tokens=batch * self.num_heads * int(selected_tokens.shape[-1]),
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None
        assert output_scale is None and output_block_scale is None
        if attn_metadata is None:
            return output.fill_(0)
        if attn_metadata.max_query_len != 1:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
            )

        triton_reshape_and_cache_flash_diffkv(
            key,
            value,
            kv_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
        num_tokens = int(attn_metadata.num_actual_tokens)
        self._run_sparse_decode(
            layer=layer,
            query=query[:num_tokens],
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table[:num_tokens],
            seq_lens=attn_metadata.seq_lens[:num_tokens],
            output=output[:num_tokens],
        )
        return output


__all__ = [
    "BasisServeQwen3_8BSparseC1Backend",
    "BasisServeQwen3_8BSparseC1Impl",
    "CACHE_PADDING_DIM",
    "CACHE_VALUE_HEAD_SIZE",
]
