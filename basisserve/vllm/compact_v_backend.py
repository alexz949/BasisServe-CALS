"""Paged K128/dynamic-V attention for the BasisServe vLLM integration.

Initial complete prefill uses BasisServe's causal compact-Value Triton kernel.
Uniform single-token decode uses vLLM's grouped Triton paged-attention kernel,
whose Q/K and Value tile widths are independent.  The cache update is kept
inside the backend because vLLM's generic split cache-update path assumes
symmetric K/V storage.
"""

from __future__ import annotations

import torch

from basisserve.kernels.compressed_v_decode_attention import (
    compressed_v_prefill_attention,
)

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn_diffkv import (
    FlashAttentionDiffKVBackend,
    FlashAttentionDiffKVImpl,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)


VALUE_HEAD_SIZE = 64
DENSE_VALUE_HEAD_SIZE = 128
NUM_KV_SPLITS = 8


def _kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    value_head_size: int,
) -> tuple[int, ...]:
    assert block_size % 16 == 0
    return (
        num_blocks,
        block_size,
        num_kv_heads,
        head_size + value_head_size,
    )


@register_backend(AttentionBackendEnum.CUSTOM)
class BasisServeCompactVBackend(FlashAttentionDiffKVBackend):
    """K128/V64 backend with Triton prefill and paged Triton decode."""

    # Diff-KV cache insertion cannot use vLLM's symmetric generic updater.
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["BasisServeDiffKVImpl"]:
        return BasisServeDiffKVImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return _kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            VALUE_HEAD_SIZE,
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # The Triton decode kernel linearizes block/page locations, so keep the
        # logical NHD page layout even if vLLM globally requests HND.
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4)
        return (0, 1, 2, 3)


class BasisServeDenseVBackend(BasisServeCompactVBackend):
    """K128/V128 control using exactly the same Triton attention template."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return _kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            DENSE_VALUE_HEAD_SIZE,
        )


class BasisServeDiffKVImpl(FlashAttentionDiffKVImpl):
    """Use the same dynamic-Value Triton kernels around the paged cache."""

    def _run_decode(
        self,
        *,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        num_sequences = int(query.shape[0])
        key_cache = kv_cache[..., : self.head_size]
        value_cache = kv_cache[..., self.head_size :]
        value_head_size = int(value_cache.shape[-1])
        intermediate = torch.empty(
            (
                num_sequences,
                self.num_heads,
                NUM_KV_SPLITS,
                value_head_size + 1,
            ),
            dtype=torch.float32,
            device=query.device,
        )
        lse = torch.empty(
            (num_sequences, self.num_heads),
            dtype=torch.float32,
            device=query.device,
        )
        decode_attention_fwd(
            query,
            key_cache,
            value_cache,
            output,
            lse,
            block_table,
            seq_lens,
            intermediate,
            NUM_KV_SPLITS,
            self.scale,
            page_size=int(kv_cache.shape[1]),
            logit_cap=self.logits_soft_cap,
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
        )

    @staticmethod
    def _host_sequence_layout(attn_metadata) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Copy dynamic varlen metadata to CPU once for all layers in this step."""

        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        num_sequences = int(query_start_loc.shape[0]) - 1
        cache_key = (
            query_start_loc.data_ptr(),
            query_start_loc._version,
            seq_lens.data_ptr(),
            seq_lens._version,
            int(attn_metadata.num_actual_tokens),
            num_sequences,
        )
        cached = getattr(attn_metadata, "_basisserve_host_sequence_layout", None)
        if cached is None or cached[0] != cache_key:
            starts = tuple(
                map(int, query_start_loc[: num_sequences + 1].cpu().tolist())
            )
            lengths = tuple(map(int, seq_lens[:num_sequences].cpu().tolist()))
            cached = (cache_key, starts, lengths)
            setattr(attn_metadata, "_basisserve_host_sequence_layout", cached)
        return cached[1], cached[2]

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
        if output is None:
            raise ValueError("BasisServe compact-V attention requires an output buffer")
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("compact-V output quantization is not supported")
        if attn_metadata is None:
            return output.fill_(0)
        if self.attn_type != AttentionType.DECODER:
            raise NotImplementedError("BasisServe compact-V supports decoder attention")
        if self.kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("BasisServe compact-V currently requires BF16 cache")

        triton_reshape_and_cache_flash_diffkv(
            key,
            value,
            kv_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        if attn_metadata.max_query_len == 1 and not attn_metadata.use_cascade:
            if self.alibi_slopes is not None or self.sinks is not None:
                raise NotImplementedError("Triton compact-V decode excludes ALiBi/sinks")
            if self.sliding_window != (-1, -1):
                raise NotImplementedError("Triton compact-V decode excludes sliding window")
            if self.dcp_world_size != 1:
                raise NotImplementedError("Triton compact-V decode excludes DCP")

            num_tokens = int(attn_metadata.num_actual_tokens)
            query_view = query[:num_tokens]
            output_view = output[:num_tokens]
            if query_view.shape[0] != attn_metadata.seq_lens[:num_tokens].shape[0]:
                raise RuntimeError("single-token decode requires one query per sequence")

            self._run_decode(
                layer=layer,
                query=query_view,
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table[:num_tokens],
                seq_lens=attn_metadata.seq_lens[:num_tokens],
                output=output_view,
            )
            return output

        num_tokens = int(attn_metadata.num_actual_tokens)
        if not attn_metadata.causal:
            raise NotImplementedError("BasisServe compact-V prefill requires causal attention")
        if attn_metadata.use_cascade:
            raise NotImplementedError("BasisServe compact-V prefill excludes cascade attention")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("Triton compact-V prefill excludes ALiBi/sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("Triton compact-V prefill excludes sliding window")
        if self.dcp_world_size != 1:
            raise NotImplementedError("Triton compact-V prefill excludes DCP")

        starts, sequence_lengths = self._host_sequence_layout(attn_metadata)
        for sequence_index, (query_start, query_stop, sequence_length) in enumerate(
            zip(starts[:-1], starts[1:], sequence_lengths, strict=True)
        ):
            query_length = query_stop - query_start
            query_slice = query[query_start:query_stop]
            output_slice = output[query_start:query_stop]
            if query_length == 1:
                self._run_decode(
                    layer=layer,
                    query=query_slice,
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table[
                        sequence_index : sequence_index + 1
                    ],
                    seq_lens=attn_metadata.seq_lens[
                        sequence_index : sequence_index + 1
                    ],
                    output=output_slice,
                )
                continue
            if query_length != sequence_length:
                raise NotImplementedError(
                    "BasisServe compact-V does not yet support chunked prefill"
                )

            # vLLM supplies packed token-major tensors [T,H,D]. Each complete
            # prompt is exposed to the causal kernel as a metadata-only
            # [1,H,T,D] view; no padding or packed-QKV copy is introduced.
            compact_output = compressed_v_prefill_attention(
                query_slice.transpose(0, 1).unsqueeze(0),
                key[query_start:query_stop].transpose(0, 1).unsqueeze(0),
                value[query_start:query_stop].transpose(0, 1).unsqueeze(0),
                scale=self.scale,
            )
            output_slice.copy_(compact_output.squeeze(0).transpose(0, 1))
        return output


__all__ = [
    "BasisServeCompactVBackend",
    "BasisServeDenseVBackend",
    "BasisServeDiffKVImpl",
    "DENSE_VALUE_HEAD_SIZE",
    "NUM_KV_SPLITS",
    "VALUE_HEAD_SIZE",
]
