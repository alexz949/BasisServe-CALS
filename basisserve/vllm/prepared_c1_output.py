"""Stream-bound prepared NCCL AllGather and one replicated decoder GEMM."""

from __future__ import annotations

import torch
from torch import Tensor

from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.ragged_allgather import StaticRaggedPlan
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.utils.torch_utils import direct_register_custom_op


class PreparedC1Output:
    """One communicator shared by sequential attention layers.

    Plans are warmed on the actual capture stream before capture begins.
    Returning the GEMM result instead of the mutable arena also gives the
    custom operator an ordinary, non-aliasing output contract for compilation.
    """

    def __init__(self, group, device, local_width: int, max_tokens: int):
        self.communicator = FeatureRaggedCommunicator.from_distributed(group, device=device)
        self.plan = StaticRaggedPlan.from_source_widths(
            (local_width,) * self.communicator.world_size
        )
        self.communicator.configure_direct_workspace(
            tokens=max_tokens,
            max_total_width=self.plan.total_width,
            dtype=torch.bfloat16,
        )
        self.prepared = {}
        self.capture_calls = 0
        self.eager_calls = 0
        self.direct_decode_calls = 0

    def _prepared(
        self,
        tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        stream = torch.cuda.current_stream(device)
        key = (tokens, stream.cuda_stream)
        capturing = torch.cuda.is_current_stream_capturing()
        prepared = self.prepared.get(key)
        if prepared is None:
            assert not capturing, "Warm this token count on the capture stream first"
            prepared = self.communicator.prepare_uniform(
                self.plan,
                tokens=tokens,
                dtype=dtype,
                backend="uniform_nccl",
            )
            self.prepared[key] = prepared
        if capturing:
            self.capture_calls += 1
        else:
            self.eager_calls += 1
        return prepared

    @staticmethod
    def _gather_decode(prepared, decoder: Tensor) -> Tensor:
        arena = prepared.gather_inplace_fast()
        return torch.mm(arena.T, decoder)

    def __call__(self, local: Tensor, decoder: Tensor) -> Tensor:
        prepared = self._prepared(
            int(local.shape[0]), local.dtype, local.device,
        )
        # Native vLLM attention emits token-major coordinates. The source-local
        # pack is currently required; the gathered arena needs no global pack.
        prepared.local_feature_major_view_fast().copy_(local.T)
        return self._gather_decode(prepared, decoder)

    def attention(
        self,
        attn_layer,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kv_cache: Tensor,
        attn_metadata,
        decoder: Tensor,
    ) -> Tensor:
        tokens = int(query.shape[0])
        if (
            attn_metadata is not None
            and attn_metadata.max_query_len <= 1
        ):
            prepared = self._prepared(tokens, query.dtype, query.device)
            local = prepared.local_feature_major_view_fast()
            output = local.T.view(
                tokens, attn_layer.impl.num_heads, attn_layer.head_size_v
            )
            attn_layer.impl.forward(
                attn_layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
            )
            self.direct_decode_calls += 1
            return self._gather_decode(prepared, decoder)

        output = torch.empty(
            (tokens, attn_layer.impl.num_heads, attn_layer.head_size_v),
            dtype=query.dtype,
            device=query.device,
        )
        attn_layer.impl.forward(
            attn_layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
        )
        return self(output.flatten(1), decoder)

    def statistics(self):
        return dict(backend="uniform_nccl", decoder_gemms=1,
                    capture_calls=self.capture_calls, eager_calls=self.eager_calls,
                    direct_decode_calls=self.direct_decode_calls,
                    prepared_shapes=sorted({int(key[0]) for key in self.prepared}))


def prepared_c1_output(local: Tensor, decoder: Tensor, layer_name: str) -> Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    return layer._basisserve_c1_output(local, decoder)


def prepared_c1_output_fake(local: Tensor, decoder: Tensor, layer_name: str) -> Tensor:
    return local.new_empty((local.shape[0], decoder.shape[1]))


direct_register_custom_op(
    op_name="basisserve_prepared_c1_output", op_func=prepared_c1_output,
    fake_impl=prepared_c1_output_fake,
)


def c1_attention_output(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decoder: Tensor,
    layer_name: str,
) -> Tensor:
    attn_metadata, attn_layer, kv_cache, slot_mapping = get_attention_context(
        layer_name
    )
    query = query.view(-1, attn_layer.num_heads, attn_layer.head_size)
    key = key.view(-1, attn_layer.num_kv_heads, attn_layer.head_size)
    value = value.view(-1, attn_layer.num_kv_heads, attn_layer.head_size_v)
    if slot_mapping is not None:
        attn_layer.impl.do_kv_cache_update(
            attn_layer,
            key,
            value,
            kv_cache,
            slot_mapping,
        )
    boundary = attn_layer._basisserve_c1_output
    return boundary.attention(
        attn_layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        decoder,
    )


def c1_attention_output_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decoder: Tensor,
    layer_name: str,
) -> Tensor:
    return query.new_empty((query.shape[0], decoder.shape[1]))


direct_register_custom_op(
    op_name="basisserve_c1_attention_output",
    op_func=c1_attention_output,
    fake_impl=c1_attention_output_fake,
)
