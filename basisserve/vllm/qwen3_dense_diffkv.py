"""Qwen3-32B dense V128 control using the BasisServe Triton attention path."""

from __future__ import annotations

import torch

from basisserve.core.qwen3_32b_tp4_decode import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_QUERY_HEADS,
    TP_SIZE,
)
from basisserve.vllm.compact_v_backend import BasisServeDenseVBackend

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM


class BasisServeQwen3DenseDiffKVForCausalLM(Qwen3ForCausalLM):
    """Dense Qwen3-32B with only its attention kernel changed to Triton."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        observed = (
            int(config.hidden_size),
            int(config.num_attention_heads),
            int(config.num_key_value_heads),
            int(getattr(config, "head_dim", 0)),
            int(config.num_hidden_layers),
        )
        assert observed == (
            HIDDEN_SIZE,
            NUM_QUERY_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            NUM_LAYERS,
        )
        assert get_tensor_model_parallel_world_size() == TP_SIZE
        assert get_pp_group().world_size == 1
        assert vllm_config.parallel_config.decode_context_parallel_size == 1
        assert vllm_config.quant_config is None
        assert vllm_config.lora_config is None
        assert vllm_config.model_config.dtype == torch.bfloat16
        assert vllm_config.cache_config.cache_dtype in ("auto", "bfloat16")

        super().__init__(vllm_config=vllm_config, prefix=prefix)
        static_context = vllm_config.compilation_config.static_forward_context
        for layer in self.model.layers:
            attention = layer.self_attn
            assert attention.dual_chunk_attention_config is None
            attention_prefix = attention.attn.layer_name
            registered = static_context.pop(attention_prefix)
            assert registered is attention.attn
            attention.attn = Attention(
                attention.num_heads,
                attention.head_dim,
                attention.scaling,
                num_kv_heads=attention.num_kv_heads,
                cache_config=vllm_config.cache_config,
                quant_config=None,
                prefix=attention_prefix,
                attn_type=registered.attn_type,
                attn_backend=BasisServeDenseVBackend,
                head_size_v=attention.head_dim,
            )


__all__ = ["BasisServeQwen3DenseDiffKVForCausalLM"]
