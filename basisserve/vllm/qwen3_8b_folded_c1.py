"""Qwen3-8B C1 quality-evaluation model for the pinned vLLM 0.18 build.

This path folds authenticated C1 factors into vLLM's standard dense V/O slots.
It preserves the checkpoint's logits while using vLLM continuous batching; it
does not claim compact-cache serving measurements.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from basisserve.core.qwen3_8b_vllm_folded_c1 import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_QUERY_HEADS,
    fold_qwen3_8b_c1_layer_into_vllm_weights,
    load_qwen3_8b_folded_c1_checkpoint,
    load_qwen3_8b_folded_c1_layer,
)

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM


MODEL_ARCHITECTURE = "BasisServeQwen3_8BFoldedC1ForCausalLM"
_CONFIG_CHECKPOINT_DIR = "basisserve_c1_checkpoint_dir"
_CONFIG_MANIFEST_SHA256 = "basisserve_c1_manifest_sha256"


def _required_config_string(config: Any, name: str) -> str:
    value = getattr(config, name, None)
    assert isinstance(value, str) and value.strip()
    return value.strip()


class BasisServeQwen3_8BFoldedC1ForCausalLM(Qwen3ForCausalLM):
    """TP1 Qwen3-8B with C1 folded into standard vLLM projection slots."""

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
        assert get_tensor_model_parallel_world_size() == 1
        assert get_pp_group().world_size == 1
        assert vllm_config.parallel_config.decode_context_parallel_size == 1
        assert vllm_config.quant_config is None
        assert vllm_config.lora_config is None
        assert vllm_config.model_config.dtype == torch.bfloat16
        checkpoint_dir = Path(
            _required_config_string(config, _CONFIG_CHECKPOINT_DIR)
        )
        expected_manifest_sha256 = _required_config_string(
            config,
            _CONFIG_MANIFEST_SHA256,
        )
        checkpoint = load_qwen3_8b_folded_c1_checkpoint(
            checkpoint_dir,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        assert checkpoint.model_config_sha256
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.basisserve_c1_checkpoint = checkpoint

    def load_weights(
        self,
        weights: Iterable[tuple[str, Tensor]],
    ) -> set[str]:
        loaded = super().load_weights(weights)
        for layer_index, layer in enumerate(self.model.layers):
            record = self.basisserve_c1_checkpoint.layers[layer_index]
            encoders, decoders = load_qwen3_8b_folded_c1_layer(record)
            attention = layer.self_attn
            fold_qwen3_8b_c1_layer_into_vllm_weights(
                attention.qkv_proj.weight,
                attention.o_proj.weight,
                encoders,
                decoders,
                record.ranks,
            )
            del encoders, decoders
        return loaded


__all__ = ["BasisServeQwen3_8BFoldedC1ForCausalLM", "MODEL_ARCHITECTURE"]
