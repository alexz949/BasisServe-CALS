"""Qwen3-8B PaLU-M quality-evaluation model for the pinned vLLM build."""

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
)
from basisserve.core.qwen3_8b_vllm_folded_palu import (
    fold_qwen3_8b_palu_layer_into_vllm_weights,
    load_qwen3_8b_folded_palu_checkpoint,
    load_qwen3_8b_palu_factors,
)

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM


MODEL_ARCHITECTURE = "BasisServeQwen3_8BFoldedPaLUForCausalLM"
_CONFIG_CHECKPOINT_DIR = "basisserve_palu_checkpoint_dir"
_CONFIG_MANIFEST_SHA256 = "basisserve_palu_manifest_sha256"


def _required_config_string(config: Any, name: str) -> str:
    value = getattr(config, name, None)
    assert isinstance(value, str) and value.strip()
    return value.strip()


class BasisServeQwen3_8BFoldedPaLUForCausalLM(Qwen3ForCausalLM):
    """TP1 Qwen3-8B with PaLU M/G2/G4 folded into dense V slots."""

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
        checkpoint_dir = Path(_required_config_string(config, _CONFIG_CHECKPOINT_DIR))
        expected_manifest_sha256 = _required_config_string(
            config,
            _CONFIG_MANIFEST_SHA256,
        )
        checkpoint = load_qwen3_8b_folded_palu_checkpoint(
            checkpoint_dir,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        assert checkpoint.model_config_sha256
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.basisserve_palu_checkpoint = checkpoint

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        factors = load_qwen3_8b_palu_factors(self.basisserve_palu_checkpoint)
        for layer_index, layer in enumerate(self.model.layers):
            prefix = f"layers.{layer_index}.v_"
            fold_qwen3_8b_palu_layer_into_vllm_weights(
                layer.self_attn.qkv_proj.weight,
                factors[prefix + "writer.weight"],
                factors[prefix + "decoder.weight"],
                self.basisserve_palu_checkpoint.layer_ranks[layer_index],
            )
        del factors
        return loaded


__all__ = ["BasisServeQwen3_8BFoldedPaLUForCausalLM", "MODEL_ARCHITECTURE"]
