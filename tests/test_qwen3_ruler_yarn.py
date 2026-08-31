from __future__ import annotations

import pytest
import torch
from transformers import DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3RotaryEmbedding,
)

from evaluation.eval_qwen3_8b_c1_kq_routing_ruler import _restore_prompt_cache
from evaluation.eval_qwen3_dense_ruler import _chunked_exact_prefill, _yarn_override


def test_qwen3_128k_uses_official_yarn_factor_four_geometry() -> None:
    maximum, rope = _yarn_override(
        sequence_length=131072,
        original_max_position_embeddings=32768,
        yarn_factor=4.0,
    )

    assert maximum == 131072
    assert rope == {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
    }


def test_transformers_qwen3_yarn_keeps_rope_theta() -> None:
    _, yarn = _yarn_override(
        sequence_length=131072,
        original_max_position_embeddings=32768,
        yarn_factor=4.0,
    )
    assert yarn is not None
    rope_parameters = {"rope_theta": 1_000_000.0, **yarn}
    config = Qwen3Config(
        max_position_embeddings=131072,
        rope_parameters=rope_parameters,
    )

    rotary = Qwen3RotaryEmbedding(config)

    assert rotary.inv_freq.shape == (64,)
    assert torch.isfinite(rotary.inv_freq).all()


def test_context_extension_requires_explicit_yarn() -> None:
    with pytest.raises(ValueError, match="set --yarn-factor"):
        _yarn_override(
            sequence_length=131072,
            original_max_position_embeddings=32768,
            yarn_factor=None,
        )


def test_yarn_scaled_maximum_must_cover_evaluation() -> None:
    with pytest.raises(ValueError, match="shorter"):
        _yarn_override(
            sequence_length=131072,
            original_max_position_embeddings=32768,
            yarn_factor=2.0,
        )


def test_paired_arm_cache_transaction_restores_prompt() -> None:
    cache = DynamicCache()
    key = torch.randn(1, 2, 7, 4)
    value = torch.randn(1, 2, 7, 3)
    cache.update(key, value, 0)

    _restore_prompt_cache(cache, 5)

    assert cache.get_seq_length() == 5
    assert cache.layers[0].keys.shape[-2] == 5
    assert cache.layers[0].values.shape[-2] == 5


def test_dense_chunked_prefill_matches_direct_causal_prefill() -> None:
    torch.manual_seed(20260830)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
    )
    model = Qwen3ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])

    direct = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    logits, cache = _chunked_exact_prefill(model, input_ids, chunk_size=3)

    torch.testing.assert_close(logits, direct.logits[:, -1], rtol=1.0e-5, atol=1.0e-6)
    assert cache.get_seq_length() == input_ids.shape[1]
