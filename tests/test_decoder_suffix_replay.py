from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from basisserve.core.qwen_suffix_replay import (
    capture_qwen_anchor_replay,
    replay_qwen_suffix,
)


def test_suffix_replay_is_exact_for_llama_decoder_layers() -> None:
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.arange(26).reshape(2, 13) % config.vocab_size
    anchor = capture_qwen_anchor_replay(model, input_ids=input_ids)

    for layer in range(config.num_hidden_layers):
        replayed = replay_qwen_suffix(model, anchor, intervention_layer=layer)
        torch.testing.assert_close(replayed, anchor.final_hidden, rtol=0, atol=0)
