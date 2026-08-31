from __future__ import annotations

import copy

import pytest
import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from basisserve.checkpoint.pairwise_k_qwen3 import (
    PairwiseKCacheState,
    PairwiseQuestConfig,
    install_qwen3_pairwise_k_runtime,
)
from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention
from basisserve.core.pairwise_kq_svd import block_diagonal_pair_factors


def _contribution(value: float, tokens: int, rank: int) -> torch.Tensor:
    return torch.full((1, 2, tokens, rank), value)


def test_pair_state_exposes_only_completed_history() -> None:
    state = PairwiseKCacheState()
    state.append(
        layer_slot=0,
        independent_contribution=_contribution(1.0, 3, 2),
        pair_contribution=_contribution(10.0, 3, 4),
    )
    assert state.history("pairwise", 0) is None
    assert state.history_length("pairwise", 1) == 0
    assert state.history_length("independent", 0) == 3
    assert state.history_length("independent", 1) == 0

    state.append(
        layer_slot=1,
        independent_contribution=_contribution(2.0, 3, 2),
        pair_contribution=_contribution(20.0, 3, 4),
    )
    torch.testing.assert_close(
        state.history("pairwise", 0),
        _contribution(30.0, 3, 4),
    )
    assert state.history_length("independent", 0) == 3
    assert state.history_length("independent", 1) == 3


def test_pair_state_appends_next_block_transactionally() -> None:
    state = PairwiseKCacheState()
    for tokens in (3, 2):
        state.append(
            layer_slot=0,
            independent_contribution=_contribution(1.0, tokens, 2),
            pair_contribution=_contribution(10.0, tokens, 4),
        )
        state.append(
            layer_slot=1,
            independent_contribution=_contribution(2.0, tokens, 2),
            pair_contribution=_contribution(20.0, tokens, 4),
        )
    assert state.history_length("pairwise", 0) == 5
    assert state.history_length("independent", 0) == 5
    assert state.history_length("independent", 1) == 5
    state.reset()
    assert state.history("pairwise", 0) is None
    assert state.pending_layer0 is None


def test_pair_state_rejects_out_of_order_layer1() -> None:
    state = PairwiseKCacheState()
    with pytest.raises(RuntimeError, match="without layer-0"):
        state.append(
            layer_slot=1,
            independent_contribution=_contribution(2.0, 1, 2),
            pair_contribution=_contribution(20.0, 1, 4),
        )


@torch.inference_mode()
def test_full_rank_pair_runtime_matches_dense_block_cache() -> None:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        _attn_implementation="eager",
    )
    dense = Qwen3ForCausalLM(config).float().eval()
    candidate = copy.deepcopy(dense)
    independent_key = (
        torch.eye(8).view(1, 1, 8, 8).expand(2, 2, 8, 8).clone()
    )
    grouped_query = torch.eye(8).view(1, 1, 8, 8).expand(2, 2, 8, 8).clone()
    pair_key, pair_query = block_diagonal_pair_factors(
        independent_key,
        grouped_query,
    )
    independent_query = (
        grouped_query[:, :, None]
        .expand(2, 2, 2, 8, 8)
        .reshape(2, 4, 8, 8)
        .clone()
    )
    expanded_pair_query = (
        pair_query[:, :, None]
        .expand(2, 2, 2, 8, 16)
        .reshape(2, 4, 8, 16)
        .clone()
    )
    runtime = install_qwen3_pairwise_k_runtime(
        candidate,
        independent_key_projector=independent_key,
        independent_query_projector=independent_query,
        pair_key_projector=pair_key.unsqueeze(0),
        pair_query_projector=expanded_pair_query,
    )
    runtime.set_mode("pairwise")
    input_ids = torch.arange(12).view(1, 12) % config.vocab_size

    outputs = []
    for model in (dense, candidate):
        cache = DynamicCache()
        chunks = []
        for start in (0, 4, 8):
            chunks.append(
                model(
                    input_ids=input_ids[:, start : start + 4],
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=0,
                ).logits
            )
        outputs.append(torch.cat(chunks, dim=1))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-6, atol=2e-6)
    assert runtime.states[0].history_length("pairwise", 0) == 12


@torch.inference_mode()
def test_full_rank_pair_runtime_matches_c1_value_block_cache() -> None:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        _attn_implementation="eager",
    )
    dense_c1 = Qwen3ForCausalLM(config).float().eval()
    generator = torch.Generator().manual_seed(20260828)
    for layer in dense_c1.model.layers:
        base_attention = layer.self_attn
        dense_v = base_attention.v_proj.weight.detach().reshape(2, 8, 32)
        encoder = torch.randn(2, 8, 4, generator=generator)
        encoder, _ = torch.linalg.qr(encoder, mode="reduced")
        compressed_v = torch.bmm(encoder.mT, dense_v).reshape(8, 32)
        decoder = torch.randn(32, 16, generator=generator) / 8.0
        layer.self_attn = GQATiedVOQwen3Attention(
            base_attention,
            v_proj_compressed_weight=compressed_v,
            o_decoder_weight=decoder,
            attention_backend="native",
        )
    candidate = copy.deepcopy(dense_c1)
    sparse_candidate = copy.deepcopy(dense_c1)
    independent_key = (
        torch.eye(8).view(1, 1, 8, 8).expand(2, 2, 8, 8).clone()
    )
    grouped_query = torch.eye(8).view(1, 1, 8, 8).expand(2, 2, 8, 8).clone()
    pair_key, pair_query = block_diagonal_pair_factors(
        independent_key,
        grouped_query,
    )
    independent_query = (
        grouped_query[:, :, None]
        .expand(2, 2, 2, 8, 8)
        .reshape(2, 4, 8, 8)
        .clone()
    )
    expanded_pair_query = (
        pair_query[:, :, None]
        .expand(2, 2, 2, 8, 16)
        .reshape(2, 4, 8, 16)
        .clone()
    )
    runtime = install_qwen3_pairwise_k_runtime(
        candidate,
        independent_key_projector=independent_key,
        independent_query_projector=independent_query,
        pair_key_projector=pair_key.unsqueeze(0),
        pair_query_projector=expanded_pair_query,
    )
    runtime.set_mode("pairwise")
    sparse_runtime = install_qwen3_pairwise_k_runtime(
        sparse_candidate,
        independent_key_projector=independent_key,
        independent_query_projector=independent_query,
        pair_key_projector=pair_key.unsqueeze(0),
        pair_query_projector=expanded_pair_query,
    )
    sparse_runtime.set_mode("pairwise")
    sparse_runtime.set_sparse_policy(
        PairwiseQuestConfig(
            page_size=2,
            historical_token_budget=64,
            landmark_dtype="float32",
        )
    )
    input_ids = torch.arange(12).view(1, 12) % config.vocab_size

    outputs = []
    for model in (dense_c1, candidate, sparse_candidate):
        cache = DynamicCache()
        chunks = []
        for start in (0, 4, 8):
            chunks.append(
                model(
                    input_ids=input_ids[:, start : start + 4],
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=0,
                ).logits
            )
        outputs.append(torch.cat(chunks, dim=1))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=2e-6, atol=2e-6)
    assert candidate.model.layers[0].self_attn.value_head_dim == 4
    sparse_statistics = sparse_runtime.sparse_statistics()
    assert sparse_statistics["sparse_layers"] == [0, 1]
    assert sparse_statistics["totals"]["selected_token_fraction"] == 1.0
    assert sparse_statistics["totals"]["resident_selector_metadata_bytes"] == (
        sparse_statistics["per_layer"]["0"]["resident_selector_metadata_bytes"]
    )
