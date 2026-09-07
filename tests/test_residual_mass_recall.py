from __future__ import annotations

from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention
from basisserve.core.c1_conditional_page_attention import _selected_pages, c1_conditional_page_topk_attention
from basisserve.core.c1_k_reverse_shadow import ReverseShadowConfig
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from basisserve.core.residual_kl_replay import fork_routing_prefix, prefix_signature
from basisserve.core.residual_mass_recall import (
    capture_teacher_queries, rank_sidecar, routing_mass_recall, teacher_page_probabilities,
)
from evaluation.eval_qwen3_8b_residual_mass_recall import describe_pair


def inputs():
    torch.manual_seed(172)
    q, k = torch.randn(1, 8, 5, 8), torch.randn(1, 2, 11, 8)
    side, projector = torch.randn(1, 2, 11, 5), torch.randn(8, 8, 5)
    mask = torch.ones(1, 1, 5, 11, dtype=torch.bool)
    mask[..., 4] = False
    shared = dict(page_size=3, pinned_prefix_pages=1, scale=8**-.5,
                  query_block_size=2, attention_mask=mask)
    return q, k, side, projector, shared


@torch.inference_mode()
def test_teacher_mass_matches_brute_causal_token_softmax():
    q, k, _, _, shared = inputs()
    teacher = teacher_page_probabilities(q, k, **shared)
    for h in range(8):
        for t in range(5):
            scores = (q[0, h, t] @ k[0, h // 4].T) * shared["scale"]
            valid = torch.arange(11) <= 6 + t
            valid[4] = False
            scores[~valid] = -torch.inf
            p = scores.softmax(-1)
            pages = torch.stack([p[start:start+3].sum() for start in range(0, 11, 3)])
            torch.testing.assert_close(teacher["mass"][0, h, t], pages, atol=2e-7, rtol=2e-6)
            scores[:3] = -torch.inf
            non_sink = scores.softmax(-1)
            non_sink_pages = torch.stack([non_sink[start:start+3].sum() for start in range(0, 11, 3)])
            torch.testing.assert_close(teacher["non_sink_mass"][0, h, t], non_sink_pages, atol=2e-7, rtol=2e-6)
    assert bool(teacher["non_sink_valid"].all())


@torch.inference_mode()
def test_pages_match_actual_sparse_attention_and_full_budget_recalls_one():
    q, k, side, projector, shared = inputs()
    teacher = teacher_page_probabilities(q, k, **shared)
    observed = routing_mass_recall(q, side, projector, teacher, exact_token_budget=6,
                                   return_pages=True, **shared)
    captured = []

    def observe(*args, **kwargs):
        ids, valid = _selected_pages(*args, **kwargs)
        captured.append((ids, valid))
        return ids, valid

    with patch("basisserve.core.c1_conditional_page_attention._selected_pages", side_effect=observe):
        c1_conditional_page_topk_attention(q, k, torch.randn(1, 2, 11, 3), side, projector,
                                          exact_token_budget=6, **shared)
    torch.testing.assert_close(observed["page_ids"], torch.cat([p[0] for p in captured], dim=2), atol=0, rtol=0)
    torch.testing.assert_close(observed["page_valid"], torch.cat([p[1] for p in captured], dim=2), atol=0, rtol=0)
    for h in range(8):
        for t in range(5):
            ids = observed["page_ids"][0, h // 4, t]
            valid = observed["page_valid"][0, h // 4, t]
            assert ids[0] == 0
            expected = teacher["mass"][0, h, t, ids[valid]].sum()
            torch.testing.assert_close(observed["mass"][0, h, t], expected)
    full = routing_mass_recall(q, side, projector, teacher, exact_token_budget=12, **shared)
    torch.testing.assert_close(full["mass"], torch.ones_like(full["mass"]), atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(full["non_sink_mass"], torch.ones_like(full["non_sink_mass"]), atol=2e-7, rtol=2e-6)


@torch.inference_mode()
def test_sink_dominance_does_not_hide_non_sink_recall():
    q = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.zeros(1, 1, 6, 2)
    key[0, 0, 0, 0] = 1000.0
    projector = torch.eye(2)[None]
    shared = dict(page_size=2, pinned_prefix_pages=1, scale=1., query_block_size=1)
    teacher = teacher_page_probabilities(q, key, **shared)
    recall = routing_mass_recall(q, key, projector, teacher, exact_token_budget=4, **shared)
    torch.testing.assert_close(recall["mass"], torch.ones_like(recall["mass"]))
    torch.testing.assert_close(recall["non_sink_mass"], torch.full_like(recall["mass"], .5))
    assert bool(teacher["non_sink_valid"].all())
    short = teacher_page_probabilities(q, key[:, :, :2], **shared)
    assert not bool(short["non_sink_valid"].any())
    assert not bool(short["non_sink_mass"].any())


@torch.inference_mode()
def test_teacher_capture_does_not_change_forward_and_native_cache_matches_sidecar():
    torch.manual_seed(41)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    for layer in model.model.layers:
        layer.self_attn = GQATiedVOQwen3Attention(layer.self_attn,
            v_proj_compressed_weight=torch.randn(8, 32) * .03,
            o_decoder_weight=torch.randn(32, 16) * .03, attention_backend="sdpa")
    tokens = torch.arange(22)[None]
    prefix = RoutingDynamicCache()
    model.model(input_ids=tokens[:, :16], past_key_values=prefix, use_cache=True)
    signature = prefix_signature(prefix)
    original_keys = prefix.layers[0].keys.clone()
    cache = fork_routing_prefix(prefix)
    hidden, records = capture_teacher_queries(model, tokens[:, 16:], cache)
    reference = model.model(input_ids=tokens[:, 16:], past_key_values=fork_routing_prefix(prefix), use_cache=True).last_hidden_state
    torch.testing.assert_close(hidden, reference, atol=0, rtol=0)
    tensors = dict(base_left_b16=torch.randn(2, 4, 2) * .1, base_right_b16=torch.randn(2, 2, 8) * .1,
                   base_bias_b16=torch.randn(2, 8) * .1, residual_encoder_b16_r4=torch.randn(2, 8, 4),
                   residual_query_b16_r4=torch.randn(4, 8, 4))
    prefix_rotary = model.model.rotary_emb(model.model.embed_tokens.weight[:1], torch.arange(16)[None])
    side, projector = rank_sidecar(tensors, 4, prefix.layers[0], cache.layers[0], prefix_rotary,
                                   records[0]["context"]["position_embeddings"])
    module = model.model.layers[0].self_attn
    module.attention_backend = "native"
    module.set_conditional_routing_factors(base_left=tensors["base_left_b16"], base_right=tensors["base_right_b16"],
        base_bias=tensors["base_bias_b16"], residual_encoder=tensors["residual_encoder_b16_r4"],
        residual_query_projector=tensors["residual_query_b16_r4"])
    module.set_conditional_page_query_block_size(2)
    module.set_reverse_shadow_config(ReverseShadowConfig(page_size=2, exact_token_budget=4,
        selector="kq_svd", quest_support="physical_shared", pinned_prefix_pages=1))
    native_cache = fork_routing_prefix(prefix)
    native_cache._ensure_routing_layer(0)
    native_cache._routing_sidecars[0] = build_conditional_routing_sidecar(prefix.layers[0].values, prefix.layers[0].keys,
        base_left=module.conditional_base_left, base_right=module.conditional_base_right, base_bias=module.conditional_base_bias,
        residual_encoder=module.conditional_residual_encoder, cos=prefix_rotary[0], sin=prefix_rotary[1])
    shared = dict(page_size=2, pinned_prefix_pages=1, scale=module.scaling, query_block_size=2,
                  attention_mask=records[0]["context"]["attention_mask"])
    teacher = teacher_page_probabilities(records[0]["query"], cache.layers[0].keys, **shared)
    recall = routing_mass_recall(records[0]["query"], side, projector, teacher, exact_token_budget=4,
                                 return_pages=True, **shared)
    captured = []

    def observe(*args, **kwargs):
        selected = _selected_pages(*args, **kwargs)
        captured.append(selected)
        return selected

    with patch("basisserve.core.c1_conditional_page_attention._selected_pages", side_effect=observe):
        module(**dict(records[0]["context"], past_key_values=native_cache))
    torch.testing.assert_close(native_cache.routing_sidecar(0), side, atol=0, rtol=0)
    torch.testing.assert_close(torch.cat([x[0] for x in captured], dim=2), recall["page_ids"], atol=0, rtol=0)
    torch.testing.assert_close(torch.cat([x[1] for x in captured], dim=2), recall["page_valid"], atol=0, rtol=0)
    assert prefix_signature(prefix) == signature
    torch.testing.assert_close(prefix.layers[0].keys, original_keys, atol=0, rtol=0)


def test_summary_handles_identical_ranks_and_unsupported_non_sink():
    values = torch.tensor([.2, .4, .6, .8])
    raw = dict(uniform_mass=values, adaptive_mass=values.clone(), uniform_non_sink=values,
               adaptive_non_sink=values.clone(), sink_mass=torch.ones(4), non_sink_valid=torch.tensor([True, False, True, False]))
    summary = describe_pair(raw)
    assert summary["delta_mass"]["mean"] == 0 and summary["delta_non_sink"]["mean"] == 0
    assert summary["uniform"]["mass"]["count"] == 4
    assert summary["uniform"]["non_sink"]["count"] == 2 and summary["non_sink_unsupported"] == 2
    raw["non_sink_valid"].fill_(False)
    assert describe_pair(raw)["uniform"]["non_sink"]["mean"] is None
