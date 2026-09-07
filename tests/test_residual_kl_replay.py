from __future__ import annotations

import itertools

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention
from basisserve.core.c1_k_reverse_shadow import ReverseShadowConfig
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from basisserve.core.residual_kl_replay import (
    capture_cached_suffix, fork_routing_prefix, prefix_signature,
    replay_cached_suffix, terminal_metrics,
)
from evaluation.build_qwen3_8b_c1_two_sided_factorized_kl_schedule import allocate_layer_schedule


def tiny_model():
    torch.manual_seed(32)
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    for layer in model.model.layers:
        layer.self_attn = GQATiedVOQwen3Attention(
            layer.self_attn, v_proj_compressed_weight=torch.randn(8, 32) * .03,
            o_decoder_weight=torch.randn(32, 16) * .03, attention_backend="sdpa",
        )
    return model


def route(module, rank, *, budget=4):
    torch.manual_seed(123 + module.layer_idx)
    left, right = torch.randn(2, 4, 2) * .1, torch.randn(2, 2, 8) * .1
    encoder = torch.randn(2, 8, 4)[..., :rank]
    query = torch.randn(4, 8, 4)[..., :rank]
    module.attention_backend = "native"
    module.set_conditional_routing_factors(
        base_left=left, base_right=right, base_bias=torch.zeros(2, 8),
        residual_encoder=encoder, residual_query_projector=query,
    )
    module.set_conditional_page_query_block_size(2)
    module.set_reverse_shadow_config(ReverseShadowConfig(
        page_size=2, exact_token_budget=budget, selector="kq_svd", pinned_prefix_pages=1,
    ))


def sidecar(module, prefix, cos, sin):
    layer = prefix.layers[module.layer_idx]
    return build_conditional_routing_sidecar(
        layer.values, layer.keys, base_left=module.conditional_base_left,
        base_right=module.conditional_base_right, base_bias=module.conditional_base_bias,
        residual_encoder=module.conditional_residual_encoder, cos=cos, sin=sin,
    )


@torch.inference_mode()
def test_fork_isolates_kv_and_sidecar_appends():
    prefix = RoutingDynamicCache()
    prefix.update(torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 4), 0)
    prefix.update_precomputed_routing_sidecar(torch.randn(1, 2, 5, 3), 0)
    expected = [prefix.layers[0].keys.clone(), prefix.layers[0].values.clone(),
                prefix.routing_sidecar(0).clone()]
    signature = prefix_signature(prefix)
    for _ in range(2):
        fork = fork_routing_prefix(prefix)
        fork.update(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 4), 0)
        fork.update_precomputed_routing_sidecar(torch.randn(1, 2, 2, 3), 0)
        assert fork.get_seq_length() == 7 and prefix.get_seq_length() == 5
        assert fork.routing_sidecar(0).shape[-2] == 7
    assert prefix_signature(prefix) == signature
    for observed, reference in zip((prefix.layers[0].keys, prefix.layers[0].values,
                                    prefix.routing_sidecar(0)), expected, strict=True):
        torch.testing.assert_close(observed, reference, atol=0, rtol=0)


@torch.inference_mode()
def test_sparse_rank_intervention_replay_matches_full_suffix_and_is_order_independent():
    model = tiny_model()
    tokens = torch.arange(22)[None] % 64
    prefix = RoutingDynamicCache()
    model.model(input_ids=tokens[:, :16], past_key_values=prefix, use_cache=True)
    cos, sin = model.model.rotary_emb(model.model.embed_tokens.weight[:1], torch.arange(16)[None])
    for layer in model.model.layers:
        module = layer.self_attn
        route(module, 2)
        prefix._ensure_routing_layer(module.layer_idx)
        prefix._routing_sidecars[module.layer_idx] = sidecar(module, prefix, cos, sin)
    signature = prefix_signature(prefix)
    anchor = capture_cached_suffix(model, tokens[:, 16:], fork_routing_prefix(prefix))
    for index in range(3):
        observed = replay_cached_suffix(model, anchor, fork_routing_prefix(prefix), intervention_layer=index)
        torch.testing.assert_close(observed, anchor.final_hidden, atol=0, rtol=0)
    previous = {}
    for index, rank in ((1, 1), (2, 4), (1, 1)):
        module = model.model.layers[index].self_attn
        route(module, rank)
        fork = fork_routing_prefix(prefix)
        fork._routing_sidecars[index] = sidecar(module, prefix, cos, sin)
        observed = replay_cached_suffix(model, anchor, fork, intervention_layer=index)
        reference_cache = fork_routing_prefix(prefix)
        reference_cache._routing_sidecars[index] = sidecar(module, prefix, cos, sin)
        expected = model.model(input_ids=tokens[:, 16:], past_key_values=reference_cache, use_cache=True).last_hidden_state
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        if (index, rank) in previous:
            torch.testing.assert_close(observed, previous[(index, rank)], atol=0, rtol=0)
        previous[(index, rank)] = observed
        assert prefix_signature(prefix) == signature
        route(module, 2)
    assert not torch.equal(previous[(1, 1)], anchor.final_hidden)


@torch.inference_mode()
def test_full_budget_rank_independence_and_causal_future_mask():
    model = tiny_model()
    tokens = torch.arange(22)[None] % 64
    prefix = RoutingDynamicCache()
    model.model(input_ids=tokens[:, :16], past_key_values=prefix, use_cache=True)
    expected = model.model(input_ids=tokens[:, 16:], past_key_values=fork_routing_prefix(prefix), use_cache=True).last_hidden_state
    cos, sin = model.model.rotary_emb(model.model.embed_tokens.weight[:1], torch.arange(16)[None])
    for rank in (1, 4):
        for layer in model.model.layers:
            module = layer.self_attn
            route(module, rank, budget=24)
            prefix._ensure_routing_layer(module.layer_idx)
            prefix._routing_sidecars[module.layer_idx] = sidecar(module, prefix, cos, sin)
        observed = model.model(input_ids=tokens[:, 16:], past_key_values=fork_routing_prefix(prefix), use_cache=True).last_hidden_state
        torch.testing.assert_close(observed, expected, atol=3e-6, rtol=3e-6)
    altered = tokens[:, 16:].clone()
    altered[:, -1] = 43
    changed = model.model(input_ids=altered, past_key_values=fork_routing_prefix(prefix), use_cache=True).last_hidden_state
    torch.testing.assert_close(changed[:, :-1], observed[:, :-1], atol=0, rtol=0)


@torch.inference_mode()
def test_sparse_block_suffix_matches_token_by_token():
    model = tiny_model()
    tokens = torch.arange(22)[None] % 64
    prefix = RoutingDynamicCache()
    model.model(input_ids=tokens[:, :16], past_key_values=prefix, use_cache=True)
    cos, sin = model.model.rotary_emb(model.model.embed_tokens.weight[:1], torch.arange(16)[None])
    for layer in model.model.layers:
        module = layer.self_attn
        route(module, 2)
        prefix._ensure_routing_layer(module.layer_idx)
        prefix._routing_sidecars[module.layer_idx] = sidecar(module, prefix, cos, sin)
    block = model.model(input_ids=tokens[:, 16:], past_key_values=fork_routing_prefix(prefix), use_cache=True).last_hidden_state
    sequential_cache = fork_routing_prefix(prefix)
    outputs = []
    for position in range(16, 22):
        outputs.append(model.model(input_ids=tokens[:, position:position+1], past_key_values=sequential_cache,
                                   use_cache=True).last_hidden_state)
    torch.testing.assert_close(torch.cat(outputs, dim=1), block, atol=3e-6, rtol=3e-6)


def test_terminal_kl_direction_and_nll_readout():
    teacher = torch.tensor([[[2., 0.], [0., 2.], [1., 0.]]])
    student = torch.tensor([[[0., 2.], [0., 1.], [0., 0.]]])
    tokens = torch.tensor([[0, 1, 0]])
    logp = teacher.log_softmax(-1)
    metrics = terminal_metrics(student, logp, tokens)
    expected = torch.nn.functional.kl_div(student.log_softmax(-1), logp, log_target=True, reduction="none").sum(-1).mean()
    assert abs(metrics["kl_mean"] - float(expected)) < 1e-7
    assert metrics["kl_positions"] == 3 and metrics["nll_positions"] == 2
    assert terminal_metrics(teacher, logp, tokens)["kl_mean"] == 0


def test_signed_three_point_dp_matches_brute_force():
    costs = [{4: .1, 8: 0., 16: -.5}, {4: .05, 8: 0., 16: .2}, {4: -.03, 8: 0., 16: -.1}]
    ranks, total = allocate_layer_schedule(costs, candidate_ranks=(4, 8, 16), anchor_rank=8, target_average_rank=8)
    feasible = [(sum(costs[i][r] for i, r in enumerate(rs)), rs)
                for rs in itertools.product((4, 8, 16), repeat=3) if sum(rs) == 24]
    expected, expected_ranks = min(feasible)
    assert ranks == expected_ranks and abs(total - expected) < 1e-12
    assert ranks == (16, 4, 4)
