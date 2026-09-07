"""Independent routing bounds, support and model-cache integration checks."""

import torch
from torch.nn.functional import scaled_dot_product_attention

from basisserve.core.c1_routing_comparison import (
    BASELINES, QuestPageBounds, RoutingBaseline, exact_page_attention, quest_page_ids,
)
from basisserve.core.residual_kl_replay import prefix_signature
from evaluation.eval_qwen3_8b_residual_rank_ruler import greedy_decode, prepare_arm
from tests.test_residual_rank_ruler import tiny_inputs


def test_quest_bounds_incremental_partial_pages_and_upper_bound():
    torch.manual_seed(42)
    keys = torch.randn(2, 2, 17, 8)
    bounds = QuestPageBounds(keys[..., :3, :], 4)
    for length in (4, 5, 8, 11, 17):
        bounds.update(keys[..., :length, :])
        rebuilt = QuestPageBounds(keys[..., :length, :], 4)
        torch.testing.assert_close(bounds.minimum, rebuilt.minimum, atol=0, rtol=0)
        torch.testing.assert_close(bounds.maximum, rebuilt.maximum, atol=0, rtol=0)
        q = torch.randn(2, 4, 1, 8)
        scores = bounds.scores(q)
        expanded = keys[..., :length, :].repeat_interleave(2, dim=1)
        exact = (q @ expanded.mT).squeeze(2) / 8**0.5
        for page in range(scores.shape[-1]):
            actual = exact[..., page * 4:min((page + 1) * 4, length)].amax(-1)
            assert bool((scores[..., page] >= actual - 1e-6).all())


def test_quest_group_max_budget_and_pinned_page():
    scores = torch.tensor([[[1., 9., 2., 3.], [1., 2., 8., 4.],
                            [1., 2., 3., 7.], [1., 5., 4., 3.]]])
    ids = quest_page_ids(scores, kv_heads=2, page_budget=2, pinned_prefix_pages=1)
    assert ids.tolist() == [[[[0, 1]], [[0, 3]]]]


def test_selected_exact_attention_matches_independent_dense_mask():
    torch.manual_seed(13)
    q, k, v = torch.randn(1, 4, 1, 8), torch.randn(1, 2, 11, 8), torch.randn(1, 2, 11, 5)
    ids = torch.tensor([[[[0, 2]], [[1, 2]]]])
    actual, valid = exact_page_attention(q, k, v, ids, 4)
    mask = torch.zeros(1, 2, 1, 11, dtype=torch.bool)
    for group in range(2):
        for page in ids[0, group, 0].tolist():
            mask[0, group, 0, page * 4:min(page * 4 + 4, 11)] = True
    expected = scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1),
        attn_mask=mask.repeat_interleave(2, 1))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert int(valid.sum()) == 14


@torch.inference_mode()
def test_full_support_model_replay_and_cleanup_across_baselines():
    model, bank, prefix, first = tiny_inputs()
    signature = prefix_signature(prefix)
    original = [(x.keys.clone(), x.values.clone()) for x in prefix.layers]
    projector = torch.eye(8)[:, :4].expand(3, 2, 8, 4).clone()
    cache = prepare_arm(model, bank, prefix, None)
    expected_ids, _, expected_trace = greedy_decode(model, cache, first,
        maximum_tokens=5, eos_ids=set(), trace=True)
    for arm in BASELINES:
        for repeat in range(2):
            with RoutingBaseline(model, prefix, projector, arm, page_size=4, budget=64, pinned=1) as runtime:
                ids, final, trace = greedy_decode(model, runtime.cache, first,
                    maximum_tokens=5, eos_ids=set(), trace=True)
                assert ids == expected_ids and final.get_seq_length() == 20
                for a, b in zip(trace, expected_trace, strict=True):
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
                assert runtime.statistics()["selector_calls"] == 12
                assert runtime.statistics()["mean_physical_tokens_per_kv_group"] == 18.5
            assert prefix_signature(prefix) == signature
            for layer, (k, v) in zip(prefix.layers, original, strict=True):
                torch.testing.assert_close(layer.keys, k, atol=0, rtol=0)
                torch.testing.assert_close(layer.values, v, atol=0, rtol=0)
            assert all(layer.self_attn.routing_key_projector is None for layer in model.model.layers)
    cache = prepare_arm(model, bank, prefix, None)
    ids, _, _ = greedy_decode(model, cache, first, maximum_tokens=5, eos_ids=set())
    assert ids == expected_ids
