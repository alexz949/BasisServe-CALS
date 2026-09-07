from __future__ import annotations

from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention
from basisserve.core.c1_conditional_page_attention import _selected_pages
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.residual_kl_replay import prefix_signature
from evaluation.eval_qwen3_8b_residual_rank_ruler import arm_ranks, base_description, greedy_decode, prepare_arm
from evaluation.ruler_v1 import paired_summary, parse_tasks, sample_score, summarize_arm


def tiny_inputs():
    torch.manual_seed(19)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    bank = []
    for layer in model.model.layers:
        layer.self_attn = GQATiedVOQwen3Attention(layer.self_attn,
            v_proj_compressed_weight=torch.randn(8, 32) * .03,
            o_decoder_weight=torch.randn(32, 16) * .03, attention_backend="sdpa")
        row = {"base_left_b16": torch.randn(2, 4, 2) * .1,
               "base_right_b16": torch.randn(2, 2, 8) * .1, "base_bias_b16": torch.randn(2, 8) * .1}
        for rank in (4, 8, 16):
            row[f"residual_encoder_b16_r{rank}"] = torch.randn(2, 8, rank) * .1
            row[f"residual_query_b16_r{rank}"] = torch.randn(4, 8, rank) * .1
        bank.append(row)
    prefix = RoutingDynamicCache()
    output = model(input_ids=torch.arange(16)[None], past_key_values=prefix, use_cache=True, logits_to_keep=1)
    return model, bank, prefix, int(output.logits[0, -1].argmax())


def test_arm_rank_mapping_is_explicit_and_copies_schedule():
    assert arm_ranks("c1_exact_k", 3) is None
    ranks = arm_ranks("uniform_r8", 3)
    assert ranks == [8, 8, 8]
    ranks[0] = 4
    assert arm_ranks("uniform_r8", 3) == [8, 8, 8]


def test_base_description_distinguishes_closed_form_q1_and_qaware_q16():
    assert base_description({"format": "basisserve.closed_form_base_fisher_bank.v1",
                             "base_kind": "closed_form_rrr", "base_optimizer": None,
                             "base_query_positions": [],
                             "residual_query_positions": list(range(16))}) == (
        "closed-form MSE-RRR Base16 + Q16 Page-Fisher R8")
    assert base_description({"format": "basisserve.residual_kl_bank.v1",
                             "fit_query": "last token of each window"}) == (
        "closed-form MSE-RRR Base16 + Q1 Page-Fisher R8")
    assert base_description({"format": "basisserve.qaware_base_fisher_bank.v1",
                             "residual_query_positions": list(range(16))}) == (
        "Q-aware Base16 + Q16 Page-Fisher R8")


@torch.inference_mode()
def test_full_budget_rank_independence_prefix_isolation_and_native_decode_mask():
    model, bank, prefix, first = tiny_inputs()
    signature = prefix_signature(prefix)
    before = [(layer.keys.clone(), layer.values.clone()) for layer in prefix.layers]
    generated, traces = {}, {}
    for arm in ("c1_exact_k", "uniform_r8", "c1_exact_k", "uniform_r8"):
        ranks = arm_ranks(arm, 3)
        cache = prepare_arm(model, bank, prefix, ranks)
        if ranks is not None:
            for layer, rank in enumerate(ranks):
                assert cache.routing_sidecar(layer).shape == (1, 2, 16, 8 + rank)
        calls = []

        def observe(*args, **kwargs):
            calls.append(1)
            return _selected_pages(*args, **kwargs)

        with patch("basisserve.core.c1_conditional_page_attention._selected_pages", side_effect=observe):
            ids, final_cache, trace = greedy_decode(model, cache, first, maximum_tokens=5, eos_ids=set(), trace=True)
        assert len(ids) == 5 and final_cache.get_seq_length() == 20
        assert len(calls) == (12 if ranks is not None else 0)
        if ranks is not None:
            for layer, rank in enumerate(ranks):
                assert final_cache.routing_sidecar(layer).shape == (1, 2, 20, 8 + rank)
        assert prefix_signature(prefix) == signature
        for layer, (keys, values) in zip(prefix.layers, before, strict=True):
            torch.testing.assert_close(layer.keys, keys, atol=0, rtol=0)
            torch.testing.assert_close(layer.values, values, atol=0, rtol=0)
        if arm in generated:
            assert ids == generated[arm]
            for a, b in zip(trace, traces[arm], strict=True):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        generated[arm], traces[arm] = ids, trace
    assert generated["c1_exact_k"] == generated["uniform_r8"]
    for a, b in zip(traces["c1_exact_k"], traces["uniform_r8"], strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_greedy_first_token_eos_and_length_limit_do_not_append_cache():
    model, bank, prefix, first = tiny_inputs()
    for maximum, eos in ((8, {first}), (1, set())):
        cache = prepare_arm(model, bank, prefix, None)
        ids, final_cache, trace = greedy_decode(model, cache, first, maximum_tokens=maximum, eos_ids=eos, trace=True)
        assert ids == [first] and final_cache.get_seq_length() == 16 and trace == []


def test_ruler_scoring_and_paired_task_average():
    tasks = parse_tasks("niah_multivalue,qa_1")
    rows = []
    cases = [(tasks[0], ["red", "blue"], "RED", "red and blue"),
             (tasks[0], ["red", "blue"], "red blue", "nothing"),
             (tasks[1], ["Paris", "city of Paris"], "PARIS", "Paris")]
    for task, refs, uniform, adaptive in cases:
        rows.append({"task": task.name, "arms": {
            "uniform": {"score": sample_score(uniform, refs, task.match_type)},
            "adaptive": {"score": sample_score(adaptive, refs, task.match_type)},
        }})
    u = summarize_arm(rows, "uniform", tasks)
    a = summarize_arm(rows, "adaptive", tasks)
    assert u["task_balanced_accuracy"] == .875 and a["task_balanced_accuracy"] == .75
    pair = paired_summary(rows, "uniform", "adaptive", tasks)["all_samples"]
    assert pair["sparse_improvements"] == 1 and pair["sparse_regressions"] == 1 and pair["ties"] == 1
