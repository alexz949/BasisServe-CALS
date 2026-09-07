"""Check upstream invocation, independent support reference and C1 cache bridge."""

from pathlib import Path
import torch

from basisserve.core.c1_official_quest import load_quest, official_attention, OfficialQuest
from evaluation.eval_qwen3_8b_residual_rank_ruler import greedy_decode, prepare_arm
from tests.test_residual_rank_ruler import tiny_inputs
from basisserve.core.residual_kl_replay import prefix_signature

SOURCE = Path("/deac/csc/yangGrp/zhangal/Quest/evaluation/quest_attention.py")


def test_upstream_support_and_padded_value_match_reference():
    torch.manual_seed(47)
    upstream = load_quest(SOURCE)
    q = torch.randn(1, 4, 1, 8)
    k, v = torch.randn(1, 2, 37, 8), torch.randn(1, 2, 37, 5)
    seen = []
    actual = official_attention(upstream, q, k, v, page_size=4, budget=12, observe=seen.append)
    expanded = k.repeat_interleave(2, 1)
    padded = torch.nn.functional.pad(expanded, (0, 0, 0, 3))
    low, high = padded.reshape(1, 4, 10, 4, 8).amin(-2), padded.reshape(1, 4, 10, 4, 8).amax(-2)
    # Exclude zero padding when computing the final partial page's extrema.
    low[..., -1, :] = expanded[..., -1, :]
    high[..., -1, :] = expanded[..., -1, :]
    bound = torch.maximum(q * low, q * high).sum(-1)
    pages = bound.topk(3, dim=-1).indices
    mask = torch.zeros(1, 4, 10, dtype=torch.bool).scatter_(-1, pages, True)
    mask = mask.repeat_interleave(4, -1)[..., :37].unsqueeze(2)
    assert len(seen) == 1 and torch.equal(mask, seen[0])
    scores = (q @ expanded.mT) / 8**0.5
    scores.masked_fill_(~mask, torch.finfo(scores.dtype).min)
    expected = scores.softmax(-1) @ v.repeat_interleave(2, 1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_upstream_full_budget_and_first_two_full_cache_replay():
    model, bank, prefix, first = tiny_inputs()
    upstream = load_quest(SOURCE)
    signature = prefix_signature(prefix)
    original = [(layer.keys.clone(), layer.values.clone()) for layer in prefix.layers]
    cache = prepare_arm(model, bank, prefix, None)
    expected, _, expected_logits = greedy_decode(model, cache, first, maximum_tokens=5, eos_ids=set(), trace=True)
    for repeat in range(2):
        # Upstream clamps budget to length then floors budget/page_size, so an
        # oversized budget with a partial page is not full support. Page1 makes
        # this bridge-equivalence test genuinely full support at every step.
        with OfficialQuest(model, prefix, upstream, page_size=1, budget=64) as runtime:
            ids, cache, logits = greedy_decode(model, runtime.cache, first, maximum_tokens=5, eos_ids=set(), trace=True)
            assert ids == expected and cache.get_seq_length() == 20
            assert runtime.statistics()["layer_calls"] == [0, 0, 4]
            for actual, reference in zip(logits, expected_logits, strict=True):
                torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
        assert prefix_signature(prefix) == signature
        for layer, (k, v) in zip(prefix.layers, original, strict=True):
            assert torch.equal(layer.keys, k) and torch.equal(layer.values, v)
