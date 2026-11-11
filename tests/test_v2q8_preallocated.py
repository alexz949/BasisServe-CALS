"""The V2Q8 storage changes must not change arithmetic: preallocated layers and LRQK code storage
reproduce the reference DynamicLayer and c1_lrqk.LRQKState bit for bit (CPU, small shapes)."""

import torch
from transformers.cache_utils import DynamicLayer

from basisserve.core.c1_lrqk import LRQKConfig, LRQKState
from benchmarks.system import v2q8_tp1_v96 as v96


def test_preallocated_layer_matches_dynamic_layer():
    torch.manual_seed(0)
    reference, preallocated = DynamicLayer(), v96.PreallocatedLayer(capacity=16, keep_keys=True)
    steps = [(torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 6))] + [
        (torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 6)) for _ in range(3)]
    for key, value in steps:
        expected, actual = reference.update(key, value), preallocated.update(key, value)
        assert torch.equal(expected[0], actual[0]) and torch.equal(expected[1], actual[1])
        assert reference.get_seq_length() == preallocated.get_seq_length()
        assert reference.get_mask_sizes(1) == preallocated.get_mask_sizes(1)


def test_shadowkv_layer_returns_current_keys_and_all_values():
    layer = v96.PreallocatedLayer(capacity=8, keep_keys=False)
    prompt_key, prompt_value = torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 3)
    key, value = layer.update(prompt_key, prompt_value)
    assert torch.equal(key, prompt_key) and torch.equal(value, prompt_value)
    step_key, step_value = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 3)
    key, value = layer.update(step_key, step_value)
    assert torch.equal(key, step_key) and torch.equal(value, torch.cat((prompt_value, step_value), 2))
    assert layer.get_seq_length() == 6 and layer.key_storage is None


def test_preallocated_lrqk_state_matches_reference():
    torch.manual_seed(1)
    config = LRQKConfig(rank=4, topk=8, recent=4, prefill_backend="sdpa")
    length, steps = 40, 6
    q, k = torch.randn(1, 8, length, 16), torch.randn(1, 2, length, 16)
    v96.CAPACITY["tokens"] = length + steps
    reference, preallocated = LRQKState(q, k, config, layer=3), v96.PreallocatedLRQKState(q, k, config, layer=3)
    assert torch.equal(reference.selected, preallocated.selected)
    keys, values = k, torch.randn(1, 2, length, 12)
    for _ in range(steps):
        keys = torch.cat((keys, torch.randn(1, 2, 1, 16)), 2)
        values = torch.cat((values, torch.randn(1, 2, 1, 12)), 2)
        query = torch.randn(1, 8, 1, 16)
        expected = reference.decode(query, keys, values, 0.25)
        actual = preallocated.decode(query, keys, values, 0.25)
        assert torch.equal(expected, actual)
        assert torch.equal(reference.selected, preallocated.selected)
        assert torch.equal(reference.ak, preallocated.ak) and torch.equal(reference.bq, preallocated.bq)
