import torch
from evaluation.llama_sink_recent_routing import page_support


def test_sink_recent_budget_and_boundaries():
    torch.manual_seed(17)
    for length in (1, 32, 64, 2048, 2049, 2080, 2081, 4095, 4096):
        scores = torch.randn(1, 2, 4, length)
        # Make the historical/recent boundary attractive to catch duplicates.
        scores[..., max(0, length-96):] += 10
        ids, valid = page_support(scores)
        for group in range(2):
            selected = ids[0, group][valid[0, group]].tolist()
            assert len(selected) == len(set(selected)) <= 2048
            assert all(0 <= i < length for i in selected)
            assert set(range(min(32, length))) <= set(selected)
            assert set(range(max(0, length-64), length)) <= set(selected)
            if length <= 2048:
                assert selected == list(range(length))
