import torch

from evaluation.trace_multikey_pages import (
    PageTrace, page_snapshot, c1_conditional_page_topk_attention, _memory_bounded_gqa_sdpa,
)


def test_partial_page_and_tie_ranks():
    scores = torch.zeros(1, 4, 1, 259)
    valid = torch.ones_like(scores, dtype=torch.bool)
    snapshot = page_snapshot(scores, valid, kv_heads=1, page_size=4,
                             page_budget=64, pinned_prefix_pages=1)
    assert snapshot['ids'].shape == (1, 64)
    assert snapshot['rank_min'][0, 0] == 0
    assert (snapshot['rank_min'][0, 1:64] == 1).all()
    assert snapshot['rank_min'][0, 64] == 64
    torch.testing.assert_close(snapshot['full_mass'].sum(-1), torch.ones(1, 4))
    assert snapshot['full_mass'][0, 0, 64] < snapshot['full_mass'][0, 0, 63]


def test_trace_preserves_proxy_and_exact_payloads():
    torch.manual_seed(124)
    query = torch.randn(1, 32, 1, 128)
    key = torch.randn(1, 8, 2085, 128)
    value = torch.randn(1, 8, 2085, 80)
    sidecar = torch.randn(1, 8, 2085, 136)
    projector = torch.randn(32, 128, 136) / 12
    options = dict(page_size=32, exact_token_budget=2048, pinned_prefix_pages=1,
                   scale=128**-.5, query_block_size=1, collect_statistics=False,
                   attention_mask=torch.ones(1, 1, 1, 2085, dtype=torch.bool))
    native = c1_conditional_page_topk_attention(query, key, value, sidecar, projector, **options)
    traced = PageTrace('q32_proxy', [])(query, key, value, sidecar, projector, **options)
    torch.testing.assert_close(native.output, traced.output, rtol=0, atol=0)
    tracer = PageTrace('exact_pages', [])
    exact = tracer(query, key, value, sidecar, projector, **options)
    selected = tracer.tensors['step001.layer00.exact.ids']
    support = torch.zeros(8, 2085, dtype=torch.bool)
    for group in range(8):
        for page in selected[group].tolist():
            support[group, page*32:min((page+1)*32, 2085)] = True
    full_key = key.repeat_interleave(4, 1)
    full_value = value.repeat_interleave(4, 1)
    scores = (query @ full_key.transpose(-1, -2)) * options['scale']
    scores.masked_fill_(~support.repeat_interleave(4, 0)[None, :, None], -torch.inf)
    expected = scores.softmax(-1) @ full_value
    torch.testing.assert_close(exact.output, expected, rtol=1e-5, atol=1e-6)
    full = PageTrace('full_exact_k', [])(query, key, value, sidecar, projector, **options)
    expected_full = _memory_bounded_gqa_sdpa(query, key, value, attention_mask=options['attention_mask'],
                                            dropout_p=0., is_causal=False, scale=options['scale'])
    torch.testing.assert_close(full.output, expected_full, rtol=0, atol=0)
