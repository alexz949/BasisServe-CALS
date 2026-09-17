import torch

from evaluation.diagnose_page_landmarks import (
    build_subpage_landmarks,
    page_scores_to_token_proxy,
    score_landmark_pages,
    selected_page_mask,
    synthetic_audits,
    token_page_scores,
)
from evaluation.llama_sink_recent_routing import page_support


def test_synthetic_landmark_audits_pass():
    assert all(synthetic_audits().values())


def test_landmark_page_scores_preserve_current_gqa_selection():
    torch.manual_seed(11)
    batch, kv_heads, query_groups, tokens, width = 1, 2, 3, 353, 9
    state = torch.randn(batch, kv_heads, tokens, width)
    query = torch.randn(batch, kv_heads, query_groups, width)
    token_scores = torch.einsum("bghd,bgtd->bght", query, state)
    historical = tokens - 64
    teacher_pages = token_page_scores(token_scores, historical)
    proxy = page_scores_to_token_proxy(teacher_pages, historical, tokens)
    teacher_ids, teacher_valid = page_support(token_scores, budget=128)
    proxy_ids, proxy_valid = page_support(proxy, budget=128)
    assert torch.equal(teacher_ids, proxy_ids)
    assert torch.equal(teacher_valid, proxy_valid)


def test_ragged_landmark_uses_weighted_counts_and_no_padding():
    state = torch.tensor([[[[1.0], [1.0], [1.0], [1.0], [3.0]]]])
    query = torch.ones(1, 1, 1, 1)
    landmarks, counts = build_subpage_landmarks(state, 4, storage_dtype=None)
    assert counts.tolist() == [[4, 1, 0, 0, 0, 0, 0, 0]]
    score = score_landmark_pages(query, landmarks, counts, scale=1.0)
    direct = torch.logsumexp(state[:, :, None, :, 0], -1, keepdim=True)
    torch.testing.assert_close(score, direct)


def test_selected_page_mask_separates_pinned_and_routed_pages():
    scores = torch.randn(1, 2, 3, 257)
    ids, valid = page_support(scores, budget=128)
    all_pages = selected_page_mask(ids, valid, 193, routed_only=False)
    routed = selected_page_mask(ids, valid, 193, routed_only=True)
    assert all_pages[..., 0].all()
    assert not routed[..., 0].any()
    assert (all_pages.sum(-1) == 2).all()
    assert (routed.sum(-1) == 1).all()
