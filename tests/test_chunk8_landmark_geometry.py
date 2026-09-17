import torch

from evaluation.diagnose_chunk8_landmark_geometry import (
    CHUNK_SIZE,
    chunk8_support,
    chunk_logsumexp,
    chunk_mean_scores,
    chunk_means,
    selected_chunk_mask,
    synthetic_audits,
)


def test_chunk8_synthetic_audits():
    assert all(synthetic_audits().values())


def test_exact_chunk_lse_matches_direct_token_definition():
    torch.manual_seed(5)
    query = torch.randn(1, 2, 3, 11)
    key = torch.randn(1, 2, 40, 11)
    token_scores = torch.einsum("bghd,bgtd->bght", query, key)
    expected = torch.stack(
        [torch.logsumexp(token_scores[..., start : start + CHUNK_SIZE], -1)
         for start in range(0, 40, CHUNK_SIZE)],
        -1,
    )
    torch.testing.assert_close(chunk_logsumexp(token_scores, 5), expected)


def test_identical_rows_make_mean_landmark_exact():
    torch.manual_seed(7)
    query = torch.randn(1, 2, 3, 16)
    rows = torch.randn(1, 2, 5, 1, 16).expand(-1, -1, -1, CHUNK_SIZE, -1)
    key = rows.reshape(1, 2, 5 * CHUNK_SIZE, 16).contiguous()
    landmarks = chunk_means(key)
    token_scores = torch.einsum("bghd,bgtd->bght", query, key)
    teacher = chunk_logsumexp(token_scores, 5)
    student = chunk_mean_scores(query, landmarks, 5, scale=1.0)
    torch.testing.assert_close(student, teacher, atol=2e-2, rtol=2e-2)


def test_chunk8_budget_has_4_pinned_244_routed_and_recent64():
    torch.manual_seed(9)
    total_tokens = 65_283
    chunks = (total_tokens - 64) // CHUNK_SIZE
    scores = torch.randn(1, 8, 4, chunks)
    ids, valid = chunk8_support(scores, total_tokens, budget=2048)
    assert (valid.sum(-1) == 2048).all()
    historical = selected_chunk_mask(ids, valid, chunks, routed_only=False)
    routed = selected_chunk_mask(ids, valid, chunks, routed_only=True)
    assert (historical.sum(-1) == 248).all()
    assert (routed.sum(-1) == 244).all()
    assert historical[..., :4].all()
    assert not routed[..., :4].any()
    selected_recent = ids[..., -64:]
    expected_recent = torch.arange(total_tokens - 64, total_tokens)
    assert torch.equal(selected_recent[0, 0], expected_recent)
