import torch

from basisserve.core.c1_v_conditional_k_router import (
    build_conditional_routing_sidecar,
)
from basisserve.core.chunk8_fisher_routing import (
    CHUNK_SIZE,
    ChunkLandmarkState,
    chunk8_hard_budget_support,
    exact_chunk_logits,
    landmark_chunk_logits,
    predict_post_rope_base,
)


def test_base_prediction_matches_existing_conditional_sidecar() -> None:
    torch.manual_seed(1)
    values = torch.randn(1, 2, 13, 6)
    key = torch.randn(1, 2, 13, 6)
    left = torch.randn(2, 6, 3)
    right = torch.randn(2, 3, 6)
    bias = torch.randn(2, 6)
    encoder = torch.randn(2, 6, 2)
    angle = torch.randn(1, 13, 6)
    cos, sin = angle.cos(), angle.sin()
    expected = build_conditional_routing_sidecar(
        values,
        key,
        base_left=left,
        base_right=right,
        base_bias=bias,
        residual_encoder=encoder,
        cos=cos,
        sin=sin,
    )[..., :6]
    actual = predict_post_rope_base(
        values,
        base_left=left,
        base_right=right,
        base_bias=bias,
        cos=cos,
        sin=sin,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_incremental_landmarks_match_one_shot_chunk_means() -> None:
    torch.manual_seed(2)
    base = torch.randn(1, 2, 29, 7)
    residual = torch.randn(1, 2, 29, 3)
    state = ChunkLandmarkState.from_tokens(base[:, :, :13], residual[:, :, :13])
    state.append(base[:, :, 13:14], residual[:, :, 13:14])
    state.append(base[:, :, 14:22], residual[:, :, 14:22])
    state.append(base[:, :, 22:], residual[:, :, 22:])
    assert state.tokens == 29
    expected_base = base[:, :, :24].reshape(1, 2, 3, 8, 7).mean(dim=-2)
    expected_residual = residual[:, :, :24].reshape(1, 2, 3, 8, 3).mean(dim=-2)
    torch.testing.assert_close(state.base, expected_base)
    torch.testing.assert_close(state.residual, expected_residual)
    torch.testing.assert_close(state.pending_base, base[:, :, 24:])
    torch.testing.assert_close(state.pending_residual, residual[:, :, 24:])


def test_landmark_scores_match_exact_lse_for_identical_chunk_rows() -> None:
    torch.manual_seed(3)
    query = torch.randn(1, 2, 3, 5)
    rows = torch.randn(1, 2, 4, 5)
    key = rows[:, :, :, None].expand(1, 2, 4, CHUNK_SIZE, 5).reshape(1, 2, 32, 5)
    state = ChunkLandmarkState.from_tokens(key, None)
    exact = exact_chunk_logits(query, key, 4, scale=5**-0.5)
    base = landmark_chunk_logits(
        query,
        state,
        4,
        scale=5**-0.5,
        query_factor=None,
    )
    torch.testing.assert_close(base, exact, atol=1e-6, rtol=1e-6)


def test_mean_residual_correction_uses_query_head_specific_factors() -> None:
    torch.manual_seed(4)
    query = torch.randn(1, 2, 3, 5)
    base_tokens = torch.randn(1, 2, 16, 5)
    residual_codes = torch.randn(1, 2, 16, 2)
    query_factor = torch.randn(2, 3, 5, 2)
    state = ChunkLandmarkState.from_tokens(base_tokens, residual_codes)
    base = landmark_chunk_logits(
        query,
        state,
        2,
        scale=0.5,
        query_factor=None,
    )
    mean = landmark_chunk_logits(
        query,
        state,
        2,
        scale=0.5,
        query_factor=query_factor,
    )
    query_code = torch.einsum("bghd,ghdr->bghr", query, query_factor)
    expected = 0.5 * torch.einsum("bghr,bgcr->bghc", query_code, state.residual)
    torch.testing.assert_close(mean - base, expected, atol=1e-6, rtol=1e-6)


def test_hard_budget_is_exact_when_aligned() -> None:
    torch.manual_seed(5)
    total_tokens = 4096
    chunks = (total_tokens - 64) // 8
    logits = torch.randn(1, 2, 4, chunks)
    ids, stats = chunk8_hard_budget_support(logits, total_tokens)
    assert ids.shape == (1, 2, 2048)
    assert stats == {
        "logical_budget_tokens": 2048,
        "actual_support_tokens": 2048,
        "equivalent_chunk_slots": 256,
        "historical_chunks": 248,
        "pinned_sink_chunks": 4,
        "routed_historical_chunks": 244,
        "tail_chunks": 8,
        "exact_tail_tokens": 64,
        "unused_token_capacity": 0,
    }


def test_ragged_boundary_keeps_complete_recent64_without_exceeding_budget() -> None:
    torch.manual_seed(6)
    total_tokens = 4099
    chunks = (total_tokens - 64) // 8
    logits = torch.randn(1, 2, 4, chunks)
    ids, stats = chunk8_hard_budget_support(logits, total_tokens)
    assert ids.shape == (1, 2, 2043)
    assert stats["equivalent_chunk_slots"] == 256
    assert stats["historical_chunks"] == 247
    assert stats["routed_historical_chunks"] == 243
    assert stats["tail_chunks"] == 9
    assert stats["exact_tail_tokens"] == 67
    assert stats["unused_token_capacity"] == 5
    expected_recent = torch.arange(total_tokens - 64, total_tokens)
    for group in range(2):
        assert torch.isin(expected_recent, ids[0, group].cpu()).all()
        assert int(ids[0, group].unique().numel()) == int(ids.shape[-1])
