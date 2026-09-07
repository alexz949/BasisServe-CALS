import torch

from basisserve.core.c1_v_conditional_k_router import AffineReducedRankMap
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (
    _build_residual_statistics, _post_rope_rows,
)
from evaluation.eval_qwen3_8b_v80_exact_query_weighted_rrr import _query_weighted_document_loss
from evaluation.fit_qwen3_8b_qaware_base_fisher_bank import base_maps, pack_bank


def test_query_loss_matches_explicit_position_rotated_causal_scores():
    torch.manual_seed(91)
    tokens, groups, dim = 12, 2, 4
    rows = torch.randn(tokens, groups, dim * 2)
    queries = torch.randn(2, 4, dim)
    encoder = torch.randn(groups, dim, 3)
    left, right, bias = torch.randn(groups, 3, 2), torch.randn(groups, 2, dim), torch.randn(groups, dim)
    angles = torch.arange(tokens)[:, None] * torch.tensor([.3, .8])[None, :]
    angles = torch.cat((angles, angles), dim=-1)[None]
    cos, sin = angles.cos(), angles.sin()
    positions = torch.tensor([5, 8])
    kwargs = dict(query_positions=positions, value_encoder=encoder, left=left, right=right,
                  bias=bias, cos=cos, sin=sin, page_size=2, pinned_prefix_pages=1)
    loss, energy = _query_weighted_document_loss(queries, rows, **kwargs)
    expected_loss, expected_energy = torch.zeros(()), torch.zeros(())
    for sample, stop in enumerate(positions.tolist()):
        for h in range(4):
            g = h // 2
            for i in range(2, stop + 1):
                pre = rows[i, g, :dim] @ encoder[g] @ left[g] @ right[g] + bias[g]
                rotated_half = torch.cat((-pre[2:], pre[:2]))
                post = pre * cos[0, i] + rotated_half * sin[0, i]
                target = queries[sample, h] @ rows[i, g, dim:] / dim**.5
                error = queries[sample, h] @ post / dim**.5 - target
                expected_loss += .5 * error.square()
                expected_energy += .5 * target.square()
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(energy, expected_energy)
    changed = rows.clone()
    changed[:2] += 1000
    changed[9:] -= 1000
    masked_loss, masked_energy = _query_weighted_document_loss(queries, changed, **kwargs)
    torch.testing.assert_close(loss, masked_loss, rtol=0, atol=0)
    torch.testing.assert_close(energy, masked_energy, rtol=0, atol=0)
    doubled_loss, _ = _query_weighted_document_loss(queries * 2, rows, **kwargs)
    torch.testing.assert_close(doubled_loss, loss * 4)


@torch.inference_mode()
def test_residual_fisher_is_recomputed_for_changed_pre_rope_base():
    torch.manual_seed(92)
    value = torch.randn(12, 2, 4)
    angles = (torch.arange(12)[:, None] * torch.tensor([.2, .7])[None, :]).repeat(1, 2)[None]
    cos, sin = angles.cos(), angles.sin()
    key = _post_rope_rows(value, cos, sin)
    rows = torch.cat((value, key), dim=-1)[None]
    queries = torch.randn(1, 4, 4)
    identity = torch.eye(4)
    exact = tuple(AffineReducedRankMap(identity, identity, torch.zeros(4)) for _ in range(2))
    zero = tuple(AffineReducedRankMap(torch.zeros(4, 4), identity, torch.zeros(4)) for _ in range(2))
    kwargs = dict(value_encoder=identity[None].repeat(2, 1, 1), cos=cos, sin=sin,
                  page_size=2, excluded_prefix_pages=1, device=torch.device("cpu"))
    initial, _ = _build_residual_statistics(queries, rows, base_maps={16: zero}, **kwargs)
    refitted, reconstruction = _build_residual_statistics(queries, rows, base_maps={16: exact}, **kwargs)
    assert initial[16].teacher_fisher_energy > 0
    assert initial[16].fisher_grams_by_head.abs().sum() > 0
    assert refitted[16].teacher_fisher_energy == 0
    assert refitted[16].fisher_grams_by_head.count_nonzero() == 0
    assert reconstruction[16]["residual_squared_error"] == 0


def test_bank_packs_new_base_and_all_three_residual_ranks_without_mutating_input():
    torch.manual_seed(93)
    source = {"base_left_b16": torch.randn(2, 3, 2),
              "base_right_b16": torch.randn(2, 2, 4), "base_bias_b16": torch.randn(2, 4)}
    before = {name: tensor.clone() for name, tensor in source.items()}
    residuals = {(16, rank): (torch.randn(2, 4, rank), torch.randn(4, 4, rank)) for rank in (4, 8, 16)}
    packed = pack_bank(base_maps(source), residuals)
    assert len(packed) == 9
    for name, value in source.items():
        assert torch.equal(value, before[name]) and torch.equal(packed[name], value)
    for rank in (4, 8, 16):
        assert torch.equal(packed[f"residual_encoder_b16_r{rank}"], residuals[(16, rank)][0])
        assert torch.equal(packed[f"residual_query_b16_r{rank}"], residuals[(16, rank)][1])
