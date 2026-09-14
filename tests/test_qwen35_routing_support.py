import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from basisserve.core.qwen35_k_routing_runtime import page_support
from basisserve.core.c1_shadowkv import C1ShadowKVState


def test_page_support_normalizes_heads_and_counts_partial_page():
    scores = torch.zeros(1, 2, 4, 193)
    scores[:, :, 0] = 100  # A large constant must not give this head more mass.
    scores[:, :, 1, 128] = 20
    ids, valid = page_support(scores, budget=96)
    expected = torch.cat((torch.arange(128, 160), torch.arange(129, 193)))
    torch.testing.assert_close(ids[0, 0], expected)
    torch.testing.assert_close(ids[0, 1], expected)
    assert valid.sum(-1).tolist() == [[65, 65]]
    torch.testing.assert_close(ids[0, 0, valid[0, 0]], torch.arange(128, 193))


def test_shadow_reconstruction_uses_partial_rope():
    torch.manual_seed(41)
    pre = torch.randn(1, 2, 128, 16)
    angles = torch.randn(1, 128, 2).repeat(1, 1, 2)
    cos, sin = angles.cos(), angles.sin()
    post, _ = apply_rotary_pos_emb(pre, pre, cos, sin)
    state = C1ShadowKVState(pre, post, cos, sin, rank=8, budget=16, chunk=8, outliers=2)
    ids = torch.tensor([[[0, 9, 65], [1, 12, 90]]])
    reconstructed = state.reconstruct(ids)
    dense_pre = torch.einsum('btr,bhrd->bhtd', state.u, state.sv)
    dense_post, _ = apply_rotary_pos_emb(dense_pre, dense_pre, cos, sin)
    expected = dense_post.gather(2, ids[..., None].expand(1, 2, 3, 16))
    torch.testing.assert_close(reconstructed, expected, atol=1e-6, rtol=1e-5)


def test_page_support_does_not_pin_sink_and_excludes_recent_before_ranking():
    scores = torch.zeros(1, 1, 2, 192)
    scores[..., 0, 32:64] = 8
    scores[..., 1, 64:96] = 2
    scores[..., 0, 128:] = 100
    ids, valid = page_support(scores, budget=96)
    torch.testing.assert_close(ids[0, 0], torch.cat((torch.arange(32, 64), torch.arange(128, 192))))
    assert valid.all()
    scores[..., 128:] = -100
    changed_recent, _ = page_support(scores, budget=96)
    torch.testing.assert_close(ids, changed_recent)
    scores[..., :32] = 100
    selected_sink, _ = page_support(scores, budget=96)
    torch.testing.assert_close(selected_sink[0, 0, :32], torch.arange(32))


def test_page_support_prefix_only_is_finite_and_keeps_valid_tokens():
    ids, valid = page_support(torch.randn(1, 2, 4, 17))
    assert ids.shape == (1, 2, 17)
    assert valid.sum(-1).tolist() == [[17, 17]]
    torch.testing.assert_close(ids[0, 0, valid[0, 0]], torch.arange(17))


def test_page_support_hard_budget_recent_and_unique_across_page_boundaries():
    for length in (1, 32, 63, 64, 65, 2047, 2048, 2049, 4095, 4096, 4097):
        ids, valid = page_support(torch.randn(1, 2, 4, length))
        for group in range(2):
            chosen = ids[0, group, valid[0, group]]
            assert chosen.numel() <= min(length, 2048)
            assert chosen.unique().numel() == chosen.numel()
            assert torch.isin(torch.arange(max(0, length-64), length), chosen).all()
            if length <= 2048:
                torch.testing.assert_close(chosen, torch.arange(length))
