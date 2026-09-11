import pytest
import torch

from basisserve.core.qwen35_gated_v_als import GatedVCapture, GatedVBlock, gated_v_update, fit_gated_v


def fixture():
    generator = torch.Generator().manual_seed(197)
    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)
    z, gate, weight = rand(13, 4, 3), rand(13, 4, 3).sigmoid(), rand(4, 3, 5)
    capture = GatedVCapture(z, gate, weight, torch.einsum('nhd,hdo->no', z * gate, weight), torch.tensor([0, 0, 1, 1]))
    return capture, rand(2, 3, 2), rand(2, 2, 3)


@pytest.mark.parametrize('block', ['encoder', 'decoder'])
@pytest.mark.parametrize('preconditioner', ['jacobi', 'separable'])
def test_adjoint_chunks_and_damped_explicit_reference(block, preconditioner):
    capture, e, r = fixture()
    fixed, old = (r, e) if block == 'encoder' else (e, r)
    full = GatedVBlock(capture, fixed, block=block, chunk_rows=100)
    op = GatedVBlock(capture, fixed, block=block, chunk_rows=4)
    h = capture.target
    torch.testing.assert_close((op.apply(old) * h).sum(), (old * op.adjoint(h)).sum())
    torch.testing.assert_close(op.apply(old), full.apply(old))
    torch.testing.assert_close(op.adjoint(h), full.adjoint(h))
    torch.testing.assert_close(op.normal(old), full.normal(old))
    torch.testing.assert_close(op.normal(old), op.adjoint(op.apply(old)) / len(capture.z))
    eye = torch.eye(old.numel(), dtype=old.dtype)
    design = torch.stack([op.apply(col.reshape_as(old)).flatten() for col in eye], dim=1)
    updated, diagnostic = gated_v_update(op, old, relative_damping=0.01, linear_tol=1e-12, linear_max_iter=100,
        encoder_preconditioner=preconditioner)
    n = capture.z.shape[0]
    damping = diagnostic['absolute_damping']
    matrix = torch.cat([design / n**0.5, damping**0.5 * eye])
    rhs = torch.cat([(h.flatten() - design @ old.flatten()) / n**0.5, torch.zeros(old.numel(), dtype=old.dtype)])
    expected = old + torch.linalg.lstsq(matrix, rhs).solution.reshape_as(old)
    torch.testing.assert_close(updated, expected, atol=1e-9, rtol=1e-9)
    assert diagnostic['accepted'] and diagnostic['loss_after'] <= diagnostic['loss_before']
    assert diagnostic['recomputed_relative_residual'] < 1e-9


def test_identity_gauge_and_fitting():
    capture, e, r = fixture()
    identity = torch.eye(3, dtype=e.dtype).repeat(2, 1, 1)
    op = GatedVBlock(capture, identity, block='decoder')
    torch.testing.assert_close(op.apply(identity), capture.target)
    q, t = torch.linalg.qr(e)
    torch.testing.assert_close(GatedVBlock(capture, e, block='decoder').apply(r),
                               GatedVBlock(capture, q, block='decoder').apply(t @ r))
    fit = fit_gated_v(capture, 2, encoder_sweeps=2, work_dtype=torch.float64, linear_tol=1e-10)
    assert len(fit['endpoints']) == 3
    assert all(row['loss_after'] <= row['loss_before'] for row in fit['history'] if row['block'] != 'gauge')


def test_attention_folding_and_gate_noncommutation():
    capture, e, r = fixture()
    torch.manual_seed(25)
    v = torch.randn(2, 7, 3, dtype=e.dtype)
    attention = torch.randn(4, 5, 7, dtype=e.dtype).softmax(-1)
    mapping = capture.head_to_group
    latent = torch.einsum('gld,gdr->glr', v, e)
    actual = torch.einsum('hql,hlr,hrd->hqd', attention, latent[mapping], r[mapping])
    expected = torch.einsum('hql,hld,hdr,hre->hqe', attention, v[mapping], e[mapping], r[mapping])
    torch.testing.assert_close(actual, expected)
    z, gate = capture.z, capture.gate
    product = e[mapping] @ r[mapping]
    correct = torch.einsum('nhd,hde->nhe', z, product) * gate
    wrong = torch.einsum('nhd,hde->nhe', z * gate, product)
    assert not torch.allclose(correct, wrong)


def test_cross_head_mixing_cannot_fold_before_distinct_attention():
    generator = torch.Generator().manual_seed(39)
    values = torch.randn(2, 7, 3, generator=generator, dtype=torch.float64)
    probabilities = torch.randn(2, 5, 7, generator=generator, dtype=torch.float64).softmax(-1)
    mixing = torch.tensor([[1., 0.7], [-0.2, 1.]], dtype=torch.float64)
    outputs = torch.einsum('hql,hld->hqd', probabilities, values)
    mix_after_attention = torch.einsum('gh,hqd->gqd', mixing, outputs)
    mixed_cache = torch.einsum('gh,hld->gld', mixing, values)
    invalid_fold = torch.einsum('gql,gld->gqd', probabilities, mixed_cache)
    assert not torch.allclose(mix_after_attention, invalid_fold)
    # A shared probability operator is the special commuting case.
    shared = probabilities[:1].expand_as(probabilities)
    torch.testing.assert_close(
        torch.einsum('gh,hql,hld->gqd', mixing, shared, values),
        torch.einsum('gql,gld->gqd', shared, mixed_cache))


@pytest.mark.parametrize('block', ['encoder', 'decoder'])
def test_rank_deficient_block_predictions(block):
    capture, e, r = fixture()
    capture.z[:, :, 2] = capture.z[:, :, 1]
    capture = GatedVCapture(capture.z, capture.gate, capture.weight,
        torch.einsum('nhd,hdo->no', capture.z * capture.gate, capture.weight), capture.head_to_group)
    if block == 'decoder':
        e[:, :, 1] = e[:, :, 0]
    fixed, old = (r, e) if block == 'encoder' else (e, r)
    op = GatedVBlock(capture, fixed, block=block)
    eye = torch.eye(old.numel(), dtype=old.dtype)
    design = torch.stack([op.apply(col.reshape_as(old)).flatten() for col in eye], 1)
    expected = design @ torch.linalg.lstsq(design, capture.target.flatten(), driver='gelsd').solution
    actual, diagnostic = gated_v_update(op, old, relative_damping=0, linear_tol=1e-11, linear_max_iter=200)
    torch.testing.assert_close(op.apply(actual).flatten(), expected, atol=1e-7, rtol=1e-7)
    assert diagnostic['accepted']
