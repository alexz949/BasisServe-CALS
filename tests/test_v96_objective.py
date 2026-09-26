import torch

from basisserve.core.gqa_vo_svdllm import GQAVOLayout
from basisserve.core.joint_aa_gqa_o import materialize_structured_encoder, resolve_head_to_kv_group
from evaluation.run_v96_objective import independent_covariance, fit_pair


def test_independent_metric_preserves_within_source_cross_heads():
    torch.manual_seed(7)
    z = torch.randn(31, 16, dtype=torch.float64)
    r = torch.randn(16, 9, dtype=torch.float64)
    c = z.T @ z / len(z)
    local = independent_covariance(c, 8)
    expected = sum((z[:, s:s+8] @ r[s:s+8]).square().sum() for s in (0, 8)) / len(z)
    torch.testing.assert_close((r * (local @ r)).sum(), expected)
    torch.testing.assert_close(local[:8, :8], c[:8, :8])
    assert torch.count_nonzero(local[:8, 8:]) == 0


def test_shared_encoder_can_move_before_attention():
    torch.manual_seed(8)
    layout = GQAVOLayout(16, 4, 2, 4, 3)
    mapping = resolve_head_to_kv_group(layout)
    values = torch.randn(2, 11, 4, dtype=torch.float64)
    attention = torch.randn(4, 3, 11, dtype=torch.float64).softmax(-1)
    e = torch.randn(2, 4, 3, dtype=torch.float64)
    d = torch.randn(12, 16, dtype=torch.float64)
    post = torch.einsum('hts,hsd->thd', attention, values[mapping]).reshape(3, 16)
    before = torch.einsum('gsd,gdr->gsr', values, e)
    after = torch.einsum('hts,hsr->thr', attention, before[mapping]).reshape(3, 12)
    torch.testing.assert_close(after @ d, post @ materialize_structured_encoder(e, mapping) @ d)


def test_pair_matches_structure_and_monotonic_objectives():
    torch.manual_seed(9)
    z = torch.randn(64, 16, dtype=torch.float64)
    w = torch.randn(16, 16, dtype=torch.float64)
    layout = GQAVOLayout(16, 4, 2, 4, 3)
    pair = fit_pair(w, z.T @ z / len(z), layout, len(z), 2)
    for factors, audit in pair.values():
        assert factors['encoder_fp64'].shape == (2, 4, 3)
        assert factors['decoder_fp64'].shape == (12, 16)
        assert not audit['global_optimum_certified']
        for start in audit['starts']:
            scores = [item['objective'] for item in start['history']]
            assert all(b <= a for a, b in zip(scores, scores[1:]))
