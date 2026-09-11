import pytest
import torch

from basisserve.core.qwen35_gdn_private_ag import fit_qwen35_private_ag_joint_factors


@pytest.mark.parametrize('weight_scale', [1.0, 0.02])
@pytest.mark.parametrize('local_rank', [4, 6])
def test_wo_completes_six_sweeps_at_native_weight_scale(weight_scale, local_rank):
    torch.manual_seed(630)
    weight = torch.randn(24, 32) * weight_scale
    rows = torch.randn(256, 32)
    heldout = torch.randn(64, 32)
    result = fit_qwen35_private_ag_joint_factors(weight, rows.T @ rows / len(rows),
        heldout.T @ heldout / len(heldout), tp_size=4, local_rank=local_rank,
        encoder_sweeps=6, minimum_encoder_sweeps=6)
    assert result.metrics['diagnostics']['encoder_sweeps_completed'] == 6
    assert result.private_encoders.shape == (4, 8, local_rank)
    assert result.joint_decoder_weight.shape == (24, 4 * local_rank)
    assert torch.isfinite(result.private_encoders).all()
    assert torch.isfinite(result.joint_decoder_weight).all()
