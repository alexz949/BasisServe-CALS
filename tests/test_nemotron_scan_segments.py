from types import SimpleNamespace
import torch
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from evaluation.nemotron_h_scan_layout import install_contiguous_dt_scan


@torch.inference_mode()
def test_real_scan_segments_preserve_outputs_and_final_state():
    torch.manual_seed(23)
    x = torch.randn(1, 8197, 8, 64, device='cuda', dtype=torch.bfloat16) * 0.1
    dt = torch.ones(1, 8197, 8, device='cuda', dtype=torch.bfloat16) * 0.01
    A = -torch.ones(8, device='cuda')
    B = torch.randn(1, 8197, 2, 64, device='cuda', dtype=torch.bfloat16) * 0.1
    C = torch.randn_like(B) * 0.1
    options = dict(chunk_size=128, return_final_states=True)
    expected, expected_state = mamba_chunk_scan_combined(x, dt, A, B, C, **options)
    namespace = SimpleNamespace(mamba2_chunk_scan=mamba_chunk_scan_combined)
    install_contiguous_dt_scan(namespace)
    actual, actual_state = namespace.mamba2_chunk_scan(x, dt, A, B, C, **options)
    for result, reference in ((actual, expected), (actual_state, expected_state)):
        assert torch.isfinite(result).all()
        relative_rmse = ((result.float() - reference.float()).square().sum()
            / reference.float().square().sum()).sqrt()
        assert relative_rmse < 0.01, relative_rmse
