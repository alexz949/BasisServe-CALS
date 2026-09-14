import pytest
import torch

from basisserve.kernels.split_indexed_attention import split_indexed_attention


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Requires two GPUs')
def test_noncurrent_device_matches_selected_dense_attention():
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    device = torch.device('cuda:1')
    q = torch.randn(1, 4, 1, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 131072, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ids = torch.linspace(0, 131071, 2048, device=device).long().expand(1, 4, -1).clone()
    ids[:, :, 100:110] = -1
    heads = torch.arange(4, device=device) // 2
    selected_k = k[0, heads[:, None], ids[0].clamp_min(0)]
    selected_v = v[0, heads[:, None], ids[0].clamp_min(0)]
    scores = (q[0, :, 0].float()[:, None] * selected_k.float()).sum(-1) / 128**0.5
    probabilities = scores.masked_fill(ids[0] < 0, -torch.inf).softmax(-1)
    expected = (probabilities[:, :, None] * selected_v.float()).sum(1)[None, :, None]
    for _ in range(3):
        actual = split_indexed_attention(q, k, v, ids, scale=128**-0.5)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.float(), expected, atol=0.002, rtol=0.02)
        assert torch.cuda.current_device() == 0
