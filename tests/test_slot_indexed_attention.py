import pytest
import torch

from basisserve.kernels.slot_indexed_attention import slot_indexed_attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("layout", ["contiguous", "feature_major"])
def test_slot_attention_matches_reference(batch, layout):
    torch.manual_seed(42)
    device = "cuda"
    heads, rank, support, length, splits = 4, 96, 2048, 4096, 16
    q = torch.randn(batch, heads, 1, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, 1, support, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, 1, length, rank, device=device, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(length, device=device)[:support] for _ in range(batch)])[:, None]
    slots = torch.stack([torch.randperm(support, device=device) for _ in range(batch)])[:, None]
    # Empty splits and a partial page must not contaminate the softmax merge.
    ids[:, :, :128] = -1
    ids[:, :, 500:509] = -1
    slots[:, :, 900:905] = -1
    partial = torch.empty(batch, heads, splits, rank, device=device)
    lse = torch.empty(batch, heads, splits, device=device)
    if layout == "feature_major":
        arena = torch.full((heads * rank + 2, batch), 123, device=device, dtype=q.dtype)
        out = arena[1:-1].T.view(batch, heads, 1, rank)
    else:
        out = torch.empty(batch, heads, 1, rank, device=device, dtype=q.dtype)

    rows = torch.arange(batch, device=device)[:, None]
    selected_k = k[:, 0][rows, slots[:, 0].clamp_min(0)].float()
    selected_v = v[:, 0][rows, ids[:, 0].clamp_min(0)].float()
    scores = q[:, :, 0].float() @ selected_k.transpose(1, 2) * 128**-0.5
    valid = (ids[:, 0] >= 0) & (slots[:, 0] >= 0)
    expected = (scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1) @ selected_v)[:, :, None]

    actual = slot_indexed_attention(q, k, v, ids, slots, (partial, lse, out), scale=128**-0.5)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual.float(), expected, atol=0.002, rtol=0.02)
    if layout == "feature_major":
        assert torch.all(arena[[0, -1]] == 123)
        torch.testing.assert_close(arena[1:-1].T.float(), expected.reshape(batch, -1), atol=0.002, rtol=0.02)
