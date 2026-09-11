"""Analytic checks for padded latent V and native-gate placement."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from basisserve.core.qwen35_vllm_hybrid import pad_latent_v_writer, ReconstructBeforeGate, VLLMPrivateAGOutput


@pytest.mark.parametrize('rank', [2, 4])
def test_padded_writer_bias_and_group_specific_attention(rank):
    torch.manual_seed(81)
    groups, heads, width, hidden, length = 2, 4, 4, 7, 5
    x = torch.randn(length, hidden, dtype=torch.float64)
    w = torch.randn(groups * width, hidden, dtype=torch.float64)
    b = torch.randn(groups * width, dtype=torch.float64)
    e = torch.randn(groups, width, rank, dtype=torch.float64)
    r = torch.randn(groups, rank, width, dtype=torch.float64)
    pw, pb = pad_latent_v_writer(w, b, e)
    padded = F.linear(x, pw, pb).reshape(length, groups, width)
    latent = torch.einsum('ngd,gdr->ngr', F.linear(x, w, b).reshape(length, groups, width), e)
    torch.testing.assert_close(padded[..., :rank], latent)
    assert torch.count_nonzero(padded[..., rank:]) == 0
    mapping = torch.arange(heads) // (heads // groups)
    probs = torch.randn(heads, length, length, dtype=torch.float64).softmax(-1)

    class Attention(nn.Module):
        def forward(self, q, k, v):
            return torch.einsum('hnm,mhd->nhd', probs, v[:, mapping]).flatten(1)

    runtime = ReconstructBeforeGate(Attention(), r, heads)
    restored = runtime(None, None, padded).reshape(length, heads, width)
    expected = torch.einsum('hnm,mhr,hrd->nhd', probs, latent[:, mapping], r[mapping])
    torch.testing.assert_close(restored, expected)
    gate = torch.randn_like(expected).sigmoid()
    output_weight = torch.randn(heads * width, hidden, dtype=torch.float64)
    actual = (restored * gate).flatten(1) @ output_weight
    torch.testing.assert_close(actual, (expected * gate).flatten(1) @ output_weight)
    assert not torch.allclose(actual, restored.flatten(1) @ output_weight)


def test_private_ag_tuple_contract_and_bias_once():
    torch.manual_seed(7)
    x = torch.randn(9, 12, dtype=torch.float64)
    e = torch.randn(4, 3, 2, dtype=torch.float64)
    d = torch.randn(7, 8, dtype=torch.float64)
    bias = torch.randn(7, dtype=torch.float64)
    y, returned_bias = VLLMPrivateAGOutput(e, d, bias)(x)
    expected = torch.cat([x[:, i*3:(i+1)*3] @ e[i] for i in range(4)], -1) @ d.T + bias
    torch.testing.assert_close(y, expected)
    assert returned_bias is None


def test_reject_non_group_reconstruction():
    with pytest.raises(AssertionError):
        ReconstructBeforeGate(nn.Identity(), torch.zeros(3, 2, 4), 4)
