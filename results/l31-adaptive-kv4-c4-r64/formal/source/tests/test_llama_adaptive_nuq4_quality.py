from types import SimpleNamespace

import torch
import pytest
from torch import nn

from basisserve.core.llama_adaptive_nuq4_quality import fold_layer


@torch.no_grad()
@pytest.mark.parametrize("ranks", [[32, 48, 64, 80, 96, 112, 128, 96],
                                   [32, 48, 64, 32, 48, 64, 64, 64]])
def test_adaptive_folding_and_active_coordinates(ranks):
    torch.manual_seed(5)
    a = SimpleNamespace(v_proj=nn.Linear(4096, 1024, bias=False, dtype=torch.bfloat16),
                        o_proj=nn.Linear(4096, 4096, bias=False, dtype=torch.bfloat16))
    dense = a.v_proj.weight.float().clone()
    enc = torch.randn(8, 128, max(ranks)).to(torch.bfloat16)
    dec = torch.randn(32, max(ranks), 4096).to(torch.bfloat16)
    indices = fold_layer(a, dict(value_coordinate_encoders=enc, head_output_decoders=dec,
                                 source_ranks=torch.tensor(ranks)), ranks)
    assert indices.tolist() == [h * 128 + j for h, r in enumerate(ranks) for j in range(r)]
    for h, rank in enumerate(ranks):
        expected = (enc[h, :, :rank].float().T @ dense[h * 128:(h + 1) * 128]).bfloat16()
        torch.testing.assert_close(a.v_proj.weight[h * 128:h * 128 + rank], expected, rtol=0, atol=0)
        assert torch.count_nonzero(a.v_proj.weight[h * 128 + rank:(h + 1) * 128]) == 0
        for q in range(h * 4, (h + 1) * 4):
            torch.testing.assert_close(a.o_proj.weight[:, q * 128:q * 128 + rank],
                                       dec[q, :rank].T, rtol=0, atol=0)
            assert torch.count_nonzero(a.o_proj.weight[:, q * 128 + rank:(q + 1) * 128]) == 0
