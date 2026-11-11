"""TP repartition must preserve source-head ownership and latent decoder order."""

import pytest
import torch

from basisserve.core.tp4_c1_factors import fold_value_weight


@pytest.mark.parametrize('width',[64,96])
@pytest.mark.parametrize('hidden_size',[4096,5120])
def test_tp4_fold_preserves_both_physical_heads(width, hidden_size):
    generator=torch.Generator().manual_seed(17)
    values=torch.randn(8,128,hidden_size,generator=generator)/hidden_size**0.5
    encoders=torch.randn(8,128,width,generator=generator)/128**0.5
    hidden=torch.randn(3,hidden_size,generator=generator)
    expected=torch.einsum('th,skh,skr->tsr',hidden,values,encoders)
    pieces=[]
    for rank in range(4):
        weight=fold_value_weight(values[rank*2:rank*2+2].flatten(0,1),encoders[rank*2:rank*2+2])
        pieces.append((hidden @ weight.T).reshape(3,2,width))
    torch.testing.assert_close(torch.cat(pieces,dim=1),expected,rtol=2e-4,atol=2e-5)


@pytest.mark.parametrize('width',[64,96])
def test_tp4_gather_preserves_tp8_output_function(width):
    generator=torch.Generator().manual_seed(91)
    latent=torch.randn(3,8,8,width,generator=generator)
    decoder=torch.randn(64,width,17,generator=generator)
    tp4=torch.cat([latent[:,2*r:2*r+2].flatten(1).T.contiguous() for r in range(4)])
    tp8=torch.cat([latent[:,r].flatten(1).T.contiguous() for r in range(8)])
    assert torch.equal(tp4,tp8)
    expected=torch.einsum('tkgr,kgrh->th',latent,decoder.reshape(8,8,width,17))
    torch.testing.assert_close(tp4.T @ decoder.flatten(0,1),expected,rtol=2e-4,atol=1e-4)
