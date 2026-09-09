import torch
from basisserve.core.c1_residual_spectral import spectral_factors


def test_weighted_optimum_and_batch():
    torch.manual_seed(3)
    residual=torch.randn(3,27,7,dtype=torch.float64)
    q=torch.randn(3,19,7,dtype=torch.float64)
    c=residual.mT@residual/27; h=q.mT@q/19
    e,u,info=spectral_factors(c,h,rank=3)
    p=e@u.mT; delta=torch.eye(7)-p
    loss=torch.einsum('bij,bjk,bkl,bli->b',delta.mT,c,delta,info['query_metric'])
    torch.testing.assert_close(loss,info['spectrum'][...,:-3].sum(-1))
    torch.testing.assert_close(p@p,p)
    for b in range(3):
        ee,uu,_=spectral_factors(c[b],h[b],rank=3)
        torch.testing.assert_close(ee@uu.mT,p[b])


def test_rank_deficient_query_metric():
    q=torch.zeros(7,7,dtype=torch.float64); q[0,0]=1
    e,u,_=spectral_factors(torch.eye(7,dtype=torch.float64),q,rank=3)
    assert torch.isfinite(e).all() and torch.isfinite(u).all()
