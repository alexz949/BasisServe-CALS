import torch
from basisserve.core.c1_prompt_core import core_statistics,fit_cores,quadratic_loss


def test_normal_equations_and_regularized_solve():
    torch.manual_seed(7)
    q=torch.randn(5,4,6,dtype=torch.float64)
    u=torch.randn(4,6,2,dtype=torch.float64); e=torch.randn(6,2,dtype=torch.float64)
    x=torch.randn(5,4,6,6,dtype=torch.float64); grams=x@x.mT
    parts=[core_statistics(q[i],u,e,grams[i]) for i in range(5)]
    h,b,c=[sum(p[j] for p in parts) for j in range(3)]
    cores,lam=fit_cores(h,b)
    losses={}
    for name,s in cores.items():
        error=torch.einsum('nhd,hdr,hrs,ks->nhk',q,u,s,e)-q
        direct=torch.einsum('nhd,nhdk,nhk->h',error,grams,error)
        torch.testing.assert_close(quadratic_loss(h,b,c,s),direct)
        penalty=lam*(s-torch.eye(2)).square().sum((-1,-2))
        losses[name]=direct+penalty
    assert (losses['full']<=losses['diagonal']+1e-8).all()
    assert (losses['diagonal']<=losses['identity']+1e-8).all()


def test_zero_information_returns_identity():
    cores,_=fit_cores(torch.zeros(4,64,64),torch.zeros(4,64))
    for s in cores.values():
        torch.testing.assert_close(s,torch.eye(8,dtype=torch.float64).expand(4,8,8))
