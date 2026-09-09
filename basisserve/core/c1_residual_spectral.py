"""Closed-form shared-query-metric rank-r residual projection."""
import torch


def spectral_factors(residual_covariance,query_covariance,rank=8,relative_ridge=1e-5):
    """Batched dxd input moments. No QK matrix, BCD, PCG or autograd optimizer."""
    c=residual_covariance; h=query_covariance
    assert c.shape==h.shape and c.shape[-1]==c.shape[-2] and 0<rank<=c.shape[-1]
    assert relative_ridge>0 and torch.isfinite(c).all() and torch.isfinite(h).all()
    c=(c+c.mT)*.5; h=(h+h.mT)*.5
    eigenvalues,vectors=torch.linalg.eigh(h)
    ridge=relative_ridge*h.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(torch.finfo(h.dtype).tiny)
    positive=eigenvalues.clamp_min(0)+ridge[...,None]
    root=(vectors*positive.sqrt()[...,None,:])@vectors.mT
    inverse=(vectors*positive.rsqrt()[...,None,:])@vectors.mT
    weighted=root@c@root
    spectrum,basis=torch.linalg.eigh((weighted+weighted.mT)*.5)
    selected=basis[...,-rank:]
    e=root@selected; u=inverse@selected
    assert torch.isfinite(e).all() and torch.isfinite(u).all()
    return e,u,dict(ridge=ridge,spectrum=spectrum,query_metric=root@root)
