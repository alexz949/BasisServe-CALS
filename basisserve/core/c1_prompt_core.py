"""Fixed-subspace prompt core: exact damped quadratic solves, no optimizer."""
import torch


def core_statistics(q,u,e,gram):
    """Per-head row-major vec(S) normal equations; q already includes 1/sqrt(d)."""
    a=torch.einsum('hd,hdr->hr',q,u)
    metric=torch.einsum('di,hdk,kj->hij',e,gram,e)
    target=torch.einsum('hd,hdk,kr->hr',q,gram,e)
    h=torch.einsum('hi,hk,hjl->hijkl',a,a,metric).flatten(1,2).flatten(2,3)
    b=torch.einsum('hi,hj->hij',a,target).flatten(1)
    c=torch.einsum('hd,hdk,hk->h',q,gram,q)
    return h,b,c


def fit_cores(h,b,*,relative_ridge=1e-3):
    """Same Frobenius penalty toward I for diagonal/full; batched FP64 solves."""
    h=h.double(); b=b.double()
    n=h.shape[-1]; r=int(n**.5)
    assert r*r==n and relative_ridge>0
    h=(h+h.mT)*.5
    identity=torch.eye(r,device=h.device,dtype=h.dtype).flatten().expand(h.shape[0],-1)
    rhs=b-(h@identity[...,None]).squeeze(-1)
    lam=relative_ridge*h.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-12)
    cores={'identity':identity.reshape(-1,r,r)}
    for name,idx in [('diagonal',torch.arange(r,device=h.device)*(r+1)),
                     ('full',torch.arange(n,device=h.device))]:
        matrix=h[:,idx][:,:,idx]+lam[:,None,None]*torch.eye(len(idx),device=h.device,dtype=h.dtype)
        delta,info=torch.linalg.solve_ex(matrix,rhs[:,idx,None])
        assert (info==0).all() and torch.isfinite(delta).all()
        value=identity.clone(); value[:,idx]+=delta.squeeze(-1)
        cores[name]=value.reshape(-1,r,r)
    return cores,lam


def quadratic_loss(h,b,c,s):
    v=s.flatten(1).to(h.dtype)
    return torch.einsum('hi,hij,hj->h',v,h,v)-2*(b*v).sum(-1)+c
