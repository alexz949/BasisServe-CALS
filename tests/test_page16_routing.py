import torch
from basisserve.core.c1_conditional_page_attention import _selected_pages
from basisserve.core.c1_v_conditional_k_router import residual_page_fisher_gram


def test_page16_budget_and_prefix():
    torch.manual_seed(41)
    for length in (1192,2048,2049,4097):
        scores=torch.randn(1,8,1,length)
        valid=torch.ones_like(scores,dtype=torch.bool)
        ids,mask=_selected_pages(scores,valid,kv_heads=2,page_size=16,page_budget=128,pinned_prefix_pages=2)
        assert ids.shape==(1,2,1,min(128,(length+15)//16))
        assert mask.all() and (ids[...,0]==0).all() and (ids[...,1]==1).all()
        assert (ids.sort(-1).values.diff(dim=-1)>0).all()
        assert ((length-ids*16).clamp(min=0,max=16).sum(-1)<=2048).all()


def test_page16_fisher_reference_and_prefix():
    torch.manual_seed(42)
    q=torch.randn(4,8,dtype=torch.float64)
    k=torch.randn(113,8,dtype=torch.float64)
    r=torch.randn(113,8,dtype=torch.float64)
    g,energy=residual_page_fisher_gram(q,k,r,scaling=8**-.5,page_size=16,excluded_prefix_pages=2)
    prob=(q@k[32:].mT/8**.5).softmax(-1)
    mass=[];representatives=[]
    for start in range(0,81,16):
        w=prob[:,start:start+16]; m=w.sum(-1)
        mass.append(m);representatives.append(w@r[32+start:32+start+16]/m[:,None])
    mass=torch.stack(mass,1);v=torch.stack(representatives,1)
    center=v-(mass[...,None]*v).sum(1,keepdim=True)
    ref=torch.einsum('hp,hpd,hpe->hde',mass,center,center)
    torch.testing.assert_close(g,ref)
    expected=.5*torch.einsum('hd,hde,he->',q,g,q)/8
    torch.testing.assert_close(torch.tensor(energy,dtype=torch.float64),expected)
    k[:32]=1000;r[:32]=-1000
    other,_=residual_page_fisher_gram(q,k,r,scaling=8**-.5,page_size=16,excluded_prefix_pages=2)
    torch.testing.assert_close(g,other)
