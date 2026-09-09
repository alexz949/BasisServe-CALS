import torch
from basisserve.core.c1_conditional_recent_attention import append_recent_tokens,c1_conditional_page_recent64_attention
from basisserve.core.c1_conditional_page_attention import c1_conditional_page_topk_attention

def test_union():
    for length in (1,31,64,65,2048,2049,2100):
        n=min(length,2048)
        ids=torch.arange(n).reshape(1,1,1,n)
        valid=torch.ones_like(ids,dtype=torch.bool)
        support=torch.ones(1,1,1,length,dtype=torch.bool)
        out,mask=append_recent_tokens(ids,valid,support,torch.tensor([length-1]))
        values=out[mask].tolist()
        assert len(values)==len(set(values))
        assert set(values)==set(range(n))|set(range(max(0,length-64),length))
        assert len(values)<=2112 and torch.equal(out[...,:n],ids)

def test_attention_full_support():
    torch.manual_seed(3)
    q=torch.randn(1,4,1,8);k=torch.randn(1,1,67,8);v=torch.randn(1,1,67,6)
    side=torch.randn(1,1,67,4);proj=torch.randn(4,8,4)
    args=dict(page_size=32,exact_token_budget=96,pinned_prefix_pages=1,scale=8**-.5,query_block_size=1)
    a=c1_conditional_page_topk_attention(q,k,v,side,proj,**args)
    b=c1_conditional_page_recent64_attention(q,k,v,side,proj,**args)
    torch.testing.assert_close(a.output,b.output,atol=1e-6,rtol=1e-5)
    assert b.statistics['selected_tokens']==67

def test_sparse_recent_attention():
    torch.manual_seed(5)
    q=torch.randn(1,4,1,8);k=torch.randn(1,1,97,8);v=torch.randn(1,1,97,6)
    side=torch.randn(1,1,97,4);proj=torch.randn(4,8,4)
    b=c1_conditional_page_recent64_attention(q,k,v,side,proj,page_size=32,
        exact_token_budget=32,pinned_prefix_pages=1,scale=8**-.5,query_block_size=1)
    # Original pinned page0 plus recent positions33..96, not whole recent pages.
    scores=(q@k.repeat_interleave(4,1).mT)*8**-.5
    scores[...,32]=-torch.inf
    expected=scores.softmax(-1)@v.repeat_interleave(4,1)
    torch.testing.assert_close(b.output,expected,atol=1e-6,rtol=1e-5)
    assert b.statistics['selected_tokens']==96
