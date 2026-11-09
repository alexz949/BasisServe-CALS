import pytest
import torch

from basisserve.core.exact_page_routing import exact_grouped_scores, exact_page4_support
from evaluation.llama_sink_recent_routing_p16 import page_support


@pytest.mark.parametrize('length',[1,63,64,65,2111,2112,2113,2121,4097])
def test_exact_gqa_scores_and_production_page4_selection(length):
    generator = torch.Generator().manual_seed(71)
    query = torch.randn(1,8,1,16,generator=generator).to(torch.bfloat16)
    key = torch.randn(1,2,length,16,generator=generator).to(torch.bfloat16)
    reference = torch.einsum('bhqd,bhtd->bhqt',query.float(),key.float().repeat_interleave(4,dim=1))*.25
    actual = exact_grouped_scores(query,key,scale=.25,chunk_size=137)
    torch.testing.assert_close(actual,reference.reshape(1,2,4,length),rtol=0,atol=2e-6)
    expected_ids,expected_valid = page_support(reference.reshape(1,2,4,length),budget=2112,page_size=4,pinned=0)
    ids,valid = exact_page4_support(query,key,scale=.25,chunk_size=137)
    assert torch.equal(ids,expected_ids) and torch.equal(valid,expected_valid)
    assert int(valid.sum(-1).max()) <= min(length,2112)
    for group in range(2):
        support = ids[0,group][valid[0,group]].tolist()
        assert len(support) == len(set(support))
        assert all(0 <= t < length for t in support)
        assert set(range(max(0,length-64),length)) <= set(support)


def test_global_normalization_not_chunk_local_and_no_pinned_sink():
    # Head 0 favors a page late in the stream; head 1 favors another page.
    scores = torch.full((1,1,2,2200),-100.)
    scores[:,:,:,100:2140] = torch.linspace(-1,1,2040)
    scores[:,:,0,2100:2104] = 15
    scores[:,:,1,2120:2124] = 15
    ids,valid = page_support(scores,budget=2112,page_size=4,pinned=0)
    support = set(ids[0,0][valid[0,0]].tolist())
    assert 0 not in support
    assert set(range(2100,2104)) | set(range(2120,2124)) <= support
    assert set(range(2136,2200)) <= support


@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU validation is also repeated before real-model smoke')
@pytest.mark.parametrize('length',[2201,131008,131009])
def test_gpu_lossless_cpu_chunked_scores_and_selected_attention(length):
    from basisserve.kernels.split_indexed_attention import split_indexed_attention
    from evaluation.llama_b16r16_k_offload_p1 import _selected_attention
    from types import SimpleNamespace
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(19)
    q = torch.randn(1,8,1,128,device='cuda',dtype=torch.bfloat16)
    k = torch.randn(1,2,length,128,device='cuda',dtype=torch.bfloat16)
    v = torch.randn(1,2,length,96,device='cuda',dtype=torch.bfloat16)
    resident = exact_grouped_scores(q,k,scale=128**-.5,chunk_size=length)
    offloaded = exact_grouped_scores(q,k.cpu().pin_memory(),scale=128**-.5,chunk_size=137 if length<3000 else 16381)
    torch.testing.assert_close(resident,offloaded,rtol=1e-6,atol=2e-6)
    ids,valid = page_support(resident,budget=2112,page_size=4,pinned=0)
    other,other_valid = page_support(offloaded,budget=2112,page_size=4,pinned=0)
    assert torch.equal(ids,other) and torch.equal(valid,other_valid)
    expanded = ids.masked_fill(~valid,-1).repeat_interleave(4,dim=1)
    output = split_indexed_attention(q,k,v,expanded,scale=128**-.5)
    keys = k.repeat_interleave(4,dim=1).gather(2,expanded.clamp_min(0)[...,None].expand(-1,-1,-1,128))
    values = v.repeat_interleave(4,dim=1).gather(2,expanded.clamp_min(0)[...,None].expand(-1,-1,-1,96))
    scores = (q.float() @ keys.float().transpose(-1,-2))*128**-.5
    scores.masked_fill_(expanded[:, :, None] < 0,-torch.inf)
    expected = scores.softmax(-1) @ values.float()
    torch.testing.assert_close(output.float(),expected,rtol=0.02,atol=0.002)
    fetched = _selected_attention(q,k.cpu().pin_memory(),v,ids,valid,128**-.5,
        SimpleNamespace(key_staging={}),0)
    torch.testing.assert_close(output,fetched,rtol=0,atol=0)
    print(dict(length=length,score_chunk_max_abs=float((resident-offloaded).abs().max()),
        attention_max_abs=float((output.float()-expected).abs().max()),
        score_tolerance=dict(rtol=1e-6,atol=2e-6),attention_tolerance=dict(rtol=.02,atol=.002),
        lossless_selected_attention_bitwise_equal=True))


def test_paired_task_balanced_bootstrap():
    from evaluation.eval_llama_exact_sparse_page4 import bootstrap
    assert bootstrap([[1]*100,[3]*100],seed=7,replicates=1000)==[2.,2.]
