import torch
from basisserve.core.c1_lrqk import LRQKConfig,LRQKState
from evaluation.eval_longbench_lrqk_fp32route import FP32RoutingState


def test_fp16_payload_fp32_routing():
    torch.manual_seed(0)
    q=torch.randn(1,4,8,8).half()
    k=torch.randn(1,1,8,8).half()
    v=torch.randn(1,1,9,6).half()
    cfg=LRQKConfig(rank=2,topk=4,recent=2,prefill_backend='sdpa')
    state=FP32RoutingState(q,k,cfg)
    output=state.decode(q[:,:,-1:],torch.cat((k,k[:,:,-1:]),2),v,8**-.5)
    assert state.ak.dtype==state.bq.dtype==state.bk.dtype==torch.float32
    assert output.dtype==torch.float16 and torch.isfinite(output).all()


def test_original_fp32_equation_parity():
    torch.manual_seed(0)
    q=torch.randn(1,4,8,8)
    k=torch.randn(1,1,8,8)
    v=torch.randn(1,1,9,6)
    cfg=LRQKConfig(rank=2,topk=4,recent=2,prefill_backend='sdpa')
    old=LRQKState(q,k,cfg)
    new=FP32RoutingState(q,k,cfg)
    full_k=torch.cat((k,k[:,:,-1:]),2)
    a=old.decode(q[:,:,-1:],full_k,v,8**-.5)
    b=new.decode(q[:,:,-1:],full_k,v,8**-.5)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for name in ('ak','bq','bk','selected'):
        torch.testing.assert_close(getattr(old,name),getattr(new,name),rtol=0,atol=0)
