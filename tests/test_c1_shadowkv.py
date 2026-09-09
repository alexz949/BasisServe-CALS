import ast
import math
from pathlib import Path
import torch
from torch import nn
from basisserve.core.c1_shadowkv import C1ShadowKVState,gather_group
from transformers.models.qwen3.modeling_qwen3 import rotate_half


def fixture():
    torch.manual_seed(11)
    pre=torch.randn(1,2,96,128)
    theta=torch.randn(1,96,64).repeat(1,1,2)
    cos,sin=theta.cos(),theta.sin()
    post=pre*cos[:,None]+rotate_half(pre)*sin[:,None]
    return pre,post,cos,sin


def test_official_accuracy_equations():
    source=Path(__file__).resolve().parents[1]/'external/ShadowKV/models/kv_cache.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ShadowKVCache')
    scope={'torch':torch,'nn':nn,'math':math}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),scope)
    ref=scope['ShadowKVCache'].__new__(scope['ShadowKVCache'])
    ref.batch_size=1;ref.num_key_value_heads=2;ref.num_key_value_groups=2;ref.head_dim=128
    ref.rank=8;ref.num_layers=1;ref.dtype=torch.float32;ref.device='cpu'
    ref.chunk_size=4;ref.local_chunk=4;ref.outlier_chunk=2;ref.sparse_budget=16
    ref.kv_offset=0;ref.gen_offset=0
    ref.v_cache_cpu=torch.zeros(1,1,2,128,128)
    ref.k_cache_buffer=torch.zeros(1,1,2,128,128)
    ref.v_cache_buffer=torch.zeros(1,1,2,128,128)
    ref.selected_chunk_idx=torch.zeros(1,1,2,4,dtype=torch.long)
    pre,post,cos,sin=fixture()
    ref.get_svd(pre,0);ref.prefill_kv_cache(post,0,post)
    state=C1ShadowKVState(pre,post,cos,sin,rank=8,budget=16,chunk=4,outliers=2)
    torch.testing.assert_close(state.u,ref.U[0],atol=0,rtol=0)
    torch.testing.assert_close(state.sv,ref.SV[0],atol=0,rtol=0)
    torch.testing.assert_close(state.landmarks,ref.k_landmark[0],atol=0,rtol=0)
    assert torch.equal(state.landmark_ids,ref.k_landmark_idx[0])
    q=torch.randn(1,4,1,128)
    ids=ref.get_retrieval_position_ids(0,q)
    assert torch.equal(ids,state.select(q))
    def rope(k,p):
        c=cos[:,None].expand(1,2,-1,-1).gather(2,p[...,None].expand(*p.shape,128))
        s=sin[:,None].expand(1,2,-1,-1).gather(2,p[...,None].expand(*p.shape,128))
        return k*c+rotate_half(k)*s
    rebuilt=ref.get_key_cache(0,ids,rope,None)[:,:,ref.sparse_start:ref.sparse_end]
    torch.testing.assert_close(rebuilt,state.reconstruct(ids),atol=0,rtol=0)


def test_resident_v96_decode():
    pre,post,cos,sin=fixture()
    state=C1ShadowKVState(pre,post,cos,sin,rank=8,budget=16,chunk=4,outliers=2)
    v=torch.randn(1,2,98,96)
    for step in range(2):
        q=torch.randn(1,4,1,128);k=torch.randn(1,2,1,128)
        output=state.decode(q,k,v[:,:,:97+step],128**-.5)
        assert output.shape==(1,4,1,96) and torch.isfinite(output).all()
        ids=state.selected_ids
        assert (ids.sort(-1).values.diff(dim=-1)>0).all()
        assert ids.shape[-1]==16+8+16+step+1
        assert (ids[:,:,-1]==96+step).all()
        selected_key=torch.cat((state.fixed_key,state.reconstruct(state.select(q)),state.generated_key),2)
        probs=(q@selected_key.repeat_interleave(2,1).mT*128**-.5).softmax(-1)
        expected=probs@gather_group(v[:,:,:97+step],ids).repeat_interleave(2,1)
        torch.testing.assert_close(output,expected,atol=1e-6,rtol=1e-5)


def test_zero_decode_statistics():
    pre,post,cos,sin=fixture()
    state=C1ShadowKVState(pre,post,cos,sin,rank=8,budget=16,chunk=4,outliers=2)
    stats=state.statistics()
    assert stats['length']==96 and stats['decode_steps']==0
    assert stats['physical_tokens_per_group']==stats['routed_tokens']==0
    assert state.selected_ids is None
