"""Compare the upstream accuracy-cache equations with our resident adapter."""
import ast
import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from flash_attn import flash_attn_with_kvcache
from basisserve.core.c1_shadowkv import C1ShadowKVState, gather_group
from transformers.models.llama.modeling_llama import rotate_half


@torch.inference_mode()
def main():
    source=Path('external/ShadowKV/models/kv_cache.py')
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ShadowKVCache')
    namespace=dict(torch=torch,math=math,gc=gc,nn=nn)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),namespace)
    cfg=SimpleNamespace(num_attention_heads=32,num_key_value_heads=8,hidden_size=4096,num_hidden_layers=1)
    torch.manual_seed(42)
    length=4099
    pre=torch.randn(1,8,length,128,device='cuda',dtype=torch.bfloat16)
    value=torch.randn_like(pre)
    positions=torch.arange(length+8,device='cuda').float()
    frequency=1/(500000**(torch.arange(0,128,2,device='cuda').float()/128))
    angles=(positions[:,None]*frequency[None]).repeat(1,2)
    cos=angles.cos().to(pre.dtype);sin=angles.sin().to(pre.dtype)
    post=pre*cos[None,None,:length]+rotate_half(pre)*sin[None,None,:length]
    ours=C1ShadowKVState(pre,post,cos[None,:length],sin[None,:length])
    official=namespace['ShadowKVCache'](cfg,max_length=length+8)
    official.get_svd(pre,0)
    official.prefill_kv_cache(value,0,post)
    assert torch.equal(official.k_landmark[0],ours.landmarks)
    assert torch.equal(official.k_landmark_idx[0],ours.landmark_ids)
    assert torch.equal(official.k_cache_buffer[0,:,:,:official.sparse_start],ours.fixed_key)
    def rope(x,ids):
        return x*cos[ids]+rotate_half(x)*sin[ids]
    rows=[]
    for step in range(8):
        query=torch.randn(1,32,1,128,device='cuda',dtype=pre.dtype)
        key=torch.randn(1,8,1,128,device='cuda',dtype=pre.dtype)
        new_v=torch.randn_like(key);value=torch.cat([value,new_v],2)
        official.update_kv_cache(key,new_v,0)
        ids=official.get_retrieval_position_ids(0,query)
        assert torch.equal(ids,ours.select(query))
        keys=official.get_key_cache(0,ids,rope,None)
        values=official.get_value_cache(0,ids)
        actual=ours.decode(query,key,value,128**-0.5)
        expected_keys=torch.cat([ours.fixed_key,ours.reconstruct(ids),ours.generated_key],2)
        torch.testing.assert_close(keys,expected_keys,atol=0,rtol=0)
        assert torch.equal(values,gather_group(value,ours.selected_ids))
        expected=flash_attn_with_kvcache(query.transpose(1,2),keys.transpose(1,2),values.transpose(1,2),causal=True).transpose(1,2)
        error=float((actual.float()-expected.float()).abs().max())
        torch.testing.assert_close(actual,expected,atol=0.004,rtol=0.02)
        rows.append(dict(step=step,selected_ids_equal=True,key_equal=True,value_equal=True,attention_max_abs_error=error))
    print(json.dumps(dict(status='passed',length=length,rank=160,budget=2048,steps=rows,
        limitation='Synthetic same-input accuracy-cache comparison; common PyTorch RoPE, not upstream CUDA RoPE or full model'),indent=2),flush=True)


if __name__=='__main__':main()
