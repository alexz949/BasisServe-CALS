import ast
import copy
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from basisserve.core.c1_lrqk import (LRQKConfig,LRQKState,prefill_factors,decode_factors,
                                    select_tokens,gather_heads,selected_attention)
from basisserve.checkpoint.c1_lrqk_qwen3 import C1LRQKCache,install_c1_lrqk
from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention


def upstream_equations():
    path = Path(__file__).resolve().parents[1]/'external/LRQK/lrqk_attention.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '2bfbc6df2bf97316ce5b086249ce8af48481b0a78f89f6650e19463fe783fab5'
    names = {'_lrqk_prefill_inv_w1','_lrqk_decode_inv_w1','_lrqk_decode_gd_B_lr'}
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = []
    namespace = dict(torch=torch,torchF=F)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def test_factor_equations_match_pinned_upstream():
    torch.manual_seed(19)
    q,k = torch.randn(1,4,37,8),torch.randn(1,4,37,8)
    aq,ak = torch.randn(1,4,37,4),torch.randn(1,4,37,4)
    upstream = upstream_equations()
    ref = upstream['_lrqk_prefill_inv_w1'](q,k,aq,ak,2,0.)
    actual = prefill_factors(q,k,aq,ak,2,0.)
    for a,b in zip(actual,ref,strict=True):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    _,bq,ak,bk = ref
    qt,kt = torch.randn(1,4,1,8),torch.randn(1,4,1,8)
    ref = upstream['_lrqk_decode_inv_w1'](bq,ak,bk,k,qt,kt,2,0.)
    actual = decode_factors(bq,ak,bk,k,qt,kt,2,0.)
    for a,b in zip(actual,ref,strict=True):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_topk_recent_and_budget():
    cfg = LRQKConfig(rank=1,topk=2,recent=2)
    scores = torch.tensor([8.,2.,9.,1.,0.,-1.]).view(1,1,6,1)
    ids = select_tokens(torch.ones(1,1,1,1),scores,cfg)
    assert ids.tolist() == [[[0,2,4,5]]]
    assert select_tokens(torch.ones(1,1,1,1),scores[:,:,:3],cfg).tolist() == [[[0,1,2]]]


def test_selected_attention_matches_explicit_logits_with_gqa_v96():
    torch.manual_seed(3)
    q,k,v = torch.randn(1,4,1,8),torch.randn(1,2,11,8),torch.randn(1,2,11,96)
    ids = torch.tensor([[[0,2,10],[1,3,10],[4,5,10],[6,7,10]]])
    actual = selected_attention(q,k,v,ids,8**-.5)
    kk = k.repeat_interleave(2,dim=1)
    vv = v.repeat_interleave(2,dim=1)
    mask = torch.full((1,4,1,11),-torch.inf)
    mask.scatter_(-1,ids[:,:,None],0.)
    probability = (q @ kk.transpose(-1,-2)*8**-.5+mask).softmax(-1)
    torch.testing.assert_close(actual,probability @ vv,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(gather_heads(k,ids),kk.gather(2,ids[...,None].expand(-1,-1,-1,8)))


@torch.inference_mode()
def test_incremental_state_growth_recent_eviction_and_fork():
    torch.manual_seed(4)
    q,k,v = torch.randn(1,4,13,8),torch.randn(1,2,13,8),torch.randn(1,2,13,6)
    state = LRQKState(q,k,LRQKConfig(rank=4,topk=3,recent=2))
    frozen = copy.deepcopy(state)
    for _ in range(4):
        old = state.ak.clone()
        k = torch.cat((k,torch.randn(1,2,1,8)),dim=2)
        v = torch.cat((v,torch.randn(1,2,1,6)),dim=2)
        output = state.decode(torch.randn(1,4,1,8),k,v,8**-.5)
        assert output.shape == (1,4,1,6) and state.length == k.shape[2]
        assert torch.equal(state.ak[:,:,:-1],old)
        assert torch.equal(state.selected[:,:,-2:],torch.arange(k.shape[2]-2,k.shape[2]).expand(1,4,-1))
        assert all(row.unique().numel() == 5 for row in state.selected[0])
    assert frozen.length == 13 and state.length == 17
    stats = state.statistics(2)
    assert stats['selected_per_query_head'] == 5
    assert all(5 <= n <= 10 for n in stats['physical_union_per_kv_group'][0])


def tiny_c1():
    config = Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,
        attention_dropout=0.,_attn_implementation='sdpa')
    model = Qwen3ForCausalLM(config).eval()
    for layer in model.model.layers:
        a = layer.self_attn
        layer.self_attn = GQATiedVOQwen3Attention(a,
            v_proj_compressed_weight=torch.randn(12,32)*.02,
            o_decoder_weight=torch.randn(32,24)*.02,
            value_coordinate_encoder=torch.randn(2,8,6)*.02,attention_backend='sdpa')
    return model.eval()


@torch.inference_mode()
def test_qwen3_cache_owned_state_and_full_budget_equivalence():
    torch.manual_seed(13)
    baseline = tiny_c1()
    candidate = copy.deepcopy(baseline)
    install_c1_lrqk(candidate,LRQKConfig(rank=4,topk=100,recent=2,prefill_backend='sdpa'))
    for _ in range(2):
        tokens = torch.randint(0,64,(1,12))
        ordinary = DynamicCache(config=baseline.config)
        routed = C1LRQKCache(config=candidate.config)
        for step in range(4):
            mask = None if step == 0 else {'full_attention':torch.ones(1,1,1,12+step,dtype=torch.bool)}
            expected = baseline(input_ids=tokens,past_key_values=ordinary,use_cache=True,attention_mask=mask).logits
            actual = candidate(input_ids=tokens,past_key_values=routed,use_cache=True,attention_mask=mask).logits
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
            assert len(routed.lrqk_states) == 2
            assert all(s.length == 12+step for s in routed.lrqk_states.values())
            tokens = torch.randint(0,64,(1,1))
