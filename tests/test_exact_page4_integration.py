from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.mark.parametrize('length',[2201,4097])
def test_independent_page_mass_reference(length):
    from evaluation.llama_sink_recent_routing_p16 import page_support

    generator=torch.Generator().manual_seed(94)
    scores=torch.randn(1,2,4,length,generator=generator)
    historical=length-64
    page_log_mass=torch.stack([
        torch.logsumexp(scores[...,start:min(start+4,historical)],dim=-1)
        for start in range(0,historical,4)],dim=-1)
    normalized=torch.exp(page_log_mass-torch.logsumexp(page_log_mass,dim=-1,keepdim=True))
    group_mass=normalized.max(dim=2).values
    pages=group_mass.argsort(dim=-1,descending=True)[...,:512].sort(dim=-1).values
    ids,valid=page_support(scores,budget=2112,page_size=4,pinned=0)
    # Per-head additive offsets must disappear before the GQA maximum.
    shifted=scores+torch.tensor([0.,10.,-20.,30.]).view(1,1,4,1)
    shifted_ids,shifted_valid=page_support(shifted,budget=2112,page_size=4,pinned=0)
    assert torch.equal(ids,shifted_ids) and torch.equal(valid,shifted_valid)
    for group in range(2):
        reference=sorted(t for p in pages[0,group].tolist() for t in range(p*4,min(p*4+4,historical)))
        reference+=list(range(historical,length))
        assert ids[0,group][valid[0,group]].tolist()==reference
        assert len(reference)==len(set(reference))<=2112


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA integration')
@torch.inference_mode()
def test_llama_prefill_decode_proxy_and_exact_storage(tmp_path, monkeypatch):
    from evaluation import eval_k_routing_ruler as runtime
    from evaluation import eval_llama_cal128_p1 as matched
    from evaluation import llama_b16r16_k_offload_p1 as offload
    from evaluation.eval_llama_exact_sparse_page4 import set_arm
    from evaluation.deterministic_evaluation import configure_deterministic_evaluation

    configure_deterministic_evaluation()
    torch.manual_seed(61)
    monkeypatch.setattr(runtime, 'RoutingCache', runtime.RoutingCache)
    monkeypatch.setattr(offload, 'PAGE_SIZE', 4)
    monkeypatch.setattr(offload, 'PINNED_PAGES', 0)
    monkeypatch.setattr(offload, 'BUDGET', 2112)
    config=LlamaConfig(vocab_size=64, hidden_size=512, intermediate_size=1024,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=128, max_position_embeddings=4096, eos_token_id=[128001,128008,128009])
    config._attn_implementation='sdpa'
    model=LlamaForCausalLM(config).to(device='cuda',dtype=torch.bfloat16).eval()
    factors=dict(value_coordinate_encoders=torch.randn(2,128,96)*.01,
        head_output_decoders=torch.randn(4,96,512)*.01,
        source_ranks=torch.tensor([96,96]))
    save_file(factors,str(tmp_path/'layer.safetensors'))
    bank={0:dict(base_left_b16=torch.randn(2,96,16)*.01,
        base_right_b16=torch.randn(2,16,128)*.01,base_bias_b16=torch.zeros(2,128),
        residual_encoder_b16_r16=torch.randn(2,128,16)*.01,
        residual_query_b16_r16=torch.randn(4,128,16)*.01)}
    runtime.install(model,tmp_path,dict(layers=[dict(layer=0,file='layer.safetensors',ranks=[96,96])]),'ours',bank)
    offload.install(model)
    counts=matched.instrument_kernels()
    tokenizer=SimpleNamespace(eos_token_id=128009)
    row=dict(input_ids=[i%64 for i in range(2201)])
    results=[]
    for arm,placement in [('proxy_page4','cpu'),('exact_sparse_page4','cuda'),('exact_sparse_page4','cpu')]:
        set_arm(model,arm,dict(exact_score=dict(key_placement=placement,chunk_size=137)))
        counts.clear()
        result=runtime.generate(model,tokenizer,row,'ours',3)
        assert counts==dict(prefill=1,split_decode=2)
        assert len(result[0])==3 and not result[3] and torch.isfinite(result[1]).all()
        assert result[2][0]['token_budget']==2112
        if arm=='exact_sparse_page4':
            assert result[2][0]['decode_steps']==2
            assert result[2][0]['key_tokens_scanned']==2202+2203
        results.append(result)
    assert torch.equal(results[0][1],results[1][1]) and torch.equal(results[1][1],results[2][1])
    assert results[1][0]==results[2][0]
