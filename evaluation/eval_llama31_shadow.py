"""Llama-3.1-8B Base: exact V with full K versus ShadowKV reconstructed K."""
import argparse
import json
from pathlib import Path
import shlex
import sys
import time
from types import MethodType
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.v96kl_common import configure,read_json,write_json,sha256
from evaluation.ruler_v1 import parse_tasks,ruler_prompt,sample_score
from evaluation.eval_qwen3_8b_residual_rank_ruler import TASKS,_eos_ids
from basisserve.core.c1_shadowkv import C1ShadowKVState
from basisserve.checkpoint.c1_shadowkv_qwen3 import C1ShadowKVCache
from basisserve.kernels.compressed_v_decode_attention import compressed_v_prefill_attention

MODEL=Path('/home/lz299/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b')
DATA=ROOT/'results/datasets/llama31_base_ruler_seed42_margin8'
OUTPUT=ROOT/'results/evaluation/llama31_base_shadow'


@torch.inference_mode()
def forward(self,hidden_states,position_embeddings,attention_mask,past_key_values=None,cache_position=None,**kwargs):
    batch,length,_=hidden_states.shape
    assert batch==1 and isinstance(past_key_values,C1ShadowKVCache)
    q=self.q_proj(hidden_states).view(batch,length,self.config.num_attention_heads,self.head_dim).transpose(1,2)
    pre=self.k_proj(hidden_states).view(batch,length,self.config.num_key_value_heads,self.head_dim).transpose(1,2)
    v=self.v_proj(hidden_states).view(batch,length,self.config.num_key_value_heads,self.head_dim).transpose(1,2)
    cos,sin=position_embeddings
    q,k=apply_rotary_pos_emb(q,pre,cos,sin)
    previous=past_key_values.get_seq_length(self.layer_idx)
    assert previous==0 or length==1
    if attention_mask is not None:
        expected=torch.arange(previous+length,device=q.device)[None,:]<=(previous+torch.arange(length,device=q.device)[:,None])
        valid=attention_mask if attention_mask.dtype==torch.bool else attention_mask==0
        assert torch.equal(valid.expand(batch,1,length,previous+length)[0,0],expected)
    current=k
    k,v=past_key_values.update(k,v,self.layer_idx,{'cos':cos,'sin':sin,'cache_position':cache_position})
    if previous==0:
        if self.shadow_enabled:
            past_key_values.shadow_states[self.layer_idx]=C1ShadowKVState(pre,k,cos,sin)
        output=compressed_v_prefill_attention(q,k,v,scale=self.scaling)
    elif self.shadow_enabled:
        output=past_key_values.shadow_states[self.layer_idx].decode(q,current,v,self.scaling)
    else:
        groups=q.shape[1]//k.shape[1]
        output=torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(groups,1),
                    v.repeat_interleave(groups,1),scale=self.scaling,dropout_p=0,is_causal=False)
    return self.o_proj(output.transpose(1,2).contiguous().reshape(batch,length,-1)),None


def inputs(tokenizer):
    data=read_json(DATA/'manifest.json');tasks=parse_tasks(TASKS)
    assert data['status']=='complete'
    assert data['protocol']==dict(model_template_type='base',sequence_length=32768,
                                 samples_per_task=8,random_seed=42,prompt_margin=8,tasks=[t.name for t in tasks])
    assert data['tokenizer_config_sha256']==sha256(MODEL/'tokenizer_config.json')
    rows=[]
    for t in tasks:
        path=DATA/t.name/'validation.jsonl'
        assert sha256(path)==data['artifacts'][t.name]['sha256']
        sources=[json.loads(line) for line in path.read_text().splitlines()]
        assert len(sources)==8
        for source in sources:
            ids=tokenizer(ruler_prompt(source),add_special_tokens=True)['input_ids']
            assert len(ids)+t.tokens_to_generate<=32768
            rows.append(dict(index=len(rows),task=t.name,input_ids=ids,answers=source['outputs'],
                             match_type=t.match_type,maximum_tokens=t.tokens_to_generate))
    spec=dict(model='meta-llama/Llama-3.1-8B',revision=MODEL.name,model_config_sha256=sha256(MODEL/'config.json'),
              data_sha256=sha256(DATA/'manifest.json'),ruler_revision=data['ruler']['revision'],
              dtype='bfloat16',value='original exact V128, original output projection',
              prefill='common full causal Triton',generation='greedy, official caps and model EOS',
              rank=160,chunk=8,routed=2048,outlier_chunks=48,local='last4 full prompt chunks plus remainder and generated tokens',
              seed=42,prompt_margin=8,primary_excluded_indices=[86],implementation='HF Llama adapter, official ShadowKV accuracy equations; resident cache',
              code_sha256={p:sha256(ROOT/p) for p in ['evaluation/eval_llama31_shadow.py','basisserve/core/c1_shadowkv.py',
                  'basisserve/kernels/compressed_v_decode_attention.py']})
    return rows,spec


@torch.inference_mode()
def generate(model,tokenizer,row,cap):
    cache=C1ShadowKVCache(model.config)
    out=model(input_ids=torch.tensor([row['input_ids']],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1)
    assert torch.isfinite(out.logits).all()
    first=out.logits[0,-1].float().cpu();ids=[int(first.argmax())];del out
    eos=_eos_ids(tokenizer,model)
    while len(ids)<cap and ids[-1] not in eos:
        mask=torch.ones(1,1,1,cache.get_seq_length()+1,dtype=torch.bool,device='cuda')
        out=model(input_ids=torch.tensor([[ids[-1]]],device='cuda'),past_key_values=cache,
                  attention_mask=mask,use_cache=True,logits_to_keep=1)
        assert torch.isfinite(out.logits).all()
        ids.append(int(out.logits[0,-1].argmax()));del out
    stats=[s.statistics() for s in cache.shadow_states.values()]
    return ids,first,stats


def summarize(rows,spec,tokenizer):
    records={}
    eos=read_json(MODEL/'config.json')['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos])|{tokenizer.eos_token_id}
    for arm in ['full','shadowkv']:
        records[arm]=[]
        for row in rows:
            d=read_json(OUTPUT/arm/'evaluate'/f"sample_{row['index']:03d}.json")
            assert d['status']=='complete' and d['protocol']==spec and d['sample']==row
            r=d['result'];ids=r['ids']
            assert 0<len(ids)<=row['maximum_tokens'] and not any(i in eos for i in ids[:-1])
            assert ids[-1] in eos or len(ids)==row['maximum_tokens']
            assert tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['prediction']
            assert sample_score(r['prediction'],row['answers'],row['match_type'])==r['score']
            records[arm].append(r)
    assert all(a['ids'][0]==b['ids'][0] for a,b in zip(records['full'],records['shadowkv'],strict=True))
    means87={a:100*sum(r['score'] for i,r in enumerate(rs) if i!=86)/87 for a,rs in records.items()}
    means88={a:100*sum(r['score'] for r in rs)/88 for a,rs in records.items()}
    tasks={t.name:{a:100*sum(rs[row['index']]['score'] for row in rows if row['task']==t.name)/8 for a,rs in records.items()} for t in parse_tasks(TASKS)}
    write_json(OUTPUT/'result.json',dict(status='complete',verified=176,protocol=spec,means87=means87,means88=means88,tasks=tasks))
    print('VERIFIED',means87,means88,tasks,flush=True)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['smoke','evaluate','summarize'])
    p.add_argument('--arm',choices=['full','shadowkv'],default='full')
    args=p.parse_args();configure();torch.manual_seed(0)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    rows,spec=inputs(tokenizer)
    if args.stage=='summarize':summarize(rows,spec,tokenizer);return
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').cuda().eval()
    for layer in model.model.layers:
        layer.self_attn.shadow_enabled=args.arm=='shadowkv'
        layer.self_attn.forward=MethodType(forward,layer.self_attn)
    selected=[rows[64],rows[0]] if args.stage=='smoke' else rows[64:72]+rows[:64]+rows[72:]
    for row in selected:
        path=OUTPUT/args.arm/args.stage/f"sample_{row['index']:03d}.json"
        if path.exists():
            assert read_json(path)['protocol']==spec
            continue
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        cap=min(4,row['maximum_tokens']) if args.stage=='smoke' else row['maximum_tokens']
        ids,first,stats=generate(model,tokenizer,row,cap)
        if args.stage=='smoke':
            again,other,_=generate(model,tokenizer,row,cap)
            assert again==ids;torch.testing.assert_close(first,other,atol=0,rtol=0)
        prediction=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        result=dict(ids=ids,prediction=prediction,score=sample_score(prediction,row['answers'],row['match_type']),
                    routing=stats,seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(path,dict(status='complete',protocol=spec,sample=row,result=result,command=shlex.join(sys.argv),
                            python=sys.executable,gpu=torch.cuda.get_device_name(0)))
        print(args.arm,row['index'],row['task'],result['score'],result['seconds'],repr(prediction[:120]),flush=True)


if __name__=='__main__':main()
