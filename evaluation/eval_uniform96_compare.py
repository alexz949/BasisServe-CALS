"""Complete uniform96 RULER controls beside the frozen ShadowKV results."""
import argparse
from pathlib import Path
import shlex
import sys
import time
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.uniform96_common import paths, checkpoint_manifest, install
from evaluation.v96kl_common import configure, read_json, write_json, sha256
from evaluation.profile_qwen3_8b_residual_two_sided_kl import full_attention
from evaluation.eval_qwen3_8b_residual_rank_ruler import prepare_arm, _eos_ids
from evaluation.eval_longbench_lrqk_fp32route import FP32RoutingState
from evaluation.ruler_v1 import sample_score
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.c1_lrqk import LRQKConfig
from basisserve.checkpoint import c1_lrqk_qwen3 as lrqk
from basisserve.checkpoint import gqa_vo_qwen3 as attention
from basisserve.core.c1_conditional_recent_attention import c1_conditional_page_recent64_attention

ARMS = ['full', 'lrqk', 'recent_extra', 'recent_fixed']


def inputs(p):
    manifest = checkpoint_manifest(p['checkpoint'], p['model'])
    shadow = read_json(p['shadow'] / 'result.json')
    assert shadow['status'] == 'complete' and shadow['verified'] == 88
    assert shadow['protocol']['checkpoint_sha256'] == sha256(p['checkpoint'] / 'manifest.json')
    saved = [read_json(p['shadow'] / 'evaluate' / f'sample_{i:03d}.json') for i in range(88)]
    assert all(d['status'] == 'complete' and d['protocol'] == shadow['protocol'] for d in saved)
    rows = [d['sample'] for d in saved]
    spec = dict(checkpoint_sha256=sha256(p['checkpoint'] / 'manifest.json'),
        shadow_summary_sha256=sha256(p['shadow'] / 'result.json'),
        shadow_samples_sha256={str(i):sha256(p['shadow'] / 'evaluate' / f'sample_{i:03d}.json') for i in range(88)},
        dtype='bfloat16', prefill='full causal C1 Triton', generation='greedy, frozen official caps and EOS',
        seed=0, v_rank=96, base_rank=16, residual_rank=16, page_size=32, budget=2048,
        recent_extra='2048 page-selected tokens including sink32 union sliding recent64; max2112',
        recent_fixed='sink32 + 61 disjoint historical pages + sliding recent64; max2048',
        lrqk=dict(rank=32, topk=1152, recent=64, iterations=[2,2], seed=0, state_dtype='float32'),
        primary_excluded_indices=[86],
        code_sha256={name:sha256(ROOT/name) for name in [
            'evaluation/eval_uniform96_compare.py','evaluation/uniform96_common.py',
            'evaluation/profile_qwen3_8b_residual_two_sided_kl.py','evaluation/eval_qwen3_8b_residual_rank_ruler.py',
            'evaluation/eval_longbench_lrqk_fp32route.py','basisserve/core/c1_conditional_recent_attention.py',
            'basisserve/core/c1_conditional_page_attention.py','basisserve/core/c1_lrqk.py',
            'basisserve/checkpoint/c1_lrqk_qwen3.py','basisserve/checkpoint/gqa_vo_qwen3.py',
            'basisserve/core/c1_k_routing_sidecar.py','basisserve/core/c1_v_conditional_k_router.py',
            'basisserve/kernels/compressed_v_decode_attention.py']})
    return manifest, rows, saved, spec


def load_bank(p, layers):
    bank, hashes, protocol = [], {}, None
    for i in range(layers):
        path = p['bank'] / f'layer_{i:03d}.safetensors'
        d = read_json(path.with_suffix('.json'))
        assert d['status'] == 'complete' and d['layer'] == i and d['sha256'] == sha256(path)
        if protocol is None: protocol = d['protocol']
        assert d['protocol'] == protocol
        assert protocol['checkpoint_sha256'] == sha256(p['checkpoint'] / 'manifest.json')
        assert (protocol['fit_windows'], protocol['validation_windows'], protocol['query_count']) == (64,16,32)
        bank.append(load_file(str(path)))
        hashes[str(i)] = d['sha256']
    return bank, hashes


@torch.inference_mode()
def generate(model, tokenizer, row, arm, bank, cap, family):
    modules = [l.self_attn for l in model.model.layers]
    for m in modules: m.reset_reverse_shadow_statistics()
    if arm != 'lrqk': full_attention(modules, 'triton')
    cache = lrqk.C1LRQKCache(config=model.config) if arm == 'lrqk' else RoutingDynamicCache(config=model.config)
    out = model(input_ids=torch.tensor([row['input_ids']],device='cuda'), past_key_values=cache,
                use_cache=True, logits_to_keep=1)
    assert torch.isfinite(out.logits).all()
    first = out.logits[0,-1].float().cpu(); ids = [int(first.argmax())]; del out
    if arm == 'full':
        for m in modules: m.attention_backend = 'sdpa'
    if arm.startswith('recent_'):
        cache = prepare_arm(model, bank, cache, [16]*len(modules))
        for m in modules: m.set_conditional_page_query_block_size(1,collect_statistics=True)
    eos = _eos_ids(tokenizer, model)
    while len(ids) < cap and ids[-1] not in eos:
        valid = torch.ones(1,1,1,cache.get_seq_length()+1,device='cuda',dtype=torch.bool)
        out = model(input_ids=torch.tensor([[ids[-1]]],device='cuda'),past_key_values=cache,use_cache=True,
            attention_mask={'full_attention':valid} if family=='qwen3' else valid,logits_to_keep=1)
        assert torch.isfinite(out.logits).all()
        ids.append(int(out.logits[0,-1].argmax())); del out
    stats = []
    for i, (layer,m) in enumerate(zip(cache.layers,modules,strict=True)):
        assert layer.keys.shape == (1,8,len(row['input_ids'])+len(ids)-1,128)
        assert layer.values.shape == (1,8,len(row['input_ids'])+len(ids)-1,96)
        if arm == 'lrqk': stats.append(cache.lrqk_states[i].statistics(8))
        elif arm.startswith('recent_'): stats.append(m.reverse_shadow_statistics())
    return ids, first, stats


def summarize(p, rows, shadows, spec, tokenizer):
    records = {'shadowkv':[d['result'] for d in shadows]}
    _, hashes = load_bank(p, len(read_json(p['checkpoint']/'manifest.json')['layers']))
    config_eos=read_json(p['model']/'config.json')['eos_token_id']
    eos=set(config_eos if isinstance(config_eos,list) else [config_eos])|{tokenizer.eos_token_id}
    for arm in ARMS:
        records[arm]=[]
        for row in rows:
            d=read_json(p['output']/arm/'evaluate'/f"sample_{row['index']:03d}.json")
            assert d['status']=='complete' and d['protocol']==spec and d['sample']==row
            assert d['bank_sha256']==(hashes if arm.startswith('recent_') else {})
            r=d['result'];ids=r['ids']
            assert 0<len(ids)<=row['maximum_tokens'] and not any(i in eos for i in ids[:-1])
            assert ids[-1] in eos or len(ids)==row['maximum_tokens']
            assert tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['prediction']
            assert sample_score(r['prediction'],row['answers'],row['match_type'])==r['score']
            records[arm].append(r)
        assert all(a['ids'][0]==b['ids'][0] for a,b in zip(records[arm],records['shadowkv'],strict=True))
    means87={a:100*sum(r['score'] for i,r in enumerate(rs) if i!=86)/87 for a,rs in records.items()}
    means88={a:100*sum(r['score'] for r in rs)/88 for a,rs in records.items()}
    tasks={t:{a:100*sum(rs[r['index']]['score'] for r in rows if r['task']==t)/8 for a,rs in records.items()}
           for t in dict.fromkeys(r['task'] for r in rows)}
    write_json(p['output']/'result.json',dict(status='complete',verified=440,protocol=spec,means87=means87,
        means88=means88,tasks=tasks,bank_sha256=hashes,first_token_agreement=True))
    print('VERIFIED',means87,means88,tasks,flush=True)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-family',choices=['qwen3','llama31'],required=True)
    parser.add_argument('--arm',choices=ARMS,default='full')
    parser.add_argument('--stage',choices=['run','summarize'],default='run')
    args=parser.parse_args();configure();torch.manual_seed(0)
    p=paths(args.model_family);manifest,rows,shadows,spec=inputs(p)
    tokenizer=AutoTokenizer.from_pretrained(p['model'],local_files_only=True)
    if args.stage=='summarize': summarize(p,rows,shadows,spec,tokenizer); return
    bank,hashes=load_bank(p,len(manifest['layers'])) if args.arm.startswith('recent_') else ([],{})
    model=AutoModelForCausalLM.from_pretrained(p['model'],dtype=torch.bfloat16,local_files_only=True,
                                           attn_implementation='sdpa').cuda().eval()
    install(model,p['checkpoint'],manifest,args.model_family);model.eval()
    support=[]
    if args.arm=='lrqk':
        lrqk.LRQKState=FP32RoutingState
        lrqk.install_c1_lrqk(model,LRQKConfig(rank=32,topk=1152,recent=64,prefill_backend='triton'))
    if args.arm.startswith('recent_'):
        def routing(*a,**kw):
            result=c1_conditional_page_recent64_attention(*a,**kw,recent_within_budget=args.arm=='recent_fixed')
            count=result.statistics['selected_tokens'];groups=a[1].shape[0]*a[1].shape[1]*a[0].shape[2]
            assert count<=(2048 if args.arm=='recent_fixed' else 2112)*groups
            support.append(count/groups)
            return result
        attention.c1_conditional_page_topk_attention=routing
    for stage, selected in [('smoke',[rows[64],rows[0]]),('evaluate',rows[64:72]+rows[:64]+rows[72:])]:
        for row in selected:
            path=p['output']/args.arm/stage/f"sample_{row['index']:03d}.json"
            if path.exists():
                d=read_json(path)
                assert d['status']=='complete' and d['protocol']==spec and d['sample']==row and d['bank_sha256']==hashes
                continue
            cap=min(4,row['maximum_tokens']) if stage=='smoke' else row['maximum_tokens']
            support.clear();started=time.monotonic();torch.cuda.reset_peak_memory_stats()
            ids,first,stats=generate(model,tokenizer,row,args.arm,bank,cap,args.model_family)
            physical=dict(mean=sum(support)/len(support),max=max(support)) if support else {}
            assert ids[0]==shadows[row['index']]['result']['ids'][0]
            if stage=='smoke':
                again,other,_=generate(model,tokenizer,row,args.arm,bank,cap,args.model_family)
                assert ids==again;torch.testing.assert_close(first,other,rtol=0,atol=0)
            pred=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            result=dict(ids=ids,prediction=pred,score=sample_score(pred,row['answers'],row['match_type']),
                        routing=stats,physical_support=physical,seconds=time.monotonic()-started,
                        peak_gib=torch.cuda.max_memory_allocated()/2**30)
            write_json(path,dict(status='complete',protocol=spec,bank_sha256=hashes,sample=row,result=result,
                                command=shlex.join(sys.argv),python=sys.executable,gpu=torch.cuda.get_device_name(0)))
            print(stage,args.arm,row['index'],row['task'],result['score'],result['seconds'],flush=True)


if __name__=='__main__': main()
