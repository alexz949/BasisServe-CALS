"""Full LongBench-v1: matched C1-KL96 full-K, Base16/R16, and LRQK."""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.v96kl_common import (
    ROOT, MODEL, CHECKPOINT, DATA, BANK, OUTPUT, OFFICIAL, configure, checkpoint_manifest,
    read_json, write_json, sha256, tensor_hash, code_hashes,
)
from evaluation.prepare_v96kl_data import TASKS
from evaluation.eval_longbench_c1_twosided_denseprefill import install
from evaluation.profile_qwen3_8b_residual_two_sided_kl import full_attention
from evaluation.eval_qwen3_8b_residual_rank_ruler import prepare_arm, _eos_ids
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.c1_lrqk import LRQKConfig
from basisserve.checkpoint.c1_lrqk_qwen3 import C1LRQKCache, install_c1_lrqk
from basisserve.checkpoint import c1_lrqk_qwen3 as lrqk_adapter
from evaluation.eval_longbench_lrqk_fp32route import FP32RoutingState

ARMS=('full','b16r16','lrqk')


def scorer_module():
    sys.path.insert(0,str(ROOT/'results/tools/longbench_deps'))
    sys.path.insert(0,str(OFFICIAL))
    spec=importlib.util.spec_from_file_location('official_longbench_eval',OFFICIAL/'eval.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_sample(scorer,row,prediction):
    # This preprocessing is part of the official scorer, not task-independent QA scoring.
    if row['task'] in ('trec','triviaqa','samsum','lsht'):
        prediction=prediction.lstrip('\n').split('\n')[0]
    return max(float(scorer.dataset2metric[row['task']](prediction,answer,all_classes=row['all_classes']))
               for answer in row['answers'])


def inputs(args):
    manifest=checkpoint_manifest(args.checkpoint,args.model)
    data=read_json(args.data/'manifest.json')
    assert data['status']=='complete' and data['tasks']==list(TASKS)
    assert data['tokens_sha256']==sha256(args.data/'tokens.safetensors')
    assert data['samples_sha256']==sha256(args.data/'samples.json')
    for name,digest in data['official_sha256'].items(): assert sha256(OFFICIAL/name)==digest
    rows=read_json(args.data/'samples.json')
    tokens=load_file(str(args.data/'tokens.safetensors'))
    assert len(rows)==len(tokens)==data['total']
    assert dict(Counter(row['task'] for row in rows))==data['counts']
    assert data['model_config_sha256']==manifest['model']['config_sha256']
    for i,row in enumerate(rows):
        assert row['index']==i and tensor_hash(tokens[f'sample_{i:05d}'])==row['input_ids_sha256']
        assert len(tokens[f'sample_{i:05d}'])==row['prompt_tokens']
        assert row['prompt_tokens']+row['maximum_tokens']<=32768
    # All arms have the same protocol. The bank is required only for the router arm.
    protocol=dict(format='basisserve.full_longbench_v96kl.v1',
        checkpoint_sha256=sha256(args.checkpoint/'manifest.json'),data_sha256=sha256(args.data/'manifest.json'),
        model_config_sha256=sha256(args.model/'config.json'),model_revision=manifest['model']['revision'],
        prefill='full causal C1-KL96 BF16 Triton for every arm; fresh unpadded cache per sample',
        dtype='bfloat16',lrqk_state_dtype='float32',base_rank=16,residual_rank=16,
        page_size=32,physical_page_token_budget=2048,pinned_prefix_pages=1,force_current_page=False,
        lrqk_rank=32,lrqk_topk=args.lrqk_topk,lrqk_recent=64,lrqk_iterations=[2,2],seed=0,
        lrqk_budget='per-query-head top-k plus recent64; physical GQA union measured, not capped',
        storage='resident exact K and C1 latent; materialized Base128+R16; quality comparison, not offload timing',
        generation='greedy, official task caps; model/tokenizer EOS; samsum additionally newline after first token',
        code_sha256=code_hashes(['evaluation/eval_v96kl_longbench.py','evaluation/v96kl_common.py',
            'evaluation/eval_longbench_c1_twosided_denseprefill.py',
            'evaluation/eval_qwen3_8b_residual_rank_ruler.py',
            'evaluation/profile_qwen3_8b_residual_two_sided_kl.py',
            'basisserve/core/c1_lrqk.py','basisserve/checkpoint/c1_lrqk_qwen3.py',
            'evaluation/eval_longbench_lrqk_fp32route.py',
            'basisserve/core/c1_conditional_page_attention.py','basisserve/core/c1_v_conditional_k_router.py',
            'basisserve/checkpoint/gqa_vo_qwen3.py']))
    return manifest,rows,tokens,protocol


def load_bank(args):
    bank=[];hashes={};protocol=None
    for layer in range(36):
        path=args.bank/f'layer_{layer:03d}.safetensors'
        saved=read_json(path.with_suffix('.json'))
        assert saved['status']=='complete' and saved['sha256']==sha256(path)
        if protocol is None: protocol=saved['protocol']
        assert saved['protocol']==protocol
        assert protocol['checkpoint_sha256']==sha256(args.checkpoint/'manifest.json')
        assert protocol['base_rank']==protocol['residual_rank']==16
        assert protocol['fit_windows']==64 and protocol['query_count']==32
        bank.append(load_file(str(path)));hashes[str(layer)]=saved['sha256']
    return bank,hashes


@torch.inference_mode()
def generate(model,tokenizer,tokens,row,arm,bank,cap):
    modules=[layer.self_attn for layer in model.model.layers]
    for m in modules: m.reset_reverse_shadow_statistics()
    if arm!='lrqk': full_attention(modules,'triton')
    cache=C1LRQKCache(config=model.config) if arm=='lrqk' else RoutingDynamicCache(config=model.config)
    out=model(input_ids=tokens.long()[None].cuda(),past_key_values=cache,use_cache=True,logits_to_keep=1)
    logits=out.logits[0,-1]
    assert torch.isfinite(logits).all()
    first=int(logits.argmax());first_logits=logits.float().cpu();del out,logits
    if arm=='full':
        for m in modules: m.attention_backend='sdpa'
    if arm=='b16r16':
        cache=prepare_arm(model,bank,cache,[16]*36)
        for m in modules: m.set_conditional_page_query_block_size(1,collect_statistics=True)
    eos=_eos_ids(tokenizer,model)
    newline=tokenizer.encode('\n',add_special_tokens=False)[-1]
    ids=[first]
    while len(ids)<cap and ids[-1] not in eos:
        # Official samsum min_length permits the first generated token before newline stopping.
        if row['task']=='samsum' and len(ids)>1 and ids[-1]==newline: break
        length=cache.get_seq_length()+1
        valid=torch.ones(1,1,1,length,dtype=torch.bool,device='cuda')
        out=model(input_ids=torch.tensor([[ids[-1]]],device='cuda'),past_key_values=cache,use_cache=True,
                  attention_mask={'full_attention':valid},logits_to_keep=1)
        logits=out.logits[0,-1]
        assert torch.isfinite(logits).all()
        ids.append(int(logits.argmax()));del out,logits
    stats=[]
    for i,(layer,m) in enumerate(zip(cache.layers,modules,strict=True)):
        assert layer.keys.shape==(1,8,len(tokens)+len(ids)-1,128)
        assert layer.values.shape==(1,8,len(tokens)+len(ids)-1,m.value_head_dim)
        if arm=='b16r16':
            assert cache.routing_sidecar(i).shape[-1]==144
            stats.append(m.reverse_shadow_statistics())
        elif arm=='lrqk': stats.append(cache.lrqk_states[i].statistics(8))
    stopped=ids[-1] in eos or (row['task']=='samsum' and len(ids)>1 and ids[-1]==newline)
    del cache
    return ids,first_logits,stats,stopped


def summarize(args,rows,protocol,tokenizer,scorer):
    bank,hashes=load_bank(args)
    del bank
    records={};tasks={};means={};budget={}
    for arm in ARMS:
        records[arm]=[]
        for row in rows:
            saved=read_json(args.output/arm/'evaluate'/f"sample_{row['index']:05d}.json")
            assert saved['status']=='complete' and saved['protocol']==protocol
            assert saved['bank_sha256']==(hashes if arm=='b16r16' else {})
            r=saved['result'];ids=r['generated_token_ids']
            assert r['sample']==row and 0<len(ids)<=row['maximum_tokens']
            assert r['stopped'] or len(ids)==row['maximum_tokens']
            assert tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['prediction']
            assert score_sample(scorer,row,r['prediction'])==r['score']
            records[arm].append(r)
        for task in TASKS:
            rs=[r for r in records[arm] if r['sample']['task']==task]
            mean=100*sum(r['score'] for r in rs)/len(rs)
            assert round(mean,2)==scorer.scorer(task,[r['prediction'] for r in rs],
                [r['sample']['answers'] for r in rs],rs[0]['sample']['all_classes'])
            tasks.setdefault(task,dict(samples=len(rs)))[arm]=mean
        means[arm]=sum(tasks[t][arm] for t in TASKS)/len(TASKS)
    for arm in ('b16r16','lrqk'):
        assert all(a['generated_token_ids'][0]==b['generated_token_ids'][0]
                   for a,b in zip(records['full'],records[arm],strict=True))
    unions=torch.tensor([n for r in records['lrqk'] for s in r['routing']
                         for group in s['physical_union_per_kv_group'] for n in group],dtype=torch.float64)
    budget['lrqk']=dict(scope='final decode step per prompt/layer/KV group',mean=float(unions.mean()),
        max=float(unions.max()),p95=float(unions.quantile(.95)),fraction_above_2048=float((unions>2048).double().mean()))
    chosen=sum(s['physical_selected_tokens'] if 'physical_selected_tokens' in s else s['selected_tokens']
               for r in records['b16r16'] for s in r['routing'])
    valid=sum(s['physical_valid_tokens'] for r in records['b16r16'] for s in r['routing'])
    budget['b16r16']=dict(scope='all decode steps',selected_tokens=chosen,valid_tokens=valid,
                         selected_fraction=chosen/valid)
    write_json(args.output/'result.json',dict(status='complete',protocol=protocol,tasks=tasks,means=means,budget=budget,
        samples_per_arm=len(rows),first_token_agreement=True,bank_sha256=hashes))
    lines=['# Qwen3-8B C1 two-sided KL average V96: full LongBench-v1','',
        'All 21 tasks and all samples. Shared BF16 C1 prefill. Base16/R16: 64 x 32K fit, 16 x 32K validation, fit-only Query-Gram Q32.',
        'Scores use official task metrics (0–100); overall mean is the arithmetic mean of 21 task scores.',
        'LRQK uses FP32 routing state and per-query top-k + recent64; its physical GQA union is not a hard 2048-token budget.','',
        '| Task | N | Full K | Base16 + R16 | LRQK |','|---|---:|---:|---:|---:|']
    for task,r in tasks.items(): lines.append(f"| {task} | {r['samples']} | {r['full']:.4f} | {r['b16r16']:.4f} | {r['lrqk']:.4f} |")
    lines.append('| Mean | | '+' | '.join(f'{means[a]:.4f}' for a in ARMS)+' |')
    path=args.output/'summary.md'
    text='\n'.join(lines)+'\n'
    if path.exists(): assert path.read_text()==text
    else: path.write_text(text)
    print('VERIFIED',len(rows),'samples x 3 arms',means,budget,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('smoke','evaluate','summarize'))
    p.add_argument('--arm',choices=ARMS,default='full')
    for name,default in [('model',MODEL),('checkpoint',CHECKPOINT),('data',DATA),('bank',BANK),('output',OUTPUT)]:
        p.add_argument('--'+name,type=Path,default=default)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=2)
    p.add_argument('--lrqk-topk',type=int,default=2048)
    args=p.parse_args();configure()
    assert 0<=args.shard_index<args.num_shards
    manifest,rows,tokens,protocol=inputs(args)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    scorer=scorer_module()
    if args.stage=='summarize':
        summarize(args,rows,protocol,tokenizer,scorer);return
    bank,hashes=load_bank(args) if args.arm=='b16r16' else (None,{})
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
        local_files_only=True,attn_implementation='sdpa').to('cuda').eval()
    originals,modules=install(model,args.checkpoint,manifest)
    del originals
    for m in modules: m.attention_backend='triton'
    if args.arm=='lrqk':
        lrqk_adapter.LRQKState=FP32RoutingState
        install_c1_lrqk(model,LRQKConfig(rank=32,topk=args.lrqk_topk,recent=64,prefill_backend='triton'))
    smoke=args.stage=='smoke'
    selected=([min(rows,key=lambda r:r['prompt_tokens']),max(rows,key=lambda r:r['prompt_tokens'])]
              if smoke else rows[args.shard_index::args.num_shards])
    for row in selected:
        path=args.output/args.arm/args.stage/f"sample_{row['index']:05d}.json"
        if path.exists():
            saved=read_json(path)
            assert saved['status']=='complete' and saved['protocol']==protocol and saved['bank_sha256']==hashes
            continue
        cap=min(4,row['maximum_tokens']) if smoke else row['maximum_tokens']
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            ids,first,stats,stopped=generate(model,tokenizer,tokens[f"sample_{row['index']:05d}"],row,args.arm,bank,cap)
            if smoke:
                repeated,other,_,_=generate(model,tokenizer,tokens[f"sample_{row['index']:05d}"],row,args.arm,bank,cap)
                assert repeated==ids
                torch.testing.assert_close(first,other,rtol=0,atol=0)
        prediction=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        result=dict(sample=row,prediction=prediction,generated_token_ids=ids,score=score_sample(scorer,row,prediction),
            stopped=stopped,routing=stats,elapsed_seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,smoke_repeat_verified=smoke)
        write_json(path,dict(status='complete',protocol=protocol,bank_sha256=hashes,result=result,
            command=sys.argv,python=sys.executable,gpu=torch.cuda.get_device_name(0)))
        print(args.arm,row['index'],row['task'],'score',result['score'],'tokens',len(ids),
              'seconds',result['elapsed_seconds'],flush=True)


if __name__=='__main__': main()
