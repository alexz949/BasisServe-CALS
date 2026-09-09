"""Matched V100 FP16 RULER: exact K versus FP32-state LRQK k1152."""
import argparse
import json
from pathlib import Path
import shlex
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.eval_longbench_lrqk_fp16 import v100_prefill,kernel_check
from evaluation.eval_longbench_lrqk_fp32route import FP32RoutingState
from evaluation.eval_qwen3_8b_residual_rank_ruler import TASKS,greedy_decode,_eos_ids
from evaluation.eval_qwen3_c1_quest_ruler import _load_dataset_manifest,_build_work
from evaluation.ruler_v1 import parse_tasks,ruler_prompt,sample_score
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256,write_json
from basisserve.checkpoint import gqa_vo_qwen3 as c1
from basisserve.checkpoint import c1_lrqk_qwen3 as lrqk
from basisserve.core.c1_lrqk import LRQKConfig
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache

ARMS=('full','k1152')

def inputs(args):
    tasks=parse_tasks(TASKS)
    manifest,digest=_load_dataset_manifest(args.data_dir,sequence_length=32768,samples=8,tasks=tasks,tokenizer_path=args.model)
    payload=json.loads((args.c1_checkpoint/'results.json').read_text())
    assert payload['status']=='complete' and payload['fit_config']['cache_rank_per_head']==96
    assert payload['fit_config']['model_config_sha256']==sha256(args.model/'config.json')
    hashes={}
    for l in range(36):
        a=payload['artifacts'][str(l)]
        hashes[str(l)]=sha256(args.c1_checkpoint/a['file'])
        assert hashes[str(l)]==a['sha256']
    spec=dict(format='basisserve.ruler_lrqk_v96.v1',arms=list(ARMS),tasks=[t.name for t in tasks],
        samples_per_task=8,sequence_length=32768,dataset_manifest_sha256=digest,ruler_revision=manifest['ruler']['revision'],
        c1_manifest_sha256=sha256(args.c1_checkpoint/'results.json'),c1_layer_sha256=hashes,
        model_config_sha256=sha256(args.model/'config.json'),dtype='float16',routing_dtype='float32',gpu='V100',
        rank=32,topk=1152,recent=64,iterations=[2,2],seed=0,
        prefill='full causal C1-V96; FP16 memory-efficient SDPA with repeated GQA',
        decode='exact full K or LRQK selected exact K; same resident C1-V96',
        prompt='official base completion plus answer_prefix; add_special_tokens=True',
        scope='reused88-prompt pilot; resident cache accuracy oracle, not official CPU-ring implementation; no hard B2048 cap',
        code_sha256={n:sha256(ROOT/n) for n in ('evaluation/eval_ruler_lrqk_v96.py',
            'evaluation/eval_longbench_lrqk_fp16.py','evaluation/eval_longbench_lrqk_fp32route.py',
            'basisserve/core/c1_lrqk.py','basisserve/checkpoint/c1_lrqk_qwen3.py',
            'basisserve/checkpoint/gqa_vo_qwen3.py','evaluation/ruler_v1.py',
            'evaluation/eval_qwen3_8b_residual_rank_ruler.py')})
    work=_build_work(args.data_dir,tasks,8)
    assert len(work)==88
    return spec,work

@torch.inference_mode()
def generate(model,tokenizer,tokens,maximum,sparse,trace=False):
    cache=lrqk.C1LRQKCache(config=model.config) if sparse else RoutingDynamicCache()
    for layer in model.model.layers: layer.self_attn.attention_backend='triton'
    out=model(input_ids=tokens,past_key_values=cache,use_cache=True,logits_to_keep=1)
    first_logits=out.logits[0,-1].float().cpu()
    assert torch.isfinite(first_logits).all()
    first=int(first_logits.argmax());del out
    if not sparse:
        for layer in model.model.layers: layer.self_attn.attention_backend='sdpa'
    ids,cache,logits=greedy_decode(model,cache,first,maximum_tokens=maximum,eos_ids=_eos_ids(tokenizer,model),trace=trace)
    length=tokens.shape[-1]+len(ids)-1
    for layer in cache.layers:
        assert layer.keys.shape==(1,8,length,128) and layer.values.shape==(1,8,length,96)
        assert layer.keys.dtype==layer.values.dtype==torch.float16
    stats=[cache.lrqk_states[l].statistics(8) for l in range(36)] if sparse else []
    assert all(s['length']==length and s['decode_steps']==len(ids)-1 for s in stats)
    return ids,first_logits,logits,stats

def summarize(args,spec,work,tokenizer):
    eos=json.loads((args.model/'config.json').read_text())['eos_token_id']
    eos=set(eos if isinstance(eos,list) else [eos])|{tokenizer.eos_token_id}
    records={};hashes={}
    for arm in ARMS:
        rows=[];hashes[arm]={}
        for index,task,ordinal,source in work:
            path=args.output_dir/arm/'evaluate'/f'sample_{index:03d}.json'
            saved=json.loads(path.read_text());r=saved['result'];ids=r['generated_token_ids']
            assert saved['status']=='complete' and saved['protocol']==spec and saved['arm']==arm
            assert r['index']==index and r['task']==task.name and r['references']==source['outputs']
            assert 0<len(ids)==r['generated_tokens']<=r['maximum_tokens']==task.tokens_to_generate
            assert r['stopped_on_eos']==(ids[-1] in eos) and not any(i in eos for i in ids[:-1])
            assert r['stopped_on_eos'] or len(ids)==task.tokens_to_generate
            assert r['prediction']==tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            assert r['score']==sample_score(r['prediction'],source['outputs'],task.match_type)
            rows.append(r);hashes[arm][str(index)]=sha256(path)
        for shard in range(4):
            done=json.loads((args.output_dir/arm/'evaluate'/f'shard_{shard}.json').read_text())
            assert done['status']=='complete' and done['protocol']==spec and done['indices']==list(range(shard,88,4))
        records[arm]=rows
    assert all(a['generated_token_ids'][0]==b['generated_token_ids'][0] for a,b in zip(records['full'],records['k1152']))
    tasks={t:{a:100*sum(r['score'] for r in rows if r['task']==t)/8 for a,rows in records.items()} for t in spec['tasks']}
    means={a:sum(t[a] for t in tasks.values())/11 for a in ARMS}
    union=[v for r in records['k1152'] for s in r['routing'] for g in s['physical_union_per_kv_group'] for v in g]
    assert union
    budget=dict(scope='last decode step per prompt; all layers/groups, not all-step mean',mean=sum(union)/len(union),minimum=min(union),maximum=max(union))
    paired=dict(improved=sum(b['score']>a['score'] for a,b in zip(records['full'],records['k1152'])),
        regressed=sum(b['score']<a['score'] for a,b in zip(records['full'],records['k1152'])))
    write_json(args.output_dir/'result.json',dict(status='complete',protocol=spec,tasks=tasks,means=means,paired=paired,
        physical_budget=budget,records=records,input_sha256=hashes))
    write_json(args.output_dir/'audit.json',dict(status='complete',verified=176,first_tokens_match=True,
        scores_and_shard_coverage_verified=True,result_sha256=sha256(args.output_dir/'result.json')))
    lines=['# LRQK RULER32K: V100 FP16 C1-V96','',
        '11 tasks x8 reused prompts; model/K/V FP16, LRQK routing FP32. Full C1 prefill in both arms. No V128 baseline.',
        'LRQK R32, per-head k1152 plus recent64; not Page32 and not a hard physical B2048 cap.','',
        '| Task | Exact K | LRQK k1152 |','|---|---:|---:|']
    lines += [f"| {t} | {v['full']:.4f} | {v['k1152']:.4f} |" for t,v in [*tasks.items(),('Mean',means)]]
    lines += ['',f'Physical union: {json.dumps(budget)}',f'Paired: {json.dumps(paired)}',
        '', 'Environment and commands: docs/lrqk_ruler_v96_protocol.md. Per-record commands preserved.']
    (args.output_dir/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(means=means,paired=paired,physical_budget=budget)),flush=True)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','evaluate','summarize'),required=True)
    p.add_argument('--arm',choices=ARMS,default='k1152')
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--c1-checkpoint',type=Path,default=ROOT/'results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6')
    p.add_argument('--data-dir',type=Path,default=ROOT/'results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/evaluation/ruler_lrqk_v96_fp16')
    args=p.parse_args();assert 0<=args.shard_index<4
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    spec,work=inputs(args);tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if args.stage=='summarize': summarize(args,spec,work,tokenizer);return
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0)
    c1.compressed_v_prefill_attention=v100_prefill
    lrqk.compressed_v_prefill_attention=v100_prefill;lrqk.LRQKState=FP32RoutingState
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float16,local_files_only=True,
        low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda').eval()
    c1.install_qwen3_gqa_vo_als_export(model,args.c1_checkpoint,attention_backend='triton')
    sparse=args.arm=='k1152';smoke=args.stage=='smoke'
    assigned=[work[0],work[32]] if smoke else work[args.shard_index::4]
    reference=None
    if smoke:
        print('kernel_error',kernel_check(),flush=True)
        tokens=tokenizer(ruler_prompt(assigned[0][3]),add_special_tokens=True,return_tensors='pt')['input_ids'].cuda()
        _,reference,_,_=generate(model,tokenizer,tokens,1,False)
    if sparse: lrqk.install_c1_lrqk(model,LRQKConfig(rank=32,topk=1152,recent=64))
    for index,task,ordinal,source in assigned:
        path=args.output_dir/args.arm/args.stage/f'sample_{index:03d}.json'
        if path.exists():
            saved=json.loads(path.read_text());assert saved['status']=='complete' and saved['protocol']==spec
            continue
        tokens=tokenizer(ruler_prompt(source),add_special_tokens=True,return_tensors='pt')['input_ids'].cuda()
        assert tokens.shape[-1]+task.tokens_to_generate<=32768
        maximum=min(4,task.tokens_to_generate) if smoke else task.tokens_to_generate
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        ids,first,trace,stats=generate(model,tokenizer,tokens,maximum,sparse,trace=smoke)
        if smoke:
            if index==assigned[0][0]: torch.testing.assert_close(first,reference,atol=0,rtol=0)
            repeated,other,traces,_=generate(model,tokenizer,tokens,maximum,sparse,trace=True)
            assert repeated==ids and len(ids)>1
            torch.testing.assert_close(first,other,atol=0,rtol=0)
            for x,y in zip(trace,traces,strict=True): torch.testing.assert_close(x,y,atol=0,rtol=0)
        prediction=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        r=dict(index=index,task=task.name,ordinal=ordinal,references=source['outputs'],prompt_tokens=tokens.shape[-1],
            maximum_tokens=maximum,generated_token_ids=ids,generated_tokens=len(ids),prediction=prediction,
            score=None if smoke else sample_score(prediction,source['outputs'],task.match_type),
            stopped_on_eos=ids[-1] in _eos_ids(tokenizer,model),routing=stats,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-started)
        write_json(path,dict(status='complete',protocol=spec,arm=args.arm,result=r,command=shlex.join(sys.argv),
            python=sys.executable,gpu=torch.cuda.get_device_name(0)))
        print(f'{args.arm} sample={index} score={r["score"]} tokens={len(ids)} seconds={r["seconds"]:.1f}',flush=True)
    write_json(args.output_dir/args.arm/args.stage/f'shard_{args.shard_index}.json',
        dict(status='complete',protocol=spec,indices=[r[0] for r in assigned]))

if __name__=='__main__': main()
