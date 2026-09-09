"""L40S BF16 RULER: ShadowKV K reconstruction/routing with resident C1-V96."""
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
from basisserve.kernels.compressed_v_decode_attention import compressed_v_prefill_attention
from evaluation.eval_qwen3_8b_residual_rank_ruler import TASKS,greedy_decode,_eos_ids
from evaluation.eval_qwen3_c1_quest_ruler import _load_dataset_manifest,_build_work
from evaluation.ruler_v1 import parse_tasks,ruler_prompt,sample_score
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256,write_json
from basisserve.checkpoint import gqa_vo_qwen3 as c1
from basisserve.checkpoint import c1_shadowkv_qwen3 as shadow
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache

ARMS=('full','shadowkv')

def verified_record(path,args,spec):
    saved=json.loads(path.read_text())
    assert saved['status']=='complete'
    if saved['protocol']!=spec:
        repair=json.loads((args.output_dir/'repair_inputs.json').read_text())
        assert saved['protocol']==repair['original_protocol']
        assert sha256(path)==repair['artifacts'][str(path.relative_to(args.output_dir))]
        old=dict(saved['protocol']);current=dict(spec)
        old_code=old.pop('code_sha256');new_code=current.pop('code_sha256')
        assert old==current and old_code.keys()==new_code.keys()
        changed={k for k in old_code if old_code[k]!=new_code[k]}
        assert changed=={'basisserve/core/c1_shadowkv.py','evaluation/eval_ruler_shadowkv_v96.py'}
    return saved

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
    spec=dict(format='basisserve.ruler_shadowkv_v96.v1',arms=list(ARMS),tasks=[t.name for t in tasks],
        samples_per_task=8,sequence_length=32768,dataset_manifest_sha256=digest,ruler_revision=manifest['ruler']['revision'],
        c1_manifest_sha256=sha256(args.c1_checkpoint/'results.json'),c1_layer_sha256=hashes,
        model_config_sha256=sha256(args.model/'config.json'),dtype='bfloat16',routing_dtype='bfloat16',gpu='L40S',
        rank=160,sparse_budget=2048,chunk_size=8,outlier_chunks=48,local_chunks=4,
        upstream_revision='e51904cdeab7d4d34013370f09f2cf5fcd655e15',svd='FP32 torch.svd of concatenated normalized pre-RoPE K; BF16 U/SV storage',
        prefill='full causal C1-V96 BF16 Triton in both arms',
        decode='exact full K or ShadowKV reconstructed selected K plus exact outlier/local/generated K; resident C1-V96',
        prompt='official base completion plus answer_prefix; add_special_tokens=True',
        scope='reused88-prompt pilot; resident cache accuracy oracle, not official CPU-ring implementation; no hard B2048 cap',
        code_sha256={n:sha256(ROOT/n) for n in ('evaluation/eval_ruler_shadowkv_v96.py',
            'basisserve/core/c1_shadowkv.py','basisserve/checkpoint/c1_shadowkv_qwen3.py',
            'external/ShadowKV/models/kv_cache.py',
            'basisserve/checkpoint/gqa_vo_qwen3.py','evaluation/ruler_v1.py',
            'evaluation/eval_qwen3_8b_residual_rank_ruler.py')})
    work=_build_work(args.data_dir,tasks,8)
    assert len(work)==88
    return spec,work

@torch.inference_mode()
def generate(model,tokenizer,tokens,maximum,sparse,trace=False):
    cache=shadow.C1ShadowKVCache(config=model.config) if sparse else RoutingDynamicCache()
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
        assert layer.keys.dtype==layer.values.dtype==torch.bfloat16
    stats=[cache.shadow_states[l].statistics() for l in range(36)] if sparse else []
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
            saved=verified_record(path,args,spec);r=saved['result'];ids=r['generated_token_ids']
            assert saved['arm']==arm
            assert r['index']==index and r['task']==task.name and r['references']==source['outputs']
            assert 0<len(ids)==r['generated_tokens']<=r['maximum_tokens']==task.tokens_to_generate
            assert r['stopped_on_eos']==(ids[-1] in eos) and not any(i in eos for i in ids[:-1])
            assert r['stopped_on_eos'] or len(ids)==task.tokens_to_generate
            assert r['prediction']==tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            assert r['score']==sample_score(r['prediction'],source['outputs'],task.match_type)
            rows.append(r);hashes[arm][str(index)]=sha256(path)
        for shard in range(4):
            done=verified_record(args.output_dir/arm/'evaluate'/f'shard_{shard}.json',args,spec)
            assert done['indices']==list(range(shard,88,4))
        records[arm]=rows
    assert all(a['generated_token_ids'][0]==b['generated_token_ids'][0] for a,b in zip(records['full'],records['shadowkv']))
    tasks={t:{a:100*sum(r['score'] for r in rows if r['task']==t)/8 for a,rows in records.items()} for t in spec['tasks']}
    means={a:sum(t[a] for t in tasks.values())/11 for a in ARMS}
    for r in records['shadowkv']:
        assert len(r['routing'])==36
        assert all(s['decode_steps']==len(r['generated_token_ids'])-1 for s in r['routing'])
    union=[s['physical_tokens_per_group'] for r in records['shadowkv'] for s in r['routing'] if s['decode_steps']>0 for _ in range(8)]
    assert union
    budget=dict(scope='last decode step per prompt with at least one decode; zero-decode prompts excluded; not all-step mean',mean=sum(union)/len(union),minimum=min(union),maximum=max(union),
        zero_decode_prompts=sum(len(r['generated_token_ids'])==1 for r in records['shadowkv']))
    paired=dict(improved=sum(b['score']>a['score'] for a,b in zip(records['full'],records['shadowkv'])),
        regressed=sum(b['score']<a['score'] for a,b in zip(records['full'],records['shadowkv'])))
    write_json(args.output_dir/'result.json',dict(status='complete',protocol=spec,tasks=tasks,means=means,paired=paired,
        physical_budget=budget,records=records,input_sha256=hashes,
        repair_input_sha256=sha256(args.output_dir/'repair_inputs.json') if (args.output_dir/'repair_inputs.json').exists() else None))
    write_json(args.output_dir/'audit.json',dict(status='complete',verified=176,first_tokens_match=True,
        scores_and_shard_coverage_verified=True,result_sha256=sha256(args.output_dir/'result.json')))
    lines=['# ShadowKV K + resident C1-V96: L40S BF16 RULER32K','',
        '11 tasks x8 reused prompts; model/K/V and stored SVD factors BF16; SVD solve FP32. Full C1 Triton prefill in both arms.',
        'ShadowKV rank160 across concatenated KV heads, chunk8, selected2048 plus exact48 outlier chunks, prompt local tail and all generated tokens.','',
        '| Task | Exact K | ShadowKV |','|---|---:|---:|']
    lines += [f"| {t} | {v['full']:.4f} | {v['shadowkv']:.4f} |" for t,v in [*tasks.items(),('Mean',means)]]
    lines += ['',f'Physical union: {json.dumps(budget)}',f'Paired: {json.dumps(paired)}',
        '', 'Environment and commands: docs/shadowkv_v96_protocol.md. Per-record commands preserved.']
    (args.output_dir/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(means=means,paired=paired,physical_budget=budget)),flush=True)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','evaluate','summarize'),required=True)
    p.add_argument('--arm',choices=ARMS,default='shadowkv')
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--c1-checkpoint',type=Path,default=ROOT/'results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6')
    p.add_argument('--data-dir',type=Path,default=ROOT/'results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/evaluation/ruler_shadowkv_v96_bf16')
    args=p.parse_args();assert 0<=args.shard_index<4
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    spec,work=inputs(args);tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if args.stage=='summarize': summarize(args,spec,work,tokenizer);return
    assert torch.cuda.is_available() and 'L40S' in torch.cuda.get_device_name(0)
    c1.compressed_v_prefill_attention=compressed_v_prefill_attention
    shadow.compressed_v_prefill_attention=compressed_v_prefill_attention
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,local_files_only=True,
        low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda').eval()
    c1.install_qwen3_gqa_vo_als_export(model,args.c1_checkpoint,attention_backend='triton')
    sparse=args.arm=='shadowkv';smoke=args.stage=='smoke'
    assigned=[work[0],work[32]] if smoke else work[args.shard_index::4]
    reference=None
    if smoke:
        print('ShadowKV BF16 smoke: prefill equality and repeated logits',flush=True)
        tokens=tokenizer(ruler_prompt(assigned[0][3]),add_special_tokens=True,return_tensors='pt')['input_ids'].cuda()
        _,reference,_,_=generate(model,tokenizer,tokens,1,False)
    if sparse: shadow.install_c1_shadowkv(model)
    for index,task,ordinal,source in assigned:
        path=args.output_dir/args.arm/args.stage/f'sample_{index:03d}.json'
        if path.exists():
            saved=verified_record(path,args,spec)
            assert saved['arm']==args.arm and saved['result']['index']==index
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
