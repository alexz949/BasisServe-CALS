"""RULER V96/Base16/R16 matched to completed V100 exact-K and LRQK arms."""
from functools import partial
import argparse
import json
from pathlib import Path
import shlex
import sys
import time
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.eval_ruler_lrqk_v96 import inputs as reference_inputs
from evaluation.eval_longbench_c1_v96_r16_fp16 import generate
from evaluation.eval_longbench_lrqk_fp16 import v100_prefill,kernel_check
from evaluation.eval_qwen3_8b_residual_rank_ruler import _eos_ids
from evaluation.ruler_v1 import ruler_prompt,sample_score
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256,write_json
from basisserve.checkpoint import gqa_vo_qwen3 as c1
from basisserve.core.c1_conditional_recent_attention import c1_conditional_page_recent64_attention

def inputs(args):
    reference_spec,work=reference_inputs(args)
    work=[row for row in work if row[0] != 86]
    ref=json.loads((args.reference/'result.json').read_text())
    audit=json.loads((args.reference/'audit.json').read_text())
    assert ref['status']==audit['status']=='complete' and audit['verified']==176
    assert sha256(args.reference/'result.json')==audit['result_sha256'] and ref['protocol']==reference_spec
    bank=[];hashes={};source=None
    for l in range(36):
        path=args.bank/f'layer_{l:03d}.safetensors';record=json.loads(path.with_suffix('.json').read_text())
        assert record['status']=='complete' and record['layer']==l and sha256(path)==record['sha256']
        if source is None: source=record['protocol']
        assert record['protocol']==source
        tensors=load_file(str(path))
        shapes={'base_left_b16':(8,96,16),'base_right_b16':(8,16,128),'base_bias_b16':(8,128),
            'residual_encoder_b16_r16':(8,128,16),'residual_query_b16_r16':(32,128,16)}
        assert set(tensors)==set(shapes)
        for k,shape in shapes.items():
            assert tuple(tensors[k].shape)==shape and tensors[k].dtype==torch.float32 and torch.isfinite(tensors[k]).all()
        bank.append(tensors);hashes[str(l)]=record['sha256']
    assert source['c1_manifest_sha256']==reference_spec['c1_manifest_sha256']
    assert source['c1_layer_sha256']==reference_spec['c1_layer_sha256']
    assert source['query_count']==32 and source['residual_rank']==16 and source['base_rank']==16
    assert source['page_size']==32 and source['physical_token_budget']==2048 and source['excluded_prefix_pages']==1
    assert source['bcd_sweeps']==40 and source['pcg_iterations']==100
    for positions in source['query_positions'].values():
        assert all(sum(p//8192==b for p in positions)==8 for b in range(4))
    spec=dict(format='basisserve.ruler_v96_r16.v1',reference_protocol=reference_spec,
        reference_result_sha256=audit['result_sha256'],bank_protocol=source,bank_sha256=hashes,
        dtype='float16',page_size=32,physical_token_budget=2048,pinned_prefix_pages=1,recent_tokens_within_budget=64,maximum_union_tokens=2048,excluded_indices=[86],
        code_sha256={n:sha256(ROOT/n) for n in ('evaluation/eval_ruler_c1_v96_recent64_fixed.py',
            'evaluation/eval_longbench_c1_v96_r16_fp16.py','basisserve/core/c1_conditional_page_attention.py','basisserve/core/c1_conditional_recent_attention.py',
            'basisserve/core/c1_k_routing_sidecar.py','basisserve/core/c1_v_conditional_k_router.py')})
    return spec,work,bank,ref

def summarize(args,spec,work,reference,tokenizer):
    config=json.loads((args.model/'config.json').read_text());eos=config['eos_token_id']
    eos=set(eos if isinstance(eos,list) else [eos])|{tokenizer.eos_token_id}
    records=[];hashes={}
    for index,task,ordinal,source in work:
        path=args.output_dir/'evaluate'/f'sample_{index:03d}.json'
        saved=json.loads(path.read_text());r=saved['result'];ids=r['generated_token_ids']
        assert saved['status']=='complete' and saved['protocol']==spec and r['index']==index
        assert r['task']==task.name and r['references']==source['outputs']
        assert 0<len(ids)==r['generated_tokens']<=r['maximum_tokens']==task.tokens_to_generate
        assert r['stopped_on_eos']==(ids[-1] in eos) and not any(i in eos for i in ids[:-1])
        assert r['stopped_on_eos'] or len(ids)==task.tokens_to_generate
        assert r['prediction']==tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        assert r['score']==sample_score(r['prediction'],source['outputs'],task.match_type)
        assert ids[0]==reference['records']['full'][index]['generated_token_ids'][0]
        assert r['sparse_calls']==36*(len(ids)-1) and r['prefill_calls']==36
        records.append(r);hashes[str(index)]=sha256(path)
    for shard in range(4):
        saved=json.loads((args.output_dir/'evaluate'/f'shard_{shard}.json').read_text())
        assert saved['status']=='complete' and saved['protocol']==spec and saved['indices']==[row[0] for row in work[shard::4]]
    reference={**reference,'records':{a:[r for r in rs if r['index']!=86] for a,rs in reference['records'].items()}}
    tasks={t:dict(v,ours=100*sum(r['score'] for r in records if r['task']==t)/sum(r['task']==t for r in records)) for t,v in reference['tasks'].items()}
    for t in tasks:
        for a in ('full','k1152'):
            rs=[r for r in reference['records'][a] if r['task']==t]
            tasks[t][a]=100*sum(r['score'] for r in rs)/len(rs)
    means={a:100*sum(r['score'] for r in (records if a=='ours' else reference['records'][a]))/87 for a in ('full','k1152','ours')}
    paired={a:dict(improved=sum(y['score']>x['score'] for x,y in zip(reference['records'][a],records)),
        regressed=sum(y['score']<x['score'] for x,y in zip(reference['records'][a],records))) for a in ('full','k1152')}
    write_json(args.output_dir/'result.json',dict(status='complete',protocol=spec,tasks=tasks,means=means,paired=paired,
        records=records,input_sha256=hashes,cap_without_eos=sum(not r['stopped_on_eos'] for r in records)))
    write_json(args.output_dir/'audit.json',dict(status='complete',verified=87,reference_verified=176,
        first_tokens_scores_cache_dispatch_verified=True,result_sha256=sha256(args.output_dir/'result.json')))
    lines=['# RULER32K: C1-V96/Base16/R16 versus exact K and LRQK','',
        'Same87 paired prompts (exclude index86), sample-weighted mean,11 tasks, FP16 C1-V96 full prefill and decode. Ours: full-window Query-Gram Q32, Page32/B2048, pinned page0, exact recent64 included within hard B2048 (sink32 + recent64 + 61 disjoint historical pages).',
        'LRQK: FP32 routing state, R32/k1152/recent64. Frozen offline R16 bank reused; no fitting in this experiment.','',
        '| Task | Full K | LRQK | Ours R16 |','|---|---:|---:|---:|']
    lines += [f"| {t} | {v['full']:.4f} | {v['k1152']:.4f} | {v['ours']:.4f} |" for t,v in [*tasks.items(),('Mean',means)]]
    lines+=['',f'Paired changes: {json.dumps(paired)}','', 'Environment: basis. Commands: docs/ruler_v96_recent64_protocol.md.']
    (args.output_dir/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(means=means,paired=paired)),flush=True)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','evaluate','summarize'),required=True)
    p.add_argument('--shard-index',type=int,default=0)
    for name,path in {'c1-checkpoint':'results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6',
        'data-dir':'results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8',
        'bank':'results/checkpoints/c1_v96_b16r16_qgram','reference':'results/evaluation/ruler_lrqk_v96_fp16',
        'output-dir':'results/evaluation/ruler_v96_r16_recent64_fixed87_fp16'}.items():
        p.add_argument('--'+name,type=Path,default=ROOT/path)
    args=p.parse_args();assert 0<=args.shard_index<4
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    spec,work,bank,reference=inputs(args)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if args.stage=='summarize': summarize(args,spec,work,reference,tokenizer);return
    assert torch.cuda.is_available()
    c1.compressed_v_prefill_attention=v100_prefill
    c1.c1_conditional_page_topk_attention=partial(c1_conditional_page_recent64_attention,recent_within_budget=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float16,local_files_only=True,
        low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda').eval()
    c1.install_qwen3_gqa_vo_als_export(model,args.c1_checkpoint,attention_backend='triton')
    smoke=args.stage=='smoke'
    if smoke: print('kernel_error',kernel_check(),flush=True)
    assigned=[work[0],work[32]] if smoke else work[args.shard_index::4]
    for index,task,ordinal,source in assigned:
        path=args.output_dir/args.stage/f'sample_{index:03d}.json'
        if path.exists():
            saved=json.loads(path.read_text());assert saved['status']=='complete' and saved['protocol']==spec
            continue
        tokens=tokenizer(ruler_prompt(source),add_special_tokens=True,return_tensors='pt')['input_ids'][0]
        assert len(tokens)+task.tokens_to_generate<=32768
        maximum=min(4,task.tokens_to_generate) if smoke else task.tokens_to_generate
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        ids,trace,calls,prefill_calls=generate(model,tokenizer,bank,tokens,maximum,trace=smoke)
        assert ids[0]==reference['records']['full'][index]['generated_token_ids'][0]
        if smoke:
            repeated,other,_,_=generate(model,tokenizer,bank,tokens,maximum,trace=True)
            assert repeated==ids and len(ids)>1
            for x,y in zip(trace,other,strict=True): torch.testing.assert_close(x,y,atol=0,rtol=0)
        prediction=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        r=dict(index=index,task=task.name,references=source['outputs'],prompt_tokens=len(tokens),
            maximum_tokens=maximum,generated_token_ids=ids,generated_tokens=len(ids),prediction=prediction,
            score=None if smoke else sample_score(prediction,source['outputs'],task.match_type),
            stopped_on_eos=ids[-1] in _eos_ids(tokenizer,model),sparse_calls=calls,prefill_calls=prefill_calls,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-started)
        write_json(path,dict(status='complete',protocol=spec,result=r,command=shlex.join(sys.argv),
            python=sys.executable,gpu=torch.cuda.get_device_name(0)))
        print(f'sample={index} score={r["score"]} tokens={len(ids)} seconds={r["seconds"]:.1f}',flush=True)
    write_json(args.output_dir/args.stage/f'shard_{args.shard_index}.json',
        dict(status='complete',protocol=spec,indices=[r[0] for r in assigned]))

if __name__=='__main__': main()
