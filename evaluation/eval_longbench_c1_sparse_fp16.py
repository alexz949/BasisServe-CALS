"""Matched V100/FP16 C1-V96 Base16/R8 control for LRQK budget experiments."""
import argparse
import json
from pathlib import Path
import shlex
import sys
import time
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import eval_longbench_c1_v96_router as source
from evaluation.eval_longbench_lrqk_fp16 import v100_prefill,kernel_check
from evaluation.eval_qwen3_8b_residual_rank_ruler import prepare_arm,greedy_decode,_eos_ids
from evaluation.profile_qwen3_8b_residual_two_sided_kl import full_attention
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256,write_json
from basisserve.checkpoint import gqa_vo_qwen3 as attention
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.residual_kl_replay import prefix_signature


@torch.inference_mode()
def generate(model,tokenizer,bank,tokens,maximum,smoke):
    modules=[l.self_attn for l in model.model.layers]
    full_attention(modules,'triton')
    prefix=RoutingDynamicCache()
    out=model(input_ids=tokens.long()[None].cuda(),past_key_values=prefix,use_cache=True,logits_to_keep=1)
    logits=out.logits[0,-1]
    assert torch.isfinite(logits).all()
    first=int(logits.argmax()); reference=logits.cpu() if smoke else None
    del out,logits
    signature=prefix_signature(prefix)
    cache=prepare_arm(model,bank,prefix,[8]*36)
    calls=0
    native=attention.c1_conditional_page_topk_attention
    def observe(*a,**kw):
        nonlocal calls
        calls+=1
        return native(*a,**kw)
    with patch.object(attention,'c1_conditional_page_topk_attention',new=observe):
        ids,cache,_=greedy_decode(model,cache,first,maximum_tokens=maximum,eos_ids=_eos_ids(tokenizer,model))
    assert calls==36*(len(ids)-1) and prefix_signature(prefix)==signature
    length=len(tokens)+len(ids)-1
    for l,layer in enumerate(cache.layers):
        assert layer.keys.shape==(1,8,length,128) and layer.values.shape==(1,8,length,96)
        assert cache.routing_sidecar(l).shape==(1,8,length,136)
        assert layer.keys.dtype==layer.values.dtype==cache.routing_sidecar(l).dtype==torch.float16
    del cache,prefix
    if smoke:
        full_attention(modules,'triton')
        check=RoutingDynamicCache()
        out=model(input_ids=tokens.long()[None].cuda(),past_key_values=check,use_cache=True,logits_to_keep=1)
        torch.testing.assert_close(reference,out.logits[0,-1].cpu(),rtol=0,atol=0)
    return ids,calls


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','evaluate','summarize'),required=True)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--smoke-index',type=int,choices=(0,1),default=0)
    for name,path in {
        'data-dir':'results/datasets/longbench_c1_32k','c1-results':'results/evaluation/longbench_c1_32k',
        'dense-results':'results/evaluation/longbench_dense_32k','full-results':'results/evaluation/longbench_c1_v96',
        'c1-checkpoint':'results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6',
        'bank':'results/checkpoints/c1_v96_b16r8_qgram','output-dir':'results/evaluation/longbench_c1_sparse_fp16',
    }.items(): p.add_argument('--'+name,type=Path,default=ROOT/path)
    args=p.parse_args(); args.num_shards=4
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    rows,tokens,_,_,_,bank,provenance,scorer=source.inputs(args)
    baseline=[]; hashes={}; baseline_protocol=None
    for row in rows:
        path=ROOT/'results/evaluation/longbench_lrqk_fp16/full/evaluate'/f"sample_{row['index']:03d}.json"
        saved=json.loads(path.read_text())
        if baseline_protocol is None: baseline_protocol=saved['protocol']
        assert saved['status']=='complete' and saved['arm']=='full' and saved['protocol']==baseline_protocol
        assert saved['result']['sample']==row
        baseline.append(saved['result']); hashes[str(row['index'])]=sha256(path)
    assert baseline_protocol['source']==provenance['full_protocol']
    settings=dict(format='basisserve.c1_sparse_fp16.v1',bank_provenance=provenance,full_fp16_protocol=baseline_protocol,
        full_record_sha256=hashes,dtype='float16',gpu='V100',routing_dtype='float16; page mass normalization FP32',
        prefill='same explicit C1-V96 CUTLASS memory-efficient SDPA as FP16 LRQK/full-K',
        decode='native Base16/R8 Page32 B2048; pinned page0; all36 layers; no adaptive budget',
        storage='resident exact K/C1-V96 and materialized Base128+R8; accuracy reference, not offload timing',
        code_sha256={n:sha256(ROOT/n) for n in ('evaluation/eval_longbench_c1_sparse_fp16.py',
            'evaluation/eval_longbench_lrqk_fp16.py')})
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if args.stage=='summarize':
        records=[]
        config_eos=json.loads((args.model/'config.json').read_text())['eos_token_id']
        eos=set(config_eos if isinstance(config_eos,list) else [config_eos])|{tokenizer.eos_token_id}
        for row,full in zip(rows,baseline,strict=True):
            saved=json.loads((args.output_dir/'evaluate'/f"sample_{row['index']:03d}.json").read_text())
            assert saved['status']=='complete' and saved['protocol']==settings
            r=saved['result']; ids=r['generated_token_ids']
            assert r['sample']==row and ids[0]==full['generated_token_ids'][0]
            assert 0<len(ids)==r['generated_tokens']<=row['maximum_tokens']
            assert r['stopped_on_eos']==(ids[-1] in eos) and not any(t in eos for t in ids[:-1])
            assert r['stopped_on_eos'] or len(ids)==row['maximum_tokens']
            assert r['sparse_calls']==36*(len(ids)-1)
            assert r['prediction']==tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            assert r['score']==source.score_prediction(scorer,row['task'],r['prediction'],row['answers'],row['all_classes'])
            records.append(r)
        for shard in range(4):
            saved=json.loads((args.output_dir/'evaluate'/f'shard_{shard}.json').read_text())
            assert saved['status']=='complete' and saved['protocol']==settings and saved['indices']==list(range(shard,192,4))
        tasks={t:{a:100*sum(r['score'] for r in rs if r['sample']['task']==t)/32
                  for a,rs in [('full',baseline),('base16_r8',records)]} for t in source.TASKS}
        means={a:sum(t[a] for t in tasks.values())/len(tasks) for a in ('full','base16_r8')}
        write_json(args.output_dir/'result.json',dict(status='complete',protocol=settings,tasks=tasks,means=means,records=records))
        write_json(args.output_dir/'audit.json',dict(status='complete',verified=192,result_sha256=sha256(args.output_dir/'result.json')))
        lines=['# V100 FP16 C1-V96 Base16/R8','',
            'Same192 prompts; same FP16 C1-V96 memory-efficient SDPA prefill as LRQK and full-K.',
            'Page32/shared B2048, pinned page0, uniform Base16/R8, all36 layers; no refit.',
            'Native FP16 routing; LRQK comparison uses FP32 routing state to avoid confirmed FP16 overflow.',
            'Accuracy reference, not a matched-kernel speed benchmark.','',
            '| Task | Full K | Base16/R8 |','|---|---:|---:|']
        for t,v in [*tasks.items(),('Mean',means)]: lines.append('| '+t+' | '+' | '.join(f'{x:.4f}' for x in v.values())+' |')
        (args.output_dir/'summary.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(means),flush=True)
        return
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0)
    assert 0<=args.shard_index<4
    smoke=args.stage=='smoke'
    if smoke: print('prefill_error',kernel_check(),flush=True)
    attention.compressed_v_prefill_attention=v100_prefill
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float16,local_files_only=True,
        low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
    attention.install_qwen3_gqa_vo_als_export(model,args.c1_checkpoint,attention_backend='triton')
    assigned=([min(rows,key=lambda r:r['prompt_tokens']),max(rows,key=lambda r:r['prompt_tokens'])][args.smoke_index:args.smoke_index+1]
              if smoke else rows[args.shard_index::4])
    for row in assigned:
        path=args.output_dir/args.stage/f"sample_{row['index']:03d}.json"
        if path.exists():
            saved=json.loads(path.read_text()); assert saved['status']=='complete' and saved['protocol']==settings
            continue
        started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
        ids,calls=generate(model,tokenizer,bank,tokens[f"sample_{row['index']:03d}"],
            min(8,row['maximum_tokens']) if smoke else row['maximum_tokens'],smoke)
        assert ids[0]==baseline[row['index']]['generated_token_ids'][0]
        prediction=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        r=dict(sample=row,prediction=prediction,generated_token_ids=ids,generated_tokens=len(ids),
            stopped_on_eos=ids[-1] in _eos_ids(tokenizer,model),sparse_calls=calls,
            score=None if smoke else source.score_prediction(scorer,row['task'],prediction,row['answers'],row['all_classes']),
            elapsed_seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(path,dict(status='complete',protocol=settings,result=r,command=shlex.join(sys.argv),
            python=sys.executable,gpu=torch.cuda.get_device_name(0)))
        print(f"sample={row['index']} tokens={len(ids)} seconds={r['elapsed_seconds']:.2f}",flush=True)
    label=f'smoke_{args.smoke_index}' if smoke else f'shard_{args.shard_index}'
    write_json(args.output_dir/args.stage/(label+'.json'),dict(status='complete',protocol=settings,
        indices=[r['index'] for r in assigned]))


if __name__=='__main__': main()
