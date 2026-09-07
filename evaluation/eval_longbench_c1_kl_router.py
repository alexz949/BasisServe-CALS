"""Dense prefill then allocated C1 + matched Base16/R8 sparse decode."""
import argparse
import json
from pathlib import Path
import shlex
import sys
import time
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from evaluation.eval_longbench_c1_twosided_denseprefill import inputs as full_inputs, install
from evaluation.eval_longbench_c1_denseprefill import activate
from evaluation.eval_longbench_dense import generate as dense_generate
from evaluation.eval_longbench_c1_fourarm import score_prediction
from evaluation.eval_qwen3_8b_residual_rank_ruler import prepare_arm,greedy_decode,_eos_ids
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256,write_json
from evaluation.prepare_longbench_c1 import TASKS
from basisserve.checkpoint.gqa_vo_qwen3 import transition_qwen3_dense_prefill_cache_to_c1
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.c1_conditional_page_attention import c1_conditional_page_topk_attention
from basisserve.core.residual_kl_replay import prefix_signature


def inputs(args):
    rows,tokens,dense,uniform,manifest,full_settings,scorer=full_inputs(args)
    full=json.loads((args.full_results/'result.json').read_text())
    audit=json.loads((args.full_results/'audit.json').read_text())
    assert full['status']==audit['status']=='complete' and full['protocol']==full_settings
    assert sha256(args.full_results/'result.json')==audit['result_sha256']
    banks=[]; bank_hashes={}; source=None
    for l,rank in enumerate(full_settings['layer_ranks']):
        path=args.bank/f'layer_{l:03d}.safetensors'
        record=json.loads(path.with_suffix('.json').read_text())
        if source is None: source=record['protocol']
        assert record['status']=='complete' and record['layer']==l and record['protocol']==source
        assert record['sha256']==sha256(path)
        tensors=load_file(str(path))
        expected={'base_left_b16':(8,rank,16),'base_right_b16':(8,16,128),'base_bias_b16':(8,128),
            'residual_encoder_b16_r8':(8,128,8),'residual_query_b16_r8':(32,128,8)}
        assert set(tensors)==set(expected)
        for name,t in tensors.items():
            assert tuple(t.shape)==expected[name] and t.dtype==torch.float32 and torch.isfinite(t).all()
        banks.append(tensors);bank_hashes[str(l)]=record['sha256']
    assert source['allocation_manifest_sha256']==full_settings['manifest_sha256']
    assert source['payload_ranks']==full_settings['layer_ranks']
    assert source['base_rank']==16 and source['residual_rank']==8 and source['query_count']==32
    assert source['page_size']==32 and source['excluded_prefix_pages']==1 and source['physical_token_budget']==2048
    assert source['fit_windows']==64 and source['diagnostic_windows']==16
    settings=dict(format='basisserve.longbench_allocated_c1_router.v1',full_protocol=full_settings,
        full_result_sha256=audit['result_sha256'],bank=str(args.bank.resolve()),bank_sha256=bank_hashes,
        bank_protocol=source,page_size=32,physical_token_budget=2048,pinned_prefix_pages=1,
        base_rank=16,residual_rank=8,adaptive_budget=False,force_current_page=False,
        prefill='original dense K128/V128 SDPA, then allocated C1 V-cache conversion',
        decode='all36 layers native BF16 Base16+R8 page routing and selected exact-K/C1 latent attention',
        storage='GPU-resident exact K; materialized post-RoPE Base128+R8 sidecar; accuracy oracle, not offload/latency benchmark',
        code_sha256={n:sha256(ROOT/n) for n in (
            'evaluation/eval_longbench_c1_kl_router.py','evaluation/eval_qwen3_8b_residual_rank_ruler.py',
            'evaluation/profile_qwen3_8b_residual_two_sided_kl.py','basisserve/core/c1_conditional_page_attention.py',
            'basisserve/core/c1_v_conditional_k_router.py','basisserve/checkpoint/gqa_vo_qwen3.py')})
    return rows,tokens,dense,uniform,full,manifest,banks,settings,scorer


@torch.inference_mode()
def generate(model,tokenizer,original,compressed,bank,tokens,cap,trace=False):
    activate(model,original)
    prefix=RoutingDynamicCache()
    out=model(input_ids=tokens.long()[None].to('cuda:0'),past_key_values=prefix,use_cache=True,logits_to_keep=1)
    logits=out.logits[0,-1]
    assert torch.isfinite(logits).all()
    first=int(logits.argmax());first_logits=logits.cpu() if trace else None
    prefix=out.past_key_values;del out,logits
    keys=[l.keys for l in prefix.layers]
    assert all(l.values.shape==(1,8,len(tokens),128) for l in prefix.layers)
    activate(model,compressed)
    transition_qwen3_dense_prefill_cache_to_c1(model,prefix,attention_backend='sdpa')
    for l,(cache,attention) in enumerate(zip(prefix.layers,compressed,strict=True)):
        assert cache.keys is keys[l]
        assert cache.values.shape==(1,8,len(tokens),attention.value_coordinate_encoder.shape[-1])
    del keys
    signature=prefix_signature(prefix)
    cache=prepare_arm(model,bank,prefix,[8]*36)
    calls=0
    def observe(*args,**kwargs):
        nonlocal calls
        calls+=1
        return c1_conditional_page_topk_attention(*args,**kwargs)
    with patch('basisserve.checkpoint.gqa_vo_qwen3.c1_conditional_page_topk_attention',new=observe):
        ids,cache,traces=greedy_decode(model,cache,first,maximum_tokens=cap,eos_ids=_eos_ids(tokenizer,model),trace=trace)
    assert calls==36*(len(ids)-1)
    assert prefix_signature(prefix)==signature
    assert cache.get_seq_length()==len(tokens)+len(ids)-1
    for l,(layer,attention) in enumerate(zip(cache.layers,compressed,strict=True)):
        assert layer.keys.shape==(1,8,cache.get_seq_length(),128)
        assert layer.values.shape==(1,8,cache.get_seq_length(),attention.value_coordinate_encoder.shape[-1])
        assert cache.routing_sidecar(l).shape==(1,8,cache.get_seq_length(),136)
        assert layer.keys.dtype==layer.values.dtype==cache.routing_sidecar(l).dtype==torch.bfloat16
    torch.cuda.synchronize()
    return ids,([first_logits]+traces if trace else []),calls


def summarize(args,rows,dense,uniform,full,settings,scorer,tokenizer):
    model_eos=json.loads((args.model/'config.json').read_text())['eos_token_id']
    eos=set(model_eos if isinstance(model_eos,list) else [model_eos])|{tokenizer.eos_token_id}
    records=[]
    for row,d in zip(rows,dense['records'],strict=True):
        saved=json.loads((args.output_dir/'evaluate'/f"sample_{row['index']:03d}.json").read_text())
        assert saved['status']=='complete' and saved['protocol']==settings
        r=saved['result'];ids=r['generated_token_ids']
        assert r['sample']==row and r['cache_verified'] and r['first_token_matches_dense']
        assert ids[0]==d['generated_token_ids'][0]
        assert 0<len(ids)==r['generated_tokens']<=r['maximum_tokens']==row['maximum_tokens']
        assert r['native_sparse_calls']==36*(len(ids)-1)
        assert r['stopped_on_eos']==(ids[-1] in eos) and not any(i in eos for i in ids[:-1])
        assert r['stopped_on_eos'] or len(ids)==row['maximum_tokens']
        assert r['prediction']==tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        assert r['score']==score_prediction(scorer,row['task'],r['prediction'],row['answers'],row['all_classes'])
        records.append(r)
    for shard in range(args.num_shards):
        saved=json.loads((args.output_dir/'evaluate'/f'shard_{shard}.json').read_text())
        assert saved['status']=='complete' and saved['protocol']==settings
        assert saved['indices']==list(range(shard,192,args.num_shards))
    tasks={}
    for task in TASKS:
        rs=[r for r in records if r['sample']['task']==task];assert len(rs)==32
        mean=100*sum(r['score'] for r in rs)/32
        assert round(mean,2)==scorer.scorer(task,[r['prediction'] for r in rs],[r['sample']['answers'] for r in rs],rs[0]['sample']['all_classes'])
        tasks[task]=dict(dense=dense['tasks'][task]['dense_k_dense_v'],
            uniform_full=uniform['tasks'][task]['dense_prefill_c1_decode'],
            kl_full=full['tasks'][task]['two_sided80'],kl_b16r8=mean)
    means={a:sum(t[a] for t in tasks.values())/len(TASKS) for a in next(iter(tasks.values()))}
    paired={}
    for name,baseline in [('dense',dense),('uniform_full',uniform),('kl_full',full)]:
        delta=[r['score']-b['score'] for r,b in zip(records,baseline['records'],strict=True)]
        paired[name]=dict(improvements=sum(d>0 for d in delta),regressions=sum(d<0 for d in delta),
            ties=sum(d==0 for d in delta),mean_delta_pp=100*sum(delta)/len(delta))
    result=dict(status='complete',protocol=settings,tasks=tasks,means=means,paired=paired,records=records,
        first_token_matches_dense=sum(r['first_token_matches_dense'] for r in records),
        cap_without_eos=sum(not r['stopped_on_eos'] for r in records),
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in records),command=shlex.join(sys.argv),python=sys.executable)
    write_json(args.output_dir/'result.json',result)
    write_json(args.output_dir/'audit.json',dict(status='complete',predictions_verified=192,
        first_tokens_match_dense=True,native_sparse_dispatch_verified=True,actual_rank_caches_verified=True,
        official_scores_verified=True,shard_coverage_verified=True,result_sha256=sha256(args.output_dir/'result.json')))
    lines=['# LongBench: two-sided KL C1 with matched Base16 + residual R8','',
        'Qwen3-8B-Base, basis, BF16, four L40S workers. All arms use original dense prefill. Frozen 192 prompts, six tasks with32 prompts each; official QA F1 / summary ROUGE-L on0–100 scale. Six-task arithmetic mean. Not full LongBench.','',
        '| Task | Dense | Uniform80 full K | KL avg80 full K | KL avg80 Base16/R8 sparse |','|---|---:|---:|---:|---:|']
    for task,values in [*tasks.items(),('Mean',means)]: lines.append('| '+task+' | '+' | '.join(f'{v:.4f}' for v in values.values())+' |')
    lines+=['','C1 is the unchanged alpha1 two-sided-KL allocation, average rank80. Dense V128 cache is projected into each layer\'s actual C1 coordinates after prefill. First token is dense argmax.',
        'Base16 was refitted by closed-form affine MSE reduced-rank regression in the selected C1 coordinates. Residual R8 was refitted with non-sink Page-Fisher,40 BCD sweeps and PCG. No Adam or model-weight training.',
        'Router calibration: existing C4 captures,64 x32K fit and16 x32K diagnostic. Existing Query-Gram Q32 positions,8 per8K stratum; positions and C1 factors are not reselected. Diagnostic windows do not select the final factors.',
        'Sparse decode on all36 layers: Page32, physical token budget2048 (64 pages), pinned prefix page0, no adaptive budget or forced current page. Exact K and resident C1 values are used within selected support.',
        'Same prompts, greedy decoding, EOS and task caps as full-K controls. Input-plus-generation cap32K; actual prompts1,192–30,431 tokens.',
        'Accuracy oracle: GPU-resident exact K and materialized Base128+R8 sidecar, not actual CPU offload or a latency benchmark. Old uniform80 router weights were NOT attached to incompatible allocated coordinates.',
        'The old uniform versus KL checkpoint qualification still applies: KL export includes encoder gauge canonicalization and decoder closure. The KL sparse/full comparison uses exactly the same exported payload.',
        '',f"First-token agreement: {result['first_token_matches_dense']}/192.",f"Cap exits without EOS: {result['cap_without_eos']}/192.",
        f"Peak allocated memory: {result['peak_allocated_gib']:.3f} GiB.",'','## Paired score changes','','```json',json.dumps(paired,indent=2),'```','']
    (args.output_dir/'summary.md').write_text('\n'.join(lines))
    print(json.dumps({k:result[k] for k in ('means','paired','first_token_matches_dense','cap_without_eos')},indent=2),flush=True)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('smoke','evaluate','summarize'),required=True)
    p.add_argument('--model',type=Path,required=True)
    for name,path in {
        'data-dir':'results/datasets/longbench_c1_32k','c1-results':'results/evaluation/longbench_c1_32k',
        'dense-results':'results/evaluation/longbench_dense_32k','uniform-results':'results/evaluation/longbench_c1_denseprefill_32k',
        'full-results':'results/evaluation/longbench_c1_kl_denseprefill_32k',
        'c1-checkpoint':'results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6',
        'allocation':'results/checkpoints/qwen3_8b_c1_twosided_r80_c4_32x32k',
        'bank':'results/checkpoints/c1_kl_b16r8_qgram','output-dir':'results/evaluation/longbench_c1_kl_b16r8',
    }.items():p.add_argument('--'+name,type=Path,default=ROOT/path)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=4)
    args=p.parse_args();assert 0<=args.shard_index<args.num_shards
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    rows,tokens,dense,uniform,full,manifest,bank,settings,scorer=inputs(args)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if args.stage=='summarize':
        summarize(args,rows,dense,uniform,full,settings,scorer,tokenizer);return
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA L40S'
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,local_files_only=True,
        low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda:0').eval()
    original,compressed=install(model,args.allocation,manifest)
    smoke=args.stage=='smoke'
    assigned=([min(rows,key=lambda r:r['prompt_tokens']),max(rows,key=lambda r:r['prompt_tokens'])]
              if smoke else rows[args.shard_index::args.num_shards])
    for row in assigned:
        path=args.output_dir/args.stage/f"sample_{row['index']:03d}.json"
        if path.exists():
            saved=json.loads(path.read_text())
            assert saved['status']=='complete' and saved['protocol']==settings and saved['result']['sample']==row
            continue
        tensor=tokens[f"sample_{row['index']:03d}"];cap=4 if smoke else row['maximum_tokens']
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        ids,trace,calls=generate(model,tokenizer,original,compressed,bank,tensor,cap,smoke)
        assert ids[0]==dense['records'][row['index']]['generated_token_ids'][0]
        if smoke:
            repeated,other,_=generate(model,tokenizer,original,compressed,bank,tensor,cap,True)
            assert repeated==ids
            for a,b in zip(trace,other,strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)
            activate(model,original)
            _,dense_trace,_=dense_generate(model,tokenizer,tensor,cap,trace=True)
            torch.testing.assert_close(trace[0],dense_trace[0],rtol=0,atol=0)
        prediction=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        score=None if smoke else score_prediction(scorer,row['task'],prediction,row['answers'],row['all_classes'])
        result=dict(sample=row,prediction=prediction,score=score,generated_token_ids=ids,generated_tokens=len(ids),
            maximum_tokens=cap,stopped_on_eos=ids[-1] in _eos_ids(tokenizer,model),native_sparse_calls=calls,
            first_token_matches_dense=True,cache_verified=True,smoke_repeated_and_dense_logits_verified=smoke,
            elapsed_seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(path,dict(status='complete',protocol=settings,result=result,command=shlex.join(sys.argv),
            python=sys.executable,gpu=torch.cuda.get_device_name(0)))
        print(f"sample={row['index']} task={row['task']} score={score} tokens={len(ids)} seconds={result['elapsed_seconds']:.2f}",flush=True)
    write_json(args.output_dir/args.stage/f'shard_{args.shard_index}.json',dict(status='complete',
        protocol=settings,indices=[r['index'] for r in assigned]))


if __name__=='__main__':main()
