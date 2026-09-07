"""V100 FP16 pilot: matched full-K and LRQK arms; separate from BF16 artifacts."""
import argparse
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basisserve.checkpoint.gqa_vo_qwen3 import install_qwen3_gqa_vo_als_export
from basisserve.checkpoint import gqa_vo_qwen3 as attention_impl
from basisserve.checkpoint import c1_lrqk_qwen3 as lrqk_impl
from basisserve.checkpoint.c1_lrqk_qwen3 import C1LRQKCache, install_c1_lrqk
from basisserve.core.c1_lrqk import LRQKConfig
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from evaluation.eval_longbench_c1_v96 import inputs
from evaluation.eval_longbench_c1_fourarm import score_prediction
from evaluation.eval_qwen3_8b_residual_rank_ruler import greedy_decode, _eos_ids
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from evaluation.prepare_longbench_c1 import TASKS


def v100_prefill(q,k,v,*,scale=None):
    """Explicit CUTLASS memory-efficient backend; no quadratic math fallback."""
    assert q.dtype==k.dtype==v.dtype==torch.float16 and q.shape[-2]==k.shape[-2]
    groups=q.shape[1]//k.shape[1]
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(q,k.repeat_interleave(groups,1),v.repeat_interleave(groups,1),
            is_causal=True,scale=scale)


def kernel_check():
    torch.manual_seed(0)
    q = torch.randn(1, 8, 129, 128, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 2, 129, 128, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 2, 129, 96, device='cuda', dtype=torch.float16)
    actual = v100_prefill(q, k, v)
    scores = q.float() @ k.float().repeat_interleave(4, 1).transpose(-1, -2) / 128**.5
    scores.masked_fill_(~torch.ones(129,129,device='cuda',dtype=torch.bool).tril(), -torch.inf)
    expected = scores.softmax(-1) @ v.float().repeat_interleave(4, 1)
    assert torch.isfinite(actual).all()
    error = (actual.float()-expected).norm()/expected.norm()
    assert error < .005
    return float(error)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--stage', choices=('smoke','evaluate','summarize'), required=True)
    p.add_argument('--arm', choices=('full','k1152','k1280'), default='k1152')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--smoke-index', type=int, choices=(0,1), default=0)
    for name, path in {
        'data-dir':'results/datasets/longbench_c1_32k',
        'c1-results':'results/evaluation/longbench_c1_32k',
        'dense-results':'results/evaluation/longbench_dense_32k',
        'c1-checkpoint':'results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6',
        'output-dir':'results/evaluation/longbench_lrqk_fp16',
    }.items():
        p.add_argument('--'+name,type=Path,default=ROOT/path)
    args = p.parse_args()
    args.num_shards = 4
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows,tokens,_,_,source,scorer = inputs(args)
    protocol = dict(format='basisserve.lrqk_fp16.v1',source=source,dtype='float16',gpu='V100',
        prefill='full causal C1-V96 FP16 CUTLASS memory-efficient SDPA, explicit repeated GQA K/V',
        decode='full-K SDPA or LRQK exact selected K/C1-V96',
        rank=32,recent=64,iterations=[2,2],seed=0,
        deviations='resident cache; aligned previous-active K/AK; exact recent suffix; not upstream CPU ring cache',
        code_sha256={n:sha256(ROOT/n) for n in (
            'evaluation/eval_longbench_lrqk_fp16.py','basisserve/core/c1_lrqk.py',
            'basisserve/checkpoint/c1_lrqk_qwen3.py')})
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if args.stage == 'summarize':
        all_records = {}
        for arm in ('full','k1152','k1280'):
            records = []
            for row in rows:
                saved = json.loads((args.output_dir/arm/'evaluate'/f"sample_{row['index']:03d}.json").read_text())
                assert saved['status']=='complete' and saved['protocol']==protocol and saved['arm']==arm
                r = saved['result']; ids = r['generated_token_ids']
                assert r['sample']==row and 0<len(ids)<=row['maximum_tokens']
                assert r['prediction']==tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                assert r['score']==score_prediction(scorer,row['task'],r['prediction'],row['answers'],row['all_classes'])
                assert r['stopped_on_eos'] or len(ids)==row['maximum_tokens']
                records.append(r)
            for shard in range(4):
                done=json.loads((args.output_dir/arm/'evaluate'/f'shard_{shard}.json').read_text())
                assert done['protocol']==protocol and done['indices']==list(range(shard,192,4))
            all_records[arm]=records
        for arm in ('k1152','k1280'):
            assert all(a['generated_token_ids'][0]==b['generated_token_ids'][0]
                       for a,b in zip(all_records['full'],all_records[arm],strict=True))
        tasks={t:{a:100*sum(r['score'] for r in records if r['sample']['task']==t)/32
                  for a,records in all_records.items()} for t in TASKS}
        means={a:sum(t[a] for t in tasks.values())/len(TASKS) for a in all_records}
        budgets={}
        for arm,records in all_records.items():
            u=torch.tensor([v for r in records for s in r['routing']
                            for group in s['physical_union_per_kv_group'] for v in group],dtype=torch.float64)
            if u.numel():
                budgets[arm]=dict(scope='Final decode step per prompt, all36 layers/8 KV groups; not all steps',
                    mean=u.mean().item(),minimum=u.min().item(),maximum=u.max().item(),
                    p50=u.quantile(.5).item(),p90=u.quantile(.9).item(),p95=u.quantile(.95).item(),
                    fraction_above_2048=(u>2048).double().mean().item())
        write_json(args.output_dir/'result.json',dict(status='complete',protocol=protocol,tasks=tasks,
            means=means,physical_budget=budgets,records=all_records))
        write_json(args.output_dir/'audit.json',dict(status='complete',verified=576,
            result_sha256=sha256(args.output_dir/'result.json')))
        lines=['# V100 FP16 LRQK LongBench pilot','',
            'Same192 prompts, six tasks x32. C1-V96 FP16 memory-efficient SDPA prefill; no dense prefill.',
            'FP16 full-K is the matched reference. BF16 results are a different precision/hardware experiment.',
            'LRQK: rank32 per query head, top-k plus recent64, no shared physical budget cap.',
            'Adapted resident-cache equations, not official CPU/ring-cache reproduction.','',
            '| Task | Full K | k1152 | k1280 |','|---|---:|---:|---:|']
        for task,values in [*tasks.items(),('Mean',means)]:
            lines.append('| '+task+' | '+' | '.join(f'{x:.4f}' for x in values.values())+' |')
        lines+=['','## Physical token union','','```json',json.dumps(budgets,indent=2),'```','']
        (args.output_dir/'summary.md').write_text('\n'.join(lines))
        print(json.dumps(means),flush=True)
        return
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0)
    assert 0<=args.shard_index<4
    smoke=args.stage=='smoke'
    kernel_error=kernel_check() if smoke else None
    # Process-local dispatch only; original L40S source and jobs remain unchanged.
    attention_impl.compressed_v_prefill_attention=v100_prefill
    lrqk_impl.compressed_v_prefill_attention=v100_prefill
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float16,local_files_only=True,
        low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda').eval()
    install_qwen3_gqa_vo_als_export(model,args.c1_checkpoint,attention_backend='triton')
    modules=[l.self_attn for l in model.model.layers]
    sparse=args.arm!='full'
    assigned=([min(rows,key=lambda r:r['prompt_tokens']),max(rows,key=lambda r:r['prompt_tokens'])][args.smoke_index:args.smoke_index+1]
              if smoke else rows[args.shard_index::4])
    reference=None
    if smoke:
        row=assigned[0]; cache=RoutingDynamicCache()
        out=model(input_ids=tokens[f"sample_{row['index']:03d}"].long()[None].cuda(),
                  past_key_values=cache,use_cache=True,logits_to_keep=1)
        reference=out.logits[0,-1].cpu()
        assert torch.isfinite(reference).all()
        del out,cache
    if sparse:
        install_c1_lrqk(model,LRQKConfig(rank=32,topk=int(args.arm[1:]),recent=64))
    for row in assigned:
        path=args.output_dir/args.arm/args.stage/f"sample_{row['index']:03d}.json"
        if path.exists():
            saved=json.loads(path.read_text())
            assert saved['status']=='complete' and saved['protocol']==protocol
            continue
        started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
        cache=C1LRQKCache(config=model.config) if sparse else RoutingDynamicCache()
        for module in modules: module.attention_backend='triton'
        out=model(input_ids=tokens[f"sample_{row['index']:03d}"].long()[None].cuda(),
                  past_key_values=cache,use_cache=True,logits_to_keep=1)
        logits=out.logits[0,-1]
        assert torch.isfinite(logits).all()
        if smoke: torch.testing.assert_close(logits.cpu(),reference,rtol=0,atol=0)
        first=int(logits.argmax()); del out,logits
        if not sparse:
            for module in modules: module.attention_backend='sdpa'
        maximum=min(8,row['maximum_tokens']) if smoke else row['maximum_tokens']
        ids,cache,_=greedy_decode(model,cache,first,maximum_tokens=maximum,eos_ids=_eos_ids(tokenizer,model))
        length=row['prompt_tokens']+len(ids)-1
        for layer in cache.layers:
            assert layer.keys.shape==(1,8,length,128) and layer.values.shape==(1,8,length,96)
            assert layer.keys.dtype==layer.values.dtype==torch.float16
        stats=[cache.lrqk_states[l].statistics(8) for l in range(36)] if sparse else []
        assert all(s['length']==length and s['decode_steps']==len(ids)-1 for s in stats)
        prediction=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        result=dict(sample=row,prediction=prediction,generated_token_ids=ids,generated_tokens=len(ids),
            score=None if smoke else score_prediction(scorer,row['task'],prediction,row['answers'],row['all_classes']),
            stopped_on_eos=ids[-1] in _eos_ids(tokenizer,model),routing=stats,
            elapsed_seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            smoke_prefill_logits_equal=smoke,kernel_smoke_relative_error=kernel_error)
        write_json(path,dict(status='complete',arm=args.arm,protocol=protocol,result=result,
            command=shlex.join(sys.argv),python=sys.executable,gpu=torch.cuda.get_device_name(0)))
        print(f"arm={args.arm} sample={row['index']} tokens={len(ids)} seconds={result['elapsed_seconds']:.2f}",flush=True)
        del cache
    label=f'smoke_{args.smoke_index}' if smoke else f'shard_{args.shard_index}'
    write_json(args.output_dir/args.arm/args.stage/(label+'.json'),dict(status='complete',protocol=protocol,
        indices=[r['index'] for r in assigned]))


if __name__=='__main__':
    main()
