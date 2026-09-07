"""LRQK-equation token routing + frozen C1-V96 on the existing LongBench pilot."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from basisserve.checkpoint.gqa_vo_qwen3 import install_qwen3_gqa_vo_als_export
from basisserve.checkpoint.c1_lrqk_qwen3 import C1LRQKCache,install_c1_lrqk
from basisserve.core.c1_lrqk import LRQKConfig
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from evaluation.eval_longbench_c1_v96 import inputs as full_inputs
from evaluation.eval_longbench_c1_fourarm import score_prediction
from evaluation.eval_qwen3_8b_residual_rank_ruler import greedy_decode,_eos_ids
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256,write_json
from evaluation.prepare_longbench_c1 import TASKS


def summarize(args,rows,full,settings,scorer,tokenizer):
    records = []
    eos_config = json.loads((args.model/'config.json').read_text())['eos_token_id']
    eos = set(eos_config if isinstance(eos_config,list) else [eos_config]) | {tokenizer.eos_token_id}
    for row,old in zip(rows,full['records'],strict=True):
        saved = json.loads((args.output_dir/'evaluate'/f"sample_{row['index']:03d}.json").read_text())
        assert saved['status'] == 'complete' and saved['protocol'] == settings
        r = saved['result']; ids = r['generated_token_ids']
        assert r['sample'] == row and ids[0] == old['generated_token_ids'][0]
        assert 0 < len(ids) == r['generated_tokens'] <= row['maximum_tokens'] == r['maximum_tokens']
        assert r['stopped_on_eos'] == (ids[-1] in eos) and not any(t in eos for t in ids[:-1])
        assert r['stopped_on_eos'] or len(ids) == row['maximum_tokens']
        assert r['prediction'] == tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        assert r['score'] == score_prediction(scorer,row['task'],r['prediction'],row['answers'],row['all_classes'])
        assert len(r['routing']) == 36 and all(s['decode_steps'] == len(ids)-1 for s in r['routing'])
        records.append(r)
    for shard in range(args.num_shards):
        saved = json.loads((args.output_dir/'evaluate'/f'shard_{shard}.json').read_text())
        assert saved['status'] == 'complete' and saved['protocol'] == settings
        assert saved['indices'] == list(range(shard,len(rows),args.num_shards))
    tasks = {}
    for task in TASKS:
        subset = [r for r in records if r['sample']['task'] == task]
        mean = 100*sum(r['score'] for r in subset)/len(subset)
        assert len(subset) == 32 and round(mean,2) == scorer.scorer(task,
            [r['prediction'] for r in subset],[r['sample']['answers'] for r in subset],subset[0]['sample']['all_classes'])
        tasks[task] = dict(v96_full=full['tasks'][task]['c1_v96'],lrqk_c1_v96=mean)
    means = {a:sum(t[a] for t in tasks.values())/len(tasks) for a in ('v96_full','lrqk_c1_v96')}
    unions = torch.tensor([v for r in records for s in r['routing']
                           for group in s['physical_union_per_kv_group'] for v in group],dtype=torch.float64)
    budget = dict(scope='Final decode step per sample, all layers and KV groups; not all-step average',
        count=unions.numel(),mean=unions.mean().item(),minimum=unions.min().item(),
        maximum=unions.max().item(),p50=unions.quantile(.5).item(),p90=unions.quantile(.9).item(),
        p95=unions.quantile(.95).item(),fraction_above_2048=(unions>2048).double().mean().item())
    write_json(args.output_dir/'result.json',dict(status='complete',protocol=settings,tasks=tasks,means=means,
        physical_budget=budget,records=records,command=shlex.join(sys.argv),python=sys.executable))
    write_json(args.output_dir/'audit.json',dict(status='complete',predictions_verified=len(records),
        result_sha256=sha256(args.output_dir/'result.json')))
    lines = ['# LRQK-equation routing + C1-V96 LongBench','',
        'Same192 prompts, six tasks x32; C1-V96 prefill; exact selected K / resident C1-V96.',
        'Adapted resident-cache reference, not a bitwise reproduction of upstream CPU/ring-cache behavior.',
        f'Per-query-head token Top-{args.topk} + recent{args.recent}; NOT a GQA-shared2048-token or Page32 budget.',
        'QA F1 / summary ROUGE-L; six-task arithmetic mean. Not full LongBench or an offload speed benchmark.','',
        '| Task | V96 full K | LRQK + V96 |','|---|---:|---:|']
    for task,scores in [*tasks.items(),('Mean',means)]:
        lines.append('| '+task+' | '+' | '.join(f'{v:.4f}' for v in scores.values())+' |')
    lines += ['', '## Physical token union', '', budget['scope'], '',
              '```json', json.dumps(budget,indent=2), '```']
    lines += ['', 'Settings and caveats: docs/c1_lrqk_integration.md. Commands are preserved per sample.', '']
    (args.output_dir/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(means,indent=2),flush=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','evaluate','summarize'),required=True)
    for name,path in {
        'data-dir':'results/datasets/longbench_c1_32k','c1-results':'results/evaluation/longbench_c1_32k',
        'dense-results':'results/evaluation/longbench_dense_32k','full-results':'results/evaluation/longbench_c1_v96',
        'c1-checkpoint':'results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6',
        'output-dir':'results/evaluation/longbench_c1_lrqk',
    }.items():
        p.add_argument('--'+name,type=Path,default=ROOT/path)
    p.add_argument('--rank',type=int,default=32)
    p.add_argument('--topk',type=int,default=2048)
    p.add_argument('--recent',type=int,default=64)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=4)
    p.add_argument('--smoke-index',type=int,choices=(0,1),default=0)
    args = p.parse_args()
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    rows,tokens,previous,dense,full_settings,scorer = full_inputs(args)
    full = json.loads((args.full_results/'result.json').read_text())
    audit = json.loads((args.full_results/'audit.json').read_text())
    assert full['status'] == audit['status'] == 'complete' and full['protocol'] == full_settings
    assert sha256(args.full_results/'result.json') == audit['result_sha256']
    cfg = LRQKConfig(rank=args.rank,topk=args.topk,recent=args.recent,seed=args.seed)
    settings = dict(format='basisserve.c1_lrqk.v1',full_protocol=full_settings,lrqk=asdict(cfg),
        full_result_sha256=audit['result_sha256'],upstream_commit='caf16293db2e4423a84ab2e895bacf64479f1eb7',
        prefill='C1-V96 full causal Triton; LRQK online fit on actual post-RoPE Q/K',
        decode='upstream lambda1 factor equations; per-query-head token Top-k + recent; exact selected K/C1-V',
        deviations='resident cache; aligned previous-active K/AK for decode fit; exact recent suffix; no CPU hit/miss ring implementation',
        optimizer='FP32 alternating solves plus upstream analytic-gradient B updates; no Adam/autograd; not pure closed-form/BCD',
        budget='topk+recent per query head; physical GQA union recorded, not capped to topk',
        code_sha256={n:sha256(ROOT/n) for n in ('evaluation/eval_longbench_c1_lrqk.py',
            'basisserve/core/c1_lrqk.py','basisserve/checkpoint/c1_lrqk_qwen3.py')})
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if args.stage == 'summarize':
        summarize(args,rows,full,settings,scorer,tokenizer)
        return
    assert 0 <= args.shard_index < args.num_shards
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA L40S'
    model = AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
        local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda:0').eval()
    install_qwen3_gqa_vo_als_export(model,args.c1_checkpoint,attention_backend='triton')
    model.eval()
    smoke = args.stage == 'smoke'
    assigned = ([min(rows,key=lambda r:r['prompt_tokens']),max(rows,key=lambda r:r['prompt_tokens'])][args.smoke_index:args.smoke_index+1]
                if smoke else rows[args.shard_index::args.num_shards])
    reference_logits = None
    if smoke:
        row = assigned[0]
        prefix = RoutingDynamicCache()
        out = model(input_ids=tokens[f"sample_{row['index']:03d}"].long()[None].to('cuda:0'),
            past_key_values=prefix,use_cache=True,logits_to_keep=1)
        reference_logits = out.logits[0,-1].cpu()
        del out,prefix
    install_c1_lrqk(model,cfg)
    for row in assigned:
        path = args.output_dir/args.stage/f"sample_{row['index']:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['status'] == 'complete' and saved['protocol'] == settings
            continue
        maximum = min(8,row['maximum_tokens']) if smoke else row['maximum_tokens']
        torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
        cache = C1LRQKCache(config=model.config)
        out = model(input_ids=tokens[f"sample_{row['index']:03d}"].long()[None].to('cuda:0'),
            past_key_values=cache,use_cache=True,logits_to_keep=1)
        first_logits = out.logits[0,-1]
        assert torch.isfinite(first_logits).all()
        first = int(first_logits.argmax())
        assert first == full['records'][row['index']]['generated_token_ids'][0]
        if smoke:
            torch.testing.assert_close(first_logits.cpu(),reference_logits,rtol=0,atol=0)
        del out,first_logits
        ids,cache,_ = greedy_decode(model,cache,first,maximum_tokens=maximum,eos_ids=_eos_ids(tokenizer,model))
        assert len(cache.lrqk_states) == 36
        stats = [cache.lrqk_states[l].statistics(8) for l in range(36)]
        for l,s in enumerate(stats):
            length = row['prompt_tokens']+len(ids)-1
            assert s['length'] == length and s['decode_steps'] == len(ids)-1
            assert cache.layers[l].keys.shape == (1,8,length,128)
            assert cache.layers[l].values.shape == (1,8,length,96)
            assert s['selected_per_query_head'] == min(length,cfg.topk+cfg.recent)
        prediction = tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        score = None if smoke else score_prediction(scorer,row['task'],prediction,row['answers'],row['all_classes'])
        result = dict(sample=row,prediction=prediction,generated_token_ids=ids,generated_tokens=len(ids),
            maximum_tokens=maximum,stopped_on_eos=ids[-1] in _eos_ids(tokenizer,model),score=score,
            routing=stats,full_prefill_logits_equal=smoke,elapsed_seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(path,dict(status='complete',protocol=settings,result=result,command=shlex.join(sys.argv),
            python=sys.executable,gpu=torch.cuda.get_device_name(0)))
        print(f"sample={row['index']} score={score} tokens={len(ids)} seconds={result['elapsed_seconds']:.2f}",flush=True)
        del cache
    label = f'smoke_{args.smoke_index}' if smoke else f'shard_{args.shard_index}'
    write_json(args.output_dir/args.stage/(label+'.json'),dict(status='complete',protocol=settings,indices=[r['index'] for r in assigned]))


if __name__ == '__main__':
    main()
