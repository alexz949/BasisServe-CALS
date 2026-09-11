"""Fresh paired RULER pilot: HF KL96 payload, five BF16 attention arms."""
import argparse
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import MODEL, CHECKPOINT, BANK, configure, read_json, write_json, sha256, checkpoint_manifest
from evaluation.eval_v96kl_longbench import load_bank, generate as original_generate
from evaluation.eval_longbench_c1_twosided_denseprefill import install
from evaluation.eval_qwen3_8b_residual_rank_ruler import TASKS, _eos_ids
from evaluation.ruler_v1 import parse_tasks, ruler_prompt, sample_score
from basisserve.checkpoint import gqa_vo_qwen3 as attention
from basisserve.checkpoint import c1_lrqk_qwen3 as lrqk
from basisserve.core.c1_lrqk import LRQKConfig
from evaluation.eval_longbench_lrqk_fp32route import FP32RoutingState
from basisserve.checkpoint.c1_shadowkv_qwen3 import C1ShadowKVCache, install_c1_shadowkv
from basisserve.core.c1_conditional_recent_attention import c1_conditional_page_recent64_attention

ARMS = ('full', 'recent_extra', 'recent_fixed', 'lrqk', 'shadowkv')


def inputs(args, tokenizer):
    manifest = checkpoint_manifest(args.checkpoint, args.model)
    data = read_json(args.data/'manifest.json')
    tasks = parse_tasks(TASKS)
    assert data['status'] == 'complete'
    assert data['protocol'] == dict(model_template_type='base', sequence_length=32768,
                                   samples_per_task=8, random_seed=42, tasks=[t.name for t in tasks])
    rows = []
    for task in tasks:
        path = args.data/task.name/'validation.jsonl'
        assert sha256(path) == data['artifacts'][task.name]['sha256']
        sources = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(sources) == 8
        for ordinal, source in enumerate(sources):
            tokens = tokenizer(ruler_prompt(source), add_special_tokens=True)['input_ids']
            assert len(tokens)+task.tokens_to_generate <= 32768
            rows.append(dict(index=len(rows), task=task.name, ordinal=ordinal, answers=source['outputs'],
                             match_type=task.match_type, input_ids=tokens, maximum_tokens=task.tokens_to_generate))
    bank, hashes = load_bank(args)
    code = ['evaluation/eval_ruler_kl96.py', 'evaluation/eval_v96kl_longbench.py',
            'basisserve/core/c1_conditional_recent_attention.py', 'basisserve/core/c1_shadowkv.py',
            'basisserve/checkpoint/c1_shadowkv_qwen3.py', 'basisserve/core/c1_lrqk.py',
            'basisserve/checkpoint/c1_lrqk_qwen3.py', 'evaluation/ruler_v1.py',
            'basisserve/checkpoint/gqa_vo_qwen3.py', 'evaluation/eval_longbench_lrqk_fp32route.py']
    spec = dict(checkpoint_sha256=sha256(args.checkpoint/'manifest.json'), bank_sha256=hashes,
                data_sha256=sha256(args.data/'manifest.json'), ruler_revision=data['ruler']['revision'],
                dtype='bfloat16', generation='greedy, official caps and EOS, full C1 Triton prefill',
                seed=42, tasks=[t.name for t in tasks], samples_per_task=8,
                primary_excluded_indices=[86], primary_mean='sample weighted over 87',
                base_rank=16, residual_rank=16, query_count=32, page_size=32,
                recent_extra='page budget2048 including sink32, union sliding recent64; max2112',
                recent_fixed='sink32 + sliding recent64 + 61 disjoint historical pages; max2048',
                lrqk=dict(rank=32, topk=1152, recent=64, iterations=[2,2], seed=0, state_dtype='float32'),
                shadowkv=dict(rank=160, chunk=8, routed=2048, outlier_chunks=48,
                              local='32-39 prompt tokens plus generated tokens'),
                storage='resident accuracy comparison; no CPU offload timing',
                source_data_sha256={p.name:sha256(p) for p in
                    (ROOT/'results/tools/RULER/scripts/data/synthetic/json').glob('*.json')},
                code_sha256={p:sha256(ROOT/p) for p in code})
    return manifest, rows, bank, spec


@torch.inference_mode()
def shadow_generate(model, tokenizer, tokens, cap):
    cache = C1ShadowKVCache(model.config)
    out = model(input_ids=tokens[None].cuda(), past_key_values=cache, use_cache=True, logits_to_keep=1)
    logits = out.logits[0,-1]
    assert torch.isfinite(logits).all()
    first = logits.float().cpu(); ids = [int(logits.argmax())]
    del out, logits
    eos = _eos_ids(tokenizer, model)
    while len(ids)<cap and ids[-1] not in eos:
        mask = torch.ones(1,1,1,cache.get_seq_length()+1, device='cuda', dtype=torch.bool)
        out = model(input_ids=torch.tensor([[ids[-1]]],device='cuda'), past_key_values=cache,
                    use_cache=True, attention_mask={'full_attention':mask}, logits_to_keep=1)
        assert torch.isfinite(out.logits).all()
        ids.append(int(out.logits[0,-1].argmax()))
        del out
    stats = [cache.shadow_states[l].statistics() for l in range(36)]
    return ids, first, stats, ids[-1] in eos


def summarize(args, rows, spec, tokenizer):
    records = {}
    eos = read_json(args.model/'config.json')['eos_token_id']
    eos = set(eos if isinstance(eos,list) else [eos]) | {tokenizer.eos_token_id}
    for arm in ARMS:
        records[arm] = []
        for row in rows:
            saved = read_json(args.output/arm/'evaluate'/f"sample_{row['index']:03d}.json")
            assert saved['status']=='complete' and saved['protocol']==spec and saved['sample']==row
            r = saved['result']; ids = r['ids']
            assert 0<len(ids)<=row['maximum_tokens']
            assert r['stopped']==(ids[-1] in eos) and not any(i in eos for i in ids[:-1])
            assert r['stopped'] or len(ids)==row['maximum_tokens']
            assert r['prediction']==tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            assert r['score']==sample_score(r['prediction'],row['answers'],row['match_type'])
            records[arm].append(r)
        if arm!='full':
            assert all(a['ids'][0]==b['ids'][0] for a,b in zip(records['full'],records[arm],strict=True))
    means87 = {a:100*sum(r['score'] for i,r in enumerate(rs) if i!=86)/87 for a,rs in records.items()}
    means88 = {a:100*sum(r['score'] for r in rs)/88 for a,rs in records.items()}
    write_json(args.output/'result.json', dict(status='complete', protocol=spec, verified=440,
               means87=means87, means88=means88, first_token_agreement=True))
    lines = ['# Fresh RULER KL96 BF16 pilot', '', '11 tasks x8; primary excludes global index86. Seed42. Sample-weighted official scores.', '',
             '| Arm | 87 samples | All88 |','|---|---:|---:|']
    lines += [f'| {a} | {means87[a]:.6f} | {means88[a]:.6f} |' for a in ARMS]
    path=args.output/'summary.md'; text='\n'.join(lines)+'\n'
    if path.exists(): assert path.read_text()==text
    else: path.write_text(text)
    print('VERIFIED', means87, means88, flush=True)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['smoke','evaluate','summarize'])
    p.add_argument('--arm',choices=ARMS,default='full')
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=2)
    for name,default in dict(model=MODEL,checkpoint=CHECKPOINT,bank=BANK,
         data=ROOT/'results/datasets/ruler_kl96_seed42',output=ROOT/'results/evaluation/ruler_kl96_seed42').items():
        p.add_argument('--'+name,type=Path,default=default)
    args=p.parse_args(); configure()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    manifest,rows,bank,spec=inputs(args,tokenizer)
    if args.stage=='summarize': summarize(args,rows,spec,tokenizer); return
    torch.manual_seed(0)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
          local_files_only=True,attn_implementation='sdpa').cuda().eval()
    originals,modules=install(model,args.checkpoint,manifest); del originals
    support=[]
    if args.arm.startswith('recent_'):
        def routing(*a,**kw):
            result=c1_conditional_page_recent64_attention(*a,**kw,recent_within_budget=args.arm=='recent_fixed')
            n=result.statistics['selected_tokens']
            groups=a[1].shape[0]*a[1].shape[1]*a[0].shape[2]
            limit=2048 if args.arm=='recent_fixed' else 2112
            assert n<=limit*groups
            support.append(n/groups)
            return result
        attention.c1_conditional_page_topk_attention=routing
    if args.arm=='lrqk':
        lrqk.LRQKState=FP32RoutingState
        lrqk.install_c1_lrqk(model,LRQKConfig(rank=32,topk=1152,recent=64,prefill_backend='triton'))
    if args.arm=='shadowkv': install_c1_shadowkv(model)
    selected=[rows[0],rows[32]] if args.stage=='smoke' else rows[args.shard_index::args.num_shards]
    for row in selected:
        path=args.output/args.arm/args.stage/f"sample_{row['index']:03d}.json"
        if path.exists():
            saved=read_json(path); assert saved['status']=='complete' and saved['protocol']==spec
            continue
        tokens=torch.tensor(row['input_ids'],dtype=torch.long)
        cap=min(4,row['maximum_tokens']) if args.stage=='smoke' else row['maximum_tokens']
        def run():
            if args.arm=='shadowkv': return shadow_generate(model,tokenizer,tokens,cap)
            return original_generate(model,tokenizer,tokens,row,
                   'b16r16' if args.arm.startswith('recent_') else args.arm,bank,cap)
        support.clear(); started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
        ids,first,stats,stopped=run()
        support_stats=dict(mean=sum(support)/len(support),max=max(support)) if support else {}
        if args.stage=='smoke':
            again,first2,_,_=run(); assert again==ids
            torch.testing.assert_close(first,first2,atol=0,rtol=0)
        prediction=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        result=dict(ids=ids,prediction=prediction,score=sample_score(prediction,row['answers'],row['match_type']),
                    stopped=stopped,routing=stats,physical_support=support_stats,
                    seconds=time.monotonic()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(path,dict(status='complete',protocol=spec,sample=row,result=result,
                   command=shlex.join(sys.argv),python=sys.executable,gpu=torch.cuda.get_device_name(0)))
        print(args.arm,row['index'],result['score'],result['seconds'],support_stats,flush=True)


if __name__=='__main__': main()
