"""Paired LongBench-v1 C1-V80: full K, exact sparse K, Gram-Q32, terminal-Q32."""

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shlex
import sys
import time
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from basisserve.checkpoint.gqa_vo_qwen3 import install_qwen3_gqa_vo_als_export
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.residual_kl_replay import prefix_signature
from evaluation.eval_qwen3_8b_residual_rank_ruler import prepare_arm, greedy_decode, _eos_ids
from evaluation.profile_qwen3_8b_residual_two_sided_kl import full_attention, load_bank
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from evaluation.longbench_exact_pages import ExactPageAttention
from evaluation.prepare_longbench_c1 import TASKS

ARMS = ('full_exact_k','sparse_exact_k','qgram32','terminal32')


def official_scorer(root):
    sys.path.insert(0,str(ROOT/'results/tools/longbench_deps'))
    sys.path.insert(0,str(root/'LongBench'))
    spec = importlib.util.spec_from_file_location('longbench_official_eval',root/'LongBench/eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_prediction(scorer, task, prediction, references, all_classes):
    assert task in TASKS and references
    score = max(float(scorer.dataset2metric[task](prediction,answer,all_classes=all_classes))
                for answer in references)
    assert math.isfinite(score) and 0 <= score <= 1
    return score


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('smoke','evaluate','summarize'),required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--c1-checkpoint',type=Path,required=True)
    p.add_argument('--terminal-bank',type=Path,default=ROOT/'results/checkpoints/mse_base_q32_r8')
    p.add_argument('--qgram-bank',type=Path,default=ROOT/'results/checkpoints/mse_base_qgram32_r8')
    p.add_argument('--data-dir',type=Path,default=ROOT/'results/datasets/longbench_c1_32k')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/evaluation/longbench_c1_32k')
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=4)
    return p


def inputs_and_protocol(args):
    manifest = json.loads((args.data_dir/'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    spec = manifest['protocol']
    assert spec['sequence_length'] == 32768 and spec['samples_per_task'] == 32 and spec['tasks'] == list(TASKS)
    assert spec['model_config_sha256'] == sha256(args.model/'config.json')
    assert manifest['tokens_sha256'] == sha256(args.data_dir/'tokens.safetensors')
    assert manifest['samples_sha256'] == sha256(args.data_dir/'samples.json')
    official = Path(spec['official_root'])
    for name,expected in spec['official_sha256'].items():
        assert sha256(official/'LongBench'/name) == expected
    rows = json.loads((args.data_dir/'samples.json').read_text())
    tokens = load_file(str(args.data_dir/'tokens.safetensors'))
    assert len(rows) == len(tokens) == 192
    for i,row in enumerate(rows):
        tensor = tokens[f'sample_{i:03d}']
        assert row['index'] == i and len(tensor) == row['prompt_tokens']
        assert len(tensor)+row['maximum_tokens'] <= 32768
        assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == row['input_ids_sha256']
    c1 = json.loads((args.c1_checkpoint/'results.json').read_text())
    c1_hashes = {str(l):sha256(args.c1_checkpoint/c1['artifacts'][str(l)]['file']) for l in range(36)}
    banks = {name:load_bank(path) for name,path in [('terminal32',args.terminal_bank),('qgram32',args.qgram_bank)]}
    bank_hashes,bank_protocols = {},{}
    for name,path in [('terminal32',args.terminal_bank),('qgram32',args.qgram_bank)]:
        bank_hashes[name] = {}
        for l in range(36):
            record = json.loads((path/f'layer_{l:03d}.json').read_text())
            assert record['status'] == 'complete' and record['layer'] == l
            source = record['protocol']
            if l == 0:
                bank_protocols[name] = source
            assert source == bank_protocols[name]
            assert source['base_kind'] == 'closed_form_rrr' and source['base_optimizer'] is None
            assert source['base_rank'] == 16 and source['ranks'] == [8]
            assert source['page_size'] == 32 and source['excluded_prefix_pages'] == 1
            assert source['model_config_sha256'] == spec['model_config_sha256']
            assert source['c1_manifest_sha256'] == sha256(args.c1_checkpoint/'results.json')
            assert source['inputs'][str(l)]['c1_layer_sha256'] == c1_hashes[str(l)]
            positions = source['residual_query_positions']
            if name == 'terminal32':
                assert positions == list(range(24831,32768,256))
            else:
                assert source['residual_query_count'] == 32
                assert all(sum(8192*b <= p < 8192*(b+1) for p in positions[str(l)]) == 8 for b in range(4))
            file = path/f'layer_{l:03d}.safetensors'
            assert sha256(file) == record['sha256']
            bank_hashes[name][str(l)] = record['sha256']
            data = banks[name][l]
            assert all(torch.isfinite(t).all() for t in data.values())
            assert data['residual_encoder_b16_r8'].shape == (8,128,8)
            assert data['residual_query_b16_r8'].shape == (32,128,8)
            for key in ('base_left_b16','base_right_b16','base_bias_b16'):
                assert torch.equal(data[key],banks['terminal32'][l][key])
    settings = dict(format='basisserve.longbench_c1_fourarm.v1',arms=list(ARMS),
        dataset_manifest_sha256=sha256(args.data_dir/'manifest.json'),dataset_protocol=spec,
        c1_manifest_sha256=sha256(args.c1_checkpoint/'results.json'),c1_layer_sha256=c1_hashes,
        bank_sha256=bank_hashes,bank_protocols=bank_protocols,model=str(args.model.resolve()),
        model_dtype='bfloat16',gpu='NVIDIA L40S',sequence_length=32768,page_size=32,
        physical_token_budget=2048,pinned_prefix_pages=1,
        prefill='one shared full C1-V80 Triton prefill; first generated token is shared',
        decode='full exact-K SDPA vs native BF16 sparse Page32; all36 layers; independent prefix forks',
        exact_sparse='FP32 QK on exact BF16 K; per-head non-sink page-LSE normalization then GQA max; same64-page budget and native exact-K/C1-V payload path',
        exact_sparse_sidecar='terminal32 sidecar computed as plumbing only; overridden selector never uses its scores',
        sampling='greedy, tokenizer/model EOS, official per-task output caps; base prompts without chat template',
        metrics='official LongBench-v1 QA F1 or ROUGE-L; max over alternate references; task arithmetic means',
        scope='fixed192-prompt6-task pilot, not full LongBench; no refit/tuning on benchmark; actual token lengths retained, not every prompt32K',
        storage='accuracy oracle with GPU-resident exact K and C1-V, not CPU offload/latency benchmark',
        code_sha256={n:sha256(ROOT/n) for n in ('evaluation/eval_longbench_c1_fourarm.py','evaluation/longbench_exact_pages.py',
            'evaluation/eval_qwen3_8b_residual_rank_ruler.py','evaluation/profile_qwen3_8b_residual_two_sided_kl.py',
            'basisserve/core/c1_conditional_page_attention.py','basisserve/core/c1_v_conditional_k_router.py',
            'basisserve/core/residual_kl_replay.py','basisserve/checkpoint/gqa_vo_qwen3.py')})
    return rows,tokens,banks,settings,official_scorer(official)


@torch.inference_mode()
def run_arm(model,tokenizer,banks,prefix,first,arm,maximum,trace=False):
    bank = banks['qgram32' if arm=='qgram32' else 'terminal32']
    ranks = None if arm=='full_exact_k' else [8]*36
    cache = prepare_arm(model,bank,prefix,ranks)
    exact = ExactPageAttention()
    context = patch('basisserve.checkpoint.gqa_vo_qwen3.c1_conditional_page_topk_attention',new=exact) if arm=='sparse_exact_k' else nullcontext()
    with context:
        ids,cache,logits = greedy_decode(model,cache,first,maximum_tokens=maximum,eos_ids=_eos_ids(tokenizer,model),trace=trace)
    assert cache.get_seq_length() == prefix.get_seq_length()+len(ids)-1
    if arm=='sparse_exact_k':
        assert exact.calls == exact.selector_calls == 36*(len(ids)-1)
    if ranks is not None:
        for l in range(36):
            sidecar = cache.routing_sidecar(l)
            assert sidecar.shape[-1] == 136 and sidecar.shape[-2] == cache.get_seq_length(l)
    return ids,logits,exact.calls


@torch.inference_mode()
def evaluate(model,tokenizer,banks,tokens,row,args,scorer):
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    full_attention([layer.self_attn for layer in model.model.layers],'triton')
    prefix = RoutingDynamicCache()
    output = model(input_ids=tokens.long()[None].to('cuda:0'),past_key_values=prefix,use_cache=True,logits_to_keep=1)
    assert torch.isfinite(output.logits).all()
    first = int(output.logits[0,-1].argmax())
    del output
    torch.cuda.synchronize()
    prefill_seconds = time.monotonic()-started
    signature = prefix_signature(prefix)
    smoke = args.stage=='smoke'
    cap = min(4,row['maximum_tokens']) if smoke else row['maximum_tokens']
    results = {}
    for arm in ARMS:
        arm_started = time.monotonic()
        ids,trace,calls = run_arm(model,tokenizer,banks,prefix,first,arm,cap,trace=smoke)
        assert prefix_signature(prefix) == signature
        if smoke:
            repeated,repeated_trace,_ = run_arm(model,tokenizer,banks,prefix,first,arm,cap,trace=True)
            assert repeated == ids and len(trace) == len(repeated_trace)
            for x,y in zip(trace,repeated_trace,strict=True):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
            assert prefix_signature(prefix) == signature
        text = tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        score = None if smoke else score_prediction(scorer,row['task'],text,row['answers'],row['all_classes'])
        results[arm] = dict(prediction=text,generated_token_ids=ids,generated_tokens=len(ids),
            score=score,maximum_tokens=cap,stopped_on_eos=ids[-1] in _eos_ids(tokenizer,model),
            exact_selector_calls=calls,elapsed_seconds=time.monotonic()-arm_started)
        print(f"sample={row['index']} task={row['task']} arm={arm} score={score} tokens={len(ids)} seconds={results[arm]['elapsed_seconds']:.2f}",flush=True)
    return dict(sample=row,arms=results,first_token=first,prefix_unchanged=True,
        smoke_repeated_logits_equal=smoke,prefill_seconds=prefill_seconds,
        wall_seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)


def summarize(args,rows,settings,scorer):
    records = []
    for row in rows:
        path = args.output_dir/'evaluate'/f"sample_{row['index']:03d}.json"
        saved = json.loads(path.read_text())
        assert saved['status']=='complete' and saved['protocol']==settings
        result = saved['result']
        assert result['sample']==row and result['prefix_unchanged']
        for arm in ARMS:
            r = result['arms'][arm]
            assert r['generated_tokens']==len(r['generated_token_ids']) <= row['maximum_tokens']
            assert r['maximum_tokens']==row['maximum_tokens'] and r['generated_token_ids'][0]==result['first_token']
            assert r['score']==score_prediction(scorer,row['task'],r['prediction'],row['answers'],row['all_classes'])
        records.append(result)
    tasks = {}
    for task in TASKS:
        subset = [r for r in records if r['sample']['task']==task]
        assert len(subset)==32
        tasks[task] = {}
        for arm in ARMS:
            mean = 100*sum(r['arms'][arm]['score'] for r in subset)/len(subset)
            official = scorer.scorer(task,[r['arms'][arm]['prediction'] for r in subset],
                                     [r['sample']['answers'] for r in subset],subset[0]['sample']['all_classes'])
            assert round(mean,2)==official
            tasks[task][arm] = mean
    means = {arm:sum(t[arm] for t in tasks.values())/len(tasks) for arm in ARMS}
    pairs = {}
    for reference,arm in [('full_exact_k','sparse_exact_k'),('sparse_exact_k','qgram32'),('sparse_exact_k','terminal32'),('terminal32','qgram32')]:
        delta = [r['arms'][arm]['score']-r['arms'][reference]['score'] for r in records]
        pairs[f'{arm}_vs_{reference}'] = dict(improvements=sum(x>0 for x in delta),regressions=sum(x<0 for x in delta),ties=sum(x==0 for x in delta),mean_delta_pp=means[arm]-means[reference])
    lengths = dict(min=min(r['prompt_tokens'] for r in rows),max=max(r['prompt_tokens'] for r in rows),
        mean=sum(r['prompt_tokens'] for r in rows)/len(rows),truncated=sum(r['truncated'] for r in rows),
        at_most_2048=sum(r['prompt_tokens']<=2048 for r in rows))
    write_json(args.output_dir/'result.json',dict(status='complete',protocol=settings,tasks=tasks,means=means,
        paired=pairs,lengths=lengths,records=records,command=shlex.join(sys.argv),python=sys.executable))
    lines=['# C1-V80 LongBench-v1: four-arm32K-cap pilot','',
        '192 fixed prompts, six tasks x32; all four arms share full-C1 prefill. Only decode attention differs. Frozen Base16/R8 checkpoints; no benchmark fitting.',
        '','| Task | Full exact K | Sparse exact K | Query-Gram Q32 | Terminal Q32 |','|---|---:|---:|---:|---:|']
    for task,scores in tasks.items():
        lines.append('| '+task+' | '+' | '.join(f'{scores[a]:.4f}' for a in ARMS)+' |')
    lines += ['| Mean | '+' | '.join(f'{means[a]:.4f}' for a in ARMS)+' |','',
        'Scores are0–100; QA uses official F1, summaries use official ROUGE-L, best alternative reference. These are not all accuracies.',
        '',f'Actual input lengths: {lengths}.32K is the total input+reserved-output cap, not a fixed prompt length. Token-level middle truncation is recorded per sample; no padding.',
        '', 'Page32/B2048 with pinned page0; exact sparse uses FP32 exact-QK selection and the same BF16 exact-K/C1-V payload path. Full exact K uses SDPA decode. All36 layers included.',
        'Query-Gram means32 calibration Q across four8K bins; terminal means32 calibration Q in the final8K. Both use Base16+R8. These are not terminal-Q8 checkpoints.',
        'Qwen3-8B-Base, official completion prompts, greedy/task-specific caps. This is a6-task pilot, not full LongBench or LongBench-E. The four-arm mean is task-arithmetic, not an official all-task leaderboard score.',
        '', '## Paired score changes','',json.dumps(pairs,indent=2),'',
        'Environment: basis; four L40S workers. GPU-resident accuracy oracle, not a PCIe performance benchmark. Each sample JSON preserves exact command, provenance, tokens and scores.','']
    (args.output_dir/'summary.md').write_text('\n'.join(lines))
    print(json.dumps({'means':means,'paired':pairs,'lengths':lengths},indent=2),flush=True)


@torch.inference_mode()
def main():
    args = parser().parse_args()
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    rows,tokens,banks,settings,scorer = inputs_and_protocol(args)
    if args.stage=='summarize':
        summarize(args,rows,settings,scorer)
        return
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA L40S'
    model = AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,local_files_only=True,
             low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda:0').eval()
    install_qwen3_gqa_vo_als_export(model,args.c1_checkpoint,attention_backend='triton')
    assert len(model.model.layers)==36 and not model.model.has_sliding_layers
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    assigned = ([min(rows,key=lambda r:r['prompt_tokens']),max(rows,key=lambda r:r['prompt_tokens'])]
                if args.stage=='smoke' else rows[args.shard_index::args.num_shards])
    for row in assigned:
        path = args.output_dir/args.stage/f"sample_{row['index']:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['status']=='complete' and saved['protocol']==settings and saved['result']['sample']==row
            continue
        result = evaluate(model,tokenizer,banks,tokens[f"sample_{row['index']:03d}"],row,args,scorer)
        write_json(path,dict(status='complete',protocol=settings,result=result,command=shlex.join(sys.argv),
                            python=sys.executable,torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))
    write_json(args.output_dir/args.stage/f'shard_{args.shard_index}.json',dict(status='complete',
        indices=[r['index'] for r in assigned],protocol=settings))


if __name__ == '__main__':
    main()
