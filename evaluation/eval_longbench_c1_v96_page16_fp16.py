"""C1-V96 prefill followed by matched Base16/R16 page-sparse decoding."""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.eval_longbench_c1_v96 import inputs as full_inputs, generate as full_generate
from evaluation.eval_longbench_c1_fourarm import score_prediction
from evaluation.eval_qwen3_8b_residual_rank_ruler import greedy_decode, _eos_ids
from evaluation.profile_qwen3_8b_residual_two_sided_kl import full_attention
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from evaluation.prepare_longbench_c1 import TASKS
from basisserve.checkpoint import gqa_vo_qwen3 as attention_impl
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.residual_kl_replay import prefix_signature, fork_routing_prefix
from basisserve.core.c1_k_reverse_shadow import ReverseShadowConfig
from evaluation.profile_qwen3_8b_residual_two_sided_kl import make_sidecar
from basisserve.core import c1_conditional_page_attention as page_impl
from evaluation.eval_longbench_lrqk_fp16 import v100_prefill, kernel_check


def inputs(args):
    rows, tokens, previous, dense, full_settings, scorer = full_inputs(args)
    full = json.loads((args.full_results/'result.json').read_text())
    audit = json.loads((args.full_results/'audit.json').read_text())
    assert full['status'] == audit['status'] == 'complete' and full['protocol'] == full_settings
    assert sha256(args.full_results/'result.json') == audit['result_sha256']
    assert [r['sample'] for r in full['records']] == rows
    bank, hashes, source = [], {}, None
    for l in range(36):
        path = args.bank/f'layer_{l:03d}.safetensors'
        record = json.loads(path.with_suffix('.json').read_text())
        if source is None:
            source = record['protocol']
        assert record['status'] == 'complete' and record['layer'] == l and record['protocol'] == source
        assert record['sha256'] == sha256(path)
        tensors = load_file(str(path))
        shapes = dict(base_left_b16=(8,96,16), base_right_b16=(8,16,128), base_bias_b16=(8,128),
            residual_encoder_b16_r16=(8,128,16), residual_query_b16_r16=(32,128,16))
        assert set(tensors) == set(shapes)
        for key, t in tensors.items():
            assert tuple(t.shape) == shapes[key] and t.dtype == torch.float32 and torch.isfinite(t).all()
        bank.append(tensors)
        hashes[str(l)] = record['sha256']
    assert source['c1_manifest_sha256'] == full_settings['c1_manifest_sha256']
    assert source['c1_layer_sha256'] == full_settings['c1_layer_sha256']
    assert source['payload_ranks'] == [96]*36 and source['base_rank'] == 16 and source['residual_rank'] == 16
    assert source['query_count'] == 32 and source['fit_windows'] == 64 and source['diagnostic_windows'] == 16
    assert source['page_size'] == 16 and source['excluded_prefix_pages'] == 2 and source['physical_token_budget'] == 2048
    settings = dict(format='basisserve.longbench_c1_v96_r16.v1', full_protocol=full_settings,
        full_result_sha256=audit['result_sha256'], bank=str(args.bank.resolve()), bank_sha256=hashes,
        bank_protocol=source, page_size=16, physical_token_budget=2048, pinned_prefix_pages=2,
        base_rank=16, residual_rank=16, adaptive_budget=False, force_current_page=False,
        prefill='same full causal C1-V96 Triton as full-K baseline; first token checked against it',
        decode='all36 layers, native BF16 Base16/R16 routing and selected exact-K/C1-V96 attention',
        storage='GPU-resident accuracy oracle with materialized Base128+R16 sidecar; not CPU offload or latency benchmark',
        code_sha256={n:sha256(ROOT/n) for n in (
            'evaluation/eval_longbench_c1_v96_page16_fp16.py', 'evaluation/eval_qwen3_8b_residual_rank_ruler.py',
            'evaluation/profile_qwen3_8b_residual_two_sided_kl.py', 'basisserve/core/c1_conditional_page_attention.py',
            'basisserve/core/c1_v_conditional_k_router.py', 'basisserve/core/residual_kl_replay.py',
            'basisserve/checkpoint/gqa_vo_qwen3.py')})
    fp16_records = []
    reference_hashes = {}
    for row in rows:
        path = ROOT/'results/evaluation/longbench_lrqk_fp16/full/evaluate'/f"sample_{row['index']:03d}.json"
        saved = json.loads(path.read_text())
        assert saved['status'] == 'complete' and saved['arm'] == 'full'
        assert saved['protocol']['dtype'] == 'float16' and saved['result']['sample'] == row
        r = saved['result']
        assert r['score'] == score_prediction(scorer,row['task'],r['prediction'],row['answers'],row['all_classes'])
        fp16_records.append(r)
        reference_hashes[str(row['index'])] = sha256(path)
    full = dict(records=fp16_records,tasks={t:dict(c1_v96=100*sum(r['score'] for r in fp16_records if r['sample']['task']==t)/32) for t in TASKS})
    settings.update(format='basisserve.longbench_c1_v96_r16_fp16.v1',dtype='float16',gpu='V100',
        prefill='FP16 C1-V96 explicit memory-efficient SDPA; same as matched FP16 full-K reference',
        decode='all36 layers native FP16 Base16/offline-R16 selected exact-K/C1-V96',
        fp16_full_reference_sha256=reference_hashes,
        comparison='full-K reference FP16; dense baseline BF16 is context only')
    settings['code_sha256']['evaluation/eval_longbench_lrqk_fp16.py'] = sha256(ROOT/'evaluation/eval_longbench_lrqk_fp16.py')
    return rows, tokens, previous, dense, full, bank, settings, scorer



@torch.inference_mode()
def prepare_arm(model, bank, prefix, ranks):
    assert ranks == [16]*36
    cache=fork_routing_prefix(prefix)
    device=model.model.embed_tokens.weight.device
    positions=torch.arange(prefix.get_seq_length(),device=device)[None]
    cos,sin=model.model.rotary_emb(model.model.embed_tokens.weight[:1],positions)
    for layer,(block,t) in enumerate(zip(model.model.layers,bank,strict=True)):
        module=block.self_attn
        module.attention_backend='native'
        module.set_reverse_shadow_config(None)
        module.set_conditional_routing_factors(base_left=t['base_left_b16'],
            base_right=t['base_right_b16'],base_bias=t['base_bias_b16'],
            residual_encoder=t['residual_encoder_b16_r16'],
            residual_query_projector=t['residual_query_b16_r16'])
        module.set_conditional_page_query_block_size(1,collect_statistics=False)
        module.set_reverse_shadow_config(ReverseShadowConfig(page_size=16,exact_token_budget=2048,
            selector='kq_svd',quest_support='physical_shared',pinned_prefix_pages=2))
        cache._ensure_routing_layer(layer)
        cache._routing_sidecars[layer]=make_sidecar(module,prefix,cos,sin)
    return cache

@torch.inference_mode()
def generate(model, tokenizer, bank, tokens, maximum, trace=False):
    full_attention([l.self_attn for l in model.model.layers], 'triton')
    prefix = RoutingDynamicCache()
    prefill_calls = 0
    kernel = attention_impl.compressed_v_prefill_attention
    def observe_prefill(q, k, v, **kwargs):
        nonlocal prefill_calls
        assert q.shape[-2] == k.shape[-2] == len(tokens) and v.shape[-1] == 96
        prefill_calls += 1
        return kernel(q, k, v, **kwargs)
    with patch.object(attention_impl, 'compressed_v_prefill_attention', new=observe_prefill):
        output = model(input_ids=tokens.long()[None].to('cuda:0'), past_key_values=prefix,
                       use_cache=True, logits_to_keep=1)
    logits = output.logits[0,-1]
    assert prefill_calls == 36 and torch.isfinite(logits).all()
    first = int(logits.argmax())
    first_logits = logits.cpu() if trace else None
    prefix = output.past_key_values
    del output, logits
    signature = prefix_signature(prefix)
    cache = prepare_arm(model, bank, prefix, [16]*36)
    sparse_calls = 0
    sparse_kernel = attention_impl.c1_conditional_page_topk_attention
    def observe_sparse(*args, **kwargs):
        nonlocal sparse_calls
        assert kwargs['page_size']==16 and kwargs['exact_token_budget']==2048 and kwargs['pinned_prefix_pages']==2
        sparse_calls += 1
        return sparse_kernel(*args, **kwargs)
    native_selection=page_impl._selected_pages
    def observe_selection(scores,valid,**kwargs):
        assert kwargs['page_size']==16 and kwargs['page_budget']==128 and kwargs['pinned_prefix_pages']==2
        ids,mask=native_selection(scores,valid,**kwargs)
        if trace:
            expected=min(128,(scores.shape[-1]+15)//16)
            assert ids.shape[-1]==expected and mask.all()
            assert (ids[...,0]==0).all() and (ids[...,1]==1).all()
            assert (ids.sort(-1).values.diff(dim=-1)>0).all()
        return ids,mask
    with patch.object(attention_impl, 'c1_conditional_page_topk_attention', new=observe_sparse), patch.object(page_impl,'_selected_pages',new=observe_selection):
        ids, cache, traces = greedy_decode(model, cache, first, maximum_tokens=maximum,
                                          eos_ids=_eos_ids(tokenizer, model), trace=trace)
    assert sparse_calls == 36*(len(ids)-1) and prefix_signature(prefix) == signature
    length = len(tokens)+len(ids)-1
    assert cache.get_seq_length() == length and len(cache.layers) == 36
    for l, layer in enumerate(cache.layers):
        assert layer.keys.shape == (1,8,length,128) and layer.values.shape == (1,8,length,96)
        assert cache.routing_sidecar(l).shape == (1,8,length,144)
        assert layer.keys.dtype == layer.values.dtype == cache.routing_sidecar(l).dtype == torch.float16
    torch.cuda.synchronize()
    return ids, ([first_logits]+traces if trace else []), sparse_calls, prefill_calls


def summarize(args, rows, previous, dense, full, settings, scorer, tokenizer):
    r8_path = ROOT/'results/evaluation/longbench_c1_v96_b16r8/result.json'
    r8 = json.loads(r8_path.read_text())
    r8_audit = json.loads(r8_path.with_name('audit.json').read_text())
    assert r8['status'] == r8_audit['status'] == 'complete'
    assert sha256(r8_path) == r8_audit['result_sha256']
    assert [r['sample'] for r in r8['records']] == rows
    assert r8['protocol']['full_protocol'] == settings['full_protocol']
    config_eos = json.loads((args.model/'config.json').read_text())['eos_token_id']
    eos = set(config_eos if isinstance(config_eos,list) else [config_eos]) | {tokenizer.eos_token_id}
    records = []
    for row, baseline in zip(rows,full['records'],strict=True):
        saved = json.loads((args.output_dir/'evaluate'/f"sample_{row['index']:03d}.json").read_text())
        assert saved['status'] == 'complete' and saved['protocol'] == settings
        r = saved['result']; ids = r['generated_token_ids']
        assert r['sample'] == row and r['cache_verified'] and r['first_token_matches_full_v96']
        assert ids[0] == baseline['generated_token_ids'][0]
        assert r['prefill_calls'] == 36 and r['sparse_calls'] == 36*(len(ids)-1)
        assert 0 < len(ids) == r['generated_tokens'] <= r['maximum_tokens'] == row['maximum_tokens']
        assert r['stopped_on_eos'] == (ids[-1] in eos) and not any(t in eos for t in ids[:-1])
        assert r['stopped_on_eos'] or len(ids) == row['maximum_tokens']
        assert r['prediction'] == tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        assert r['score'] == score_prediction(scorer,row['task'],r['prediction'],row['answers'],row['all_classes'])
        records.append(r)
    for shard in range(args.num_shards):
        saved = json.loads((args.output_dir/'evaluate'/f'shard_{shard}.json').read_text())
        assert saved['status'] == 'complete' and saved['protocol'] == settings
        assert saved['indices'] == list(range(shard,len(rows),args.num_shards))
    tasks = {}
    for task in TASKS:
        subset = [r for r in records if r['sample']['task'] == task]
        assert len(subset) == 32
        mean = 100*sum(r['score'] for r in subset)/32
        assert round(mean,2) == scorer.scorer(task,[r['prediction'] for r in subset],
            [r['sample']['answers'] for r in subset],subset[0]['sample']['all_classes'])
        tasks[task] = dict(dense=dense['tasks'][task]['dense_k_dense_v'],
            v96_full=full['tasks'][task]['c1_v96'],v96_r8=r8['tasks'][task]['v96_sparse'],v96_sparse=mean)
    means = {a:sum(t[a] for t in tasks.values())/len(TASKS) for a in ('dense','v96_full','v96_r8','v96_sparse')}
    paired = {}
    for name,baseline in [('dense',dense['records']),('v96_full',full['records']),('v96_r8',r8['records'])]:
        delta = [r['score']-b['score'] for r,b in zip(records,baseline,strict=True)]
        paired[name] = dict(improvements=sum(d>0 for d in delta),regressions=sum(d<0 for d in delta),
            ties=sum(d==0 for d in delta),mean_delta_pp=means['v96_sparse']-means[name])
    result = dict(status='complete',protocol=settings,tasks=tasks,means=means,paired=paired,records=records,
        first_token_matches_full_v96=sum(r['first_token_matches_full_v96'] for r in records),
        cap_without_eos=sum(not r['stopped_on_eos'] for r in records),
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in records),
        r8_reference_sha256=sha256(r8_path),command=shlex.join(sys.argv),python=sys.executable)
    write_json(args.output_dir/'result.json',result)
    write_json(args.output_dir/'audit.json',dict(status='complete',predictions_verified=len(records),
        first_tokens_match_full_v96=True,cache_and_dispatch_verified=True,
        official_scores_verified=True,shard_coverage_verified=True,result_sha256=sha256(args.output_dir/'result.json')))
    lines = ['# LongBench: C1-V96 with Base16/R16 sparse decode','',
        'Qwen3-8B-Base, FP16, basis, four independent V100 workers. Same192 frozen prompts, six tasks x32.',
        'Full-K and R16 use FP16 memory-efficient prefill. Dense and offline R8 columns are BF16 context only.', '',
        '| Task | Dense BF16 context | Full-K FP16 | R8 BF16 context | R16 FP16 |','|---|---:|---:|---:|---:|']
    for task,scores in [*tasks.items(),('Mean',means)]:
        lines.append('| '+task+' | '+' | '.join(f'{v:.4f}' for v in scores.values())+' |')
    lines += ['', 'Sparse: all36 layers, Page16/B2048, pinned pages0/1, no adaptive budget or forced current page.',
        'Selected exact K and resident C1-V96 payload. GPU-resident Base128+R16 sidecars: accuracy oracle, not offload/latency benchmark.',
        'Matched Base16 fitted by closed-form affine MSE RRR. Matched R16 fitted with non-sink Page-Fisher, Q32 across four8K bins,BCD/PCG settings are recorded in bank_protocol. No Adam.',
        'Router uses C4 64x32K fit and16x32K diagnostic; frozen C1 uses32x32K fit and4x32K diagnostic. No benchmark fitting.',
        'QA F1 and summary ROUGE-L, scores0–100, six-task arithmetic mean. Not full LongBench; actual inputs1192–30431, total cap32K.',
        'Greedy sampling, original EOS and task caps. Old baselines reused without modification.', '',
        f"First-token agreement with full-K V96: {result['first_token_matches_full_v96']}/192.",
        f"Generation-cap exits without EOS: {result['cap_without_eos']}/192.",
        f"Peak allocated GPU memory: {result['peak_allocated_gib']:.3f} GiB.", '',
        '## Paired score changes','','```json',json.dumps(paired,indent=2),'```','',
        'Commands and protocol: docs/longbench_c1_v96_r16_protocol.md. Exact commands preserved per sample.', '']
    (args.output_dir/'summary.md').write_text('\n'.join(lines))
    print(json.dumps({k:result[k] for k in ('means','paired','cap_without_eos')},indent=2),flush=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','evaluate','summarize'),required=True)
    for name,path in {
        'data-dir':'results/datasets/longbench_c1_32k','c1-results':'results/evaluation/longbench_c1_32k',
        'dense-results':'results/evaluation/longbench_dense_32k','full-results':'results/evaluation/longbench_c1_v96',
        'c1-checkpoint':'results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6',
        'bank':'results/checkpoints/c1_v96_b16r16_p16_q32','output-dir':'results/evaluation/longbench_r16_p16_q32_fp16',
    }.items():
        p.add_argument('--'+name,type=Path,default=ROOT/path)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=4)
    args = p.parse_args()
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    rows,tokens,previous,dense,full,bank,settings,scorer = inputs(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if args.stage == 'summarize':
        summarize(args,rows,previous,dense,full,settings,scorer,tokenizer)
        return
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0)
    attention_impl.compressed_v_prefill_attention = v100_prefill
    if args.stage == 'smoke': print('kernel_error',kernel_check(),flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float16,
        local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda:0').eval()
    attention_impl.install_qwen3_gqa_vo_als_export(model,args.c1_checkpoint,attention_backend='triton')
    model.eval()
    smoke = args.stage == 'smoke'
    assigned = ([min(rows,key=lambda r:r['prompt_tokens']),max(rows,key=lambda r:r['prompt_tokens'])]
                if smoke else rows[args.shard_index::args.num_shards])
    for row in assigned:
        path = args.output_dir/args.stage/f"sample_{row['index']:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['status'] == 'complete' and saved['protocol'] == settings and saved['result']['sample'] == row
            continue
        maximum = min(4,row['maximum_tokens']) if smoke else row['maximum_tokens']
        tensor = tokens[f"sample_{row['index']:03d}"]
        torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
        ids,trace,calls,prefill_calls = generate(model,tokenizer,bank,tensor,maximum,trace=smoke)
        assert ids[0] == full['records'][row['index']]['generated_token_ids'][0]
        if smoke:
            repeated,other,_,_ = generate(model,tokenizer,bank,tensor,maximum,trace=True)
            assert repeated == ids and len(other) == len(trace)
            for a,b in zip(trace,other,strict=True):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            full_attention([l.self_attn for l in model.model.layers], 'triton')
            reference = model(input_ids=tensor.long()[None].cuda(),past_key_values=RoutingDynamicCache(),use_cache=True,logits_to_keep=1)
            torch.testing.assert_close(trace[0],reference.logits[0,-1].cpu(),rtol=0,atol=0)
            del reference
        prediction = tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        score = None if smoke else score_prediction(scorer,row['task'],prediction,row['answers'],row['all_classes'])
        result = dict(sample=row,prediction=prediction,generated_token_ids=ids,generated_tokens=len(ids),
            maximum_tokens=maximum,stopped_on_eos=ids[-1] in _eos_ids(tokenizer,model),score=score,
            cache_verified=True,first_token_matches_full_v96=True,sparse_calls=calls,prefill_calls=prefill_calls,
            smoke_repeated_and_full_prefill_logits_verified=smoke,elapsed_seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(path,dict(status='complete',protocol=settings,result=result,
            command=shlex.join(sys.argv),python=sys.executable,gpu=torch.cuda.get_device_name(0)))
        print(f"sample={row['index']} task={row['task']} score={score} tokens={len(ids)} seconds={result['elapsed_seconds']:.2f}",flush=True)
    write_json(args.output_dir/args.stage/f'shard_{args.shard_index}.json',dict(status='complete',
        protocol=settings,indices=[r['index'] for r in assigned]))


if __name__ == '__main__':
    main()
