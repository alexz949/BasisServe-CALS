"""LongBench with fixed Base16 and prompt-specific closed-form residual R8."""
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
from evaluation.eval_qwen3_8b_residual_rank_ruler import prepare_arm, greedy_decode, _eos_ids
from evaluation.profile_qwen3_8b_residual_two_sided_kl import full_attention
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from evaluation.prepare_longbench_c1 import TASKS
from basisserve.checkpoint import gqa_vo_qwen3 as attention_impl
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.residual_kl_replay import prefix_signature
from basisserve.core.c1_residual_spectral import spectral_factors
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from evaluation.diagnose_c1_prompt_spectral import query_positions
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
            residual_encoder_b16_r8=(8,128,8), residual_query_b16_r8=(32,128,8))
        assert set(tensors) == set(shapes)
        for key, t in tensors.items():
            assert tuple(t.shape) == shapes[key] and t.dtype == torch.float32 and torch.isfinite(t).all()
        bank.append(tensors)
        hashes[str(l)] = record['sha256']
    assert source['c1_manifest_sha256'] == full_settings['c1_manifest_sha256']
    assert source['c1_layer_sha256'] == full_settings['c1_layer_sha256']
    assert source['payload_ranks'] == [96]*36 and source['base_rank'] == 16 and source['residual_rank'] == 8
    assert source['query_count'] == 32 and source['fit_windows'] == 64 and source['diagnostic_windows'] == 16
    assert source['page_size'] == 32 and source['excluded_prefix_pages'] == 1 and source['physical_token_budget'] == 2048
    settings = dict(format='basisserve.longbench_c1_v96_spectral.v1', full_protocol=full_settings,
        full_result_sha256=audit['result_sha256'], bank=str(args.bank.resolve()), bank_sha256=hashes,
        bank_protocol=source, page_size=32, physical_token_budget=2048, pinned_prefix_pages=1,
        residual_fit='per prompt/layer/group; FP32 shared-query metric eigensolve; rank8 ridge1e-5; full residual rows',
        fit_queries='256 positions from diagnostic query_positions rule; four associated GQA heads pooled; no answers or generated Q',
        frozen_decode_basis=True,
        base_rank=16, residual_rank=8, adaptive_budget=False, force_current_page=False,
        prefill='same full causal C1-V96 Triton as full-K baseline; first token checked against it',
        decode='all36 layers, native BF16 Base16/R8 routing and selected exact-K/C1-V96 attention',
        storage='GPU-resident accuracy oracle with materialized Base128+R8 sidecar; not CPU offload or latency benchmark',
        code_sha256={n:sha256(ROOT/n) for n in (
            'evaluation/eval_longbench_c1_v96_spectral.py', 'basisserve/core/c1_residual_spectral.py', 'evaluation/diagnose_c1_prompt_spectral.py', 'evaluation/eval_qwen3_8b_residual_rank_ruler.py',
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
    settings.update(format='basisserve.longbench_c1_v96_spectral_fp16.v1',dtype='float16',gpu='V100',
        prefill='FP16 C1-V96 explicit memory-efficient SDPA; same as matched FP16 full-K reference',
        decode='all36 layers native FP16 Base16/prompt-R8 selected exact-K/C1-V96',
        fp16_full_reference_sha256=reference_hashes,
        comparison='full-K reference FP16; dense baseline BF16 is context only')
    settings['code_sha256']['evaluation/eval_longbench_lrqk_fp16.py'] = sha256(ROOT/'evaluation/eval_longbench_lrqk_fp16.py')
    return rows, tokens, previous, dense, full, bank, settings, scorer


@torch.inference_mode()
def generate(model, tokenizer, bank, tokens, maximum, trace=False):
    full_attention([l.self_attn for l in model.model.layers], 'triton')
    prefix = RoutingDynamicCache()
    prefill_calls = 0
    prompt_bank = []
    fit_positions, _ = query_positions(len(tokens))
    positions = torch.arange(len(tokens), device='cuda:0')[None]
    cos, sin = model.model.rotary_emb(model.model.embed_tokens.weight[:1], positions)
    kernel = attention_impl.compressed_v_prefill_attention
    def observe_prefill(q, k, v, **kwargs):
        nonlocal prefill_calls
        assert q.shape[-2] == k.shape[-2] == len(tokens) and v.shape[-1] == 96
        tensors = {n: t.to(q.device) for n,t in bank[prefill_calls].items()}
        side = build_conditional_routing_sidecar(v.float(), k.float(),
            base_left=tensors['base_left_b16'], base_right=tensors['base_right_b16'],
            base_bias=tensors['base_bias_b16'], residual_encoder=tensors['residual_encoder_b16_r8'],
            cos=cos.float(), sin=sin.float())
        residual = k[0].float() - side[0,:,:,:128]
        queries = q[0,:,fit_positions].float().reshape(8,4*len(fit_positions),128)
        c = residual.mT @ residual / residual.shape[1]
        h = queries.mT @ queries / queries.shape[1]
        e,u,_ = spectral_factors(c,h,rank=8,relative_ridge=1e-5)
        tensors['residual_encoder_b16_r8'] = e
        tensors['residual_query_b16_r8'] = u.repeat_interleave(4,dim=0)
        prompt_bank.append(tensors)
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
    assert len(prompt_bank) == 36
    cache = prepare_arm(model, prompt_bank, prefix, [8]*36)
    sparse_calls = 0
    sparse_kernel = attention_impl.c1_conditional_page_topk_attention
    def observe_sparse(*args, **kwargs):
        nonlocal sparse_calls
        sparse_calls += 1
        return sparse_kernel(*args, **kwargs)
    with patch.object(attention_impl, 'c1_conditional_page_topk_attention', new=observe_sparse):
        ids, cache, traces = greedy_decode(model, cache, first, maximum_tokens=maximum,
                                          eos_ids=_eos_ids(tokenizer, model), trace=trace)
    assert sparse_calls == 36*(len(ids)-1) and prefix_signature(prefix) == signature
    length = len(tokens)+len(ids)-1
    assert cache.get_seq_length() == length and len(cache.layers) == 36
    for l, layer in enumerate(cache.layers):
        assert layer.keys.shape == (1,8,length,128) and layer.values.shape == (1,8,length,96)
        assert cache.routing_sidecar(l).shape == (1,8,length,136)
        assert layer.keys.dtype == layer.values.dtype == cache.routing_sidecar(l).dtype == torch.float16
    torch.cuda.synchronize()
    return ids, ([first_logits]+traces if trace else []), sparse_calls, prefill_calls


def summarize(args, rows, previous, dense, full, settings, scorer, tokenizer):
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
            v96_full=full['tasks'][task]['c1_v96'],v96_sparse=mean)
    means = {a:sum(t[a] for t in tasks.values())/len(TASKS) for a in ('dense','v96_full','v96_sparse')}
    paired = {}
    for name,baseline in [('dense',dense['records']),('v96_full',full['records'])]:
        delta = [r['score']-b['score'] for r,b in zip(records,baseline,strict=True)]
        paired[name] = dict(improvements=sum(d>0 for d in delta),regressions=sum(d<0 for d in delta),
            ties=sum(d==0 for d in delta),mean_delta_pp=means['v96_sparse']-means[name])
    result = dict(status='complete',protocol=settings,tasks=tasks,means=means,paired=paired,records=records,
        first_token_matches_full_v96=sum(r['first_token_matches_full_v96'] for r in records),
        cap_without_eos=sum(not r['stopped_on_eos'] for r in records),
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in records),
        command=shlex.join(sys.argv),python=sys.executable)
    write_json(args.output_dir/'result.json',result)
    write_json(args.output_dir/'audit.json',dict(status='complete',predictions_verified=len(records),
        first_tokens_match_full_v96=True,cache_and_dispatch_verified=True,
        official_scores_verified=True,shard_coverage_verified=True,result_sha256=sha256(args.output_dir/'result.json')))
    lines = ['# LongBench: C1-V96 with Base16/R8 sparse decode','',
        'Qwen3-8B-Base, FP16, basis, four independent V100 workers. Same192 frozen prompts, six tasks x32.',
        'V96 arms use FP16 memory-efficient C1 prefill. Dense BF16 is contextual, not precision-matched.', '',
        '| Task | Dense BF16 (context) | FP16 V96 full exact K | FP16 V96 Base16/prompt-R8 sparse |','|---|---:|---:|---:|']
    for task,scores in [*tasks.items(),('Mean',means)]:
        lines.append('| '+task+' | '+' | '.join(f'{v:.4f}' for v in scores.values())+' |')
    lines += ['', 'Sparse: all36 layers, Page32/B2048, pinned page0, no adaptive budget or forced current page.',
        'Selected exact K and resident C1-V96 payload. GPU-resident Base128+R8 sidecars: accuracy oracle, not offload/latency benchmark.',
        'Matched Base16 fitted by closed-form affine MSE RRR. R8 is fitted per prompt using FP32 shared-query covariance and residual covariance eigensolves. No BCD or Adam; E/U frozen during decode.',
        'Base uses the existing C4 bank; frozen C1 uses C4 32x32K fit and4x32K diagnostic. Residual uses current prompt activations only, not benchmark answers.',
        'QA F1 and summary ROUGE-L, scores0–100, six-task arithmetic mean. Not full LongBench; actual inputs1192–30431, total cap32K.',
        'Greedy sampling, original EOS and task caps. Old baselines reused without modification.', '',
        f"First-token agreement with full-K V96: {result['first_token_matches_full_v96']}/192.",
        f"Generation-cap exits without EOS: {result['cap_without_eos']}/192.",
        f"Peak allocated GPU memory: {result['peak_allocated_gib']:.3f} GiB.", '',
        '## Paired score changes','','```json',json.dumps(paired,indent=2),'```','',
        'Commands and protocol: docs/longbench_c1_v96_spectral_protocol.md. Exact commands preserved per sample.', '']
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
        'bank':'results/checkpoints/c1_v96_b16r8_qgram','output-dir':'results/evaluation/longbench_c1_v96_spectral_fp16',
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
