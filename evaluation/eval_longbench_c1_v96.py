"""Frozen LongBench pilot: uniform C1-V96 prefill and full-exact-K decode."""

import argparse
import json
from pathlib import Path
import shlex
import sys
import time
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from basisserve.checkpoint import gqa_vo_qwen3 as attention_impl
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from evaluation.eval_longbench_dense import inputs as common_inputs
from evaluation.eval_longbench_c1_fourarm import score_prediction
from evaluation.eval_qwen3_8b_residual_rank_ruler import greedy_decode, _eos_ids
from evaluation.profile_qwen3_8b_residual_two_sided_kl import full_attention
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from evaluation.prepare_longbench_c1 import TASKS


def inputs(args):
    rows, tokens, previous, common, scorer = common_inputs(args)
    dense = json.loads((args.dense_results / 'result.json').read_text())
    audit = json.loads((args.dense_results / 'audit.json').read_text())
    assert dense['status'] == audit['status'] == 'complete'
    assert sha256(args.dense_results / 'result.json') == audit['result_sha256']
    assert [r['sample'] for r in dense['records']] == rows
    assert dense['protocol']['dataset_manifest_sha256'] == common['dataset_manifest_sha256']
    c1 = json.loads((args.c1_checkpoint / 'results.json').read_text())
    old_path = ROOT / 'results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6/results.json'
    old = json.loads(old_path.read_text())
    assert sha256(old_path) == previous['protocol']['c1_manifest_sha256']
    assert c1['status'] == old['status'] == 'complete'
    config = c1['fit_config']
    assert config['cache_rank_per_head'] == 96
    varying = {'cache_rank_per_head', 'total_v_cache_rank', 'v_retained_ratio', 'total_kv_retained_ratio_with_dense_k'}
    assert {k: v for k, v in config.items() if k not in varying} == {
        k: v for k, v in old['fit_config'].items() if k not in varying}
    hashes = {str(l): sha256(args.c1_checkpoint / c1['artifacts'][str(l)]['file']) for l in range(36)}
    settings = dict(format='basisserve.longbench_c1_v96.v1', common_input_protocol=common,
        dense_result_sha256=audit['result_sha256'], checkpoint=str(args.c1_checkpoint.resolve()),
        c1_manifest_sha256=sha256(args.c1_checkpoint / 'results.json'), c1_layer_sha256=hashes,
        fit_config=config, prefill='full causal C1-V96 Triton, fresh cache, all36 layers',
        decode='full exact-K128 / C1-V96 SDPA, all36 layers, no routing or offload',
        first_token='C1-V96 prefill argmax, not dense-prefill argmax',
        code_sha256={n: sha256(ROOT / n) for n in (
            'evaluation/eval_longbench_c1_v96.py', 'basisserve/checkpoint/gqa_vo_qwen3.py',
            'basisserve/kernels/compressed_v_decode_attention.py',
            'evaluation/profile_qwen3_8b_residual_two_sided_kl.py')})
    return rows, tokens, previous, dense, settings, scorer


@torch.inference_mode()
def generate(model, tokenizer, tokens, maximum, trace=False):
    modules = [l.self_attn for l in model.model.layers]
    full_attention(modules, 'triton')
    cache = RoutingDynamicCache()
    calls = 0
    kernel = attention_impl.compressed_v_prefill_attention

    def observe(q, k, v, **kwargs):
        nonlocal calls
        assert q.shape[-2] == k.shape[-2] == len(tokens) and v.shape[-1] == 96
        calls += 1
        return kernel(q, k, v, **kwargs)

    started = time.monotonic()
    with patch.object(attention_impl, 'compressed_v_prefill_attention', new=observe):
        output = model(input_ids=tokens.long()[None].to('cuda:0'), past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
    logits = output.logits[0, -1]
    assert torch.isfinite(logits).all() and calls == 36
    first = int(logits.argmax())
    first_logits = logits.cpu() if trace else None
    cache = output.past_key_values
    del output, logits
    torch.cuda.synchronize()
    prefill_seconds = time.monotonic() - started
    full_attention(modules, 'sdpa')
    ids, cache, traces = greedy_decode(model, cache, first, maximum_tokens=maximum,
                                      eos_ids=_eos_ids(tokenizer, model), trace=trace)
    length = len(tokens) + len(ids) - 1
    assert cache.get_seq_length() == length and len(cache.layers) == 36
    for layer in cache.layers:
        assert layer.keys.shape == (1, 8, length, 128)
        assert layer.values.shape == (1, 8, length, 96)
        assert layer.keys.dtype == layer.values.dtype == torch.bfloat16
    torch.cuda.synchronize()
    return ids, ([first_logits] + traces if trace else []), prefill_seconds, calls


def summarize(args, rows, previous, dense, settings, scorer, tokenizer):
    records = []
    config_eos = json.loads((args.model / 'config.json').read_text())['eos_token_id']
    eos = set(config_eos if isinstance(config_eos, list) else [config_eos]) | {tokenizer.eos_token_id}
    for row in rows:
        saved = json.loads((args.output_dir / 'evaluate' / f"sample_{row['index']:03d}.json").read_text())
        assert saved['status'] == 'complete' and saved['protocol'] == settings
        r = saved['result']
        ids = r['generated_token_ids']
        assert r['sample'] == row and r['cache_kv_shape_verified'] and r['prefill_kernel_calls'] == 36
        assert 0 < len(ids) == r['generated_tokens'] <= row['maximum_tokens'] == r['maximum_tokens']
        assert r['stopped_on_eos'] == (ids[-1] in eos) and not any(t in eos for t in ids[:-1])
        assert r['stopped_on_eos'] or len(ids) == row['maximum_tokens']
        assert r['prediction'] == tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assert r['score'] == score_prediction(scorer, row['task'], r['prediction'], row['answers'], row['all_classes'])
        records.append(r)
    for shard in range(args.num_shards):
        saved = json.loads((args.output_dir / 'evaluate' / f'shard_{shard}.json').read_text())
        assert saved['status'] == 'complete' and saved['protocol'] == settings
        assert saved['indices'] == list(range(shard, len(rows), args.num_shards))
    tasks = {}
    for task in TASKS:
        subset = [r for r in records if r['sample']['task'] == task]
        assert len(subset) == 32
        mean = 100 * sum(r['score'] for r in subset) / 32
        assert round(mean, 2) == scorer.scorer(task, [r['prediction'] for r in subset],
            [r['sample']['answers'] for r in subset], subset[0]['sample']['all_classes'])
        tasks[task] = dict(dense=dense['tasks'][task]['dense_k_dense_v'],
            c1_v80=previous['tasks'][task]['full_exact_k'], c1_v96=mean)
    means = {a: sum(t[a] for t in tasks.values()) / len(TASKS) for a in ('dense', 'c1_v80', 'c1_v96')}
    paired = {}
    for name, baseline in [('dense', dense['records']),
                           ('c1_v80', [r['arms']['full_exact_k'] for r in previous['records']])]:
        delta = [r['score'] - b['score'] for r, b in zip(records, baseline, strict=True)]
        paired[name] = dict(improvements=sum(d > 0 for d in delta), regressions=sum(d < 0 for d in delta),
            ties=sum(d == 0 for d in delta), mean_delta_pp=means['c1_v96'] - means[name],
            first_token_matches=sum(r['generated_token_ids'][0] == b['generated_token_ids'][0]
                                    for r, b in zip(records, baseline, strict=True)))
    result = dict(status='complete', protocol=settings, tasks=tasks, means=means, paired=paired,
        records=records, cap_without_eos=sum(not r['stopped_on_eos'] for r in records),
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in records),
        command=shlex.join(sys.argv), python=sys.executable)
    write_json(args.output_dir / 'result.json', result)
    write_json(args.output_dir / 'audit.json', dict(status='complete', predictions_verified=192,
        shard_coverage_verified=True, cache_shapes_and_prefill_dispatch_verified=True,
        official_scores_verified=True, result_sha256=sha256(args.output_dir / 'result.json')))
    lines = ['# LongBench: C1-V96 prefill and C1-V96 decode', '',
        'Qwen3-8B-Base, BF16, basis, four independent L40S workers. Exact K, no routing, no refitting.',
        'Same 192 frozen prompts: six tasks x32. C1 prefill uses Triton; full-K C1 decode uses SDPA.', '',
        '| Task | Dense | C1-V80 | C1-V96 |', '|---|---:|---:|---:|']
    for task, scores in [*tasks.items(), ('Mean', means)]:
        lines.append('| ' + task + ' | ' + ' | '.join(f'{v:.4f}' for v in scores.values()) + ' |')
    lines += ['', 'V80/V96 use the same C4 32x32K fit and 4x32K diagnostic captures and six ALS sweeps.',
        'All36 layers use uniform rank. C1 participates from the start of prefill, including the first generated token.',
        'QA uses official F1, summaries official ROUGE-L, best reference, scores 0–100. Mean is six-task arithmetic mean.',
        'This is not full LongBench. Input+reserved-output cap32K; actual inputs1,192–30,431 tokens.',
        'Greedy decoding, unchanged prompts, EOS and task-specific caps. Existing baseline results reused, not rerun.',
        'Exact K means uncompressed K generated by the C1 model trajectory, not keys copied from a separate dense trajectory.',
        'No dense-prefill V96 arm was run. This comparison changes V capacity in both prefill and decode.',
        '', f"Generation-cap exits without EOS: {result['cap_without_eos']}/192.",
        f"Peak allocated GPU memory: {result['peak_allocated_gib']:.3f} GiB.", '',
        '## Paired comparisons', '', '```json', json.dumps(paired, indent=2), '```', '',
        'Commands and protocol: docs/longbench_c1_v96_protocol.md. Exact worker commands are in each sample JSON.', '']
    (args.output_dir / 'summary.md').write_text('\n'.join(lines))
    print(json.dumps({k: result[k] for k in ('means', 'paired', 'cap_without_eos')}, indent=2), flush=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('smoke', 'evaluate', 'summarize'), required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'results/datasets/longbench_c1_32k')
    p.add_argument('--c1-results', type=Path, default=ROOT / 'results/evaluation/longbench_c1_32k')
    p.add_argument('--dense-results', type=Path, default=ROOT / 'results/evaluation/longbench_dense_32k')
    p.add_argument('--c1-checkpoint', type=Path, default=ROOT / 'results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/evaluation/longbench_c1_v96')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=4)
    args = p.parse_args()
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    rows, tokens, previous, dense, settings, scorer = inputs(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.stage == 'summarize':
        summarize(args, rows, previous, dense, settings, scorer, tokenizer)
        return
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA L40S'
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation='sdpa').to('cuda:0').eval()
    attention_impl.install_qwen3_gqa_vo_als_export(model, args.c1_checkpoint, attention_backend='triton')
    model.eval()
    smoke = args.stage == 'smoke'
    assigned = ([min(rows, key=lambda r: r['prompt_tokens']), max(rows, key=lambda r: r['prompt_tokens'])]
                if smoke else rows[args.shard_index::args.num_shards])
    for row in assigned:
        path = args.output_dir / args.stage / f"sample_{row['index']:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['status'] == 'complete' and saved['protocol'] == settings and saved['result']['sample'] == row
            continue
        tensor = tokens[f"sample_{row['index']:03d}"]
        maximum = min(4, row['maximum_tokens']) if smoke else row['maximum_tokens']
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        ids, trace, prefill, calls = generate(model, tokenizer, tensor, maximum, trace=smoke)
        if smoke:
            repeated, other_trace, _, _ = generate(model, tokenizer, tensor, maximum, trace=True)
            assert repeated == ids and len(trace) == len(other_trace)
            for a, b in zip(trace, other_trace, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        score = None if smoke else score_prediction(scorer, row['task'], prediction, row['answers'], row['all_classes'])
        result = dict(sample=row, prediction=prediction, generated_token_ids=ids, generated_tokens=len(ids),
            maximum_tokens=maximum, stopped_on_eos=ids[-1] in _eos_ids(tokenizer, model), score=score,
            cache_kv_shape_verified=True, prefill_kernel_calls=calls, smoke_repeated_logits_verified=smoke,
            prefill_seconds=prefill, elapsed_seconds=time.monotonic() - started,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        write_json(path, dict(status='complete', protocol=settings, result=result,
            command=shlex.join(sys.argv), python=sys.executable, gpu=torch.cuda.get_device_name(0)))
        print(f"sample={row['index']} task={row['task']} score={score} tokens={len(ids)} seconds={result['elapsed_seconds']:.2f}", flush=True)
    write_json(args.output_dir / args.stage / f'shard_{args.shard_index}.json', dict(status='complete',
        protocol=settings, indices=[r['index'] for r in assigned]))


if __name__ == '__main__':
    main()
