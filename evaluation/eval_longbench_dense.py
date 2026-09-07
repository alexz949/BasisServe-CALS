"""Unmodified BF16 K128/V128 baseline on the frozen C1 LongBench pilot."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

import torch
import transformers
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.eval_longbench_c1_fourarm import ARMS, official_scorer, score_prediction
from evaluation.eval_qwen3_8b_residual_rank_ruler import greedy_decode, _eos_ids
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from evaluation.prepare_longbench_c1 import TASKS


def inputs(args):
    manifest = json.loads((args.data_dir / 'manifest.json').read_text())
    spec = manifest['protocol']
    assert manifest['status'] == 'complete'
    assert sha256(args.model / 'config.json') == spec['model_config_sha256']
    assert sha256(args.data_dir / 'samples.json') == manifest['samples_sha256']
    assert sha256(args.data_dir / 'tokens.safetensors') == manifest['tokens_sha256']
    previous = json.loads((args.c1_results / 'result.json').read_text())
    audit = json.loads((args.c1_results / 'audit.json').read_text())
    assert previous['status'] == audit['status'] == 'complete'
    assert sha256(args.c1_results / 'result.json') == audit['result_sha256']
    assert sha256(args.data_dir / 'manifest.json') == previous['protocol']['dataset_manifest_sha256']
    assert str(args.model.resolve()) == previous['protocol']['model']
    rows = json.loads((args.data_dir / 'samples.json').read_text())
    tokens = load_file(str(args.data_dir / 'tokens.safetensors'))
    assert len(rows) == len(tokens) == len(previous['records']) == 192
    for i, (row, old) in enumerate(zip(rows, previous['records'], strict=True)):
        tensor = tokens[f'sample_{i:03d}']
        assert row == old['sample'] and row['index'] == i
        assert len(tensor) == row['prompt_tokens']
        assert len(tensor) + row['maximum_tokens'] <= 32768
        assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == row['input_ids_sha256']
    official = Path(spec['official_root'])
    for name, digest in spec['official_sha256'].items():
        assert sha256(official / 'LongBench' / name) == digest
    settings = dict(format='basisserve.longbench_dense.v1', model=str(args.model.resolve()),
        model_config_sha256=spec['model_config_sha256'],
        dataset_manifest_sha256=sha256(args.data_dir / 'manifest.json'),
        c1_result_sha256=audit['result_sha256'], dtype='bfloat16',
        prefill='full-sequence unmodified dense K128/V128 SDPA',
        decode='unmodified dense K128/V128 SDPA; explicit full-support mask',
        sampling='greedy; same tokenizer/model EOS and official task caps; no chat template',
        sequence_length=32768, num_shards=args.num_shards,
        torch=torch.__version__, transformers=transformers.__version__,
        code_sha256={n: sha256(ROOT / n) for n in (
            'evaluation/eval_longbench_dense.py', 'evaluation/eval_longbench_c1_fourarm.py',
            'evaluation/eval_qwen3_8b_residual_rank_ruler.py')})
    return rows, tokens, previous, settings, official_scorer(official)


@torch.inference_mode()
def generate(model, tokenizer, tokens, maximum, trace=False):
    started = time.monotonic()
    cache = DynamicCache(config=model.config)
    output = model(input_ids=tokens.long()[None].to('cuda:0'), past_key_values=cache,
                   use_cache=True, logits_to_keep=1)
    logits = output.logits[0, -1]
    assert torch.isfinite(logits).all()
    first = int(logits.argmax())
    first_logits = logits.cpu() if trace else None
    cache = output.past_key_values
    del output, logits
    torch.cuda.synchronize()
    prefill_seconds = time.monotonic() - started
    ids, cache, traces = greedy_decode(model, cache, first, maximum_tokens=maximum,
                                      eos_ids=_eos_ids(tokenizer, model), trace=trace)
    assert cache.get_seq_length() == len(tokens) + len(ids) - 1
    assert len(cache.layers) == 36
    for layer in cache.layers:
        assert layer.keys.shape == layer.values.shape == (1, 8, cache.get_seq_length(), 128)
        assert layer.keys.dtype == layer.values.dtype == torch.bfloat16
    torch.cuda.synchronize()
    return ids, ([first_logits] + traces if trace else []), prefill_seconds


def summarize(args, rows, previous, settings, scorer, tokenizer):
    records = []
    model_eos = json.loads((args.model / 'config.json').read_text())['eos_token_id']
    eos_ids = set(model_eos if isinstance(model_eos, list) else [model_eos])
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    for row in rows:
        saved = json.loads((args.output_dir / 'evaluate' / f"sample_{row['index']:03d}.json").read_text())
        assert saved['status'] == 'complete' and saved['protocol'] == settings
        r = saved['result']
        assert r['sample'] == row and r['cache_kv_shape_verified']
        ids = r['generated_token_ids']
        assert 0 < len(ids) == r['generated_tokens'] <= row['maximum_tokens'] == r['maximum_tokens']
        assert r['stopped_on_eos'] == (ids[-1] in eos_ids)
        assert not any(t in eos_ids for t in ids[:-1])
        assert r['stopped_on_eos'] or len(ids) == row['maximum_tokens']
        assert r['prediction'] == tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assert r['score'] == score_prediction(scorer, row['task'], r['prediction'], row['answers'], row['all_classes'])
        records.append(r)
    for shard in range(args.num_shards):
        s = json.loads((args.output_dir / 'evaluate' / f'shard_{shard}.json').read_text())
        assert s['status'] == 'complete' and s['protocol'] == settings
        assert s['indices'] == list(range(shard, 192, args.num_shards))
    tasks = {}
    for task in TASKS:
        subset = [r for r in records if r['sample']['task'] == task]
        assert len(subset) == 32
        mean = 100 * sum(r['score'] for r in subset) / 32
        official = scorer.scorer(task, [r['prediction'] for r in subset],
                                 [r['sample']['answers'] for r in subset], subset[0]['sample']['all_classes'])
        assert round(mean, 2) == official
        tasks[task] = {'dense_k_dense_v': mean, **previous['tasks'][task]}
    arms = ('dense_k_dense_v', *ARMS)
    means = {arm: sum(t[arm] for t in tasks.values()) / len(TASKS) for arm in arms}
    pairs = {}
    for arm in ARMS:
        delta = [old['arms'][arm]['score'] - new['score'] for new, old in zip(records, previous['records'], strict=True)]
        pairs[arm + '_vs_dense_k_dense_v'] = dict(improvements=sum(d > 0 for d in delta),
            regressions=sum(d < 0 for d in delta), ties=sum(d == 0 for d in delta),
            mean_delta_pp=means[arm] - means['dense_k_dense_v'])
    result = dict(status='complete', protocol=settings, tasks=tasks, means=means, paired=pairs,
                  records=records, lengths=previous['lengths'], command=shlex.join(sys.argv), python=sys.executable)
    write_json(args.output_dir / 'result.json', result)
    write_json(args.output_dir / 'audit.json', dict(status='complete', predictions_verified=192,
        same_inputs_as_c1=True, official_scores_verified=True, shard_coverage_verified=True,
        token_decoding_and_eos_verified=True, result_sha256=sha256(args.output_dir / 'result.json')))
    lines = ['# LongBench-v1 dense K/V and C1-V80 comparison', '',
        'Qwen3-8B-Base; 192 frozen prompts, six tasks with 32 prompts each. BF16, basis, four L40S workers.', '',
        '| Task | Dense K + dense V | Full K + V80 | Exact sparse + V80 | Query-Gram Q32 + V80 | Terminal Q32 + V80 |',
        '|---|---:|---:|---:|---:|---:|']
    for task, scores in [*tasks.items(), ('Mean', means)]:
        lines.append('| ' + task + ' | ' + ' | '.join(f'{scores[a]:.4f}' for a in arms) + ' |')
    lines += ['', 'QA: official F1; summaries: official ROUGE-L; scores on a 0–100 scale, not all accuracies.', '',
        '32K is the input plus reserved generation cap. Actual inputs: 1,192–30,431 tokens; mean 9,244.59; none truncated.',
        'The new baseline uses unmodified dense K128/V128 in BOTH prefill and decode. Its first token is not forced to match C1.',
        'The existing four C1 arms share full-C1 prefill; their attention differences apply during decode. Sparse arms use Page32/B2048 with pinned page 0.',
        'Base16/R8 banks are frozen: Query-Gram uses 8 calibration Q in each of four 8K bins; terminal uses 32 Q in the last 8K.',
        'Same saved input tokens, references, greedy decoding, EOS policy and generation caps (128/64/32/32/512/512). No benchmark fitting or tuning.',
        'This is a six-task pilot, not full LongBench. The dense versus C1 comparison also differs in prefill backend (SDPA versus C1 Triton); it is not a pure matched-kernel V-compression ablation.',
        '', '## Paired score changes relative to dense K/V', '', '```json', json.dumps(pairs, indent=2), '```', '',
        'All 192 dense predictions were re-decoded and rescored; task means checked against the official scorer. Inputs match the previously independently audited dataset.',
        'Commands and full execution settings are recorded in docs/longbench_dense_protocol.md and each sample JSON.', '']
    (args.output_dir / 'summary.md').write_text('\n'.join(lines))
    print(json.dumps({'means': means, 'paired': pairs}, indent=2), flush=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('smoke', 'evaluate', 'summarize'), required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'results/datasets/longbench_c1_32k')
    p.add_argument('--c1-results', type=Path, default=ROOT / 'results/evaluation/longbench_c1_32k')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/evaluation/longbench_dense_32k')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=4)
    args = p.parse_args()
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    rows, tokens, previous, settings, scorer = inputs(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.stage == 'summarize':
        summarize(args, rows, previous, settings, scorer, tokenizer)
        return
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA L40S'
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation='sdpa').to('cuda:0').eval()
    assert len(model.model.layers) == 36 and not model.model.has_sliding_layers
    assert all(type(layer.self_attn).__name__ == 'Qwen3Attention' for layer in model.model.layers)
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
        cap = 4 if smoke else row['maximum_tokens']
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        ids, trace, prefill = generate(model, tokenizer, tensor, cap, trace=smoke)
        if smoke:
            repeated, repeated_trace, _ = generate(model, tokenizer, tensor, cap, trace=True)
            assert ids == repeated
            for a, b in zip(trace, repeated_trace, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            native = model.generate(input_ids=tensor.long()[None].to('cuda:0'),
                max_new_tokens=cap, do_sample=False, eos_token_id=list(_eos_ids(tokenizer, model)),
                pad_token_id=tokenizer.eos_token_id, use_cache=True)
            assert native[0, len(tensor):].tolist() == ids
            del native
        prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        score = None if smoke else score_prediction(scorer, row['task'], prediction, row['answers'], row['all_classes'])
        result = dict(sample=row, prediction=prediction, generated_token_ids=ids, generated_tokens=len(ids),
            maximum_tokens=cap, stopped_on_eos=ids[-1] in _eos_ids(tokenizer, model), score=score,
            cache_kv_shape_verified=True, smoke_repeated_logits_and_native_generate_verified=smoke,
            prefill_seconds=prefill, elapsed_seconds=time.monotonic() - started,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        write_json(path, dict(status='complete', protocol=settings, result=result, command=shlex.join(sys.argv),
                             python=sys.executable, gpu=torch.cuda.get_device_name(0)))
        print(f"sample={row['index']} task={row['task']} score={score} tokens={len(ids)} seconds={result['elapsed_seconds']:.2f}", flush=True)
    write_json(args.output_dir / args.stage / f'shard_{args.shard_index}.json', dict(status='complete',
        protocol=settings, indices=[r['index'] for r in assigned]))


if __name__ == '__main__':
    main()
