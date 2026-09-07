"""Original dense prefill, then compact C1-V80 full-exact-K decode."""

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from basisserve.checkpoint.gqa_vo_qwen3 import (
    install_qwen3_gqa_vo_als_export, transition_qwen3_dense_prefill_cache_to_c1,
)
from evaluation.eval_longbench_dense import inputs as dense_inputs, generate as dense_generate
from evaluation.eval_longbench_c1_fourarm import score_prediction
from evaluation.eval_qwen3_8b_residual_rank_ruler import greedy_decode, _eos_ids
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from evaluation.prepare_longbench_c1 import TASKS


def inputs(args):
    rows, tokens, previous, base, scorer = dense_inputs(args)
    dense = json.loads((args.dense_results / 'result.json').read_text())
    audit = json.loads((args.dense_results / 'audit.json').read_text())
    assert dense['status'] == audit['status'] == 'complete'
    assert sha256(args.dense_results / 'result.json') == audit['result_sha256']
    assert [r['sample'] for r in dense['records']] == rows
    assert dense['protocol']['dataset_manifest_sha256'] == base['dataset_manifest_sha256']
    manifest_hash = sha256(args.c1_checkpoint / 'results.json')
    assert manifest_hash == previous['protocol']['c1_manifest_sha256']
    settings = dict(format='basisserve.longbench_c1_denseprefill.v1',
        common_input_protocol=base, dense_result_sha256=audit['result_sha256'],
        checkpoint=str(args.c1_checkpoint.resolve()), c1_manifest_sha256=manifest_hash,
        prefill='original Qwen3Attention modules, dense K128/V128 SDPA, fresh cache per prompt',
        transition='each dense cached V @ C1 encoder in BF16; exact K tensors unchanged',
        decode='full exact-K, compact C1-V80, C1 output decoder, sdpa backend; no routing',
        first_token='argmax from original dense prefill, checked against saved dense baseline',
        code_sha256={name: sha256(ROOT / name) for name in (
            'evaluation/eval_longbench_c1_denseprefill.py',
            'basisserve/checkpoint/gqa_vo_qwen3.py')})
    return rows, tokens, previous, dense, settings, scorer


def activate(model, attentions):
    for layer, attention in zip(model.model.layers, attentions, strict=True):
        layer.self_attn = attention


@torch.inference_mode()
def generate(model, tokenizer, original, compressed, tokens, maximum, trace=False):
    # Keep the original modules outside the model across prompts. The conversion
    # helper intentionally drops its own references to dense V/O after prefill.
    activate(model, original)
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
    keys = [layer.keys for layer in cache.layers]
    # Small independent projection check on every layer, including first/last rows.
    expected = []
    for layer, attention in zip(cache.layers, compressed, strict=True):
        assert layer.keys.shape == layer.values.shape == (1, 8, len(tokens), 128)
        if trace:
            value = layer.values[0, :, [0, len(tokens) - 1], :]
            expected.append(torch.bmm(value, attention.value_coordinate_encoder.to(value)))
    activate(model, compressed)
    started = time.monotonic()
    cache = transition_qwen3_dense_prefill_cache_to_c1(model, cache, attention_backend='sdpa')
    for i, layer in enumerate(cache.layers):
        assert layer.keys is keys[i]
        assert layer.values.shape == (1, 8, len(tokens), 80)
        if trace:
            torch.testing.assert_close(layer.values[0, :, [0, len(tokens) - 1], :],
                                       expected[i], rtol=0, atol=0)
    del keys, expected
    torch.cuda.synchronize()
    conversion_seconds = time.monotonic() - started
    ids, cache, traces = greedy_decode(model, cache, first, maximum_tokens=maximum,
                                      eos_ids=_eos_ids(tokenizer, model), trace=trace)
    assert cache.get_seq_length() == len(tokens) + len(ids) - 1
    for layer in cache.layers:
        assert layer.keys.shape == (1, 8, cache.get_seq_length(), 128)
        assert layer.values.shape == (1, 8, cache.get_seq_length(), 80)
        assert layer.keys.dtype == layer.values.dtype == torch.bfloat16
    torch.cuda.synchronize()
    return ids, ([first_logits] + traces if trace else []), prefill_seconds, conversion_seconds


def summarize(args, rows, previous, dense, settings, scorer, tokenizer):
    records = []
    config = json.loads((args.model / 'config.json').read_text())
    eos = config['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    for row, baseline in zip(rows, dense['records'], strict=True):
        saved = json.loads((args.output_dir / 'evaluate' / f"sample_{row['index']:03d}.json").read_text())
        assert saved['status'] == 'complete' and saved['protocol'] == settings
        r = saved['result']
        ids = r['generated_token_ids']
        assert r['sample'] == row and r['cache_conversion_verified']
        assert r['first_token_matches_dense'] and ids[0] == baseline['generated_token_ids'][0]
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
        mean = 100 * sum(r['score'] for r in subset) / len(subset)
        assert round(mean, 2) == scorer.scorer(task, [r['prediction'] for r in subset],
            [r['sample']['answers'] for r in subset], subset[0]['sample']['all_classes'])
        tasks[task] = dict(dense=dense['tasks'][task]['dense_k_dense_v'],
            c1_prefill=previous['tasks'][task]['full_exact_k'], dense_prefill_c1_decode=mean)
    means = {arm: sum(v[arm] for v in tasks.values()) / len(TASKS) for arm in next(iter(tasks.values()))}
    paired = {}
    for name, baseline in [('dense', dense['records']),
                           ('c1_prefill', [r['arms']['full_exact_k'] for r in previous['records']])]:
        differences = [r['score'] - b['score'] for r, b in zip(records, baseline, strict=True)]
        paired[name] = dict(improvements=sum(d > 0 for d in differences),
            regressions=sum(d < 0 for d in differences), ties=sum(d == 0 for d in differences),
            mean_delta_pp=100 * sum(differences) / len(differences))
    result = dict(status='complete', protocol=settings, tasks=tasks, means=means, paired=paired,
        records=records, first_token_matches_dense=sum(r['first_token_matches_dense'] for r in records),
        cap_without_eos=sum(not r['stopped_on_eos'] for r in records),
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in records),
        command=shlex.join(sys.argv), python=sys.executable)
    write_json(args.output_dir / 'result.json', result)
    write_json(args.output_dir / 'audit.json', dict(status='complete', predictions_verified=len(records),
        first_tokens_match_dense=True, cache_conversion_and_shapes_verified=True,
        shard_coverage_verified=True, official_scores_verified=True,
        result_sha256=sha256(args.output_dir / 'result.json')))
    lines = ['# LongBench: dense prefill followed by C1-V80 decode', '',
        'Qwen3-8B-Base; basis; BF16; four independent L40S workers; same 192 frozen prompts, six tasks with 32 prompts each.', '',
        '| Task | Dense prefill + dense decode | C1 prefill + C1 decode | Dense prefill + C1 decode |',
        '|---|---:|---:|---:|']
    for task, scores in [*tasks.items(), ('Mean', means)]:
        lines.append('| ' + task + ' | ' + ' | '.join(f'{v:.4f}' for v in scores.values()) + ' |')
    lines += ['', 'Original Qwen3Attention modules perform every prefill. Cached BF16 V128 is then multiplied by the frozen C1 encoder to obtain V80; K tensors are unchanged.',
        'Decode scans all K and C1-V tokens with the C1 output decoder and SDPA backend. No routing, sparse pages, sidecar, CPU offload or refitting.',
        'The first generated token is dense-prefill argmax, verified against the saved dense baseline on all 192 prompts. C1 first participates in computing the second generated token.',
        'Same uniform C1-V80 checkpoint, C4 32 x 32K fit / 4 x 32K held-out. Greedy decoding and unchanged EOS/task caps.',
        '32K is the input-plus-reserved-output cap, not a fixed prompt length. Actual prompts: 1,192–30,431 tokens. This is a six-task pilot, not full LongBench.',
        'QA scores are official F1; summary scores are official ROUGE-L, all on a 0–100 scale. Mean is the arithmetic mean across six tasks.',
        'Old results are reused without modification. Switching prefill also changes subsequent hidden states and therefore the keys produced during decode.',
        '', f"First-token agreement with dense: {result['first_token_matches_dense']}/192.",
        f"Generation-cap exits without EOS: {result['cap_without_eos']}/192.",
        f"Maximum allocated GPU memory: {result['peak_allocated_gib']:.3f} GiB.",
        '', '## Paired score changes', '', '```json', json.dumps(paired, indent=2), '```', '']
    (args.output_dir / 'summary.md').write_text('\n'.join(lines))
    print(json.dumps({k: result[k] for k in ('means', 'paired', 'first_token_matches_dense', 'cap_without_eos')}, indent=2), flush=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('smoke', 'evaluate', 'summarize'), required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT / 'results/datasets/longbench_c1_32k')
    p.add_argument('--c1-results', type=Path, default=ROOT / 'results/evaluation/longbench_c1_32k')
    p.add_argument('--dense-results', type=Path, default=ROOT / 'results/evaluation/longbench_dense_32k')
    p.add_argument('--c1-checkpoint', type=Path, default=ROOT / 'results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/evaluation/longbench_c1_denseprefill_32k')
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
    original = [l.self_attn for l in model.model.layers]
    assert len(original) == 36 and all(type(a).__name__ == 'Qwen3Attention' for a in original)
    install_qwen3_gqa_vo_als_export(model, args.c1_checkpoint, attention_backend='sdpa')
    compressed = [l.self_attn for l in model.model.layers]
    model.eval()
    for a, b in zip(original, compressed, strict=True):
        assert a.q_proj is b.q_proj and a.k_proj is b.k_proj
        assert a.q_norm is b.q_norm and a.k_norm is b.k_norm
    smoke = args.stage == 'smoke'
    assigned = ([min(rows, key=lambda r:r['prompt_tokens']), max(rows, key=lambda r:r['prompt_tokens'])]
                if smoke else rows[args.shard_index::args.num_shards])
    for row in assigned:
        path = args.output_dir / args.stage / f"sample_{row['index']:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['status'] == 'complete' and saved['protocol'] == settings and saved['result']['sample'] == row
            continue
        tensor = tokens[f"sample_{row['index']:03d}"]
        maximum = 4 if smoke else row['maximum_tokens']
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        ids, trace, prefill, conversion = generate(model, tokenizer, original, compressed, tensor, maximum, trace=smoke)
        assert ids[0] == dense['records'][row['index']]['generated_token_ids'][0]
        if smoke:
            repeated, other_trace, _, _ = generate(model, tokenizer, original, compressed, tensor, maximum, trace=True)
            assert repeated == ids
            for a, b in zip(trace, other_trace, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            activate(model, original)
            _, dense_trace, _ = dense_generate(model, tokenizer, tensor, maximum, trace=True)
            torch.testing.assert_close(trace[0], dense_trace[0], rtol=0, atol=0)
        prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        score = None if smoke else score_prediction(scorer, row['task'], prediction, row['answers'], row['all_classes'])
        result = dict(sample=row, prediction=prediction, generated_token_ids=ids, generated_tokens=len(ids),
            maximum_tokens=maximum, stopped_on_eos=ids[-1] in _eos_ids(tokenizer, model), score=score,
            cache_conversion_verified=True, first_token_matches_dense=True,
            smoke_repeated_logits_and_dense_prefill_logits_verified=smoke,
            prefill_seconds=prefill, conversion_seconds=conversion, elapsed_seconds=time.monotonic() - started,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        write_json(path, dict(status='complete', protocol=settings, result=result,
            command=shlex.join(sys.argv), python=sys.executable, gpu=torch.cuda.get_device_name(0)))
        print(f"sample={row['index']} task={row['task']} score={score} tokens={len(ids)} seconds={result['elapsed_seconds']:.2f}", flush=True)
    write_json(args.output_dir / args.stage / f'shard_{args.shard_index}.json', dict(status='complete',
        protocol=settings, indices=[r['index'] for r in assigned]))


if __name__ == '__main__':
    main()
