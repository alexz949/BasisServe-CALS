"""Frozen two-sided-KL rank80 allocation, original dense prefill, compact C1 decode."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention, transition_qwen3_dense_prefill_cache_to_c1
from evaluation.eval_longbench_c1_denseprefill import inputs as uniform_inputs, activate
from evaluation.eval_longbench_dense import generate as dense_generate
from evaluation.eval_longbench_c1_fourarm import score_prediction
from evaluation.eval_qwen3_8b_residual_rank_ruler import greedy_decode, _eos_ids
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from evaluation.prepare_longbench_c1 import TASKS


def inputs(args):
    rows, tokens, _, dense, base, scorer = uniform_inputs(args)
    uniform = json.loads((args.uniform_results / 'result.json').read_text())
    audit = json.loads((args.uniform_results / 'audit.json').read_text())
    assert uniform['status'] == audit['status'] == 'complete'
    assert sha256(args.uniform_results / 'result.json') == audit['result_sha256']
    assert uniform['protocol'] == base and [r['sample'] for r in uniform['records']] == rows
    manifest = json.loads((args.allocation / 'manifest.json').read_text())
    allocation = json.loads((args.allocation / 'result.json').read_text())
    assert manifest['status'] == allocation['status'] == 'complete'
    assert sha256(args.allocation / 'result.json') == manifest['artifact']['sha256']
    assert manifest['model']['config_sha256'] == sha256(args.model / 'config.json')
    selection = allocation['selection']
    ranks = manifest['compression']['layer_ranks']
    assert ranks == selection['selected_schedule']
    assert len(ranks) == 36 and all(len(r) == 8 and len(set(r)) == 1 for r in ranks)
    assert sum(map(sum, ranks)) == 36 * 8 * 80
    assert selection['factorized_method']['exponent'] == 1
    assert selection['selected_candidate'] == 'two_sided_factorized_kl'
    assert allocation['factor_sources']['80']['results_sha256'] == base['c1_manifest_sha256']
    for layer in manifest['layers']:
        assert layer['ranks'] == ranks[layer['layer']]
        assert sha256(args.allocation / layer['file']) == layer['sha256']
        assert layer['sha256'] == allocation['selected_artifacts'][str(layer['layer'])]['sha256']
    settings = dict(format='basisserve.longbench_c1_twosided_denseprefill.v1',
        uniform_input_protocol=base, uniform_result_sha256=audit['result_sha256'],
        allocation=str(args.allocation.resolve()), manifest_sha256=sha256(args.allocation / 'manifest.json'),
        allocation_result_sha256=manifest['artifact']['sha256'],
        layer_ranks=[r[0] for r in ranks], average_rank=80, alpha=1,
        rank_histogram=dict(sorted(Counter(r[0] for r in ranks).items())),
        prefill='original dense K128/V128 SDPA, fresh cache and restored original modules per prompt',
        decode='full exact-K, compact per-layer C1 latent and output decoder, sdpa backend, no routing',
        calibration=dict(factor_fit_windows=32, factor_heldout_windows=4, sequence_length=32768,
            profile_windows=32, confirmation_windows=12, anchor_rank=64, probes=[32,96],
            terminal_positions_per_window=1024, local_error='heldout relative MSE'),
        code_sha256={n: sha256(ROOT / n) for n in (
            'evaluation/eval_longbench_c1_twosided_denseprefill.py',
            'evaluation/eval_longbench_c1_denseprefill.py', 'basisserve/checkpoint/gqa_vo_qwen3.py')})
    # JSON serializes histogram keys as strings; keep in-memory and saved protocols identical.
    settings = json.loads(json.dumps(settings))
    return rows, tokens, dense, uniform, manifest, settings, scorer


@torch.inference_mode()
def install(model, checkpoint, manifest):
    originals = [l.self_attn for l in model.model.layers]
    for layer, record in zip(model.model.layers, manifest['layers'], strict=True):
        original = layer.self_attn
        payload = load_file(str(checkpoint / record['file']))
        rank = record['ranks'][0]
        assert payload['source_ranks'].tolist() == record['ranks']
        encoder, decoder = payload['value_coordinate_encoders'], payload['head_output_decoders']
        assert encoder.shape == (8, 128, rank) and decoder.shape == (32, rank, 4096)
        assert encoder.dtype == decoder.dtype == torch.bfloat16
        assert torch.isfinite(encoder).all() and torch.isfinite(decoder).all()
        assert original.v_proj.bias is None and original.o_proj.bias is None
        device = original.v_proj.weight.device
        projected = torch.bmm(encoder.to(device=device, dtype=torch.float32).transpose(1, 2),
            original.v_proj.weight.float().reshape(8, 128, 4096)).reshape(8 * rank, 4096)
        output = decoder.to(device=device, dtype=torch.float32).permute(2, 0, 1).reshape(4096, 32 * rank)
        layer.self_attn = GQATiedVOQwen3Attention(original,
            v_proj_compressed_weight=projected, o_decoder_weight=output,
            attention_backend='sdpa', value_coordinate_encoder=encoder)
        replacement = layer.self_attn
        assert replacement.q_proj is original.q_proj and replacement.k_proj is original.k_proj
        assert replacement.q_norm is original.q_norm and replacement.k_norm is original.k_norm
        assert torch.equal(replacement.v_proj.weight, projected.bfloat16())
        assert torch.equal(replacement.o_proj.weight, output.bfloat16())
        assert torch.equal(replacement.value_coordinate_encoder.cpu(), encoder)
    model.eval()
    return originals, [l.self_attn for l in model.model.layers]


@torch.inference_mode()
def generate(model, tokenizer, original, compressed, tokens, cap, trace=False):
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
    prefill = time.monotonic() - started
    keys, expected = [], []
    for layer, attention in zip(cache.layers, compressed, strict=True):
        assert layer.keys.shape == layer.values.shape == (1, 8, len(tokens), 128)
        keys.append(layer.keys)
        if trace:
            value = layer.values[0, :, [0, len(tokens)-1], :]
            expected.append(torch.bmm(value, attention.value_coordinate_encoder.to(value)))
    activate(model, compressed)
    started = time.monotonic()
    cache = transition_qwen3_dense_prefill_cache_to_c1(model, cache, attention_backend='sdpa')
    for i, (layer, attention) in enumerate(zip(cache.layers, compressed, strict=True)):
        assert layer.keys is keys[i]
        assert layer.values.shape == (1, 8, len(tokens), attention.value_coordinate_encoder.shape[-1])
        if trace:
            torch.testing.assert_close(layer.values[0, :, [0, len(tokens)-1], :], expected[i], rtol=0, atol=0)
    del keys, expected
    torch.cuda.synchronize()
    conversion = time.monotonic() - started
    ids, cache, traces = greedy_decode(model, cache, first, maximum_tokens=cap,
                                      eos_ids=_eos_ids(tokenizer, model), trace=trace)
    assert cache.get_seq_length() == len(tokens) + len(ids) - 1
    for layer, attention in zip(cache.layers, compressed, strict=True):
        assert layer.keys.shape == (1, 8, cache.get_seq_length(), 128)
        assert layer.values.shape == (1, 8, cache.get_seq_length(), attention.value_coordinate_encoder.shape[-1])
        assert layer.keys.dtype == layer.values.dtype == torch.bfloat16
    torch.cuda.synchronize()
    return ids, ([first_logits] + traces if trace else []), prefill, conversion


def summarize(args, rows, dense, uniform, settings, scorer, tokenizer):
    config = json.loads((args.model / 'config.json').read_text())
    eos = config['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    records = []
    for row, baseline in zip(rows, dense['records'], strict=True):
        saved = json.loads((args.output_dir / 'evaluate' / f"sample_{row['index']:03d}.json").read_text())
        assert saved['status'] == 'complete' and saved['protocol'] == settings
        r = saved['result']; ids = r['generated_token_ids']
        assert r['sample'] == row and r['cache_conversion_verified'] and r['first_token_matches_dense']
        assert ids[0] == baseline['generated_token_ids'][0]
        assert 0 < len(ids) == r['generated_tokens'] <= r['maximum_tokens'] == row['maximum_tokens']
        assert r['stopped_on_eos'] == (ids[-1] in eos) and not any(i in eos for i in ids[:-1])
        assert r['stopped_on_eos'] or len(ids) == row['maximum_tokens']
        assert r['prediction'] == tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assert r['score'] == score_prediction(scorer, row['task'], r['prediction'], row['answers'], row['all_classes'])
        records.append(r)
    for shard in range(args.num_shards):
        saved = json.loads((args.output_dir / 'evaluate' / f'shard_{shard}.json').read_text())
        assert saved['status'] == 'complete' and saved['protocol'] == settings
        assert saved['indices'] == list(range(shard, 192, args.num_shards))
    tasks = {}
    for task in TASKS:
        subset = [r for r in records if r['sample']['task'] == task]
        assert len(subset) == 32
        mean = 100 * sum(r['score'] for r in subset) / len(subset)
        assert round(mean, 2) == scorer.scorer(task, [r['prediction'] for r in subset],
            [r['sample']['answers'] for r in subset], subset[0]['sample']['all_classes'])
        tasks[task] = dict(dense=dense['tasks'][task]['dense_k_dense_v'],
            uniform80=uniform['tasks'][task]['dense_prefill_c1_decode'], two_sided80=mean)
    means = {arm: sum(t[arm] for t in tasks.values()) / len(TASKS) for arm in next(iter(tasks.values()))}
    paired = {}
    for name, baseline in [('dense', dense), ('uniform80', uniform)]:
        delta = [r['score'] - b['score'] for r, b in zip(records, baseline['records'], strict=True)]
        paired[name] = dict(improvements=sum(x>0 for x in delta), regressions=sum(x<0 for x in delta),
            ties=sum(x==0 for x in delta), mean_delta_pp=100*sum(delta)/len(delta))
    result = dict(status='complete', protocol=settings, tasks=tasks, means=means, paired=paired,
        records=records, cap_without_eos=sum(not r['stopped_on_eos'] for r in records),
        first_token_matches_dense=sum(r['first_token_matches_dense'] for r in records),
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in records),
        command=shlex.join(sys.argv), python=sys.executable)
    write_json(args.output_dir / 'result.json', result)
    write_json(args.output_dir / 'audit.json', dict(status='complete', predictions_verified=192,
        first_tokens_match_dense=True, shard_coverage_verified=True, official_scores_verified=True,
        rank_budget_verified=True, result_sha256=sha256(args.output_dir / 'result.json')))
    lines = ['# LongBench: two-sided KL versus uniform C1 at average rank80', '',
        'Qwen3-8B-Base, BF16, basis, four L40S workers. All arms use ORIGINAL dense prefill. C1 arms convert the dense V cache and then use full-exact-K compact C1 decode with the SDPA backend. No routing, sparse attention, offload or refitting.', '',
        '| Task | Dense decode | Uniform V80 decode | Two-sided KL avg80 decode |', '|---|---:|---:|---:|']
    for task, values in [*tasks.items(), ('Mean', means)]:
        lines.append('| '+task+' | '+' | '.join(f'{v:.4f}' for v in values.values())+' |')
    lines += ['', 'Same 192 frozen LongBench-v1 prompts, six tasks with 32 each. Official QA F1 and summary ROUGE-L, on a 0–100 scale; six-task arithmetic mean. Not full LongBench.',
        'Same input tokens, greedy decoding, EOS and task caps. Input-plus-generation cap 32K; actual prompts 1,192–30,431 tokens.',
        'Frozen allocation: alpha=1, anchor64, probes32/96, rank bank32/48/64/80/96/112/128. Rank counts: 3/3/11/6/5/3/5 layers. Average rank exactly80, same total cache budget as uniform80.',
        'Factor bank: C4 32 x 32K fit, 4 x 32K held-out local MSE, six ALS sweeps. Separate allocation profile: 32 x 32K C4 windows; confirmation: 12 x 32K; 1,024 sampled terminal positions/window. No LongBench calibration.',
        'Each layer retains its actual rank without padding to128. First token comes from original dense prefill; C1 computes the second and subsequent generated tokens.',
        '', f"First-token agreement with dense: {result['first_token_matches_dense']}/192.",
        f"Generation-cap exits without EOS: {result['cap_without_eos']}/192.",
        f"Maximum allocated GPU memory: {result['peak_allocated_gib']:.3f} GiB.",
        '', '## Layer ranks (0–35)', '', str(settings['layer_ranks']), '',
        '## Paired score changes', '', '```json', json.dumps(paired, indent=2), '```', '']
    (args.output_dir / 'summary.md').write_text('\n'.join(lines))
    print(json.dumps({k: result[k] for k in ('means','paired','first_token_matches_dense','cap_without_eos')}, indent=2), flush=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('smoke','evaluate','summarize'), required=True)
    p.add_argument('--model', type=Path, required=True)
    for name, default in {
        'data-dir':'results/datasets/longbench_c1_32k',
        'c1-results':'results/evaluation/longbench_c1_32k',
        'dense-results':'results/evaluation/longbench_dense_32k',
        'uniform-results':'results/evaluation/longbench_c1_denseprefill_32k',
        'c1-checkpoint':'results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6',
        'allocation':'results/checkpoints/qwen3_8b_c1_twosided_r80_c4_32x32k',
        'output-dir':'results/evaluation/longbench_c1_kl_denseprefill_32k',
    }.items():
        p.add_argument('--'+name, type=Path, default=ROOT / default)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=4)
    args = p.parse_args()
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    rows, tokens, dense, uniform, manifest, settings, scorer = inputs(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.stage == 'summarize':
        summarize(args, rows, dense, uniform, settings, scorer, tokenizer)
        return
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA L40S'
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation='sdpa').to('cuda:0').eval()
    original, compressed = install(model, args.allocation, manifest)
    smoke = args.stage == 'smoke'
    assigned = ([min(rows,key=lambda r:r['prompt_tokens']),max(rows,key=lambda r:r['prompt_tokens'])]
                if smoke else rows[args.shard_index::args.num_shards])
    for row in assigned:
        path = args.output_dir / args.stage / f"sample_{row['index']:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['status'] == 'complete' and saved['protocol'] == settings and saved['result']['sample'] == row
            continue
        tensor = tokens[f"sample_{row['index']:03d}"]; cap = 4 if smoke else row['maximum_tokens']
        started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
        ids, trace, prefill, conversion = generate(model, tokenizer, original, compressed, tensor, cap, smoke)
        assert ids[0] == dense['records'][row['index']]['generated_token_ids'][0]
        if smoke:
            repeated, repeated_trace, _, _ = generate(model, tokenizer, original, compressed, tensor, cap, True)
            assert repeated == ids
            for a,b in zip(trace, repeated_trace, strict=True):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            activate(model, original)
            _, dense_trace, _ = dense_generate(model,tokenizer,tensor,cap,trace=True)
            torch.testing.assert_close(trace[0],dense_trace[0],rtol=0,atol=0)
        prediction = tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        score = None if smoke else score_prediction(scorer,row['task'],prediction,row['answers'],row['all_classes'])
        result = dict(sample=row,prediction=prediction,generated_token_ids=ids,generated_tokens=len(ids),
            maximum_tokens=cap,stopped_on_eos=ids[-1] in _eos_ids(tokenizer,model),score=score,
            cache_conversion_verified=True,first_token_matches_dense=True,smoke_repeated_and_dense_logits_verified=smoke,
            prefill_seconds=prefill,conversion_seconds=conversion,elapsed_seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(path,dict(status='complete',protocol=settings,result=result,command=shlex.join(sys.argv),
            python=sys.executable,gpu=torch.cuda.get_device_name(0)))
        print(f"sample={row['index']} task={row['task']} score={score} tokens={len(ids)} seconds={result['elapsed_seconds']:.2f}",flush=True)
    write_json(args.output_dir / args.stage / f'shard_{args.shard_index}.json',dict(status='complete',
        protocol=settings,indices=[r['index'] for r in assigned]))


if __name__ == '__main__':
    main()
