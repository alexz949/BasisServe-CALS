"""Matched-calibration PaLU M/G4 full-exact-K LongBench quality controls."""

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.eval_longbench_dense import inputs as dense_inputs, generate
from evaluation.eval_longbench_c1_fourarm import ARMS, score_prediction
from evaluation.eval_qwen3_8b_residual_rank_ruler import _eos_ids
from evaluation.eval_gqa_palu_m_wikitext import install_palu_m_factors
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from evaluation.prepare_longbench_c1 import TASKS

CHECKPOINTS = {
    'm': ROOT / 'results/checkpoints/palu_m_fisher_r80_c1matched32k',
    'g4': ROOT / 'results/checkpoints/palu_g4_fisher_r80_c1matched32k',
}


def arm_inputs(args, arm):
    rows, tokens, c1, base, scorer = dense_inputs(args)
    dense = json.loads((args.dense_results / 'result.json').read_text())
    audit = json.loads((args.dense_results / 'audit.json').read_text())
    assert dense['status'] == audit['status'] == 'complete'
    assert sha256(args.dense_results / 'result.json') == audit['result_sha256']
    assert dense['protocol']['dataset_manifest_sha256'] == base['dataset_manifest_sha256']
    assert [r['sample'] for r in dense['records']] == rows
    checkpoint = CHECKPOINTS[arm]
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    compression = manifest['compression']
    assert compression['key_cache'] == 'dense' and compression['projection'] == 'v'
    assert manifest['model']['config_sha256'] == base['model_config_sha256']
    calibration = manifest['calibration']
    assert calibration['samples'] == 32 and calibration['sequence_length'] == 32768
    fisher_path = Path(manifest['fisher']['result'])
    assert sha256(fisher_path) == manifest['fisher']['result_sha256']
    fisher = json.loads(fisher_path.read_text())
    matched = fisher['matched_protocol']
    assert matched['exact_c1_fit_tokens_verified'] and not matched['heldout_windows_used']
    assert matched['c1_manifest_sha256'] == c1['protocol']['c1_manifest_sha256']
    assert calibration['windows']['sha256'] == matched['windows_sha256']
    assert calibration['whitening_artifact_sha256'] == matched['whitening_sha256']
    path = checkpoint / manifest['artifact']['file']
    assert sha256(path) == manifest['artifact']['sha256']
    factors = load_file(str(path))
    ranks = compression['layer_ranks']
    groups = 8 if arm == 'm' else 2
    assert len(ranks) == 36 and len(factors) == 72
    for l, r in enumerate(ranks):
        assert len(r) == groups and len(set(r)) == 1
        assert factors[f'layers.{l}.v_writer.weight'].shape == (sum(r), 4096)
        assert factors[f'layers.{l}.v_decoder.weight'].shape == (groups, 1024 // groups, max(r))
    assert all(t.dtype == torch.bfloat16 and torch.isfinite(t).all() for t in factors.values())
    settings = dict(format='basisserve.longbench_palu_matched.v1', arm=arm,
        common_dense_input_protocol=base, dense_result_sha256=audit['result_sha256'],
        checkpoint=str(checkpoint), checkpoint_manifest_sha256=sha256(checkpoint / 'manifest.json'),
        factor_sha256=manifest['artifact']['sha256'], fisher_sha256=manifest['fisher']['result_sha256'],
        calibration=calibration, head_group_size=8 // groups, layer_ranks=ranks,
        actual_average_rank=sum(map(sum, ranks)) / 288,
        runtime='BF16 latent V writer plus per-group V reconstruction; native dense SDPA prefill/decode; exact K unchanged',
        cache='standard dense K128/V128 cache containing approximate reconstructed V; quality-only, not compact-cache performance',
        code_sha256={name: sha256(ROOT / name) for name in (
            'evaluation/eval_longbench_palu.py', 'evaluation/eval_longbench_dense.py',
            'evaluation/eval_gqa_palu_m_wikitext.py')})
    return rows, tokens, dense, c1, settings, factors, scorer


@torch.inference_mode()
def run(args):
    rows, tokens, _, _, settings, factors, scorer = arm_inputs(args, args.arm)
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA L40S'
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation='sdpa').to('cuda:0').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    original = [(l.self_attn.q_proj, l.self_attn.k_proj, l.self_attn.o_proj) for l in model.model.layers]
    install_palu_m_factors(model, factors, layer_ranks=settings['layer_ranks'], head_dim=128,
                          require_cuda_resident=True)
    assert len(model.model.layers) == 36 and not model.model.has_sliding_layers
    for l, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        assert (attn.q_proj, attn.k_proj, attn.o_proj) == original[l]
        assert torch.equal(attn.v_proj.VT.weight.cpu(), factors[f'layers.{l}.v_writer.weight'])
        assert torch.equal(torch.stack([up.weight.cpu() for up in attn.v_proj.U]), factors[f'layers.{l}.v_decoder.weight'])
    del factors, original
    smoke = args.stage == 'smoke'
    assigned = ([min(rows, key=lambda r:r['prompt_tokens']), max(rows, key=lambda r:r['prompt_tokens'])]
                if smoke else rows[args.shard_index::args.num_shards])
    for row in assigned:
        path = args.output_dir / args.arm / args.stage / f"sample_{row['index']:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['status'] == 'complete' and saved['protocol'] == settings and saved['result']['sample'] == row
            continue
        tensor = tokens[f"sample_{row['index']:03d}"]
        cap = 4 if smoke else row['maximum_tokens']
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        ids, trace, prefill = generate(model, tokenizer, tensor, cap, trace=smoke)
        if smoke:
            repeated, repeated_trace, _ = generate(model, tokenizer, tensor, cap, trace=True)
            assert repeated == ids
            for a, b in zip(trace, repeated_trace, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            native = model.generate(input_ids=tensor.long()[None].to('cuda:0'), max_new_tokens=cap,
                do_sample=False, eos_token_id=list(_eos_ids(tokenizer, model)),
                pad_token_id=tokenizer.eos_token_id, use_cache=True)
            assert native[0, len(tensor):].tolist() == ids
            del native
        prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        score = None if smoke else score_prediction(scorer, row['task'], prediction, row['answers'], row['all_classes'])
        result = dict(sample=row, prediction=prediction, generated_token_ids=ids, generated_tokens=len(ids),
            score=score, maximum_tokens=cap, stopped_on_eos=ids[-1] in _eos_ids(tokenizer, model),
            cache_kv_shape_verified=True, factor_installation_verified=True,
            smoke_repeated_logits_and_native_generate_verified=smoke,
            prefill_seconds=prefill, elapsed_seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(path, dict(status='complete', protocol=settings, result=result, command=shlex.join(sys.argv),
                             python=sys.executable, gpu=torch.cuda.get_device_name(0)))
        print(f"arm={args.arm} sample={row['index']} task={row['task']} score={score} tokens={len(ids)} seconds={result['elapsed_seconds']:.2f}", flush=True)
    write_json(args.output_dir / args.arm / args.stage / f'shard_{args.shard_index}.json',
               dict(status='complete', protocol=settings, indices=[r['index'] for r in assigned]))


def summarize(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model_eos = json.loads((args.model / 'config.json').read_text())['eos_token_id']
    eos = set(model_eos if isinstance(model_eos, list) else [model_eos])
    if tokenizer.eos_token_id is not None:
        eos.add(tokenizer.eos_token_id)
    protocols, records = {}, {}
    for arm in CHECKPOINTS:
        rows, _, dense, c1, settings, _, scorer = arm_inputs(args, arm)
        protocols[arm] = settings
        records[arm] = []
        for row in rows:
            saved = json.loads((args.output_dir / arm / 'evaluate' / f"sample_{row['index']:03d}.json").read_text())
            assert saved['status'] == 'complete' and saved['protocol'] == settings
            r = saved['result']
            ids = r['generated_token_ids']
            assert r['sample'] == row and r['cache_kv_shape_verified'] and r['factor_installation_verified']
            assert 0 < len(ids) == r['generated_tokens'] <= row['maximum_tokens'] == r['maximum_tokens']
            assert r['stopped_on_eos'] == (ids[-1] in eos) and not any(t in eos for t in ids[:-1])
            assert r['stopped_on_eos'] or len(ids) == row['maximum_tokens']
            assert r['prediction'] == tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            assert r['score'] == score_prediction(scorer, row['task'], r['prediction'], row['answers'], row['all_classes'])
            records[arm].append(r)
        for shard in range(args.num_shards):
            s = json.loads((args.output_dir / arm / 'evaluate' / f'shard_{shard}.json').read_text())
            assert s['status'] == 'complete' and s['protocol'] == settings
            assert s['indices'] == list(range(shard, 192, args.num_shards))
    assert protocols['m']['fisher_sha256'] == protocols['g4']['fisher_sha256']
    tasks = {}
    for task in TASKS:
        tasks[task] = dict(dense['tasks'][task])
        for arm in CHECKPOINTS:
            subset = [r for r in records[arm] if r['sample']['task'] == task]
            assert len(subset) == 32
            score = 100 * sum(r['score'] for r in subset) / 32
            official = scorer.scorer(task, [r['prediction'] for r in subset], [r['sample']['answers'] for r in subset], subset[0]['sample']['all_classes'])
            assert round(score, 2) == official
            tasks[task]['palu_' + arm] = score
    arms = ('dense_k_dense_v', 'palu_m', 'palu_g4', *ARMS)
    means = {arm:sum(t[arm] for t in tasks.values()) / len(TASKS) for arm in arms}
    paired = {}
    for arm in CHECKPOINTS:
        for ref_name, ref in [('dense', dense['records']), ('c1_full', [dict(score=r['arms']['full_exact_k']['score']) for r in c1['records']])]:
            delta = [a['score']-b['score'] for a,b in zip(records[arm], ref, strict=True)]
            paired[f'palu_{arm}_vs_{ref_name}'] = dict(improvements=sum(d>0 for d in delta),
                regressions=sum(d<0 for d in delta), ties=sum(d==0 for d in delta), mean_delta_pp=100*sum(delta)/192)
    write_json(args.output_dir / 'result.json', dict(status='complete', protocols=protocols, tasks=tasks,
        means=means, paired=paired, records=records, lengths=dense['lengths'], python=sys.executable, command=shlex.join(sys.argv)))
    write_json(args.output_dir / 'audit.json', dict(status='complete', predictions_verified=384,
        same_inputs_as_dense_and_c1=True, official_scores_verified=True, shard_coverage_verified=True,
        token_decoding_and_eos_verified=True, result_sha256=sha256(args.output_dir / 'result.json')))
    lines = ['# LongBench: matched-calibration PaLU M/G4, dense, and C1', '',
        'Qwen3-8B-Base; 192 fixed prompts, six tasks x 32; BF16, basis, L40S.', '',
        '| Task | Dense K/V | PaLU M | PaLU G4 | C1 full K | C1 exact sparse | C1 Query-Gram | C1 terminal |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for task, values in [*tasks.items(), ('Mean', means)]:
        lines.append('| ' + task + ' | ' + ' | '.join(f'{values[a]:.4f}' for a in arms) + ' |')
    lines += ['', 'Scores are official QA F1 or summary ROUGE-L on a 0–100 scale, not all accuracies. Six-task arithmetic mean, not full LongBench.',
        'Exact same saved inputs, references, completion prompts, greedy/EOS policy and task generation caps. Actual prompt lengths 1,192–30,431; 32K is the total cap.',
        'Both PaLU arms use full exact K and their own approximate V throughout prefill/decode. Factors are executed as a BF16 latent writer plus per-group reconstruction, using standard V128 cache and dense SDPA. This is not a compact-cache or speed benchmark.',
        'PaLU uses exactly the C1 32 x 32K fit tokens, matched whitening and the same newly measured Fisher statistics. M actual average rank is 81.7778; G4 is 80.0000. Nominal R80 does not mean identical realized budgets.',
        'C1 uses its previously evaluated full-C1 Triton prefill; its four arms differ only during decode. PaLU and dense use SDPA prefill. Differences against C1 cannot be attributed solely to factor fitting.',
        'No factors or configurations were selected or refit using LongBench results.', '',
        '## Paired changes', '', '```json', json.dumps(paired, indent=2), '```', '',
        'All 384 new predictions were decoded and rescored; source identities, eight shard manifests, EOS/caps, and official per-task scoring were checked.',
        'See docs/longbench_palu_protocol.md for settings and exact commands.', '']
    (args.output_dir / 'summary.md').write_text('\n'.join(lines))
    print(json.dumps({'means':means, 'paired':paired}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('smoke','evaluate','summarize'), required=True)
    p.add_argument('--arm', choices=tuple(CHECKPOINTS))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT/'results/datasets/longbench_c1_32k')
    p.add_argument('--c1-results', type=Path, default=ROOT/'results/evaluation/longbench_c1_32k')
    p.add_argument('--dense-results', type=Path, default=ROOT/'results/evaluation/longbench_dense_32k')
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/evaluation/longbench_palu_32k')
    p.add_argument('--num-shards', type=int, default=4)
    p.add_argument('--shard-index', type=int, default=0)
    args = p.parse_args()
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if args.stage == 'summarize':
        summarize(args)
    else:
        assert args.arm in CHECKPOINTS
        run(args)


if __name__ == '__main__':
    main()
