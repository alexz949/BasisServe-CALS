"""Audit complete V96 runs, optionally including the full Wo compression arms."""

import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import re

import torch

from evaluation.eval_qwen35_hybrid_gsm8k import validate_gsm8k_samples
from evaluation.qwen35_hybrid_common import atomic_save, load_bank, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/q35_hybrid/gsm8k_vllm'))
    p.add_argument('--output', default='results/q35_hybrid/gsm8k_vllm/v96_summary.json')
    p.add_argument('--with-wo', action='store_true')
    p.add_argument('--dense-wo', action='store_true')
    p.add_argument('--split-wo', action='store_true')
    p.add_argument('--mixed-wo', action='store_true')
    p.add_argument('--v128-wo', action='store_true')
    a = p.parse_args()
    if a.v128_wo:
        a.mixed_wo = True
    assert not a.dense_wo or a.with_wo
    baseline_audit = json.loads((a.root / 'summary.json').read_text())
    assert baseline_audit['status'] == 'all_four_complete_and_audited'
    trusted = {r['arm']: r['result_sha256'] for r in baseline_audit['rows']}
    rows, outcomes, shared_prompts, shared_provenance = [], {}, None, None
    arms = ('dense', 'twosided64', 'twosided96') + (('twosided64_wo', 'twosided96_wo') if a.with_wo else ())
    if a.dense_wo:
        arms += ('dense_wo',)
    if a.split_wo or a.mixed_wo:
        assert not a.with_wo and not a.dense_wo
        assert not (a.split_wo and a.mixed_wo)
        arms = ('dense', 'dense_wo', 'dense_gdn_wo', 'dense_full_wo')
        previous = json.loads((a.root / 'dense_wo_summary.json').read_text())
        assert previous['status'] == 'complete_and_audited'
        trusted.update({r['arm']: r['source_sha256'] for r in previous['rows']})
        if a.mixed_wo:
            previous_split = json.loads((a.root / 'dense_split_wo_summary.json').read_text())
            assert previous_split['status'] == 'complete_and_audited'
            trusted.update({r['arm']: r['source_sha256'] for r in previous_split['rows']})
            arms += ('dense_g768_f512_wo',)
            if a.v128_wo:
                previous_mixed = json.loads((a.root / 'dense_g768_f512_wo_summary.json').read_text())
                assert previous_mixed['status'] == 'complete_and_audited'
                trusted.update({r['arm']: r['source_sha256'] for r in previous_mixed['rows']})
                arms += ('twosided128_g768_f512_wo',)
    shared_dense_factors = None
    for arm in arms:
        is_v128 = arm == 'twosided128_g768_f512_wo'
        is_mixed = arm == 'dense_g768_f512_wo' or is_v128
        path = a.root / f'result_{arm}.json'
        digest = sha256(path)
        if arm in trusted:
            assert digest == trusted[arm]
        result = json.loads(path.read_text())
        assert result['status'] == 'complete' and result['thinking'] is False
        settings = result['args']
        assert settings['limit'] is None
        assert bool(settings['wo_bank']) == arm.endswith('_wo')
        for key, expected in {'max_num_seqs': 32, 'max_num_batched_tokens': 4096,
                              'max_new_tokens': 1024, 'max_model_len': 8192, 'seed': 20260909}.items():
            assert settings[key] == expected
        provenance = result['provenance']
        identity = {k: provenance[k] for k in ('model_identity', 'tokenizer_sha256', 'chat_template_sha256')}
        if shared_provenance is None:
            shared_provenance = identity
        assert identity == shared_provenance
        ranks = None
        wo_layers = None
        active_wo_layers = []
        wo_fit = []
        if settings['bank']:
            assert sha256(settings['bank']) == provenance['v_bank_sha256']
            bank = load_bank(settings['bank'])
            assert bank['factor_sha256'] == provenance['v_factor_sha256']
            ranks = {i: f['E_V'].shape[-1] for i, f in bank['layers'].items()}
            if is_v128:
                assert bank['method'] == 'twosided' and bank['nominal_v_rank'] == 128
                assert bank['kl_anchor_rank'] == 128
                assert tuple(bank['candidate_ranks']) == (32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256)
                assert set(bank['schedule']) == {3, 7, 11, 15, 19, 23, 27, 31}
                assert sum(bank['schedule'].values()) == 8 * 128 and bank['value_retention'] == 0.5
                assert ranks == {i: r for i, r in bank['schedule'].items() if r != 256}
                ranks = bank['schedule']
                for source in bank['source_factors'].values():
                    assert sha256(source['file']) == source['sha256']
            if arm.startswith('dense_'):
                assert not bank['layers'] and bank['encoder_sweeps'] == 0
                assert set(bank['schedule']) == {3, 7, 11, 15, 19, 23, 27, 31}
                assert set(bank['schedule'].values()) == {256} and bank['value_retention'] == 1
                ranks = bank['schedule']
        if settings['wo_bank']:
            scope = {'dense_gdn_wo': 'gdn', 'dense_full_wo': 'full_attention'}.get(arm, 'all')
            assert settings.get('wo_scope', 'all') == scope
            assert sha256(settings['wo_bank']) == provenance['wo_bank_sha256']
            wo = torch.load(settings['wo_bank'], map_location='cpu', weights_only=True)
            assert wo['upstream_v_factor_sha256'] == provenance['v_factor_sha256']
            assert wo['tp_size'] == 4
            if is_v128:
                manifest_path = Path('results/q35_hybrid/wo_v128_moments/manifest.json')
                assert sha256(manifest_path) == wo['moment_manifest_sha256']
                manifest = json.loads(manifest_path.read_text())
                assert manifest['trajectory'] == 'frozen_v_compressed'
                assert manifest['upstream_v_factor_sha256'] == bank['factor_sha256']
                assert manifest['verified_model_identity'] == bank['model_identity']
                assert manifest['windows_sha256'] == wo['windows_sha256'] == bank['windows_sha256']
                assert wo['source_rank_by_type'] == {'gdn': 768, 'full_attention': 512}
                assert wo['retained_output_input_fraction'] == 0.6875 and wo['work_dtype'] == 'float64'
            if arm.startswith('dense_'):
                factor_identity = {k: provenance[k] for k in ('v_bank_sha256', 'v_factor_sha256', 'wo_bank_sha256')}
                if shared_dense_factors is None:
                    shared_dense_factors = factor_identity
                if not is_mixed:
                    assert shared_dense_factors == factor_identity
                else:
                    assert factor_identity['v_bank_sha256'] == shared_dense_factors['v_bank_sha256']
                    assert wo['source_wo_sha256'] == shared_dense_factors['wo_bank_sha256']
                    assembly = json.loads(Path(settings['wo_bank']).with_name('audit.json').read_text())
                    assert assembly['status'] == 'complete' and assembly['full_attention_exact_reuse']
                    assert assembly['bank_sha256'] == provenance['wo_bank_sha256']
                manifest_path = Path('results/q35_hybrid/wo_dense_moments/manifest.json')
                assert sha256(manifest_path) == wo['moment_manifest_sha256']
                manifest = json.loads(manifest_path.read_text())
                assert manifest['trajectory'] == 'native_dense'
                assert manifest['verified_model_identity'] == bank['model_identity']
                assert manifest['windows_sha256'] == wo['windows_sha256'] == bank['windows_sha256']
                assert manifest['upstream_v_factor_sha256'] == bank['factor_sha256']
                assert wo['work_dtype'] == 'float64'
                assert wo['retained_output_input_fraction'] == (0.6875 if is_mixed else 0.5)
            wo_layers = {family: [r['layer_index'] for r in wo[family]['layers']] for family in ('gdn', 'full_attention')}
            assert len(wo_layers['gdn']) == 24 and len(wo_layers['full_attention']) == 8
            assert sorted(wo_layers['gdn'] + wo_layers['full_attention']) == list(range(32))
            active_wo_layers = sorted(i for family, ids in wo_layers.items() if scope == 'all' or family == scope for i in ids)
            for family in wo_layers:
                for record in wo[family]['layers']:
                    if is_v128:
                        assert record['moment_manifest_sha256'] == wo['moment_manifest_sha256']
                        assert record['upstream_v_factor_sha256'] == bank['factor_sha256']
                        assert record['work_dtype'] == 'float64'
                        assert record['metrics']['diagnostics']['encoder_sweeps_completed'] == 6
                    encoder = record['private_encoders']
                    expected_rank = 768 if is_mixed and family == 'gdn' else 512
                    assert encoder.shape == (4, 1024, expected_rank)
                    assert record['joint_decoder_weight'].shape == (4096, 4 * expected_rank)
                    assert torch.isfinite(encoder).all() and torch.isfinite(record['joint_decoder_weight']).all()
                    diagnostics = record['metrics']['diagnostics']
                    wo_fit.append({'layer': record['layer_index'], 'type': family,
                                   'sweeps': diagnostics['encoder_sweeps_completed'],
                                   'max_encoder_solve_residual': diagnostics['encoder_maximum_relative_residual'],
                                   'selected_sweep': record['metrics']['selected_sweep'],
                                   'heldout_bf16_relative_mse': record['metrics']['quantized_heldout_relative_output_mse']})
                    if arm.startswith('dense_'):
                        assert record['metrics']['diagnostics']['encoder_sweeps_completed'] == 6
            del wo
        samples = result['evaluation']['samples']['gsm8k']
        validate_gsm8k_samples(samples, 1319)
        by_key = {(r['doc_id'], r['filter']): r for r in samples}
        records = result['generation_records']
        assert len(records) == 1319 and {r['doc_id'] for r in records} == set(range(1319))
        prompts, numeric, strict, flexible = {}, [], [], []
        for i in range(1319):
            s, f = (by_key[i, k] for k in ('strict-match', 'flexible-extract'))
            assert s['resps'] == f['resps'] and s['arguments'] == f['arguments'] and s['doc'] == f['doc']
            context, kwargs = s['arguments'][0]
            assert context.endswith('<think>\n\n</think>\n\n')
            assert kwargs['max_gen_toks'] == 1024 and kwargs['temperature'] == 0 and kwargs['do_sample'] is False
            prompts[i] = (s['doc'], s['target'], s['arguments'])
            gold = Decimal(s['target'].rsplit('####', 1)[1].strip().replace(',', ''))
            extracted = f['filtered_resps'][0].strip().replace('$', '').replace(',', '').rstrip('.')
            numeric.append(int(bool(re.fullmatch(r'-?\d+(?:\.\d+)?', extracted)) and Decimal(extracted) == gold))
            strict.append(int(s['exact_match']))
            flexible.append(int(f['exact_match']))
        if shared_prompts is None:
            shared_prompts = prompts
        assert prompts == shared_prompts
        import hashlib
        for r in records:
            context = prompts[r['doc_id']][2][0][0]
            assert r['prompt_sha256'] == hashlib.sha256(context.encode()).hexdigest()
            assert r['retained_prompt_tokens'] == r['original_prompt_tokens']
            assert 0 < r['generated_tokens'] <= 1024
        for key, values in [('strict-match', strict), ('flexible-extract', flexible)]:
            assert math.isclose(sum(values) / 1319, result['evaluation']['results']['gsm8k']['exact_match,' + key], abs_tol=1e-12)
        capped = sum(r['finish_reason'] == 'length' for r in records)
        assert capped == result['length_capped']
        row = {'arm': arm, 'source_sha256': digest, 'strict_correct': sum(strict),
               'flexible_correct': sum(flexible), 'numeric_equivalent_correct': sum(numeric),
               'length_capped': capped, 'generated_tokens': sum(r['generated_tokens'] for r in records),
               'closing_think_responses': sum('</think>' in by_key[i, 'strict-match']['resps'][0][0] for i in range(1319)),
               'repeated_closing_think_responses': sum(by_key[i, 'strict-match']['resps'][0][0].count('</think>') >= 3 for i in range(1319)),
               'strict_correct_flexible_wrong': sum(s and not f for s, f in zip(strict, flexible)),
               'flexible_correct_strict_wrong': sum(f and not s for s, f in zip(strict, flexible)),
               'command': result['command'], 'args': settings, 'versions': result['versions'], 'ranks': ranks,
               'wo_layers': wo_layers, 'active_wo_layers': active_wo_layers, 'wo_fit': wo_fit}
        rows.append(row)
        outcomes[arm] = {'strict': strict, 'flexible': flexible, 'numeric_equivalent': numeric}
        print(json.dumps(row), flush=True)
    paired = {}
    target = 'twosided128_g768_f512_wo' if a.v128_wo else 'dense_g768_f512_wo' if a.mixed_wo else 'dense_wo' if a.dense_wo or a.split_wo else 'twosided96_wo' if a.with_wo else 'twosided96'
    for reference in (arm for arm in arms if arm != target):
        paired[reference] = {}
        for metric, current in outcomes[target].items():
            other = outcomes[reference][metric]
            paired[reference][metric] = {
                'target_only_correct': sum(x and not y for x, y in zip(current, other)),
                'reference_only_correct': sum(y and not x for x, y in zip(current, other)),
                'difference_pp': 100 * (sum(current) - sum(other)) / 1319}
    split_interaction = {}
    if a.split_wo:
        for metric in ('strict', 'flexible', 'numeric_equivalent'):
            dense = outcomes['dense'][metric]
            gdn = outcomes['dense_gdn_wo'][metric]
            full = outcomes['dense_full_wo'][metric]
            joint = outcomes['dense_wo'][metric]
            ids = [i for i in range(1319) if dense[i] and gdn[i] and full[i] and not joint[i]]
            split_interaction[metric] = {
                'dense_and_both_single_correct_joint_wrong_count': len(ids),
                'dense_and_both_single_correct_joint_wrong_doc_ids': ids,
                'joint_correct_both_single_wrong_count': sum(joint[i] and not gdn[i] and not full[i] for i in range(1319))}
    atomic_save(a.output, {'status': 'complete_and_audited', 'questions': 1319,
                          'identical_prompts_targets_and_generation_kwargs': True,
                          'rows': rows, 'paired_target': target, 'paired_vs': paired,
                          'split_interaction': split_interaction,
                          'numeric_metric_scope': 'Decimal equality of the existing flexible extraction; not human grading or answer reselection.',
                          'runtime_scope': 'TP1 padded native cache; no actual cache-memory saving measured.'})


if __name__ == '__main__':
    main()
