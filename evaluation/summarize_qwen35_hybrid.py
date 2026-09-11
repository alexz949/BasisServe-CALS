"""Audit all requested arms before writing a completed experiment summary."""

import argparse
import json
import math
from pathlib import Path

import torch

from evaluation.qwen35_hybrid_banks import LAYERS, RANKS, factors, profile_jobs
from basisserve.core.qwen35_gated_v_runtime import factor_hash
from evaluation.qwen35_hybrid_common import atomic_save, load_bank, sha256


def read_ppl(path):
    result = json.loads(path.read_text())
    for split, windows, tokens in (('wikitext', 146, 297047), ('c4_eval', 128, 262016)):
        row = result['datasets'][split]
        assert row['windows'] == windows and row['predicted_tokens'] == tokens
        assert len(row['window_mean_nll']) == windows
        assert math.isfinite(row['ppl']) and row['ppl'] > 0
        assert math.isclose(row['ppl'], math.exp(row['nll_sum'] / tokens), rel_tol=1e-12)
    return result


def read_kl_profile(path, schedule, expected_factor_hash, identity, windows_hash):
    result = json.loads(path.read_text())
    assert result['schedule'] == schedule
    assert result['factor_sha256'] == expected_factor_hash
    assert result['verified_model_identity'] == identity
    assert result['windows_sha256'] == windows_hash
    values = result['window_kl']
    assert len(values) == 128 and all(math.isfinite(value) and value >= 0 for value in values)
    assert math.isclose(result['mean_kl'], sum(values) / len(values), rel_tol=1e-12, abs_tol=1e-15)
    return result


def read_kl_confirmation(path, identity, windows_hash):
    result = json.loads(path.read_text())
    assert result['windows_sha256'] == windows_hash and result['window_count'] == 16
    assert result['verified_model_identity'] == identity
    assert set(result['results']) == {'uniform', 'twosided'}
    for item in result['results'].values():
        values = item['window_kl']
        assert len(values) == 16 and all(math.isfinite(value) and value >= 0 for value in values)
        assert math.isclose(item['mean_kl'], sum(values) / 16, rel_tol=1e-12, abs_tol=1e-15)
    expected = [a - b for a, b in zip(result['results']['twosided']['window_kl'],
                                     result['results']['uniform']['window_kl'])]
    actual = result['paired_kl_difference_twosided_minus_uniform']
    assert len(actual) == 16
    assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(actual, expected))
    mean = sum(expected) / 16
    stderr = math.sqrt(sum((value - mean) ** 2 for value in expected) / 15 / 16)
    assert math.isclose(result['mean_difference'], mean, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(result['standard_error_across_windows'], stderr, rel_tol=1e-12, abs_tol=1e-15)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('results/q35_hybrid'))
    p.add_argument('--output', type=Path, default=Path('results/q35_hybrid/final_summary.json'))
    args = p.parse_args()
    root = args.root
    baseline = json.loads((root / 'baseline_summary.json').read_text())
    assert sha256(root / 'data/windows.pt') == baseline['windows_sha256']
    assert sha256(root / 'model/config.json') == baseline['config_sha256']
    for name, digest in baseline['model_files_sha256'].items():
        assert sha256(root / 'model' / name) == digest
    baseline_rows = {row['name']: row for row in baseline['rows']}
    names = ['dense'] + [f'palu_{family}_v{rank}' for family in ('mlrd', 'glrd2', 'glrd4') for rank in (64, 80, 96)]
    names += [f'c1_{method}_v{rank}{suffix}' for method in ('uniform', 'twosided')
              for rank in (64, 80, 96) for suffix in ('', '_wo')]
    rows = []
    for name in names:
        ppl_path = root / f'{name}_ppl.json'
        result = read_ppl(ppl_path)
        row = {'name': name, 'wikitext_ppl': result['datasets']['wikitext']['ppl'],
               'c4_ppl': result['datasets']['c4_eval']['ppl'], 'ppl_sha256': sha256(ppl_path),
               'evaluation_command': result['command']}
        if name in baseline_rows:
            assert row['ppl_sha256'] == baseline_rows[name]['ppl_file_sha256']
        if name != 'dense':
            v_name = name.removesuffix('_wo')
            bank_path = root / 'banks' / f'{v_name}.pt'
            bank = load_bank(bank_path)
            assert Path(result['bank']).resolve() == bank_path.resolve()
            assert bank['windows_sha256'] == baseline['windows_sha256']
            row.update(bank_sha256=sha256(bank_path), factor_sha256=bank['factor_sha256'])
            if name.startswith('palu_'):
                assert row['bank_sha256'] == baseline_rows[name]['bank_file_sha256']
                row['v_retention'] = bank['realized_v_retention']
                row['rank_map'] = bank['rank_map']
            else:
                assert bank['encoder_sweeps'] == 6
                assert bank['model_identity']['model_files_sha256'] == baseline['model_files_sha256']
                assert result['verified_model_identity'] == bank['model_identity']
                assert set(bank['schedule']) == set(LAYERS)
                assert sum(bank['schedule'].values()) == len(LAYERS) * bank['nominal_v_rank']
                for source in bank['source_factors'].values():
                    assert sha256(source['file']) == source['sha256']
                row.update(v_retention=bank['value_retention'], schedule=bank['schedule'])
            if name.endswith('_wo'):
                wo_path = root / v_name.replace('c1_', 'wo_', 1) / 'wo_bank.pt'
                assert Path(result['wo_bank']).resolve() == wo_path.resolve()
                wo = torch.load(wo_path, weights_only=True, map_location='cpu')
                assert wo['upstream_v_factor_sha256'] == bank['factor_sha256']
                assert wo['retained_output_input_fraction'] == 0.5 and wo['tp_size'] == 4
                assert wo['work_dtype'] == 'float64' and wo['factor_dtype'] == 'bfloat16'
                moments_path = root / (v_name.replace('c1_', 'wo_', 1) + '_moments') / 'manifest.json'
                moments = json.loads(moments_path.read_text())
                assert sha256(moments_path) == wo['moment_manifest_sha256']
                assert moments['trajectory'] == 'frozen_v_compressed'
                assert moments['verified_model_identity'] == bank['model_identity']
                assert moments['upstream_v_factor_sha256'] == bank['factor_sha256']
                records = wo['full_attention']['layers'] + wo['gdn']['layers']
                assert len(wo['full_attention']['layers']) == 8 and len(wo['gdn']['layers']) == 24
                assert sorted(record['layer_index'] for record in records) == list(range(32))
                for record in records:
                    assert record['work_dtype'] == 'float64'
                    settings = record['metrics']['fit_config']
                    assert settings['encoder_sweeps'] == settings['minimum_encoder_sweeps'] == 6
                    diagnostics = record['metrics']['diagnostics']
                    assert diagnostics['encoder_sweeps_completed'] == 6
                    assert diagnostics['encoder_group_solves'] == len(diagnostics['encoder_solves']) == 24
                    assert record['upstream_v_factor_sha256'] == bank['factor_sha256']
                    assert record['private_encoders'].shape == (4, 1024, 512)
                    assert record['joint_decoder_weight'].shape == (4096, 2048)
                    assert torch.isfinite(record['private_encoders']).all()
                    assert torch.isfinite(record['joint_decoder_weight']).all()
                error_path = root / f'{v_name}_local_errors.json'
                errors = json.loads(error_path.read_text())
                assert errors['v_factor_sha256'] == bank['factor_sha256']
                assert errors['wo_bank_sha256'] == sha256(wo_path)
                assert {(r['layer'], r['split']) for r in errors['rows']} == {
                    (layer, split) for layer in LAYERS for split in ('fit', 'heldout')}
                row.update(wo_bank_sha256=sha256(wo_path), local_errors=errors['rows'],
                           wo_layer_metrics={r['layer_index']: r['metrics'] for r in records})
        rows.append(row)
        print(json.dumps({key: row[key] for key in ('name', 'wikitext_ppl', 'c4_ppl')}), flush=True)
    candidates = []
    for layer in LAYERS:
        for rank in RANKS[:-1]:
            path = root / 'factors' / f'l{layer:02d}_r{rank:03d}.pt'
            factor = torch.load(path, weights_only=True, map_location='cpu')
            encoders = [r for r in factor['history'] if r['block'] == 'encoder']
            decoders = [r for r in factor['history'] if r['block'] == 'decoder']
            assert len(encoders) == 6 and len(decoders) == 7
            assert factor['E_V'].shape == (4, 256, rank) and factor['R_V'].shape == (4, rank, 256)
            assert torch.isfinite(factor['E_V']).all() and torch.isfinite(factor['R_V']).all()
            blocks = encoders + decoders
            candidates.append({'layer': layer, 'rank': rank, 'sha256': sha256(path),
                'metrics': factor['metrics'], 'selected_sweep': factor['selected_sweep'],
                'cap_blocks': sum(r['hit_iteration_cap'] for r in blocks),
                'unconverged_blocks': sum(not r['converged'] for r in blocks),
                'negative_curvature_blocks': sum(r['negative_curvature'] for r in blocks),
                'rejected_blocks': sum(not r['accepted'] for r in blocks),
                'max_true_relative_residual': max(r['recomputed_relative_residual'] for r in blocks)})
    identity = {key: baseline[key] for key in ('model_revision', 'config_sha256', 'model_files_sha256')}
    profiles = {}
    for anchor, name, schedule in profile_jobs((64, 80, 96)):
        path = root / 'kl' / f'a{anchor}_{name}.json'
        result = read_kl_profile(path, schedule, factor_hash(factors(root / 'factors', schedule)),
                                 identity, baseline['windows_sha256'])
        profiles[f'a{anchor}_{name}'] = {'sha256': sha256(path), **result}
    confirmation = {}
    for rank in (64, 80, 96):
        report = read_kl_confirmation(root / 'kl' / f'confirm_v{rank}.json', identity, baseline['windows_sha256'])
        assert report['anchor'] == rank
        for method, item in report['results'].items():
            bank_path = root / 'banks' / f'c1_{method}_v{rank}.pt'
            bank = load_bank(bank_path)
            assert item['bank_sha256'] == sha256(bank_path)
            assert report['verified_model_identity'] == bank['model_identity']
            assert item['factor_sha256'] == bank['factor_sha256']
            assert item['schedule'] == {str(layer): rank for layer, rank in bank['schedule'].items()}
        confirmation[rank] = report
    atomic_save(args.output, {'status': 'all_requested_arms_complete', 'rows': rows,
        'v_candidates': candidates, 'kl_profiles': profiles, 'independent_kl_confirmation': confirmation,
        'baseline_manifest_sha256': sha256(root / 'baseline_summary.json'),
        'final_compressed_arms': 21, 'dense_baselines': 1, 'environment': 'lowrank',
        'execution': 'local A100 processes, two CPU threads each, no Slurm',
        'logs': str(root / 'logs'), 'distributed_scope': 'single-process Qwen3.5 AG equivalent; no measured model TP speedup'})


if __name__ == '__main__':
    main()
