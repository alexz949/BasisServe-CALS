"""Assemble immutable C1 banks and two-sided terminal-KL schedules."""

import argparse
import json
from pathlib import Path
import time

import torch

from basisserve.core.qwen35_gated_v_runtime import GatedVRuntime, factor_hash
from evaluation.build_qwen3_8b_c1_two_sided_factorized_kl_schedule import predict_two_sided_factorized_costs, allocate_layer_schedule
from evaluation.qwen35_hybrid_common import atomic_save, load_model, load_windows, load_bank, sha256, verify_model_identity


LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
RANKS = (32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256)


def factors(root, schedule, layers=LAYERS):
    bank = {}
    for layer, rank in zip(layers, schedule, strict=True):
        if rank == 256:
            continue  # exact native endpoint; no floating point identity adapter
        payload = torch.load(Path(root) / f'l{layer:02d}_r{rank:03d}.pt', weights_only=True, map_location='cpu')
        assert payload['encoder_sweeps'] == 6 and payload['rank'] == rank and payload['layer'] == layer
        bank[layer] = {key: payload[key] for key in ('E_V', 'R_V')}
    return bank


def save_bank(args, schedule, anchor, method, extra):
    config = json.loads((Path(args.model_path) / 'config.json').read_text())['text_config']
    layers = tuple(i for i, kind in enumerate(config['layer_types']) if kind == 'full_attention')
    bank = factors(args.factors, schedule, layers)
    baseline = json.loads(Path(args.model_manifest).read_text())
    assert baseline['config_sha256'] == sha256(Path(args.model_path) / 'config.json')
    assert baseline['windows_sha256'] == sha256(Path(args.data) / 'windows.pt')
    sources = {layer: {'file': str(Path(args.factors) / f'l{layer:02d}_r{rank:03d}.pt'),
        'sha256': sha256(Path(args.factors) / f'l{layer:02d}_r{rank:03d}.pt')}
        for layer, rank in zip(layers, schedule) if rank != 256}
    atomic_save(Path(args.output) / f'c1_{method}_v{anchor}.pt', {
        'format': 'basisserve.qwen35.gated_v_als.v1', 'method': method, 'layers': bank,
        'factor_sha256': factor_hash(bank), 'nominal_v_rank': anchor,
        'schedule': dict(zip(layers, schedule)), 'value_retention': sum(schedule) / (len(layers) * 256),
        'windows_sha256': baseline['windows_sha256'], 'encoder_sweeps': 6 if bank else 0,
        'model_identity': {key: baseline[key] for key in ('model_revision', 'config_sha256', 'model_files_sha256')},
        'source_factors': sources, 'work_dtype': 'float32', 'factor_dtype': 'bfloat16',
        'capture_trajectory': 'native_dense', 'objective': 'joint_gated_attention_output_mse' if bank else 'native_dense_endpoint',
        'target_kind': 'native_attention_output_excluding_output_bias', 'initialization': 'group_pooled_pre_gate_pca' if bank else 'none',
        'v_cache_heads': config['num_key_value_heads'], 'query_heads': config['num_attention_heads'],
        'value_head_dim': config['head_dim'], **extra})


@torch.no_grad()
def teacher(args):
    model = load_model(args.model_path, args.device)
    # Store final normalized hidden states instead of 128*2048*vocabulary logits.
    # Every KL calculation applies the immutable original lm_head to both sides.
    for split in ('profile', 'confirm'):
        outputs = []
        for tokens in load_windows(args.data, split):
            outputs.append(model.model(tokens[None].to(args.device), use_cache=False).last_hidden_state[0].cpu())
        atomic_save(Path(args.output) / f'teacher_{split}.pt', {'hidden': torch.stack(outputs),
            'windows_sha256': sha256(Path(args.data) / 'windows.pt')})


@torch.no_grad()
def kl(model, windows, teacher_hidden, bank):
    values = []
    with GatedVRuntime(model, bank):
        for index, tokens in enumerate(windows):
            hidden = model.model(tokens[None].to(model.device), use_cache=False).last_hidden_state[0]
            total = 0.0
            for start in range(0, len(tokens), 128):
                student = model.lm_head(hidden[start:start + 128]).float().log_softmax(-1)
                reference = model.lm_head(teacher_hidden[index, start:start + 128].to(model.device)).float().log_softmax(-1)
                total += float((reference.exp() * (reference - student)).double().sum())
            values.append(total / len(tokens))
            if index % 16 == 0:
                print(json.dumps({'profile_window': index, 'running_kl': sum(values) / len(values)}), flush=True)
    return values


def profile_jobs(anchors, shard_index=0, num_shards=1, layers=LAYERS):
    assert num_shards > 0 and 0 <= shard_index < num_shards
    assert len(anchors) == len(set(anchors)) and set(anchors) <= {64, 80, 96, 128, 192}
    jobs = []
    for anchor in anchors:
        base = [anchor] * len(layers)
        jobs.append((anchor, 'anchor', base))
        for offset, layer in enumerate(layers):
            for probe in (anchor - 32, anchor + 32):
                schedule = base.copy()
                schedule[offset] = probe
                jobs.append((anchor, f'l{layer:02d}_r{probe}', schedule))
    return jobs[shard_index::num_shards]


@torch.no_grad()
def profile(args):
    config = json.loads((Path(args.model_path) / 'config.json').read_text())['text_config']
    layers = tuple(i for i, kind in enumerate(config['layer_types']) if kind == 'full_attention')
    anchors = (64, 80, 96) if args.all_anchors else (args.anchor,)
    jobs = [job for job in profile_jobs(anchors, args.shard_index, args.num_shards, layers)
        if not (Path(args.output) / f'a{job[0]}_{job[1]}.json').exists()]
    if not jobs:
        return
    baseline = json.loads(Path(args.model_manifest).read_text())
    identity = {key: baseline[key] for key in ('model_revision', 'config_sha256', 'model_files_sha256')}
    verify_model_identity(args.model_path, identity)
    model = load_model(args.model_path, args.device)
    windows = load_windows(args.data, 'profile')
    teacher_payload = torch.load(Path(args.teacher_dir) / 'teacher_profile.pt', weights_only=True, map_location='cpu')
    assert teacher_payload['windows_sha256'] == sha256(Path(args.data) / 'windows.pt')
    for anchor, name, schedule in jobs:
        path = Path(args.output) / f'a{anchor}_{name}.json'
        started = time.monotonic()
        bank = factors(args.factors, schedule, layers)
        print(json.dumps({'profile_start': f'a{anchor}_{name}', 'schedule': schedule}), flush=True)
        values = kl(model, windows, teacher_payload['hidden'], bank)
        atomic_save(path, {'schedule': schedule, 'factor_sha256': factor_hash(bank), 'window_kl': values,
            'mean_kl': sum(values) / len(values), 'seconds': time.monotonic() - started,
            'windows_sha256': teacher_payload['windows_sha256'], 'verified_model_identity': identity})
        print(json.dumps({'anchor': anchor, 'profile': name, 'mean_kl': sum(values) / len(values)}), flush=True)


@torch.no_grad()
def confirm(args):
    """Measure already frozen choices on independent windows; never reselect."""
    banks = {method: load_bank(Path(args.bank_dir) / f'c1_{method}_v{args.anchor}.pt')
        for method in ('uniform', 'twosided')}
    identity = banks['uniform']['model_identity']
    assert banks['twosided']['model_identity'] == identity
    verify_model_identity(args.model_path, identity)
    model = load_model(args.model_path, args.device)
    windows = load_windows(args.data, 'confirm')
    reference = torch.load(Path(args.teacher_dir) / 'teacher_confirm.pt', weights_only=True, map_location='cpu')
    windows_hash = sha256(Path(args.data) / 'windows.pt')
    assert reference['windows_sha256'] == windows_hash
    results = {}
    for method in ('uniform', 'twosided'):
        path = Path(args.bank_dir) / f'c1_{method}_v{args.anchor}.pt'
        bank = banks[method]
        assert bank['windows_sha256'] == windows_hash
        values = kl(model, windows, reference['hidden'], bank['layers'])
        results[method] = {'bank_sha256': sha256(path), 'factor_sha256': bank['factor_sha256'],
            'schedule': bank['schedule'], 'window_kl': values, 'mean_kl': sum(values) / len(values)}
    differences = torch.tensor(results['twosided']['window_kl'], dtype=torch.float64) - torch.tensor(
        results['uniform']['window_kl'], dtype=torch.float64)
    atomic_save(Path(args.output) / f'confirm_v{args.anchor}.json', {
        'purpose': 'independent confirmation only; no schedule selection', 'anchor': args.anchor,
        'verified_model_identity': identity,
        'windows_sha256': windows_hash, 'window_count': len(windows), 'results': results,
        'paired_kl_difference_twosided_minus_uniform': differences.tolist(),
        'mean_difference': float(differences.mean()),
        'standard_error_across_windows': float(differences.std() / len(differences) ** 0.5)})


def assemble(args):
    config = json.loads((Path(args.model_path) / 'config.json').read_text())['text_config']
    layers = tuple(i for i, kind in enumerate(config['layer_types']) if kind == 'full_attention')
    anchor = args.anchor
    target = args.target_average_rank if args.target_average_rank is not None else anchor
    if args.uniform:
        save_bank(args, [target] * len(layers), target, 'uniform', {})
        return
    profile_dir = Path(args.profile_dir)
    base = json.loads((profile_dir / f'a{anchor}_anchor.json').read_text())['mean_kl']
    deltas = {probe: [json.loads((profile_dir / f'a{anchor}_l{i:02d}_r{probe}.json').read_text())['mean_kl'] - base for i in layers]
        for probe in (anchor - 32, anchor + 32)}
    errors = []
    for layer in layers:
        curve = {256: 0.0}
        for rank in RANKS[:-1]:
            row = torch.load(Path(args.factors) / f'l{layer:02d}_r{rank:03d}.pt', weights_only=True, map_location='cpu')
            curve[rank] = row['metrics']['heldout_export_relative_mse']
        errors.append(curve)
    costs, minus, plus = predict_two_sided_factorized_costs(errors, deltas[anchor - 32], deltas[anchor + 32],
        candidate_ranks=RANKS, anchor_rank=anchor, compression_probe_rank=anchor - 32,
        expansion_probe_rank=anchor + 32, exponent=1.25)
    schedule, cost = allocate_layer_schedule(costs, candidate_ranks=RANKS, anchor_rank=anchor, target_average_rank=target)
    assert sum(schedule) == len(layers) * target
    save_bank(args, schedule, target, 'twosided', {'kl_anchor_rank': anchor, 'kl_exponent': 1.25, 'candidate_ranks': RANKS,
        'compression_sensitivity': minus, 'expansion_sensitivity': plus, 'predicted_cost': cost})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['teacher', 'profile', 'assemble', 'confirm'])
    p.add_argument('--model-path', default='results/q35_hybrid/model')
    p.add_argument('--model-manifest', default='results/q35_hybrid/baseline_summary.json')
    p.add_argument('--data', default='results/q35_hybrid/data')
    p.add_argument('--factors', default='results/q35_hybrid/factors')
    p.add_argument('--teacher-dir', default='results/q35_hybrid/teacher')
    p.add_argument('--profile-dir', default='results/q35_hybrid/kl')
    p.add_argument('--bank-dir', default='results/q35_hybrid/banks')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--anchor', type=int, choices=[64, 80, 96, 128, 192, 256], default=64)
    p.add_argument('--target-average-rank', type=int, choices=RANKS)
    p.add_argument('--all-anchors', action='store_true')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--uniform', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    assert args.target_average_rank is None or args.stage == 'assemble'
    assert args.anchor != 256 or (args.stage == 'assemble' and args.uniform)
    torch.set_num_threads(2)
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
