"""Allocate attention-only Nemotron V96 using native full-forward two-sided KL."""
import argparse
import inspect
from pathlib import Path
import shlex
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, save_tensors, sha256
from evaluation.allocate_qwen3_32b_c1_tp_source_global_kl import (
    _capture_teacher, _evaluate_teacher_metrics, _paired_delta,
    _fold_ragged_to_padded_weights,
)
from evaluation.build_qwen3_8b_c1_two_sided_factorized_kl_schedule import (
    local_error_curves_from_factor_results, predict_two_sided_factorized_costs,
    allocate_layer_schedule,
)
from basisserve.core.gqa_routed_ov_joint import covariance_with_trace_damping, quadratic_from_target
from basisserve.core.decoder_closed_rank_candidates import close_ragged_decoder_with_fixed_encoders
from evaluation.nemotron_h_runtime import install_mamba_device_guards

LAYERS = (17, 38, 49, 60, 86)
RANKS = (32, 48, 64, 80, 96, 112, 128)


def close_candidate(A, D, covariance, weight, *, anchor_rank=64):
    """Keep the fitted anchor; close other decoders in a fixed canonical gauge."""
    groups, dim, rank = A.shape
    heads, decoder_rank, hidden = D.shape
    assert decoder_rank == rank and heads % groups == 0
    assert weight.shape == (hidden, heads * dim)
    assert covariance.shape == (heads * dim, heads * dim)
    if rank == dim:
        A = torch.eye(dim, device=A.device).expand(groups, dim, dim).contiguous()
        D = weight.float().T.reshape(heads, dim, hidden).contiguous()
        diagnostic = dict(closure='exact_identity_dense_o_endpoint')
    elif rank == anchor_rank:
        diagnostic = dict(closure='uniform_anchor_factor_bank')
    else:
        blocks = covariance.float().reshape(heads, dim, heads, dim).permute(0, 2, 1, 3).contiguous()
        blocks = (blocks + blocks.permute(1, 0, 3, 2)) * 0.5
        blocks, damping = covariance_with_trace_damping(blocks, relative_damping=1e-5)
        objective = quadratic_from_target(covariance=blocks,
            target=weight.float().T.reshape(heads, dim, hidden).contiguous(),
            name='nemotron_attention_terminal_kl', trace_normalize=False)
        closure = close_ragged_decoder_with_fixed_encoders(objective=objective,
            initial_A=A.float(), initial_D=D.float(),
            head_to_kv_group=torch.arange(heads, device=A.device) // (heads // groups),
            group_ranks=(rank,) * groups, relative_jitter=0.0)
        A, D = closure.A_unique, closure.D_heads
        assert closure.encoder_sha256_before_solve == closure.encoder_sha256_after_solve
        diagnostic = dict(closure='full_layer_closed_form_decoder_refit',
            covariance_absolute_damping=float(damping),
            absolute_jitter=closure.decoder.absolute_jitters[0],
            condition_estimate=closure.decoder.condition_estimates[0],
            relative_residual=closure.decoder.relative_residuals[0],
            gauge_product_error=closure.gauge_product_error,
            encoder_sha256=closure.encoder_sha256_after_solve)
    return A.to(device='cpu', dtype=torch.bfloat16), D.to(device='cpu', dtype=torch.bfloat16), diagnostic


def allocate(results, compression, expansion):
    for rank in RANKS:
        assert results[rank]['layers'] == list(LAYERS)
        assert [row['layer'] for row in results[rank]['records']] == list(LAYERS)
    curves = local_error_curves_from_factor_results(results, candidate_ranks=RANKS,
        error_split='heldout', num_layers=len(LAYERS), head_dim=128)
    costs, left, right = predict_two_sided_factorized_costs(curves, compression, expansion,
        candidate_ranks=RANKS, anchor_rank=64, compression_probe_rank=32,
        expansion_probe_rank=96, exponent=1.0)
    ranks, cost = allocate_layer_schedule(costs, candidate_ranks=RANKS,
        anchor_rank=64, target_average_rank=96)
    assert len(ranks) == 5 and sum(ranks) == 480
    return dict(layer_ranks=list(ranks), predicted_cost=cost,
        local_errors=curves, predicted_costs=costs,
        compression_sensitivities=left, expansion_sensitivities=right)


@torch.inference_mode()
def main():
    import causal_conv1d
    import mamba_ssm
    from transformers.models.nemotron_h import modeling_nemotron_h as native
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit', 'full-smoke', 'windows', 'snapshots', 'bank', 'output', 'identity-output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    configure()
    audit, smoke = read_json(args.audit), read_json(args.full_smoke)
    assert audit['status'] == smoke['status'] == 'complete'
    assert smoke['full_model_tested'] and smoke['verified_tensor_count'] == 577
    assert smoke['audit_sha256'] == sha256(args.audit)
    assert smoke['device_guard_sha256'] == sha256(ROOT / 'evaluation/nemotron_h_runtime.py')
    assert sha256(inspect.getfile(native.NemotronHForCausalLM)) == audit['implementation_sha256']
    implementations = {}
    for name, package in (('mamba2_chunk_scan', 'mamba_ssm'),
            ('mamba2_selective_state_update', 'mamba_ssm'),
            ('causal_conv1d_fn', 'causal_conv1d'), ('causal_conv1d_update', 'causal_conv1d')):
        implementation = inspect.getclosurevars(getattr(native, name)).nonlocals['implementation']
        assert implementation.__module__.startswith(package)
        implementations[name] = implementation.__module__ + '.' + implementation.__name__
    assert tuple(row['layer'] for row in audit['layers'] if row['kind'] == 'full_attention') == LAYERS
    model_path = Path(audit['model'])
    assert sha256(model_path / 'config.json') == audit['config_sha256']
    assert sha256(model_path / 'model.safetensors.index.json') == audit['index_sha256']
    windows_manifest = read_json(args.windows.parent / 'manifest.json')
    assert windows_manifest['status'] == 'complete'
    assert windows_manifest['sha256'] == sha256(args.windows)
    assert windows_manifest['protocol']['audit_sha256'] == sha256(args.audit)
    assert windows_manifest['protocol']['profile_ids'] == list(range(320, 328))
    assert windows_manifest['protocol']['confirmation_ids'] == list(range(328, 336))
    snapshots = read_json(args.snapshots / 'manifest.json')
    assert snapshots['status'] == 'complete' and snapshots['dense_teacher']
    assert snapshots['layers'] == list(LAYERS) and snapshots['layer_kind'] == 'full_attention'
    assert snapshots['audit_sha256'] == sha256(args.audit)
    results, inputs = {}, {}
    for rank in RANKS:
        directory = args.bank / f'r{rank}'
        result_path = directory / 'results.json'
        result = read_json(result_path)
        assert result['status'] == 'complete' and result['layers'] == list(LAYERS)
        config = result['fit_config']
        assert config['model_config_sha256'] == audit['config_sha256']
        assert config['cache_rank_per_head'] == rank
        assert config['fit_windows'] == 256 and config['validation_windows'] == 64
        assert config['encoder_sweeps'] == 12 and config['encoder_cg_mode'] == 'fixed'
        assert config['encoder_cg_fixed_iterations'] == 16
        assert Path(config['snapshot_dir']).resolve() == args.snapshots.resolve()
        results[rank] = result
        inputs[str(rank)] = dict(path=str(directory.resolve()), sha256=sha256(result_path))
    protocol = dict(audit_sha256=sha256(args.audit), full_smoke_sha256=sha256(args.full_smoke),
        implementations=implementations,
        windows_sha256=sha256(args.windows), windows_manifest_sha256=sha256(args.windows.parent / 'manifest.json'),
        snapshots_sha256=sha256(args.snapshots / 'manifest.json'), factor_sources=inputs,
        attention_layers=list(LAYERS), anchor_rank=64, compression_probe_rank=32,
        expansion_probe_rank=96, exponent=1.0, target_mean_rank=96,
        profile_ids=list(range(320, 328)), confirmation_ids=list(range(328, 336)),
        backend='native_full_forward_padded_c1', prediction_positions_per_window=2047,
        vocabulary='full', mamba='dense_unchanged_during_attention_allocation',
        code_hashes={name: sha256(ROOT / name) for name in (
            'evaluation/allocate_nemotron_h_v96.py',
            'evaluation/nemotron_h_runtime.py',
            'evaluation/allocate_qwen3_32b_c1_tp_source_global_kl.py',
            'evaluation/build_qwen3_8b_c1_two_sided_factorized_kl_schedule.py',
            'basisserve/core/decoder_closed_rank_candidates.py',
            'basisserve/core/gqa_routed_ov_joint.py',
            'basisserve/core/global_rank_sensitivity.py',
            'basisserve/core/metric_rank_allocation.py')})
    write_json(args.output / 'protocol.json', protocol)
    if args.identity_output.exists():
        identity = read_json(args.identity_output)
        assert identity['status'] == 'complete'
        assert identity['manifest_sha256'] == sha256(args.output / 'manifest.json')
        manifest = read_json(args.output / 'manifest.json')
        assert manifest['artifact']['sha256'] == sha256(args.output / manifest['artifact']['file'])
        for entry in manifest['layers']:
            assert entry['sha256'] == sha256(args.output / entry['file'])
        print('ATTENTION V96 VERIFIED', identity['layer_ranks'], flush=True)
        return
    assert torch.cuda.device_count() == 4
    memory = {i: torch.cuda.get_device_properties(i).total_memory - 12 * 2**30 for i in range(4)}
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=False,
        local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map='balanced', max_memory=memory).eval()
    assert all(p.device.type == 'cuda' for p in model.parameters())
    assert model.config.model_type == 'nemotron_h'
    assert install_mamba_device_guards(model) == smoke['guarded_mamba_layers']
    dense_v = {layer: model.model.layers[layer].mixer.v_proj.weight.detach().cpu().clone() for layer in LAYERS}
    tokens = load_file(str(args.windows))['input_ids'].long()
    assert tokens.shape == (336, 2048)
    teachers = {split: _capture_teacher(model, tokens[ids], batch_size=1,
        vocab_chunk_size=8192, label=split) for split, ids in (
            ('profile', list(range(320, 328))), ('confirmation', list(range(328, 336))))}
    cache, closures = {}, {}

    def factors(layer, rank):
        key = (layer, rank)
        if key not in cache:
            record = results[rank]['artifacts'][str(layer)]
            path = args.bank / f'r{rank}' / record['file']
            assert sha256(path) == record['sha256']
            bank = load_file(str(path))
            snapshot_record = snapshots['artifacts'][str(layer)]
            snapshot_path = args.snapshots / snapshot_record['file']
            assert sha256(snapshot_path) == snapshot_record['sha256']
            snapshot = load_file(str(snapshot_path))
            device = model.model.layers[layer].mixer.v_proj.weight.device
            A, D, diagnostic = close_candidate(bank['value_coordinate_encoders'].to(device),
                bank['head_output_decoders'].to(device), snapshot['fit_covariance'].to(device),
                snapshot['weight'].to(device))
            assert A.shape == (8, 128, rank) and D.shape == (64, rank, 8192)
            cache[key] = (A, D)
            closures[f'{layer}:{rank}'] = diagnostic
            print('CLOSED', layer, rank, diagnostic, flush=True)
        return cache[key]

    def install(layer, rank):
        mixer = model.model.layers[layer].mixer
        A, D = factors(layer, rank)
        v, o = _fold_ragged_to_padded_weights(dense_v_weight=dense_v[layer].to(mixer.v_proj.weight.device),
            A=A, D=D, source_ranks=(rank,) * 8)
        mixer.v_proj.weight.copy_(v)
        mixer.o_proj.weight.copy_(o)

    def measure(name, split):
        path = args.output / 'measurements' / (name + '.json')
        if path.exists():
            record = read_json(path)
            assert record['protocol_sha256'] == sha256(args.output / 'protocol.json')
            assert record['split'] == split
            return record['metrics']
        metrics = _evaluate_teacher_metrics(model, teachers[split], vocab_chunk_size=8192)
        write_json(path, dict(protocol_sha256=sha256(args.output / 'protocol.json'), split=split, metrics=metrics))
        print('MEASURED', name, metrics['terminal_kl']['mean'], flush=True)
        return metrics

    for layer in LAYERS:
        install(layer, 64)
    anchor = measure('anchor64_profile', 'profile')
    anchor_confirmation = measure('anchor64_confirmation', 'confirmation')
    deltas = {32: [], 96: []}
    probes = {}
    for layer in LAYERS:
        for rank in (32, 96):
            install(layer, rank)
            metrics = measure(f'layer_{layer:03d}_r{rank}', 'profile')
            delta = _paired_delta(metrics, anchor)
            deltas[rank].append(delta['terminal_kl']['mean'])
            probes[f'{layer}:{rank}'] = dict(metrics=metrics, delta=delta)
        install(layer, 64)
    allocation = allocate(results, deltas[32], deltas[96])
    write_json(args.output / 'allocation.json', allocation)
    for layer in LAYERS:
        install(layer, 96)
    uniform = measure('uniform96_confirmation', 'confirmation')
    for layer, rank in zip(LAYERS, allocation['layer_ranks'], strict=True):
        install(layer, rank)
    selected = measure('selected_confirmation', 'confirmation')
    schedule = [[rank] * 8 for rank in allocation['layer_ranks']]
    result = dict(status='complete', protocol=protocol, allocation=allocation, probes=probes,
        selection=dict(selected_candidate='two_sided_factorized_kl',
            factorized_method=dict(exponent=1.0), selected_schedule=schedule),
        confirmation=dict(anchor64=anchor_confirmation, uniform96=uniform, selected=selected,
            selected_minus_uniform96=_paired_delta(selected, uniform)),
        closures=closures, command=shlex.join(sys.argv), python=sys.executable,
        device_map=model.hf_device_map,
        peak_gib=[torch.cuda.max_memory_allocated(i) / 2**30 for i in range(4)])
    write_json(args.output / 'results.json', result)
    entries = []
    for layer, rank in zip(LAYERS, allocation['layer_ranks'], strict=True):
        A, D = factors(layer, rank)
        path = args.output / f'layer_{layer:03d}.safetensors'
        save_tensors(path, dict(value_coordinate_encoders=A, head_output_decoders=D,
            source_ranks=torch.tensor([rank] * 8, dtype=torch.int64)))
        entries.append(dict(layer=layer, file=path.name, ranks=[rank] * 8, sha256=sha256(path)))
    manifest = dict(status='complete', model=dict(path=str(model_path), config_sha256=audit['config_sha256'],
        safetensors_index_sha256=audit['index_sha256']),
        compression=dict(method='c1-two-sided-kl', allocation='two_sided_factorized_terminal_kl_alpha1',
            equivalent_rank_target=96, layer_ranks=schedule), layers=entries,
        artifact=dict(file='results.json', sha256=sha256(args.output / 'results.json')))
    write_json(args.output / 'manifest.json', manifest)
    write_json(args.identity_output, dict(status='complete', checkpoint=str(args.output.resolve()),
        model=str(model_path), manifest_sha256=sha256(args.output / 'manifest.json'),
        model_config_sha256=audit['config_sha256'], attention_layers=list(LAYERS),
        layer_ranks=allocation['layer_ranks'], mean_rank=96, hq=64, hkv=8, head_dim=128, hidden_size=8192))
    print('ATTENTION V96 EXPORTED', allocation['layer_ranks'], flush=True)


if __name__ == '__main__':
    main()
