"""Stage B: frozen-V recapture and reuse of source-private C1 output ALS."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

from basisserve.core.qwen35_gated_v_runtime import GatedVRuntime, factor_hash
from basisserve.core.qwen35_gdn_private_ag import fit_qwen35_private_ag_joint_factors
from basisserve.core.qwen35_gdn_private_ag_runtime import FACTOR_FORMAT as GDN_FORMAT
from basisserve.core.qwen35_full_attention_private_ag_runtime import FACTOR_FORMAT as FULL_FORMAT
from evaluation.qwen35_hybrid_common import atomic_save, load_model, load_windows, load_bank, sha256, verify_model_identity


@torch.no_grad()
def capture(args):
    bank = load_bank(args.bank)
    upstream = bank['factor_sha256']
    assert bank['windows_sha256'] == sha256(Path(args.data) / 'windows.pt')
    verify_model_identity(args.model_path, bank['model_identity'])
    model = load_model(args.model_path, args.device)
    assert not (Path(args.output) / 'manifest.json').exists()
    with GatedVRuntime(model, bank['layers']):
        projections = {}
        for index, layer in enumerate(model.model.layers):
            kind = 'full_attention' if hasattr(layer, 'self_attn') else 'gdn'
            projection = layer.self_attn.o_proj if kind == 'full_attention' else layer.linear_attn.out_proj
            projections[index] = (kind, projection)
        for split in ('fit', 'heldout'):
            grams = {i: torch.zeros(p.in_features, p.in_features, device=args.moment_device, dtype=torch.float64) for i, (_, p) in projections.items()}
            counts = {i: 0 for i in projections}
            handles = []
            for i, (_, p) in projections.items():
                def hook(module, inputs, i=i):
                    x = inputs[0].detach().flatten(0, 1).double()
                    grams[i] += (x.T @ x).to(grams[i].device)
                    counts[i] += len(x)
                handles.append(p.register_forward_pre_hook(hook))
            windows = load_windows(args.data, split)
            for index, tokens in enumerate(windows):
                model.model(tokens[None].to(args.device), use_cache=False)
                if index % 16 == 0:
                    print(json.dumps({'split': split, 'window': index, 'upstream_v_factor_sha256': upstream}), flush=True)
            for handle in handles:
                handle.remove()
            for i, (kind, projection) in projections.items():
                assert counts[i] == windows.numel()
                atomic_save(Path(args.output) / f'{split}_l{i:02d}.pt', {'second_moment': (grams[i] / counts[i]).cpu(),
                    'weight': projection.weight.detach().cpu(), 'layer_type': kind, 'rows': counts[i],
                    'upstream_v_factor_sha256': upstream, 'windows_sha256': bank['windows_sha256']})
            del grams
    assert factor_hash(bank['layers']) == upstream
    atomic_save(Path(args.output) / 'manifest.json', {'trajectory': 'frozen_v_compressed' if bank['layers'] else 'native_dense',
        'target': 'current_post_gate_input_times_original_output_weight', 'upstream_v_factor_sha256': upstream,
        'verified_model_identity': bank['model_identity'],
        'windows_sha256': bank['windows_sha256'], 'layer_count': len(projections), 'tp_size': args.tp_size,
        'source_layout': 'contiguous equal input blocks; physical KV group ownership for TP4'})


@torch.no_grad()
def fit(args):
    root = Path(args.moments)
    manifest = json.loads((root / 'manifest.json').read_text())
    bank = load_bank(args.bank)
    assert manifest['upstream_v_factor_sha256'] == bank['factor_sha256']
    gdn, full = [], []
    assert manifest['windows_sha256'] == bank['windows_sha256']
    assert manifest['verified_model_identity'] == bank['model_identity']
    manifest_sha = sha256(root / 'manifest.json')
    config = json.loads((Path(args.model_path) / 'config.json').read_text())['text_config']
    widths = {'gdn': config['linear_num_value_heads'] * config['linear_value_head_dim'],
              'full_attention': config['num_attention_heads'] * config['head_dim']}
    for i in range(manifest['layer_count']):
        if args.stage == 'fit' and i % args.num_shards != args.shard_index:
            continue
        checkpoint = Path(args.output) / f'wo_l{i:02d}.pt'
        if checkpoint.exists():
            record = torch.load(checkpoint, weights_only=True, map_location='cpu')
            assert record['upstream_v_factor_sha256'] == bank['factor_sha256']
            assert record['work_dtype'] == args.work_dtype
            expected_rank = args.gdn_rank if record['layer_type'] == 'gdn' else args.full_rank
            assert record['private_encoders'].shape == (args.tp_size, widths[record['layer_type']] // args.tp_size, expected_rank)
        else:
            assert args.stage == 'fit', f'Missing fitted layer: {checkpoint}'
            started = time.monotonic()
            train = torch.load(root / f'fit_l{i:02d}.pt', weights_only=True, map_location=args.device)
            heldout = torch.load(root / f'heldout_l{i:02d}.pt', weights_only=True, map_location=args.device)
            assert train['upstream_v_factor_sha256'] == heldout['upstream_v_factor_sha256'] == bank['factor_sha256']
            width = train['weight'].shape[1]
            rank = args.gdn_rank if train['layer_type'] == 'gdn' else args.full_rank
            assert width % args.tp_size == 0 and 0 < rank <= width // args.tp_size
            assert train['windows_sha256'] == heldout['windows_sha256'] == bank['windows_sha256']
            assert train['layer_type'] == heldout['layer_type']
            fitted = fit_qwen35_private_ag_joint_factors(train['weight'], train['second_moment'], heldout['second_moment'],
                tp_size=args.tp_size, local_rank=rank, encoder_sweeps=6, minimum_encoder_sweeps=6,
                work_dtype=getattr(torch, args.work_dtype), factor_dtype=torch.bfloat16)
            record = {'layer_index': i, 'layer_type': train['layer_type'], 'private_encoders': fitted.private_encoders,
                'joint_decoder_weight': fitted.joint_decoder_weight, 'metrics': fitted.metrics,
                'work_dtype': args.work_dtype,
                'moment_manifest_sha256': manifest_sha,
                'upstream_v_factor_sha256': bank['factor_sha256'], 'seconds': time.monotonic() - started}
            atomic_save(checkpoint, record)
            print(json.dumps({'layer': i, 'seconds': record['seconds'], 'metrics': fitted.metrics}), flush=True)
        (full if record['layer_type'] == 'full_attention' else gdn).append(record)
    if args.stage == 'fit' and args.num_shards > 1:
        return
    assert len(gdn) == config['layer_types'].count('linear_attention')
    assert len(full) == config['layer_types'].count('full_attention')
    for record in gdn + full:
        rank = args.gdn_rank if record['layer_type'] == 'gdn' else args.full_rank
        assert record['moment_manifest_sha256'] == manifest_sha
        assert record['private_encoders'].shape == (args.tp_size, widths[record['layer_type']] // args.tp_size, rank)
        assert record['joint_decoder_weight'].shape == (config['hidden_size'], args.tp_size * rank)
        assert record['metrics']['diagnostics']['encoder_sweeps_completed'] == 6
        for key in ('private_encoders', 'joint_decoder_weight'):
            assert record[key].dtype == torch.bfloat16 and torch.isfinite(record[key]).all()
    atomic_save(Path(args.output) / 'wo_bank.pt', {'format': 'basisserve.qwen35.hybrid_output_bank.v1',
        'upstream_v_factor_sha256': bank['factor_sha256'], 'tp_size': args.tp_size,
        'retained_output_input_fraction': args.tp_size * (len(gdn) * args.gdn_rank + len(full) * args.full_rank) / (len(gdn) * widths['gdn'] + len(full) * widths['full_attention']),
        'source_rank_by_type': {'gdn': args.gdn_rank, 'full_attention': args.full_rank},
        'retained_output_input_fraction_by_type': {'gdn': args.tp_size * args.gdn_rank / widths['gdn'], 'full_attention': args.tp_size * args.full_rank / widths['full_attention']},
        'work_dtype': args.work_dtype, 'factor_dtype': 'bfloat16',
        'runtime': 'single_process_private_allgather_equivalent',
        'gdn': {'format': GDN_FORMAT, 'schema_version': 1, 'layers': gdn},
        'full_attention': {'format': FULL_FORMAT, 'schema_version': 1, 'layers': full},
        'moment_manifest_sha256': sha256(root / 'manifest.json'), 'windows_sha256': bank['windows_sha256']})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['capture', 'fit', 'assemble'])
    p.add_argument('--model-path', default='results/q35_hybrid/model')
    p.add_argument('--data', default='results/q35_hybrid/data')
    p.add_argument('--bank', required=True)
    p.add_argument('--moments')
    p.add_argument('--output', required=True)
    p.add_argument('--tp-size', type=int, choices=[4], default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--moment-device', choices=['cpu', 'cuda:0'], default='cuda:0')
    p.add_argument('--work-dtype', choices=['float32', 'float64'], default='float64')
    p.add_argument('--gdn-rank', type=int, default=512)
    p.add_argument('--full-rank', type=int, default=512)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    args = p.parse_args()
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(2)
    print(json.dumps({'args': vars(args), 'python': sys.executable, 'torch': torch.__version__}), flush=True)
    (capture if args.stage == 'capture' else fit)(args)


if __name__ == '__main__':
    main()
