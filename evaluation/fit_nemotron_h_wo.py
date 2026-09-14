"""Fit Mamba Wo factors at the frozen attention V96 AllGather wire budget."""
import argparse
from dataclasses import asdict
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, save_tensors, sha256
from basisserve.core.tp_source_wo_c1 import TPSourceWOLayout
from basisserve.core.tp_source_wo_fit import TPSourceWOFitConfig, fit_tp_source_wo_c1


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--covariances', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tp', type=int, default=4)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=4)
    parser.add_argument('--layers', type=str)
    args = parser.parse_args()
    configure()
    audit, identity = read_json(args.audit), read_json(args.identity)
    assert audit['status'] == identity['status'] == 'complete'
    assert identity['model_config_sha256'] == audit['config_sha256']
    assert identity['mean_rank'] == 96 and sum(identity['layer_ranks']) == 96*len(identity['layer_ranks'])
    checkpoint = Path(identity['checkpoint'])
    assert sha256(checkpoint / 'manifest.json') == identity['manifest_sha256']
    attention_layers = [row['layer'] for row in audit['layers'] if row['kind'] == 'full_attention']
    assert identity['attention_layers'] == attention_layers
    targets = {row['layer']: row for row in audit['layers'] if row['kind'] == 'linear_attention'}
    assert len(targets) == 45
    manifest_path = args.covariances / 'manifest.json'
    manifest = read_json(manifest_path)
    assert manifest['status'] == 'complete' and manifest['dense_teacher']
    assert manifest['layer_kind'] == 'linear_attention'
    assert manifest['audit_sha256'] == sha256(args.audit)
    assert set(manifest['layers']) == set(targets)
    assert manifest['calibration']['fit_windows'] == 256 and manifest['calibration']['heldout_windows'] == 64
    assert manifest['calibration']['sequence_length'] == 2048
    total_rank = identity['hq'] * identity['mean_rank']
    assert total_rank == 6144 and args.tp > 1 and total_rank % args.tp == 0
    source_rank = total_rank // args.tp
    attention_layout = TPSourceWOLayout(input_width=identity['hq']*identity['head_dim'],
        output_width=identity['hidden_size'], tp_size=args.tp, source_rank=source_rank)
    config = TPSourceWOFitConfig(encoder_sweeps=12, minimum_encoder_sweeps=12)
    protocol = dict(identity_sha256=sha256(args.identity), audit_sha256=sha256(args.audit),
        covariance_manifest_sha256=sha256(manifest_path), tp=args.tp, total_rank=total_rank,
        source_rank=source_rank, attention_reference=attention_layout.accounting(), fit=asdict(config),
        selection='existing TP-source solver held-out MSE selector over decoder-closed checkpoints',
        encoder_solver='existing exact single-source two-sided Cholesky',
        work_dtype='float64', factor_dtype='bfloat16',
        source_sha256={name:sha256(ROOT/name) for name in ('evaluation/fit_nemotron_h_wo.py',
            'basisserve/core/tp_source_wo_fit.py', 'basisserve/core/tp_source_wo_c1.py',
            'basisserve/core/gqa_routed_ov_joint.py')})
    assert 0 <= args.shard_index < args.num_shards <= len(targets)
    layers = ([int(value) for value in args.layers.split(',')] if args.layers
        else sorted(targets)[args.shard_index::args.num_shards])
    assert layers and len(layers)==len(set(layers)) and set(layers)<=set(targets)
    for layer in layers:
        output = args.output / f'layer_{layer:03d}.safetensors'
        if output.with_suffix('.json').exists():
            record = read_json(output.with_suffix('.json'))
            assert record['status'] == 'complete' and record['protocol'] == protocol
            assert record['sha256'] == sha256(output)
            continue
        target = targets[layer]
        layout = TPSourceWOLayout(input_width=target['input_width'], output_width=target['output_width'],
            tp_size=args.tp, source_rank=source_rank)
        assert layout.gathered_width == attention_layout.gathered_width
        assert layout.reduction_vs_dense_allreduce == attention_layout.reduction_vs_dense_allreduce
        source = manifest['artifacts'][str(layer)]
        path = args.covariances / source['file']
        assert sha256(path) == source['sha256']
        tensors = load_file(str(path))
        print('WO FIT START', layer, layout.accounting(), flush=True)
        started = time.monotonic()
        fitted = fit_tp_source_wo_c1(tensors['weight'], tensors['fit_covariance'],
            tensors['heldout_covariance'], layout, config=config, work_device='cuda',
            work_dtype=torch.float64, factor_dtype=torch.bfloat16,
            objective_name=f'nemotron_h_mamba_wo_{layer:03d}')
        factors = dict(source_encoders=fitted.source_encoders, source_decoders=fitted.source_decoders)
        assert factors['source_encoders'].shape == (args.tp, layout.source_width, source_rank)
        assert factors['source_decoders'].shape == (args.tp, source_rank, layout.output_width)
        assert all(t.dtype == torch.bfloat16 and torch.isfinite(t).all() for t in factors.values())
        save_tensors(output, factors)
        write_json(output.with_suffix('.json'), dict(status='complete', layer=layer,
            protocol=protocol, layout=layout.accounting(), covariance_sha256=source['sha256'],
            sha256=sha256(output), fit_relative_mse=fitted.fit_relative_mse,
            heldout_relative_mse=fitted.heldout_relative_mse,
            quantized_fit_relative_mse=fitted.quantized_fit_relative_mse,
            quantized_heldout_relative_mse=fitted.quantized_heldout_relative_mse,
            selected_boundary=fitted.selected_boundary, selected_sweep=fitted.selected_sweep,
            checkpoints=fitted.checkpoints, diagnostics=fitted.diagnostics,
            command=shlex.join(sys.argv), python=sys.executable, seconds=time.monotonic()-started))
        print('WO FIT COMPLETE', layer, 'heldout', fitted.quantized_heldout_relative_mse, flush=True)
        del tensors, factors, fitted
        torch.cuda.empty_cache()
    write_json(args.output / f'complete_{args.shard_index}.json',
        dict(status='complete', protocol=protocol, layers=layers))


if __name__ == '__main__':
    main()
