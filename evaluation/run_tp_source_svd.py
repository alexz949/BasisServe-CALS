"""Fit/evaluate independent full-covariance TP-source SVD, without KV compression."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basisserve.core.tp_source_svd import fit_layer
from basisserve.core.tp_source_wo_c1 import TPSourceWOLayout

FORMAT = 'basisserve.tp_source_full_covariance_svd.v1'
COVARIANCE_FORMAT = 'basisserve.tp_source_svd_covariance.v1'
C4_REVISION = '1588ec454efa1a09f29cd18ddd04fe05fc8653a2'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(path):
    parts = Path(path).parts
    snapshots = parts.index('snapshots')
    assert snapshots > 0 and parts[snapshots - 1].startswith('models--')
    return {'repo': parts[snapshots - 1][len('models--'):].replace('--', '/'),
            'revision': parts[snapshots + 1]}


def validate_inputs(model, covariance_dir, phase):
    config = json.loads((model / 'config.json').read_text())
    manifest = json.loads((covariance_dir / 'manifest.json').read_text())
    model_id = identity(model)
    assert model_id['repo'] == 'Qwen/Qwen3-8B-Base'
    assert identity(manifest['model']['path']) == model_id, 'Covariance/model snapshot mismatch'
    assert manifest['format'] == COVARIANCE_FORMAT and manifest['status'] == 'complete'
    assert manifest['phase'] == phase
    assert config['model_type'] == 'qwen3'
    assert (config['hidden_size'], config['num_hidden_layers'], config['num_attention_heads'],
            config['num_key_value_heads'], config['head_dim']) == (4096, 36, 32, 8, 128)
    calibration = manifest['calibration']
    assert calibration['storage'] == 'normalized_covariance_sufficient_statistics'
    count = 2 if phase == 'smoke' else 256
    assert (calibration['fit_windows'], calibration['heldout_windows'],
            calibration['sequence_length']) == (count, 0, 2048)
    assert calibration['fit_rows'] == count * 2048 and calibration['heldout_rows'] == 0
    assert calibration['positions_per_window'] == 2048 and calibration['window_count'] == count
    assert calibration['dataset'] == 'allenai/c4' and calibration['split'] == 'train'
    assert calibration['revision'] == C4_REVISION and calibration['moment'] == 'uncentered_ZtZ_div_N'
    assert len({r['document_id'] for r in calibration['windows']}) == count
    layers = [0] if phase == 'smoke' else list(range(36))
    assert sorted(map(int, manifest['artifacts'])) == layers
    assert sorted(manifest['layers']) == layers
    for key in ('tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'):
        assert (model / key).is_file(), f'Missing model input: {key}'
    return config, manifest, model_id


@torch.inference_mode()
def capture(args):
    import random
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    from evaluation.capture_attention_o_proj_covariances import _StreamingCovarianceCapture
    model_id = identity(args.model)
    assert model_id['repo'] == 'Qwen/Qwen3-8B-Base'
    covariance_dir = args.output / 'covariance'
    assert not covariance_dir.exists(), 'Preserve collected statistics'
    covariance_dir.mkdir(parents=True)
    count = 2 if args.phase == 'smoke' else 256
    layers = [0] if args.phase == 'smoke' else list(range(36))
    payload = {'format': COVARIANCE_FORMAT, 'status': 'running', 'phase': args.phase,
               'command': shlex.join(sys.argv), 'model': {'path': str(args.model.resolve()), **model_id},
               'environment': os.environ.get('CONDA_DEFAULT_ENV'), 'layers': layers, 'artifacts': {}}
    write_json(covariance_dir / 'manifest.json', payload)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    stream = load_dataset('allenai/c4', 'en', split='train', revision=C4_REVISION,
                          streaming=True).shuffle(seed=20260821, buffer_size=10000)
    rng = random.Random(20260821)
    windows, records, seen = [], [], set()
    for row in stream:
        document_id = row.get('url')
        if not document_id or document_id in seen:
            continue
        ids = tokenizer(row['text'], return_tensors='pt').input_ids[0]
        if ids.numel() < 2048:
            continue
        start = rng.randint(0, ids.numel() - 2048)
        windows.append(ids[start:start + 2048])
        records.append({'document_id': document_id, 'start': start, 'document_tokens': ids.numel()})
        seen.add(document_id)
        if len(windows) == count:
            break
    assert len(windows) == count
    windows = torch.stack(windows)
    save_file({'input_ids': windows}, str(covariance_dir / 'windows.safetensors'))
    payload['calibration'] = {'dataset': 'allenai/c4', 'split': 'train', 'revision': C4_REVISION,
                              'seed': 20260821, 'shuffle_buffer': 10000, 'windows': records,
                              'fit_windows': count, 'heldout_windows': 0, 'window_count': count,
                              'sequence_length': 2048, 'positions_per_window': 2048,
                              'fit_rows': count * 2048, 'heldout_rows': 0,
                              'storage': 'normalized_covariance_sufficient_statistics',
                              'moment': 'uncentered_ZtZ_div_N', 'covariance_dtype': 'float64'}
    write_json(covariance_dir / 'manifest.json', payload)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                      attn_implementation='sdpa', local_files_only=True).to(args.device).eval()
    assert model.config.model_type == 'qwen3' and model.config.num_hidden_layers == 36
    assert model.config.hidden_size == model.config.num_attention_heads * model.config.head_dim == 4096
    modules = {i: model.layers[i].self_attn.o_proj for i in layers}
    collector = _StreamingCovarianceCapture(modules, 4096, torch.float64)
    started = time.perf_counter()
    try:
        for i, window in enumerate(windows):
            collector.begin('fit', rows=2048)
            model(input_ids=window[None].to(args.device), use_cache=False)
            collector.finish()
            print(f'CAPTURE {i + 1}/{count}', flush=True)
        collector.normalize_and_offload('fit', expected_rows=count * 2048)
    finally:
        collector.close()
    for layer in layers:
        file = f'layer_{layer:03d}.safetensors'
        save_file({'fit_covariance': collector.sums['fit'].pop(layer).contiguous(),
                   'weight': modules[layer].weight.detach().cpu().contiguous()}, str(covariance_dir / file))
        payload['artifacts'][str(layer)] = {'file': file}
        write_json(covariance_dir / 'manifest.json', payload)
    payload.update(status='complete', elapsed_seconds=time.perf_counter() - started,
                   heldout_status='not_collected_by_user_request')
    write_json(covariance_dir / 'manifest.json', payload)


def dense_weight(model, layer):
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    name = f'model.layers.{layer}.self_attn.o_proj.weight'
    with safe_open(str(model / index[name]), framework='pt', device='cpu') as stream:
        return stream.get_tensor(name)


def summary(output):
    config = json.loads((output / 'config.json').read_text())
    lines = ['# Full-Covariance TP-Source SVD', '',
             f"Phase: {config['phase']}. Model: {config['model_identity']['repo']}.",
             f"TP source partition: {config['tp_size']}; source ranks: {config['ranks']}.",
             'Free post-attention source encoders; dense Q/K/V and KV cache. Not V64/V96.',
             'Zero ridge; support-truncated FP64 optimum audited separately from BF16 export.',
             'PPL uses FP64-materialized weights cast once to BF16, not BF16 factor execution.', '',
             '## Commands', '', '```bash', config['command'], '```',
             f"Conda environment: {config['environment']['conda']}", '',
             '## Results', '', '| Arm | PPL | Scored tokens | Status |', '| --- | ---: | ---: | --- |']
    ppl_path = output / 'ppl.json'
    ppl = json.loads(ppl_path.read_text()) if ppl_path.exists() else {'arms': {}}
    for arm in ['dense', *(f'r{rank}' for rank in config['ranks'])]:
        result = ppl['arms'].get(arm)
        lines.append(f"| {arm} | {result['ppl']:.8f} | {result['tokens']} | complete |" if result
                     else f'| {arm} | - | - | not run |')
    lines += ['', 'No joint-vs-independent claim is made: no matched joint factors are fitted here.',
              'Synthetic smoke is not real-model PPL; historical TP4 results are not substituted.',
              'Held-out statistics were not collected, per user request; fit errors are not generalization evidence.',
              'See fit_metrics.json for local/final fit errors, denominators and rounding diagnostics.',
              'Inputs use snapshot identity, structure and direct weight equality checks, not SHA256.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')


@torch.inference_mode()
def fit(args):
    model_config, manifest, model_id = validate_inputs(args.model, args.covariance, args.phase)
    layers = [0] if args.phase == 'smoke' else list(range(model_config['num_hidden_layers']))
    layout = TPSourceWOLayout(4096, 4096, args.tp_size, args.ranks[0])
    assert all(0 < rank <= layout.source_width for rank in args.ranks)
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'config.json').exists(), 'Preserve existing experiment'
    config = {'format': FORMAT, 'status': 'running', 'phase': args.phase,
              'command': shlex.join(sys.argv), 'model': str(args.model.resolve()),
              'model_identity': model_id, 'covariance': str(args.covariance.resolve()),
              'calibration': manifest['calibration'], 'layers': layers,
              'tp_size': args.tp_size, 'ranks': args.ranks, 'source_width': layout.source_width,
              'source_partition': 'contiguous o_proj input; four query heads/source at TP8',
              'moment_convention': 'uncentered Z.T@Z/N, as recorded by covariance collector',
              'split_convention': 'fit only; no heldout, per user request',
              'ridge': 0.0, 'support_rtol': args.support_rtol, 'support_atol': args.support_atol,
              'negative_rtol': args.negative_rtol, 'fit_dtype': 'float64',
              'export': 'FP64 product then single BF16 cast',
              'environment': {'conda': os.environ.get('CONDA_DEFAULT_ENV'), 'torch': torch.__version__},
              'started_utc': datetime.now(timezone.utc).isoformat()}
    write_json(args.output / 'config.json', config)
    metrics = {'format': FORMAT, 'status': 'running', 'layers': []}
    write_json(args.output / 'fit_metrics.json', metrics)
    started = time.perf_counter()
    for layer in layers:
        path = args.covariance / manifest['artifacts'][str(layer)]['file']
        tensors = load_file(str(path))
        assert torch.equal(tensors['weight'].float(), dense_weight(args.model, layer).float()), 'Dense weight mismatch'
        weight, fit_cov = (tensors[key].to(args.device, dtype=torch.float64)
                          for key in ('weight', 'fit_covariance'))
        for rank in args.ranks:
            layout = TPSourceWOLayout(4096, 4096, args.tp_size, rank)
            factors, audit = fit_layer(weight, fit_cov, None, layout,
                                      fit_rows=manifest['calibration']['fit_rows'],
                                      heldout_rows=manifest['calibration']['heldout_rows'],
                                      support_rtol=args.support_rtol, support_atol=args.support_atol,
                                      negative_rtol=args.negative_rtol)
            destination = args.output / 'factors' / f'r{rank}' / f'layer_{layer:03d}.safetensors'
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_file({k: v.cpu().contiguous() for k, v in factors.items()}, str(destination),
                      metadata={'format': FORMAT, 'layout': json.dumps(asdict(layout)),
                                'model_identity': json.dumps(model_id), 'layer': str(layer), 'ridge': '0'})
            metrics['layers'].append({'layer': layer, 'rank': rank,
                                      'file': str(destination.relative_to(args.output)), **audit})
            write_json(args.output / 'fit_metrics.json', metrics)
            print(json.dumps({'layer': layer, 'rank': rank, 'status': 'complete',
                              'fit': audit['losses']['fit']['materialized_bf16']}), flush=True)
            del factors
        del tensors, weight, fit_cov
    metrics.update(status='complete', elapsed_seconds=time.perf_counter() - started)
    config.update(status='complete', elapsed_seconds=metrics['elapsed_seconds'])
    write_json(args.output / 'fit_metrics.json', metrics)
    write_json(args.output / 'config.json', config)
    summary(args.output)


@torch.inference_mode()
def install(model, originals, output, rank, config):
    for layer, original in enumerate(originals):
        model.model.layers[layer].self_attn.o_proj.weight.copy_(original)
    if rank is None:
        return
    for layer in config['layers']:
        path = output / 'factors' / f'r{rank}' / f'layer_{layer:03d}.safetensors'
        with safe_open(str(path), framework='pt', device='cpu') as stream:
            metadata = stream.metadata()
            assert metadata['format'] == FORMAT and metadata['layer'] == str(layer)
            layout = json.loads(metadata['layout'])
            assert layout['tp_size'] == config['tp_size'] and layout['source_rank'] == rank
            assert layout['input_width'] == model.config.num_attention_heads * model.config.head_dim
            assert json.loads(metadata['model_identity']) == config['model_identity']
            value = stream.get_tensor('materialized_weight_bf16')
        target = model.model.layers[layer].self_attn.o_proj.weight
        assert value.dtype == torch.bfloat16 and value.shape == target.shape and torch.isfinite(value).all()
        target.copy_(value)


@torch.inference_mode()
def ppl(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.eval_attention_o_proj_collective_ppl import _eval_ppl_fp32_loss
    from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids, WIKITEXT_REVISION
    config = json.loads((args.output / 'config.json').read_text())
    assert config['format'] == FORMAT and config['status'] == 'complete'
    assert config['phase'] == args.phase, 'Smoke and formal artifacts must be separate'
    assert identity(args.model) == config['model_identity']
    assert not (args.output / 'ppl.json').exists(), 'Preserve existing evaluation'
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = _token_ids(tokenizer, 'wikitext2', 'test', None).long().flatten()
    save_file({'input_ids': tokens}, str(args.output / 'ppl_tokens.safetensors'))
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                               attn_implementation='sdpa', local_files_only=True).to(args.device).eval()
    assert args.phase == 'smoke' or config['layers'] == list(range(model.config.num_hidden_layers))
    originals = [layer.self_attn.o_proj.weight.detach().cpu().clone() for layer in model.model.layers]
    payload = {'format': FORMAT, 'phase': args.phase, 'status': 'running',
               'command': shlex.join(sys.argv), 'environment': os.environ.get('CONDA_DEFAULT_ENV'),
               'dataset_revision': WIKITEXT_REVISION, 'model_identity': config['model_identity'],
               'tokenizer': str(args.model), 'attention': 'sdpa', 'model_dtype': 'bfloat16',
               'loss_dtype': 'float32', 'token_file': 'ppl_tokens.safetensors', 'arms': {}}
    payload['compressed_layers'] = config['layers']
    payload['scope'] = 'first-layer-only short smoke' if args.phase == 'smoke' else 'all-layer full WT2 test'
    write_json(args.output / 'ppl.json', payload)
    started = time.perf_counter()
    for rank in [None, *config['ranks']]:
        arm_start = time.perf_counter()
        install(model, originals, args.output, rank, config)
        result = _eval_ppl_fp32_loss(model, tokenizer, dataset='wikitext2', split='test',
                                    seqlen=2048, batch_size=1,
                                    max_samples=2 if args.phase == 'smoke' else None,
                                    max_tokens=None, input_ids=tokens)
        result.update(status='complete', elapsed_seconds=time.perf_counter() - arm_start)
        payload['arms']['dense' if rank is None else f'r{rank}'] = result
        write_json(args.output / 'ppl.json', payload)
        summary(args.output)
        print(json.dumps({'rank': rank, **result}), flush=True)
    payload.update(status='complete', elapsed_seconds=time.perf_counter() - started)
    write_json(args.output / 'ppl.json', payload)
    summary(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('capture', 'fit', 'ppl'), required=True)
    parser.add_argument('--phase', choices=('smoke', 'formal'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--covariance', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tp-size', type=int, default=8)
    parser.add_argument('--ranks', nargs='+', type=int, default=[256, 384])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--support-rtol', type=float, default=1e-12)
    parser.add_argument('--support-atol', type=float, default=0.0)
    parser.add_argument('--negative-rtol', type=float, default=1e-10)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert len(set(args.ranks)) == len(args.ranks)
    if args.covariance is None:
        args.covariance = args.output / 'covariance'
    success = False
    status_path = args.output / f'{args.stage}_status.json'
    write_json(status_path, {'status': 'running', 'command': shlex.join(sys.argv)})
    try:
        {'capture': capture, 'fit': fit, 'ppl': ppl}[args.stage](args)
        success = True
    finally:
        write_json(status_path, {'status': 'complete' if success else 'failed',
                                'command': shlex.join(sys.argv),
                                'environment': os.environ.get('CONDA_DEFAULT_ENV')})


if __name__ == '__main__':
    main()
