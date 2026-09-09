"""Fit closed-form Base16 and non-sink Page-Fisher R16 for frozen uniform V96."""
import argparse
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from safetensors.torch import load_file
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _load_direct, _fit_base_maps, _fit_residual_grid
from evaluation.fit_qwen3_8b_q8_fisher_residual import build_multi_query_statistics
from evaluation.select_query_positions import manifest_queries
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _discover_capture, _rotary_embeddings
from evaluation.select_query_positions import load_position_manifest
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json


def protocol(args):
    c1 = json.loads((args.allocation / 'results.json').read_text())
    config = c1['fit_config']
    assert c1['status'] == 'complete' and config['cache_rank_per_head'] == 96
    assert config['model_config_sha256'] == sha256(args.model / 'config.json')
    old = json.loads((args.reference_bank / 'layer_000.json').read_text())['protocol']
    selected = load_position_manifest(args.positions)
    assert selected['num_fit_windows'] == 64 and selected['fit_document_ids'] == list(range(64))
    assert selected['context_length'] == 32768 and selected['queries_per_bin'] == 8 and selected['num_bins'] == 4
    assert sha256(args.positions) == old['query_position_manifest_sha256']
    assert sha256(args.queries / 'manifest.json') == old['query_capture_manifest_sha256']
    assert selected['model_config_sha256'] == config['model_config_sha256']
    layers, captures = [], {}
    for l in range(36):
        artifact = c1['artifacts'][str(l)]
        assert sha256(args.allocation / artifact['file']) == artifact['sha256']
        layers.append(dict(layer=l, file=artifact['file'], sha256=artifact['sha256']))
        captures[str(l)] = {}
        for split in ('fit', 'validation'):
            root, _ = _discover_capture(args.calibration_root, split=split, layer=l)
            digest = sha256(root / 'manifest.json')
            assert digest == old['inputs'][str(l)][split]['direct_manifest_sha256']
            captures[str(l)][split] = dict(root=str(root), manifest_sha256=digest)
    spec = dict(format='basisserve.uniform_v96_base_residual.v1',
        model_config_sha256=config['model_config_sha256'], checkpoint=str(args.allocation.resolve()),
        c1_manifest_sha256=sha256(args.allocation / 'results.json'), payload_ranks=[96]*36,
        c1_layer_sha256={str(l['layer']): l['sha256'] for l in layers},
        base_rank=16, base_kind='closed_form_rrr', base_objective='affine pre-RoPE K reconstruction MSE',
        residual_rank=16, residual_objective='causal non-sink Page-Fisher', bcd_sweeps=50,
        initialization='top16 eigenvectors of fit Page-Fisher Gram summed across Q and associated heads; U0=E0',
        relative_damping=1e-5, pcg_tolerance=1e-5, pcg_iterations=150,
        fit_windows=64, diagnostic_windows=16, sequence_length=32768,
        fit_indices=list(range(64)), diagnostic_indices=list(range(64,80)),
        query_position_manifest_sha256=sha256(args.positions), query_capture_manifest_sha256=sha256(args.queries/'manifest.json'),
        query_positions={l:r['selected_positions'] for l,r in selected['layers'].items()},
        query_count=32, page_size=32, excluded_prefix_pages=1, physical_token_budget=2048,
        selection='fixed R16 and final BCD endpoint; diagnostics do not select factors',
        captures=captures, code_sha256={n:sha256(ROOT/n) for n in (
            'evaluation/fit_c1_v96_r16_s50p150.py', 'evaluation/fit_c1_kl_base_residual.py',
            'evaluation/fit_qwen3_8b_q8_fisher_residual.py',
            'evaluation/eval_qwen3_8b_v80_conditional_residual_router.py',
            'basisserve/core/c1_v_conditional_k_router.py')})
    # Shared numerical fitter consumes only per-layer artifact paths, not allocation metadata.
    return dict(layers=layers), spec


@torch.inference_mode()
def fit_layer(args, layer, manifest, spec, cos, sin):
    device = torch.device('cuda:0')
    record = manifest['layers'][layer]
    encoder = load_file(str(args.allocation/record['file']))['value_coordinate_encoders']
    rank = spec['payload_ranks'][layer]
    assert encoder.shape == (8,128,rank) and torch.isfinite(encoder).all()
    root, capture = _discover_capture(args.calibration_root, split='fit', layer=layer)
    _, rows = _load_direct(root, capture, layer)
    assert rows.shape == (64,32768,8,256)
    bases = _fit_base_maps(rows,value_encoder=encoder,base_ranks=(16,),cos=cos,sin=sin,device=device)
    del rows
    statistics, reconstruction = {}, {}
    for split, count in [('fit',64),('validation',16)]:
        root, capture = _discover_capture(args.calibration_root, split=split, layer=layer)
        _, rows = _load_direct(root,capture,layer)
        queries, positions = manifest_queries(args.positions,args.queries,split,layer)
        assert queries.shape == (count,32,32,128) and rows.shape == (count,32768,8,256)
        assert positions.tolist() == spec['query_positions'][str(layer)]
        statistics[split], reconstruction[split] = build_multi_query_statistics(queries,rows,
            query_positions=positions,value_encoder=encoder,base_maps=bases,cos=cos,sin=sin,
            page_size=32,excluded_prefix_pages=1,device=device)
        del rows, queries
    residual, diagnostics = _fit_residual_grid(statistics['fit'],statistics['validation'],
        residual_ranks=(16,),sweeps=50,relative_damping=1e-5,iterative_tolerance=1e-5,
        iterative_max_iterations=150,device=device)
    tensors = {
        'base_left_b16':torch.stack([m.left for m in bases[16]]).float(),
        'base_right_b16':torch.stack([m.right for m in bases[16]]).float(),
        'base_bias_b16':torch.stack([m.bias for m in bases[16]]).float(),
        'residual_encoder_b16_r16':residual[(16,16)][0],
        'residual_query_b16_r16':residual[(16,16)][1],
    }
    shapes = [(8,rank,16),(8,16,128),(8,128),(8,128,16),(32,128,16)]
    for t,shape in zip(tensors.values(),shapes,strict=True):
        assert tuple(t.shape)==shape and t.dtype==torch.float32 and torch.isfinite(t).all()
    return tensors,dict(residual=diagnostics,reconstruction=reconstruction)



@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--c1-checkpoint', dest='allocation', type=Path,
        default=ROOT/'results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6')
    for name, path in {
        'calibration-root':'results/calibration', 'reference-bank':'results/checkpoints/mse_base_qgram32_r8',
        'positions':'results/evaluation/qgram32/positions.json', 'queries':'results/calibration/qgram32',
        'output-dir':'results/checkpoints/c1_v96_b16r16_s50p150',
    }.items():
        p.add_argument('--'+name, type=Path, default=ROOT/path)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=4)
    p.add_argument('--layers', help='Full-protocol layer smoke; outputs reused by formal fit')
    args = p.parse_args()
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0)
    manifest, spec = protocol(args)
    cos, sin = _rotary_embeddings(args.model, sequence=32768, device=torch.device('cuda:0'))
    layers = list(map(int, args.layers.split(','))) if args.layers else list(range(args.shard_index,36,args.num_shards))
    assert len(set(layers)) == len(layers) and all(0 <= l < 36 for l in layers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        path = args.output_dir/f'layer_{layer:03d}.safetensors'
        record = path.with_suffix('.json')
        if record.exists():
            saved = json.loads(record.read_text())
            assert saved['status'] == 'complete' and saved['layer'] == layer
            assert saved['protocol'] == spec and saved['sha256'] == sha256(path)
            continue
        started = time.monotonic()
        print(f'layer={layer} payload_rank=96 start', flush=True)
        tensors, diagnostic = fit_layer(args, layer, manifest, spec, cos, sin)
        save_file({k:v.contiguous() for k,v in tensors.items()}, str(path))
        write_json(record, dict(status='complete', layer=layer, protocol=spec, sha256=sha256(path),
            diagnostic=diagnostic, wall_seconds=time.monotonic()-started,
            command=shlex.join(sys.argv), python=sys.executable, gpu=torch.cuda.get_device_name(0)))
        print(f'layer={layer} complete seconds={time.monotonic()-started:.1f}', flush=True)
    name = 'smoke_'+args.layers.replace(',','_') if args.layers else f'shard_{args.shard_index}'
    write_json(args.output_dir/(name+'.json'), dict(status='complete', layers=layers, protocol=spec))


if __name__ == '__main__':
    main()
