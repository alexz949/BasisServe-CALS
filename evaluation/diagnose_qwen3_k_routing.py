"""Recompute matching diagnostic metrics for the frozen Qwen3-8B Q64 router."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _load_direct, _rotary_embeddings
from evaluation.qwen35_hybrid_common import atomic_save, sha256
from evaluation.routing_diagnostics import window_metrics
from evaluation.select_query_positions import manifest_queries


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--routers', type=Path, default=Path('results/checkpoints/c1_v96_b16r16_q64_s40p100'))
    p.add_argument('--positions', type=Path, default=Path('results/evaluation/qgram32/positions.json'))
    p.add_argument('--queries', type=Path, default=Path('results/calibration/qgram32'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=4)
    p.add_argument('--layers', default=','.join(map(str, range(36))))
    args = p.parse_args()
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    cos, sin = _rotary_embeddings(args.model, sequence=32768, device=torch.device('cuda'))
    for layer in list(map(int, args.layers.split(',')))[args.shard_index::args.num_shards]:
        path = args.routers/f'layer_{layer:03d}.json'
        saved = json.loads(path.read_text())
        spec = saved['protocol']
        assert saved['status'] == 'complete' and saved['layer'] == layer
        assert spec['base_rank'] == spec['residual_rank'] == 16
        assert spec['bcd_sweeps'] == 40 and spec['pcg_iterations'] == 100
        assert spec['fit_windows'] == 64 and spec['diagnostic_indices'] == list(range(64, 80))
        assert spec['query_count'] == 64 and spec['diagnostic_query_count'] == 32
        assert spec['physical_token_budget'] == 2048 and spec['page_size'] == 32
        assert sha256(args.model/'config.json') == spec['model_config_sha256']
        assert sha256(args.positions) == spec['diagnostic_position_manifest_sha256']
        assert sha256(args.queries/'manifest.json') == spec['diagnostic_query_capture_manifest_sha256']
        tensor_path = path.with_suffix('.safetensors')
        assert sha256(tensor_path) == saved['sha256']
        factors = load_file(str(tensor_path), device='cuda')
        checkpoint = Path(spec['checkpoint'])
        assert sha256(checkpoint/'results.json') == spec['c1_manifest_sha256']
        source = json.loads((checkpoint/'results.json').read_text())
        artifact = source['artifacts'][str(layer)]
        v_path = checkpoint/artifact['file']
        assert sha256(v_path) == spec['c1_layer_sha256'][str(layer)] == artifact['sha256']
        encoder = load_file(str(v_path))['value_coordinate_encoders'].cuda().float()
        assert encoder.shape == (8, 128, 96)
        capture_spec = spec['captures'][str(layer)]['validation']
        capture_root = Path(capture_spec['root'])
        assert sha256(capture_root/'manifest.json') == capture_spec['manifest_sha256']
        capture = json.loads((capture_root/'manifest.json').read_text())
        row_record = capture['artifacts'][str(layer)]['routing_joint_rows']
        assert sha256(capture_root/row_record['file']) == row_record['sha256']
        _, rows = _load_direct(capture_root, capture, layer)
        queries, positions = manifest_queries(args.positions, args.queries, 'validation', layer)
        positions = positions.tolist()
        assert positions == spec['diagnostic_query_positions'][str(layer)]
        assert rows.shape == (16, 32768, 8, 256) and queries.shape == (16, 32, 32, 128)
        records = []
        for window in range(16):
            raw = rows[window:window+1].cuda()
            value = torch.einsum('btgd,gdr->bgtr', raw[..., :128].float(), encoder)
            key = raw[..., 128:].transpose(1, 2)
            metrics = window_metrics(value, key, queries[window:window+1].cuda(), positions,
                cos, sin, rank=16, factors=factors)
            records.append(dict(window=64+window, **metrics))
            print(json.dumps(dict(layer=layer, **records[-1])), flush=True)
        report = dict(status='complete', layer=layer, rank=16, v_rank=96, windows=records,
            relative_mse=sum(r['squared_error'] for r in records)/sum(r['key_energy'] for r in records),
            **{name: sum(r[name] for r in records)/16 for name in ('attention_mass',
                'non_sink_attention_mass', 'exact_attention_mass', 'exact_non_sink_attention_mass')},
            positions=positions, budget=2048, page_size=32, pinned_prefix_pages=1,
            source_router_sha256=sha256(path), source_v_sha256=sha256(v_path),
            source_capture_sha256=capture_spec['manifest_sha256'],
            source_protocol=spec,
            code_sha256={name: sha256(Path(name)) for name in (
                __file__, 'evaluation/routing_diagnostics.py', 'basisserve/core/qwen35_k_routing_runtime.py',
                'basisserve/core/c1_conditional_page_attention.py', 'basisserve/core/c1_v_k_index.py')},
            feature_protocol='FP32 maps on BF16 native captures; query positions selected on fit windows only',
            comparison_note='Qwen3 V: uniform V96, 32 fit + 4 held-out windows, ALS6; Qwen3.5 V uses different fit/allocation protocol')
        atomic_save(args.output/'b16r16'/f'l{layer:02d}.json', report)


if __name__ == '__main__':
    main()
