"""Fit Base16/Residual16 from dense window shards without loading the teacher."""
import argparse
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, sha256
from evaluation.fit_k_routing import fit_layer
from basisserve.core.query_position_sampling import candidate_positions
from evaluation.k_routing_config import routing_config, routing_position_embeddings
from evaluation.audit_k_routing_fit import attention_payloads
from evaluation.k_routing_capture_windows import read_capture_windows as read_capture


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=4)
    parser.add_argument('--layers', type=str)
    parser.add_argument('--reference-bank', type=Path)
    args = parser.parse_args()
    configure()
    identity = read_json(args.identity)
    assert identity['status'] == 'complete' and identity['mean_rank'] == 96
    checkpoint = Path(identity['checkpoint'])
    manifest = read_json(checkpoint / 'manifest.json')
    assert sha256(checkpoint / 'manifest.json') == identity['manifest_sha256']
    complete = read_json(args.captures / 'complete_0.json')
    assert complete['status'] == 'complete'
    capture_protocol = complete['protocol']
    assert capture_protocol['identity_sha256'] == sha256(args.identity)
    assert capture_protocol['windows_sha256'] == sha256(args.windows)
    for shard in range(capture_protocol['num_shards']):
        record = read_json(args.captures / f'complete_{shard}.json')
        assert record['status'] == 'complete' and record['protocol'] == capture_protocol
        assert record['layers'] == complete['layers']
    fit_ids, diagnostic_ids = capture_protocol['fit_ids'], capture_protocol['diagnostic_ids']
    args.fit_count, args.diagnostic_count = len(fit_ids), len(diagnostic_ids)
    assert fit_ids == list(range(args.fit_count))
    assert diagnostic_ids == list(range(64, 64 + args.diagnostic_count))
    args.sequence_length = capture_protocol['sequence_length']
    runtime = capture_protocol['runtime_config']
    config = routing_config(identity, rope=capture_protocol['rope'],
        sequence_length=args.sequence_length)
    assert runtime['model_type'] == config.model_type
    assert runtime['max_position_embeddings'] == config.max_position_embeddings
    if config.model_type != 'nemotron_h':
        assert runtime['rope_parameters'] == config.rope_parameters
    payloads = attention_payloads(identity, manifest, config)
    cos, sin = routing_position_embeddings(config, args.sequence_length, 'cuda')
    spec = dict(identity_sha256=sha256(args.identity), windows_sha256=sha256(args.windows),
        model_config_sha256=identity['model_config_sha256'], v96_manifest_sha256=identity['manifest_sha256'],
        layer_ranks=identity['layer_ranks'], fit_ids=fit_ids, diagnostic_ids=diagnostic_ids,
        sequence_length=args.sequence_length, runtime_config=runtime, rope=capture_protocol['rope'],
        capture_protocol=capture_protocol,
        fit_queries=64, diagnostic_queries=32, selection='fit-only stratified Query-Gram',
        base_rank=16, base_objective='affine pre-RoPE K MSE closed-form RRR from frozen V latent',
        residual_rank=16, residual_objective='causal non-sink Page-Fisher relative to frozen Base16',
        page_size=32, excluded_prefix_pages=1, bcd_sweeps=40, pcg_damping=1e-5,
        pcg_tolerance=1e-5, pcg_iterations=100, endpoint='fixed final sweep',
        intended_routing_budget=2048, factor_dtype='float32', teacher='native dense BF16 SDPA',
        capture_loading='one V/K window at a time; candidate Q materialized',
        source_sha256={name:sha256(ROOT/name) for name in ('evaluation/fit_k_routing_captures.py',
            'evaluation/k_routing_capture_windows.py',
            'evaluation/fit_k_routing.py', 'evaluation/k_routing_config.py',
            'evaluation/audit_k_routing_fit.py', 'basisserve/core/c1_v_conditional_k_router.py',
            'basisserve/core/query_position_sampling.py', 'evaluation/fit_qwen3_8b_q8_fisher_residual.py',
            'evaluation/eval_qwen3_8b_v80_conditional_residual_router.py')})
    targets = [int(x) for x in args.layers.split(',')] if args.layers else identity['attention_layers'][args.shard_index::args.num_shards]
    assert targets and set(targets) <= set(complete['layers'])
    write_json(args.output / 'manifests' / f'protocol_shard_{args.shard_index}.json', spec)
    for layer in targets:
        result = args.output / 'ours_b16r16' / f'layer_{layer:03d}.json'
        if result.exists():
            record = read_json(result)
            assert record['status'] == 'complete' and record['protocol'] == spec
            assert record['sha256'] == sha256(result.with_suffix('.safetensors'))
            continue
        tensors, shards = read_capture(args.captures, layer, fit_ids + diagnostic_ids, capture_protocol, identity)
        write_json(args.output / 'calibration' / f'layer_{layer:03d}.json', dict(status='complete',
            protocol=spec, layer=layer, storage='window_shards', shards=shards,
            row_layout='raw V concatenated with exact post-RoPE K',
            pre_rope_keys='after architecture-specific normalization; before actual model RoPE',
            candidate_positions=candidate_positions(args.sequence_length), window_ids=fit_ids + diagnostic_ids))
        payload = payloads[layer]
        assert sha256(checkpoint / payload['file']) == payload['sha256']
        encoder = load_file(str(checkpoint / payload['file']))['value_coordinate_encoders']
        assert encoder.shape == (identity['hkv'], identity['head_dim'], payload['ranks'][0])
        fit_layer(args, layer, tensors['rows'], tensors['pre_rope_keys'], tensors['candidate_queries'],
            encoder, cos, sin, spec)
        del tensors, encoder
    if args.reference_bank is not None:
        for layer in targets:
            name = f'layer_{layer:03d}.safetensors'
            actual = load_file(str(args.output / 'ours_b16r16' / name))
            reference = load_file(str(args.reference_bank / name))
            assert set(actual) == set(reference)
            for key in actual:
                torch.testing.assert_close(actual[key], reference[key], atol=0, rtol=0)
            print('BITWISE FIT REFERENCE MATCH', layer, flush=True)
    write_json(args.output / 'manifests' / f'complete_{args.shard_index}.json',
        dict(status='complete', protocol=spec, layers=targets))


if __name__ == '__main__':
    main()
