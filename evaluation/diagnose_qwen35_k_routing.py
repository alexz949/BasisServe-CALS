"""Held-out K rel-MSE and attention-mass recall using the deployed page rule."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from evaluation.routing_diagnostics import window_metrics
from evaluation.qwen35_hybrid_common import atomic_save, load_bank, sha256


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--v-bank', type=Path, required=True)
    p.add_argument('--routers', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--rank', type=int, choices=(16, 32), required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    bank = load_bank(args.v_bank)
    path = args.routers/f'b{args.rank}r{args.rank}'/f'l{args.layer:02d}.pt'
    router = torch.load(path, map_location='cpu', weights_only=True)
    assert router['status'] == 'complete' and router['layer'] == args.layer
    assert router['protocol']['v_bank_sha256'] == sha256(args.v_bank)
    factors = {key: value.cuda() for key, value in router['tensors'].items()}
    positions = router['selections']['diagnostic']['selected_positions']
    assert len(positions) == 32 and min(positions) >= 32
    v_rank = bank['schedule'][args.layer]
    encoder = (torch.eye(256).repeat(4, 1, 1) if v_rank == 256 else bank['layers'][args.layer]['E_V']).cuda().float()
    records = []
    for index in range(64, 80):
        root = args.capture/f'w{index:03d}'
        manifest = json.loads((root/'manifest.json').read_text())
        assert manifest['status'] == 'complete' and manifest['index'] == index and manifest['split'] == 'diagnostic'
        assert manifest['windows_sha256'] == bank['windows_sha256'] and manifest['wo_compression'] is False
        tensor_path = root/f'l{args.layer:02d}.safetensors'
        assert sha256(tensor_path) == manifest['files'][tensor_path.name]
        assert router['protocol']['capture_hashes'][str(tensor_path)] == manifest['files'][tensor_path.name]
        assert sha256(root/'rope.safetensors') == manifest['files']['rope.safetensors']
        payload, rope = load_file(str(tensor_path)), load_file(str(root/'rope.safetensors'))
        raw = payload['rows'].cuda()
        value = torch.einsum('btgd,gdr->bgtr', raw[..., :256].float(), encoder)
        key = raw[..., 256:].transpose(1, 2)
        selection = [manifest['candidate_positions'].index(position) for position in positions]
        queries = payload['candidate_queries'][:, selection].cuda().float()
        metrics = window_metrics(value, key, queries, positions, rope['cos'].cuda(), rope['sin'].cuda(),
            rank=args.rank, factors=factors)
        records.append(dict(window=index, **metrics))
        print(json.dumps(dict(layer=args.layer, rank=args.rank, **records[-1])), flush=True)
    report = dict(status='complete', layer=args.layer, rank=args.rank, v_rank=v_rank,
        v_bank_sha256=sha256(args.v_bank), router_sha256=sha256(path), windows=records,
        relative_mse=sum(row['squared_error'] for row in records)/sum(row['key_energy'] for row in records),
        attention_mass=sum(row['attention_mass'] for row in records)/16,
        non_sink_attention_mass=sum(row['non_sink_attention_mass'] for row in records)/16,
        exact_attention_mass=sum(row['exact_attention_mass'] for row in records)/16,
        exact_non_sink_attention_mass=sum(row['exact_non_sink_attention_mass'] for row in records)/16,
        positions=positions, budget=2048, page_size=32, pinned_prefix_pages=0, recent_tokens=64,
        source_sha256=sha256(Path(__file__)),
        routing_code_sha256={name: sha256(Path(name)) for name in (
            'evaluation/routing_diagnostics.py',
            'basisserve/core/qwen35_k_routing_runtime.py',
            'basisserve/core/c1_conditional_page_attention.py',
            'basisserve/core/c1_v_k_index.py')},
        page_selection='historical normalized page mass, then GQA max; recent64 inside hard budget; no pinned sink',
        feature_protocol='FP32 maps on BF16 native captures; query positions selected on fit windows only')
    atomic_save(args.output/f'b{args.rank}r{args.rank}'/f'l{args.layer:02d}.json', report)


if __name__ == '__main__':
    main()
