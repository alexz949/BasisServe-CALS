"""Derive a matched terminal-Q8 pair from immutable existing Q captures; no forward."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basisserve.core.query_position_sampling import uniform_query_positions
from evaluation.select_query_positions import load_position_manifest
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json


def terminal_positions(parent, policy):
    assert policy in ('uniform', 'qgram') and parent['context_length'] == 32768
    assert parent['method'] == 'stratified_query_gram_pivot'
    positions = {}
    for layer, data in parent['layers'].items():
        selected = (uniform_query_positions(32768, 8, .25) if policy == 'uniform'
                    else [p for p in data['selected_positions'] if p >= 24576])
        assert len(selected) == 8 and selected == sorted(set(selected))
        assert all(24576 <= p < 32768 for p in selected)
        positions[layer] = selected
    return positions


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-positions', type=Path, default=ROOT/'results/evaluation/qgram32/positions.json')
    p.add_argument('--qgram-capture', type=Path, default=ROOT/'results/calibration/qgram32')
    p.add_argument('--uniform-source', type=Path, default=ROOT/'results/calibration/q128_terminal8k')
    p.add_argument('--windows', type=Path, default=ROOT/'results/calibration/qwen3_8b_c4_64f16h_s32768/windows.safetensors')
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    parent = load_position_manifest(args.parent_positions)
    assert parent['num_fit_windows'] == 64 and set(parent['layers']) == set(map(str,range(36)))
    tokens = load_file(str(args.windows))['input_ids']
    assert tokens.shape == (80,32768)
    hashes = {str(d):hashlib.sha256(tokens[d].contiguous().numpy().tobytes()).hexdigest() for d in range(80)}
    assert hashlib.sha256(tokens[:64].contiguous().numpy().tobytes()).hexdigest() == parent['fit_token_sha256']
    # Freeze both position manifests before reading any diagnostic Q values.
    selected = {}
    for policy in ('uniform','qgram'):
        directory = args.output_root/policy
        assert not directory.exists()
        record = copy.deepcopy(parent)
        positions = terminal_positions(parent,policy)
        record.update(method='terminal_'+policy+'_q8', num_bins=1, queries_per_bin=8,
                      layers={l:{'selected_positions':v} for l,v in positions.items()},
                      parent_position_manifest=str(args.parent_positions.resolve()),
                      parent_position_manifest_sha256=sha256(args.parent_positions),
                      query_span=[24576,32768],
                      whitening=parent['whitening'] if policy=='qgram' else None,
                      whitening_eps=parent['whitening_eps'] if policy=='qgram' else None,
                      derivation='reuse parent final-bin pivots; no whitening or selection recomputation' if policy=='qgram' else 'fixed terminal8K uniform stride1024',
                      whitening_scope='all-window fit candidates in parent; NOT terminal-only whitening' if policy=='qgram' else None,
                      code_version={'evaluation/prepare_terminal_q8.py':sha256(Path(__file__))})
        directory.mkdir(parents=True)
        write_json(directory/'positions.json',record)
        assert load_position_manifest(directory/'positions.json') == record
        selected[policy] = record
    qgram = json.loads((args.qgram_capture/'manifest.json').read_text())
    uniform = json.loads((args.uniform_source/'manifest.json').read_text())
    assert qgram['status'] == uniform['status'] == 'complete'
    assert qgram['protocol']['position_manifest_sha256'] == sha256(args.parent_positions)
    assert qgram['protocol']['document_token_sha256'] == hashes
    assert qgram['protocol']['fit_token_sha256'] == parent['fit_token_sha256']
    assert uniform['protocol']['windows_sha256'] == sha256(args.windows)
    assert uniform['protocol']['model_config_sha256'] == parent['model_config_sha256']
    for policy, source_root, source in [('uniform',args.uniform_source,uniform),('qgram',args.qgram_capture,qgram)]:
        output = args.output_root/policy/'queries'
        output.mkdir()
        positions = {l:r['selected_positions'] for l,r in selected[policy]['layers'].items()}
        protocol = copy.deepcopy(qgram['protocol'])
        protocol.update(positions_by_layer=positions,
                        position_manifest_sha256=sha256(args.output_root/policy/'positions.json'),
                        capture='subset of verified existing BF16 post-Qnorm/post-RoPE Q; no model forward',
                        source_manifest_sha256=sha256(source_root/'manifest.json'),
                        source_capture_root=str(source_root.resolve()),
                        source_sha256=sha256(Path(__file__)))
        artifacts = {}
        for d in range(80):
            artifact = source['artifacts'][str(d)]
            file = source_root/artifact['file']
            assert sha256(file) == artifact['sha256']
            data = load_file(str(file))['queries']
            assert data.dtype == torch.bfloat16 and torch.isfinite(data).all()
            layers = []
            for layer in range(36):
                grid = (source['protocol']['query_positions'] if policy=='uniform'
                        else source['protocol']['positions_by_layer'][str(layer)])
                indices = [grid.index(v) for v in positions[str(layer)]]
                layers.append(data[layer,indices])
            tensor = torch.stack(layers).contiguous()
            assert tensor.shape == (36,8,32,128)
            if policy == 'qgram':
                prior = load_file(str(args.output_root/'uniform/queries'/f'window_{d:03d}.safetensors'))['queries']
                for layer in range(36):
                    fixed = selected['uniform']['layers'][str(layer)]['selected_positions']
                    current = positions[str(layer)]
                    overlap = [v for v in current if v in fixed]
                    assert torch.equal(tensor[layer,[current.index(v) for v in overlap]],
                                       prior[layer,[fixed.index(v) for v in overlap]])
            path = output/f'window_{d:03d}.safetensors'
            save_file({'queries':tensor},str(path))
            artifacts[str(d)] = dict(file=path.name,sha256=sha256(path),source_file_sha256=artifact['sha256'])
        write_json(output/'manifest.json',dict(status='complete',protocol=protocol,artifacts=artifacts))
        print(f'{policy}: 36 layers x 80 windows x Q8; copied verified Q values, no forward',flush=True)


if __name__ == '__main__':
    main()
