"""Package a merged Llama-3.1-8B C1 ALS factor bank for eval_llama_cal128's loader.

The evaluator's install() (evaluation/eval_k_routing_ruler.py) needs only:
  - checkpoint/manifest.json: {status: complete, layers: [{layer, file, ranks, sha256}]}
  - each layer file: value_coordinate_encoders (hkv,dim,rank), head_output_decoders
    (hq,rank,hidden), source_ranks (hkv,) int64 equal to that layer's `ranks`.
It does not check any "format" string or pinned base-model revision. The generic
build_qwen3_8b_c1_uniform_checkpoint.py routes through evaluation.build_llama31_8b_palu_m_checkpoint's
MODEL_PROFILES, whose "llama31_8b" entry pins the base (non-Instruct) model at a
fixed revision and crashes on any other snapshot, including our Instruct download.
This script writes the same contract directly, with no such check.
"""
import argparse
import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file



def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--factor-bank-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--num-kv-heads', type=int, default=8)
    p.add_argument('--rank', type=int, default=96)
    args = p.parse_args()

    import json
    results = json.loads((args.factor_bank_dir / 'results.json').read_text())
    assert results['status'] == 'complete'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranks = [args.rank] * args.num_kv_heads
    layers = []
    for record in sorted(results['records'], key=lambda r: int(r['layer'])):
        layer_index = int(record['layer'])
        src = args.factor_bank_dir / record['artifact']['file']
        assert sha256(src) == record['artifact']['sha256']
        tensors = load_file(str(src))
        assert tensors['value_coordinate_encoders'].shape == (args.num_kv_heads, 128, args.rank)
        assert tensors['head_output_decoders'].shape[1] == args.rank
        tensors['source_ranks'] = torch.tensor(ranks, dtype=torch.int64)
        dest = args.output_dir / f'layer_{layer_index:03d}.safetensors'
        save_file(tensors, str(dest))
        layers.append(dict(layer=layer_index, file=dest.name, ranks=ranks, sha256=sha256(dest)))
    assert [entry['layer'] for entry in layers] == list(range(len(layers)))
    manifest_path = args.output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(dict(
        status='complete',
        format='basisserve.llama31_8b_instruct.c1_v96_manual_package.v1',
        source_results_sha256=sha256(args.factor_bank_dir / 'results.json'),
        layers=layers,
    ), indent=1))
    print('packaged checkpoint', args.output_dir, len(layers), 'layers', flush=True)


if __name__ == '__main__':
    main()
