"""Audit existing fit-only candidate captures without model execution."""

import argparse
import json
from pathlib import Path
import shlex
import sys

import scipy
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basisserve.core.query_cpqr_audit import audit_queries
from evaluation.v96kl_common import sha256


def load_candidates(root, layer):
    """Load only one layer from each audited fit-window artifact."""
    capture = json.loads((root / 'manifest.json').read_text())
    spec = capture['protocol']
    assert capture['status'] == 'complete'
    assert spec['format'] == 'basisserve.candidates_queries.v1'
    docs = spec['fit_document_ids']
    assert docs == list(range(64)) and spec['documents'] == docs
    assert set(capture['artifacts']) == set(map(str, docs))
    rows = []
    for doc in docs:
        item = capture['artifacts'][str(doc)]
        path = root / item['file']
        assert sha256(path) == item['sha256']
        with safe_open(path, framework='pt', device='cpu') as tensors:
            query = tensors.get_slice('queries')[spec['layers'].index(layer)]
        assert query.dtype == torch.bfloat16 and torch.isfinite(query).all()
        assert query.shape == (len(spec['positions_by_layer'][str(layer)]), 32, 128)
        rows.append(query)
    return torch.stack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-capture', type=Path, required=True)
    parser.add_argument('--layers', default='0,15,35')
    parser.add_argument('--queries-per-bin', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    capture_path = args.candidate_capture / 'manifest.json'
    capture = json.loads(capture_path.read_text())
    spec = capture['protocol']
    assert capture['status'] == 'complete'
    assert spec['fit_document_ids'] == list(range(64))
    assert spec['context_length'] == 32768
    layers = [int(s) for s in args.layers.split(',')]
    assert len(set(layers)) == len(layers) and all(l in spec['layers'] for l in layers)
    assert not args.output_dir.exists(), 'Use a new output directory; existing artifacts are immutable'
    args.output_dir.mkdir(parents=True)
    for layer in layers:
        print(f'layer={layer} loading fit candidates', flush=True)
        queries = load_candidates(args.candidate_capture, layer)
        result = audit_queries(queries, spec['positions_by_layer'][str(layer)],
                               queries_per_bin=args.queries_per_bin)
        result.update(layer=layer, command=shlex.join(sys.argv), python=sys.executable,
                      scipy_version=scipy.__version__, torch_version=torch.__version__,
                      candidate_manifest_sha256=sha256(capture_path),
                      source_sha256={name: sha256(ROOT / name) for name in (
                          'basisserve/core/query_cpqr_audit.py',
                          'basisserve/core/query_position_sampling.py',
                          'evaluation/audit_query_cpqr.py')})
        with (args.output_dir / f'layer_{layer:03d}.json').open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(f'layer={layer} complete cpqr_matches_production='
              f'{[b["production_matches_cpqr"] for b in result["bins"]]}', flush=True)


if __name__ == '__main__':
    main()
