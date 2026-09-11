"""Fixed-Q32 strong-RRQR versus the completed query CPQR audit."""

import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basisserve.core.query_position_sampling import select_stratified_query_positions
from basisserve.core.query_srrqr_audit import audit_srrqr_bin
from evaluation.audit_query_cpqr import load_candidates
from evaluation.v96kl_common import sha256, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-capture', type=Path, required=True)
    parser.add_argument('--cpqr-reference', type=Path, required=True)
    parser.add_argument('--layers', default='0,15,35')
    parser.add_argument('--bounds', default='2,1.01')
    parser.add_argument('--max-swaps', type=int, default=512)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    layers = list(map(int, args.layers.split(',')))
    bounds = list(map(float, args.bounds.split(',')))
    assert layers == sorted(set(layers)) and len(bounds) == len(set(bounds))
    assert args.max_swaps > 0
    capture = read_json(args.candidate_capture / 'manifest.json')
    capture_hash = sha256(args.candidate_capture / 'manifest.json')
    assert capture['status'] == 'complete'
    assert capture['protocol']['context_length'] == 32768
    assert all(layer in capture['protocol']['layers'] for layer in layers)
    assert not args.output_dir.exists(), 'Use a fresh output directory'
    args.output_dir.mkdir(parents=True)
    for layer in layers:
        print(f'layer={layer} loading fit queries', flush=True)
        reference_path = args.cpqr_reference / f'layer_{layer:03d}.json'
        reference = read_json(reference_path)
        assert reference['candidate_manifest_sha256'] == capture_hash
        assert reference['context_length'] == 32768 and reference['query_shape'] == [64,512,32,128]
        assert reference['queries_per_bin'] == 8 and reference['num_bins'] == 4
        queries = load_candidates(args.candidate_capture, layer)
        positions = capture['protocol']['positions_by_layer'][str(layer)]
        selection, _, grams = select_stratified_query_positions(
            queries, positions, context_length=32768, num_bins=4,
            queries_per_bin=8, whitening_eps=reference['whitening_eps'])
        bins = []
        for b, gram in enumerate(grams):
            expected = reference['bins'][b]['cpqr_positions']
            assert selection['bins'][b]['pivot_order'] == expected
            local = [p for p in positions if p * 4 // 32768 == b]
            result = audit_srrqr_bin(gram.numpy(), local, bounds=bounds, max_swaps=args.max_swaps)
            assert result['cpqr_positions'] == expected, 'Full-Gram root changed CPQR initialization'
            for metric in ('condition_number', 'residual_energy_fraction', 'maximum_residual_energy'):
                assert np.isclose(result['cpqr_geometry'][metric],
                                  reference['bins'][b]['cpqr_geometry'][metric], rtol=1e-9, atol=1e-10)
            result['bin'] = b
            bins.append(result)
            print(f'layer={layer} bin={b} '
                  f'swaps={[r["diagnostics"]["swaps"] for r in result["results"]]} '
                  f'rho_initial={result["results"][0]["diagnostics"]["initial_max_rho"]:.8f}', flush=True)
        result = dict(layer=layer, bins=bins, bounds=bounds, max_swaps=args.max_swaps,
                      query_shape=list(queries.shape), queries_per_bin=8, num_bins=4,
                      whitening_eps=reference['whitening_eps'],
                      candidate_manifest_sha256=capture_hash, cpqr_reference_sha256=sha256(reference_path),
                      command=shlex.join(sys.argv), python=sys.executable,
                      scipy_version=scipy.__version__, torch_version=torch.__version__,
                      source_sha256={name: sha256(ROOT / name) for name in (
                          'basisserve/core/strong_rrqr.py', 'basisserve/core/query_srrqr_audit.py',
                          'basisserve/core/query_position_sampling.py', 'evaluation/audit_query_cpqr.py',
                          'evaluation/audit_query_srrqr.py')},
                      scope='Full-spectrum Gram isometry; selector-only, no residual fit or routing recall')
        with (args.output_dir / f'layer_{layer:03d}.json').open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(f'layer={layer} complete', flush=True)


if __name__ == '__main__':
    main()
