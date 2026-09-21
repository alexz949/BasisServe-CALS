#!/usr/bin/env python3
"""Pack existing Qwen3-32B 32K windows into one fit and one held-out 128K smoke window."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from evaluation.v96kl_common import read_json, save_tensors, sha256, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_manifest = read_json(args.source.with_name('manifest.json'))
    assert source_manifest['status'] == 'complete' and source_manifest['sha256'] == sha256(args.source)
    source = load_file(str(args.source))['input_ids']
    assert source.shape[1] == 32768
    fit_ids = list(map(int, source_manifest['fit_ids'][:4]))
    heldout_ids = list(map(int, source_manifest['validation_ids'][:4]))
    assert len(fit_ids) == len(heldout_ids) == 4
    selected = torch.stack((source[fit_ids].reshape(-1), source[heldout_ids].reshape(-1)))
    assert selected.shape == (2, 131072)
    save_tensors(args.output, {'input_ids': selected.contiguous()})
    write_json(args.output.with_name('manifest.json'), dict(
        format='basisserve.calibration.128k_smoke_repack.v1', status='complete',
        sha256=sha256(args.output), model_config_sha256=source_manifest['model_config_sha256'],
        fit_ids=[0], validation_ids=[1], shape=list(selected.shape), test_only=True,
        source_windows_sha256=sha256(args.source),
        source_manifest_sha256=sha256(args.source.with_name('manifest.json')),
        source_ids=dict(fit=fit_ids, heldout=heldout_ids)))
    print('Prepared Qwen3-32B 128K smoke windows', list(selected.shape), flush=True)


if __name__ == '__main__':
    main()
