"""Assemble the 50/50 C4 + synthetic-retrieval calibration window bank from two existing banks.

fit windows 0..F/2-1 come from the C4 bank's first fit windows, fit windows F/2..F-1 from the synthetic
bank, and the validation windows are the C4 bank's validation windows unchanged, so the mixed condition
differs from the pure-C4 condition only in the second half of its fit set.
"""
import argparse
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.v96kl_common import read_json, write_json, save_tensors, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--c4', type=Path, required=True, help='C4 bank directory (windows.safetensors + manifest.json)')
    p.add_argument('--synthetic', type=Path, required=True, help='synthetic bank directory')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    c4m, sm = read_json(args.c4 / 'manifest.json'), read_json(args.synthetic / 'manifest.json')
    assert c4m['status'] == sm['status'] == 'complete'
    assert c4m['model_config_sha256'] == sm['model_config_sha256'] and c4m['tokenizer_sha256'] == sm['tokenizer_sha256']
    assert c4m['sha256'] == sha256(args.c4 / 'windows.safetensors') and sm['sha256'] == sha256(args.synthetic / 'windows.safetensors')
    c4 = load_file(str(args.c4 / 'windows.safetensors'))['input_ids']
    synthetic = load_file(str(args.synthetic / 'windows.safetensors'))['input_ids']
    fit_ids, validation_ids = list(c4m['fit_ids']), list(c4m['validation_ids'])
    half = len(fit_ids) // 2
    assert 2 * half == len(fit_ids) and synthetic.shape[0] >= half and synthetic.shape[1] == c4.shape[1]
    packed = torch.cat((c4[fit_ids[:half]], synthetic[:half].to(c4.dtype), c4[validation_ids]), 0)
    assert packed.shape == (len(fit_ids) + len(validation_ids), c4.shape[1])
    args.output.mkdir(parents=True, exist_ok=True)
    save_tensors(args.output / 'windows.safetensors', dict(input_ids=packed.contiguous()))
    write_json(args.output / 'manifest.json', dict(
        status='complete', sha256=sha256(args.output / 'windows.safetensors'), shape=list(packed.shape),
        condition='c4_retrieval_50_50', fit_ids=fit_ids, validation_ids=validation_ids,
        composition=f'fit windows 0-{half - 1}: C4 bank windows 0-{half - 1}; fit windows {half}-{len(fit_ids) - 1}: '
                    f'synthetic retrieval windows 0-{half - 1}; validation windows {validation_ids[0]}-{validation_ids[-1]}: '
                    f'C4 bank windows {validation_ids[0]}-{validation_ids[-1]} (identical to the pure-C4 condition)',
        c4_manifest_sha256=sha256(args.c4 / 'manifest.json'), c4_windows_sha256=c4m['sha256'], c4_seed=c4m['seed'],
        synthetic_manifest_sha256=sha256(args.synthetic / 'manifest.json'), synthetic_windows_sha256=sm['sha256'],
        synthetic_seed=sm['seed'], synthetic_haystack_seed=sm['haystack_seed'], synthetic_method=sm['method'],
        model=c4m['model'], model_config_sha256=c4m['model_config_sha256'], tokenizer_sha256=c4m['tokenizer_sha256'],
        source_sha256=sha256(Path(__file__))))
    print('mixed calibration windows complete', tuple(packed.shape), args.output, flush=True)


if __name__ == '__main__':
    main()
