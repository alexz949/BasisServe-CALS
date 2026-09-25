"""Build the 48-window Qwen3.5 recalibration bank: fit windows 0-31 = the 16 C4 + 16 synthetic-retrieval mix, validation
windows 32-47 = C4 bank windows 16-31 (disjoint from the fit set). The manifest carries both the router-fitter keys
(fit_ids / validation_ids / sha256 / model_config_sha256 / tokenizer_sha256) and the GDN moments collector keys
(format basisserve.calibration.c4_document_windows.v1, artifact, records[].sample_index)."""
import argparse, json
from pathlib import Path
from safetensors.torch import load_file, save_file
import torch
from evaluation.v96kl_common import sha256, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--mixed', type=Path, required=True, help='32-window c4_retrieval_50_50 bank')
    p.add_argument('--c4', type=Path, required=True, help='pure C4 32-window bank (windows 16-31 become validation)')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    mm, cm = json.loads((args.mixed / 'manifest.json').read_text()), json.loads((args.c4 / 'manifest.json').read_text())
    assert mm['status'] == cm['status'] == 'complete' and mm['model_config_sha256'] == cm['model_config_sha256'] and mm['tokenizer_sha256'] == cm['tokenizer_sha256']
    assert mm['sha256'] == sha256(args.mixed / 'windows.safetensors') and cm['sha256'] == sha256(args.c4 / 'windows.safetensors')
    mixed, c4 = load_file(str(args.mixed / 'windows.safetensors'))['input_ids'], load_file(str(args.c4 / 'windows.safetensors'))['input_ids']
    assert mixed.shape == (32, 131072) and c4.shape == (32, 131072) and list(mm['fit_ids']) == list(range(32)) and not mm['validation_ids']
    packed = torch.cat((mixed, c4[16:32]), 0).contiguous()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'windows.safetensors'
    save_file({'input_ids': packed}, str(path))
    digest = sha256(path)
    records = [dict(sample_index=i, source='mixed' if i < 32 else 'c4_validation',
                    origin=('c4_fit' if i < 16 else 'synthetic_retrieval') if i < 32 else 'c4_bank_window_%d' % (i - 16)) for i in range(48)]
    write_json(args.output / 'manifest.json', dict(
        format='basisserve.calibration.c4_document_windows.v1', status='complete', sha256=digest, shape=[48, 131072],
        artifact=dict(file='windows.safetensors', sha256=digest, tensor='input_ids', shape=[48, 131072]),
        model=mm['model'], model_config_sha256=mm['model_config_sha256'], tokenizer_sha256=mm['tokenizer_sha256'],
        condition='c4_retrieval_50_50_plus_c4_validation', fit_ids=list(range(32)), validation_ids=list(range(32, 48)),
        composition='fit 0-15: C4 bank windows 0-15; fit 16-31: synthetic retrieval windows 0-15; validation 32-47: C4 bank windows 16-31',
        mixed_manifest_sha256=sha256(args.mixed / 'manifest.json'), mixed_windows_sha256=mm['sha256'],
        c4_manifest_sha256=sha256(args.c4 / 'manifest.json'), c4_windows_sha256=cm['sha256'],
        records=records, source_sha256=sha256(Path(__file__))))
    print(json.dumps(dict(output=str(args.output), sha256=digest, shape=[48, 131072])), flush=True)


if __name__ == '__main__':
    main()
