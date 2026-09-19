"""Check every formal native covariance artifact before rank-bank fitting."""
import argparse
from pathlib import Path
import sys

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit', 'snapshots', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    configure()
    audit = read_json(args.audit)
    assert audit['status'] == 'complete'
    common, verified = None, {}
    for kind in ('full_attention', 'linear_attention'):
        directory = args.snapshots / kind
        manifest_path = directory / 'manifest.json'
        manifest = read_json(manifest_path)
        assert manifest['status'] == 'complete' and manifest['dense_teacher']
        assert manifest['audit_sha256'] == sha256(args.audit)
        layers = [row['layer'] for row in audit['layers'] if row['kind'] == kind]
        assert layers and manifest['layers'] == layers
        widths = {row['input_width'] for row in audit['layers'] if row['kind'] == kind}
        output_widths = {row['output_width'] for row in audit['layers'] if row['kind'] == kind}
        assert len(widths) == len(output_widths) == 1
        width, output_width = widths.pop(), output_widths.pop()
        assert set(manifest['artifacts']) == {str(layer) for layer in layers}
        protocol = manifest['protocol']
        if common is None:
            common = protocol
        assert protocol == common
        assert protocol['window_ids'] == list(range(320)) and protocol['sequence_length'] == 2048
        calibration = manifest['calibration']
        assert calibration['fit_windows'] == 256 and calibration['heldout_windows'] == 64
        assert calibration['fit_rows'] == 256 * 2048 and calibration['heldout_rows'] == 64 * 2048
        assert manifest['model']['config_sha256'] == audit['config_sha256']
        for layer in layers:
            record = manifest['artifacts'][str(layer)]
            path = directory / record['file']
            assert sha256(path) == record['sha256']
            with safe_open(str(path), framework='pt', device='cpu') as tensors:
                assert set(tensors.keys()) == {'fit_covariance', 'heldout_covariance', 'weight'}
                for name in tensors.keys():
                    tensor = tensors.get_tensor(name)
                    expected = (output_width, width) if name == 'weight' else (width, width)
                    assert tuple(tensor.shape) == expected
                    assert tensor.dtype == (torch.bfloat16 if name == 'weight' else torch.float32)
                    assert torch.isfinite(tensor).all()
                    del tensor
            print('VERIFIED COVARIANCE', kind, layer, flush=True)
        verified[kind] = dict(layers=layers, manifest_sha256=sha256(manifest_path))
    write_json(args.output, dict(status='complete', artifacts=sum(
        len(value['layers']) for value in verified.values()), verified=verified,
        protocol=common, audit_sha256=sha256(args.audit), source_sha256=sha256(__file__)))


if __name__ == '__main__':
    main()
