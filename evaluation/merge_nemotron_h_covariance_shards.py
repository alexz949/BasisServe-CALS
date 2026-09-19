"""Merge memory-bounded Nemotron-H covariance captures without data copies."""
import argparse
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--full-smoke', type=Path, required=True)
    parser.add_argument('--shards-root', type=Path, required=True)
    parser.add_argument('--shard-count', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure()
    audit, smoke = read_json(args.audit), read_json(args.full_smoke)
    assert audit['status'] == smoke['status'] == 'complete'
    assert smoke['audit_sha256'] == sha256(args.audit)
    assert args.shard_count > 1
    expected = {
        kind: [row['layer'] for row in audit['layers'] if row['kind'] == kind]
        for kind in ('full_attention', 'linear_attention')
    }
    manifests = {kind: [] for kind in expected}
    for shard in range(args.shard_count):
        selected = sorted(sum((expected[kind][shard::args.shard_count] for kind in expected), []))
        selected_text = ','.join(str(layer) for layer in selected)
        for kind in expected:
            path = args.shards_root / f'shard_{shard}' / kind / 'manifest.json'
            manifest = read_json(path)
            assert manifest['status'] == 'complete' and manifest['dense_teacher']
            assert manifest['audit_sha256'] == sha256(args.audit)
            assert manifest['full_smoke_sha256'] == sha256(args.full_smoke)
            assert manifest['protocol']['selected_layers'] == selected_text
            assert manifest['layers'] == expected[kind][shard::args.shard_count]
            manifests[kind].append((path, manifest))
    args.output.mkdir(parents=True, exist_ok=True)
    common_protocol = None
    for kind in expected:
        destination = args.output / kind
        destination.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        template = None
        shard_inputs = []
        for manifest_path, manifest in manifests[kind]:
            template = manifest if template is None else template
            protocol = dict(manifest['protocol'])
            protocol['selected_layers'] = None
            if common_protocol is None:
                common_protocol = protocol
            assert protocol == common_protocol
            shard_inputs.append({'manifest': str(manifest_path), 'sha256': sha256(manifest_path)})
            for layer_text, record in manifest['artifacts'].items():
                assert layer_text not in artifacts
                source = manifest_path.parent / record['file']
                assert sha256(source) == record['sha256']
                target = destination / record['file']
                if target.exists():
                    assert sha256(target) == record['sha256']
                else:
                    os.link(source, target)
                artifacts[layer_text] = record
        assert template is not None
        assert set(artifacts) == {str(layer) for layer in expected[kind]}
        merged = dict(template)
        merged['protocol'] = common_protocol
        merged['layers'] = expected[kind]
        merged['artifacts'] = {str(layer): artifacts[str(layer)] for layer in expected[kind]}
        merged['capture_shards'] = shard_inputs
        merged['command'] = shlex.join(sys.argv)
        merged['python'] = sys.executable
        write_json(destination / 'manifest.json', merged)
        print('MERGED COVARIANCES', kind, len(expected[kind]), flush=True)
    write_json(args.output / 'merge_manifest.json', {
        'status': 'complete', 'audit_sha256': sha256(args.audit),
        'full_smoke_sha256': sha256(args.full_smoke), 'shard_count': args.shard_count,
        'layers': expected, 'source_sha256': sha256(__file__), 'command': shlex.join(sys.argv),
        'manifests': {kind: sha256(args.output / kind / 'manifest.json') for kind in expected},
    })


if __name__ == '__main__':
    main()
