"""Native 32K gated-V captures for the Qwen3.5 K-routing experiment."""

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file, save_file

from evaluation.qwen35_hybrid_common import atomic_save, full_layers, load_model, sha256
from evaluation.run_qwen35_hybrid import NativeCapture


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=('smoke', 'capture'))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((args.calibration / 'manifest.json').read_text())
    token_path = args.calibration / 'windows.safetensors'
    assert manifest['status'] == 'complete' and manifest['shape'] == [80, 32768]
    assert manifest['sha256'] == sha256(token_path)
    assert manifest['model_config_sha256'] == sha256(args.model / 'config.json')
    tokens = load_file(str(token_path))['input_ids'].long()
    assert list(tokens.shape) == [80, 32768]
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_model(str(args.model), 'cuda:0')
    layers = full_layers(model)
    assert tuple(layers) == (3, 7, 11, 15, 19, 23, 27, 31)
    provenance = dict(windows_sha256=manifest['sha256'], model_config_sha256=manifest['model_config_sha256'],
                      source_sha256=sha256(Path(__file__)), command=sys.argv, python=sys.executable,
                      trajectory='native_dense', wo_compression=False)
    with torch.inference_mode():
        if args.stage == 'smoke':
            short = tokens[:1, :64].cuda()
            reference = model.model(short, use_cache=False).last_hidden_state
            with NativeCapture(model) as collector:
                measured = model.model(short, use_cache=False).last_hidden_state
                torch.testing.assert_close(measured, reference, rtol=0, atol=0)
                started = time.monotonic()
                result = model.model(tokens[:1].cuda(), use_cache=False).last_hidden_state
                assert result.shape[1] == 32768 and torch.isfinite(result).all()
                for index in layers:
                    row = collector.rows[index]
                    assert row['z'].shape == row['gate'].shape == (1, 32768, 16, 256)
                    assert row['target'].shape == (1, 32768, 4096)
                    assert all(torch.isfinite(row[key]).all() for key in ('z', 'gate', 'target'))
            atomic_save(args.output / 'smoke.json', dict(status='complete', **provenance,
                length=32768, capture_parity='bitwise at 64 tokens; native gate closure at 32768',
                seconds=time.monotonic()-started, peak_gib=torch.cuda.max_memory_allocated()/2**30))
            print('32K NATIVE V CAPTURE SMOKE PASSED', flush=True)
            return
        smoke = json.loads((args.output / 'smoke.json').read_text())
        assert smoke['status'] == 'complete' and smoke['windows_sha256'] == manifest['sha256']
        assert smoke['source_sha256'] == provenance['source_sha256']
        assert 0 <= args.shard_index < args.num_shards
        completed = []
        with NativeCapture(model) as collector:
            for index in range(args.shard_index, 80, args.num_shards):
                destination = args.output / f'w{index:03d}'
                done = destination / 'manifest.json'
                if done.exists():
                    saved = json.loads(done.read_text())
                    assert saved['windows_sha256'] == manifest['sha256']
                    assert all(sha256(destination/name) == digest for name, digest in saved['files'].items())
                    completed.append(index)
                    continue
                started = time.monotonic()
                model.model(tokens[index:index+1].cuda(), use_cache=False)
                destination.mkdir(parents=True, exist_ok=True)
                hashes = {}
                for layer, attention in layers.items():
                    row = collector.rows[layer]
                    payload = {key: row[key].flatten(0, 1).contiguous() for key in ('z', 'gate', 'target')}
                    payload['weight'] = attention.o_proj.weight.detach().cpu().T.reshape(16, 256, -1).contiguous()
                    path = destination / f'l{layer:02d}.safetensors'
                    assert not path.exists(), f'Unfinished capture requires inspection: {path}'
                    save_file(payload, str(path))
                    hashes[path.name] = sha256(path)
                atomic_save(done, dict(status='complete', **provenance, index=index,
                    split='fit' if index < 64 else 'diagnostic', length=32768, files=hashes))
                completed.append(index)
                print(json.dumps(dict(window=index, seconds=time.monotonic()-started)), flush=True)
                gc.collect()
        atomic_save(args.output / f'shard{args.shard_index}.json', dict(status='complete', **provenance,
            windows=completed, num_shards=args.num_shards))


if __name__ == '__main__':
    main()
