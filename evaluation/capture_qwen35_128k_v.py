"""Native gated-V captures (z = attention output, gate = sigmoid attention gate, target = o_proj output) on 128K windows.

128K variant of ``qwen35_k_routing_v.py``: the window bank shape and fit/validation split come from the bank manifest,
``--layers`` captures a subset of the eight full-attention layers per pass (disk: one layer of one 128K window is
131072 x (16x256 + 16x256 + 4096) BF16 = 3 GiB before striding), and ``--row-stride`` keeps every k-th token row
(the gated-V ALS is a per-token least-squares fit, so a uniform token subsample of the same windows is used to keep
the resident inputs at the size of the earlier 64 x 32K pipeline)."""
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
    p.add_argument('--layers', default='3,7,11,15,19,23,27,31')
    p.add_argument('--row-stride', type=int, default=2)
    p.add_argument('--windows', default='fit', help="'fit', 'validation' or a comma list of window ids")
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((args.calibration / 'manifest.json').read_text())
    token_path = args.calibration / 'windows.safetensors'
    assert manifest['status'] == 'complete' and manifest['sha256'] == sha256(token_path)
    assert manifest['model_config_sha256'] == sha256(args.model / 'config.json')
    tokens = load_file(str(token_path))['input_ids'].long()
    assert list(tokens.shape) == list(manifest['shape'])
    count, length = tokens.shape
    selected_layers = [int(x) for x in args.layers.split(',')]
    window_ids = (list(manifest['fit_ids']) if args.windows == 'fit' else list(manifest['validation_ids']) if args.windows == 'validation'
                  else [int(x) for x in args.windows.split(',')])
    fit_ids = set(int(i) for i in manifest['fit_ids'])
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_model(str(args.model), 'cuda:0')
    layers = full_layers(model)
    assert tuple(layers) == (3, 7, 11, 15, 19, 23, 27, 31) and set(selected_layers) <= set(layers)
    provenance = dict(windows_sha256=manifest['sha256'], model_config_sha256=manifest['model_config_sha256'],
                      source_sha256=sha256(Path(__file__)), command=sys.argv, python=sys.executable,
                      trajectory='native_dense', wo_compression=False, sequence_length=length, row_stride=args.row_stride)
    kept = len(range(0, length, args.row_stride))
    with torch.inference_mode():
        if args.stage == 'smoke':
            short = tokens[:1, :64].cuda()
            reference = model.model(short, use_cache=False).last_hidden_state
            with NativeCapture(model) as collector:
                measured = model.model(short, use_cache=False).last_hidden_state
                torch.testing.assert_close(measured, reference, rtol=0, atol=0)
                started = time.monotonic()
                result = model.model(tokens[:1].cuda(), use_cache=False).last_hidden_state
                assert result.shape[1] == length and torch.isfinite(result).all()
                for index in layers:
                    row = collector.rows[index]
                    assert row['z'].shape == row['gate'].shape == (1, length, 16, 256)
                    assert row['target'].shape == (1, length, 4096)
                    assert all(torch.isfinite(row[key]).all() for key in ('z', 'gate', 'target'))
            atomic_save(args.output / 'smoke.json', dict(status='complete', **provenance, length=kept,
                capture_parity=f'bitwise at 64 tokens; native gate closure at {length}',
                seconds=time.monotonic() - started, peak_gib=torch.cuda.max_memory_allocated() / 2**30))
            print('128K NATIVE V CAPTURE SMOKE PASSED', flush=True)
            return
        smoke = json.loads((args.output / 'smoke.json').read_text())
        assert smoke['status'] == 'complete' and smoke['windows_sha256'] == manifest['sha256']
        assert smoke['source_sha256'] == provenance['source_sha256'] and smoke['row_stride'] == args.row_stride
        assert 0 <= args.shard_index < args.num_shards
        completed = []
        with NativeCapture(model) as collector:
            for index in window_ids[args.shard_index::args.num_shards]:
                destination = args.output / f'w{index:03d}'
                done = destination / f'manifest_l{"_".join(f"{l:02d}" for l in selected_layers)}.json'
                if done.exists():
                    saved = json.loads(done.read_text())
                    assert saved['windows_sha256'] == manifest['sha256']
                    assert all(sha256(destination / name) == digest for name, digest in saved['files'].items())
                    completed.append(index)
                    continue
                started = time.monotonic()
                model.model(tokens[index:index + 1].cuda(), use_cache=False)
                destination.mkdir(parents=True, exist_ok=True)
                hashes = {}
                for layer in selected_layers:
                    row, attention = collector.rows[layer], layers[layer]
                    payload = {key: row[key][:, ::args.row_stride].flatten(0, 1).contiguous() for key in ('z', 'gate', 'target')}
                    assert payload['z'].shape[0] == kept
                    payload['weight'] = attention.o_proj.weight.detach().cpu().T.reshape(16, 256, -1).contiguous()
                    path = destination / f'l{layer:02d}.safetensors'
                    assert not path.exists(), f'Unfinished capture requires inspection: {path}'
                    save_file(payload, str(path))
                    hashes[path.name] = sha256(path)
                for layer in layers:
                    collector.rows[layer] = {}
                atomic_save(done, dict(status='complete', **provenance, index=index, split='fit' if index in fit_ids else 'diagnostic',
                                       length=kept, layers=selected_layers, files=hashes))
                completed.append(index)
                print(json.dumps(dict(window=index, layers=selected_layers, seconds=time.monotonic() - started)), flush=True)
                gc.collect()
        atomic_save(args.output / f'shard{args.shard_index}_l{"_".join(f"{l:02d}" for l in selected_layers)}.json',
                    dict(status='complete', **provenance, windows=completed, num_shards=args.num_shards))


if __name__ == '__main__':
    main()
