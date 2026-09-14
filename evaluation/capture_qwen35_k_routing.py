"""Window-sharded native Qwen3.5 Q/K/V inputs for both router capacities."""

import argparse
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from basisserve.core.query_position_sampling import candidate_positions
from evaluation.qwen35_hybrid_common import atomic_save, full_layers, load_model, sha256


class RoutingCapture:
    def __init__(self, model, positions):
        self.model, self.positions = model, positions
        self.handles, self.rows, self.rope = [], {}, None

    def capture(self, module, args, kwargs):
        hidden = kwargs['hidden_states']
        batch, length, _ = hidden.shape
        dim = module.head_dim
        q, _ = module.q_proj(hidden).view(batch, length, -1, 2*dim).chunk(2, -1)
        q = module.q_norm(q).transpose(1, 2)
        prekey = module.k_norm(module.k_proj(hidden).view(batch, length, -1, dim))
        value = module.v_proj(hidden).view(batch, length, -1, dim)
        cos, sin = kwargs['position_embeddings']
        q, key = apply_rotary_pos_emb(q, prekey.transpose(1, 2), cos, sin)
        self.rows[module.layer_idx] = dict(
            rows=torch.cat((value, key.transpose(1, 2)), -1).cpu().contiguous(),
            pre_rope_keys=prekey.cpu().contiguous(),
            candidate_queries=q[:, :, self.positions].transpose(1, 2).cpu().contiguous())
        self.rope = dict(cos=cos.cpu().contiguous(), sin=sin.cpu().contiguous())

    def __enter__(self):
        for module in full_layers(self.model).values():
            self.handles.append(module.register_forward_pre_hook(self.capture, with_kwargs=True))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
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
    windows_path = args.calibration/'windows.safetensors'
    windows = load_file(str(windows_path))['input_ids']
    manifest = json.loads((args.calibration/'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['sha256'] == sha256(windows_path)
    assert manifest['model_config_sha256'] == sha256(args.model/'config.json')
    assert windows.shape == (80, 32768) and 0 <= args.shard_index < args.num_shards
    model = load_model(str(args.model), 'cuda:0')
    layers = list(full_layers(model))
    assert layers == [3, 7, 11, 15, 19, 23, 27, 31]
    common = dict(windows_sha256=manifest['sha256'], model_config_sha256=manifest['model_config_sha256'],
        source_sha256=sha256(Path(__file__)), wo_compression=False, trajectory='native_dense', layers=layers)
    if args.stage == 'smoke':
        tokens = windows[:1, :128].cuda()
        reference = model.model(tokens, use_cache=False).last_hidden_state
        with RoutingCapture(model, candidate_positions(128)) as capture:
            actual = model.model(tokens, use_cache=False).last_hidden_state
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            for row in capture.rows.values():
                assert row['rows'].shape == (1, 128, 4, 512)
                assert row['pre_rope_keys'].shape == (1, 128, 4, 256)
                assert row['candidate_queries'].shape == (1, 2, 16, 256)
                assert all(torch.isfinite(value).all() for value in row.values())
        atomic_save(args.output/'smoke.json', dict(status='complete', parity='bitwise', **common))
        print('NATIVE ROUTING CAPTURE SMOKE PASSED', flush=True)
        return
    smoke = json.loads((args.output/'smoke.json').read_text())
    assert smoke['status'] == 'complete' and all(smoke[key] == value for key, value in common.items())
    positions = candidate_positions(32768)
    with RoutingCapture(model, positions) as capture:
        for index in range(args.shard_index, 80, args.num_shards):
            root = args.output/f'w{index:03d}'
            done = root/'manifest.json'
            if done.exists():
                row = json.loads(done.read_text())
                assert row['status'] == 'complete' and all(row[key] == value for key, value in common.items())
                assert all(sha256(root/name) == digest for name, digest in row['files'].items())
                continue
            started = time.monotonic()
            model.model(windows[index:index+1].cuda(), use_cache=False)
            root.mkdir(parents=True, exist_ok=True)
            files = {}
            payloads = [(f'l{layer:02d}.safetensors', capture.rows[layer]) for layer in layers]
            payloads.append(('rope.safetensors', capture.rope))
            for name, payload in payloads:
                path = root/name
                assert not path.exists(), f'Inspect incomplete capture before resuming: {path}'
                save_file(payload, str(path))
                files[name] = sha256(path)
            atomic_save(done, dict(status='complete', index=index, length=32768,
                split='fit' if index < 64 else 'diagnostic', candidate_positions=positions, files=files, **common))
            print(json.dumps(dict(window=index, seconds=time.monotonic()-started)), flush=True)
    atomic_save(args.output/f'shard{args.shard_index}.json', dict(status='complete',
        windows=list(range(args.shard_index, 80, args.num_shards)), **common))


if __name__ == '__main__':
    main()
