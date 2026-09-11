"""Staged Qwen3.5 gated V fitting and evaluation; no implicit large jobs."""

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from basisserve.core.qwen35_gated_v_als import GatedVCapture, fit_gated_v, GatedVBlock
from basisserve.core.qwen35_gated_v_runtime import GatedVRuntime, factor_hash
from basisserve.core.qwen35_hybrid_output_runtime import HybridOutputRuntime
from evaluation.qwen35_hybrid_common import atomic_save, full_layers, load_bank, load_model, load_windows, sha256, verify_model_identity


class NativeCapture:
    def __init__(self, model):
        self.model, self.handles, self.rows = model, [], {}

    def __enter__(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        def capture_attention(module, query, key, value, attention_mask, **kwargs):
            output, weights = sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
            self.rows[module.layer_idx]['z'] = output.detach().cpu().contiguous()
            return output, weights
        ALL_ATTENTION_FUNCTIONS.register('basisserve_q35_capture', capture_attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register('basisserve_q35_capture', ALL_MASK_ATTENTION_FUNCTIONS['sdpa'])
        self.previous = self.model.config._attn_implementation
        for index, attention in full_layers(self.model).items():
            self.rows[index] = {}
            def q_hook(module, inputs, output, index=index, d=attention.head_dim):
                self.rows[index]['gate'] = output.view(*output.shape[:-1], -1, 2 * d).chunk(2, -1)[1].sigmoid().detach().cpu().contiguous()
            def o_hook(module, inputs, output, index=index):
                row = self.rows[index]
                expected = (row['z'] * row['gate']).reshape(inputs[0].shape)
                assert torch.equal(expected, inputs[0].detach().cpu()), 'Captured gate does not reproduce native Wo input'
                target = output if module.bias is None else output - module.bias
                self.rows[index]['target'] = target.detach().cpu().contiguous()
            def v_hook(module, inputs, output, index=index):
                self.rows[index]['raw_v'] = output.detach().cpu().contiguous()
            self.handles.extend([attention.q_proj.register_forward_hook(q_hook), attention.o_proj.register_forward_hook(o_hook), attention.v_proj.register_forward_hook(v_hook)])
        self.model.config._attn_implementation = 'basisserve_q35_capture'
        return self

    def __exit__(self, *args):
        self.model.config._attn_implementation = self.previous
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def smoke(args):
    model = load_model(args.model_path, args.device)
    tokens = load_windows(args.data, 'fit')[:1, :64].to(args.device)
    t0 = time.monotonic()
    dense = model.model(tokens, use_cache=False).last_hidden_state
    with NativeCapture(model) as capture:
        captured = model.model(tokens, use_cache=False).last_hidden_state
        torch.testing.assert_close(captured, dense, atol=0, rtol=0)
    index = next(iter(full_layers(model)))
    attention = full_layers(model)[index]
    d, groups = attention.head_dim, model.config.num_key_value_heads
    eye = torch.eye(d).repeat(groups, 1, 1).to(torch.bfloat16)
    bank = {index: {'E_V': eye, 'R_V': eye}}
    with GatedVRuntime(model, bank):
        identity = model.model(tokens, use_cache=False).last_hidden_state
    relative = float((identity.float() - dense.float()).square().sum() / dense.float().square().sum())
    assert relative < 1e-4, relative
    c = capture.rows[index]
    weight = attention.o_proj.weight.detach().float().T.reshape(model.config.num_attention_heads, d, -1)
    mapping = torch.arange(model.config.num_attention_heads) // attention.num_key_value_groups
    data = GatedVCapture(c['z'].flatten(0, 1), c['gate'].flatten(0, 1), weight,
        c['target'].flatten(0, 1), mapping)
    fitted = fit_gated_v(data, 64, encoder_sweeps=1, device=args.device, chunk_rows=64, linear_max_iter=8)
    bank = {index: {k: fitted[k].to(torch.bfloat16) for k in ['E_V', 'R_V']}}
    with GatedVRuntime(model, bank):
        uncached = model.model(tokens, use_cache=False).last_hidden_state
        prefix = model.model(tokens[:, :63], use_cache=True)
        decoded = model.model(tokens[:, 63:], past_key_values=prefix.past_key_values, use_cache=True)
        decode_error = float((decoded.last_hidden_state.float() - uncached[:, -1:].float()).square().sum() / uncached[:, -1:].float().square().sum())
        assert decode_error < 1e-3, decode_error
        cache = decoded.past_key_values
        shapes = {'key': list(cache.key_cache[index].shape), 'value': list(cache.value_cache[index].shape)}
        assert shapes['value'] == [1, groups, 64, 64]
    report = {'status': 'passed', 'layer': index, 'identity_relative_mse': relative,
        'cached_decode_relative_mse': decode_error, 'cache_shapes': shapes, 'solver': fitted['history'],
        'seconds': time.monotonic() - t0, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'device': torch.cuda.get_device_name(), 'command': sys.argv, 'python': sys.executable}
    atomic_save(Path(args.output) / 'smoke.json', report)
    print(json.dumps(report), flush=True)


@torch.no_grad()
def capture(args):
    model = load_model(args.model_path, args.device)
    output = Path(args.output)
    assert not (output / 'manifest.json').exists()
    weights = {i: a.o_proj.weight.detach().cpu().T.reshape(model.config.num_attention_heads, a.head_dim, -1).contiguous() for i, a in full_layers(model).items()}
    if not (output / 'weights.pt').exists():
        atomic_save(output / 'weights.pt', weights)
    completed = []
    with NativeCapture(model) as collector:
        for split in ('fit', 'heldout'):
            tokens = load_windows(args.data, split)
            for start in range(0, len(tokens), args.capture_chunk):
                path = output / f'{split}_{start:03d}.pt'
                if path.exists():
                    saved = torch.load(path, weights_only=True, map_location='cpu')
                    assert saved['windows_sha256'] == sha256(Path(args.data) / 'windows.pt')
                    completed.append(path.name)
                    continue
                chunks = {i: {key: [] for key in ('z', 'gate', 'target', 'raw_v')} for i in weights}
                started = time.monotonic()
                for token in tokens[start:start + args.capture_chunk]:
                    model.model(token[None].to(args.device), use_cache=False)
                    for i in weights:
                        for key in chunks[i]:
                            chunks[i][key].append(collector.rows[i][key].flatten(0, 1))
                layers = {i: {key: torch.cat(values) for key, values in row.items()} for i, row in chunks.items()}
                atomic_save(path, {'layers': layers, 'split': split, 'start': start,
                    'windows_sha256': sha256(Path(args.data) / 'windows.pt')})
                completed.append(path.name)
                print(json.dumps({'split': split, 'start': start, 'seconds': time.monotonic() - started, 'file': str(path)}), flush=True)
    atomic_save(output / 'manifest.json', {'format': 'basisserve.qwen35.gated_captures.v1',
        'trajectory': 'native_dense', 'gate': 'native_sigmoid_bfloat16',
        'gate_closure': 'every captured native Wo input exactly reproduced by captured pre-gate output times captured gate',
        'target': 'native_o_proj_output_excluding_bias',
        'weights_sha256': sha256(output / 'weights.pt'), 'windows_sha256': sha256(Path(args.data) / 'windows.pt'),
        'model_config_sha256': sha256(Path(args.model_path) / 'config.json'), 'files': completed})


def read_capture(path, split, layer, *, device, model_path):
    root = Path(path)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['model_config_sha256'] == sha256(Path(model_path) / 'config.json')
    rows = [torch.load(p, weights_only=True, map_location='cpu', mmap=True)['layers'][layer] for p in sorted(root.glob(f'{split}_*.pt'))]
    assert rows
    data = {key: torch.cat([row[key] for row in rows]).to(device) for key in ('z', 'gate', 'target')}
    assert len(data['z']) == {'fit': 256, 'heldout': 64}[split] * 2048
    weight = torch.load(root / 'weights.pt', weights_only=True, map_location='cpu')[layer].to(device)
    config = json.loads((Path(model_path) / 'config.json').read_text())['text_config']
    mapping = torch.arange(weight.shape[0]) // (config['num_attention_heads'] // config['num_key_value_heads'])
    return GatedVCapture(data['z'], data['gate'], weight, data['target'], mapping)


@torch.no_grad()
def fit(args):
    for layer in map(int, args.layers.split(',')):
        train = read_capture(args.capture_dir, 'fit', layer, device=args.device, model_path=args.model_path)
        heldout = read_capture(args.capture_dir, 'heldout', layer, device=args.device, model_path=args.model_path)
        for rank in map(int, args.ranks.split(',')):
            path = Path(args.output) / f'l{layer:02d}_r{rank:03d}.pt'
            if path.exists():
                continue
            started = time.monotonic()
            result = fit_gated_v(train, rank, heldout=heldout, encoder_sweeps=args.encoder_sweeps, device=args.device,
                chunk_rows=args.chunk_rows, linear_max_iter=args.linear_max_iter,
                encoder_preconditioner=args.encoder_preconditioner,
                progress=lambda row: print(json.dumps({'layer': layer, 'rank': rank, **row}), flush=True))
            e, r = result['E_V'].to(torch.bfloat16), result['R_V'].to(torch.bfloat16)
            metrics = {}
            for name, data in [('fit', train), ('heldout', heldout)]:
                op = GatedVBlock(data, e.float(), block='decoder', chunk_rows=args.chunk_rows)
                denominator = sum(float(data.target[start:start + args.chunk_rows].double().square().sum())
                    for start in range(0, len(data.target), args.chunk_rows))
                metrics[name + '_export_relative_mse'] = 2 * len(data.z) * op.loss(r.float()) / denominator
            result.update({'E_V': e.cpu(), 'R_V': r.cpu(), 'layer': layer, 'rank': rank,
                'encoder_sweeps': args.encoder_sweeps,
                'linear_max_iter': args.linear_max_iter,
                'encoder_preconditioner': args.encoder_preconditioner,
                'metrics': metrics, 'seconds': time.monotonic() - started,
                'capture_manifest_sha256': sha256(Path(args.capture_dir) / 'manifest.json')})
            atomic_save(path, result)
            print(json.dumps({'layer': layer, 'rank': rank, 'seconds': result['seconds'], **metrics}), flush=True)


@torch.no_grad()
def evaluate(args):
    payload = load_bank(args.bank) if args.bank else None
    identity = None
    if payload is not None:
        assert payload['windows_sha256'] == sha256(Path(args.data) / 'windows.pt')
        if payload['method'] in ('uniform', 'twosided'):
            identity = payload['model_identity']
            verify_model_identity(args.model_path, identity)
        else:
            assert payload['method'] in ('palu_mlrd', 'palu_glrd2', 'palu_glrd4')
    model = load_model(args.model_path, args.device)
    bank = payload['layers'] if payload is not None else None
    result = {'command': sys.argv, 'bank': args.bank, 'wo_bank': args.wo_bank, 'python': sys.executable,
        'verified_model_identity': identity, 'datasets': {}}
    context = HybridOutputRuntime(model, bank, args.wo_bank) if args.wo_bank else GatedVRuntime(model, bank) if bank else nullcontext()
    with context:
        for split in ('wikitext', 'c4_eval'):
            tokens = load_windows(args.data, split)
            if tokens.ndim == 1:
                windows = [tokens[s:s + 2048] for s in range(0, len(tokens), 2048) if len(tokens[s:s + 2048]) > 1]
            else:
                windows = list(tokens)
            total, count, per_window = 0., 0, []
            for index, token in enumerate(windows):
                ids = token[None].to(args.device)
                hidden = model.model(ids, use_cache=False).last_hidden_state
                nll = 0.
                for start in range(0, ids.shape[1] - 1, 128):
                    stop = min(start + 128, ids.shape[1] - 1)
                    logits = model.lm_head(hidden[:, start:stop]).float()
                    nll += float(F.cross_entropy(logits.flatten(0, 1), ids[:, start + 1:stop + 1].flatten(), reduction='sum'))
                total += nll
                count += ids.shape[1] - 1
                per_window.append(nll / (ids.shape[1] - 1))
                if index % 8 == 0:
                    print(json.dumps({'dataset': split, 'window': index, 'running_ppl': math.exp(total / count)}), flush=True)
            result['datasets'][split] = {'ppl': math.exp(total / count), 'nll_sum': total,
                'predicted_tokens': count, 'windows': len(windows), 'window_mean_nll': per_window}
    atomic_save(args.output, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['smoke', 'capture', 'fit', 'evaluate'])
    parser.add_argument('--model-path', default='results/q35_hybrid/model')
    parser.add_argument('--data', default='results/q35_hybrid/data')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True)
    parser.add_argument('--capture-dir', default='results/q35_hybrid/capture')
    parser.add_argument('--capture-chunk', type=int, default=8)
    parser.add_argument('--layers', default='3,7,11,15,19,23,27,31')
    parser.add_argument('--ranks', default='64,80,96')
    parser.add_argument('--chunk-rows', type=int, default=2048)
    parser.add_argument('--linear-max-iter', type=int, default=200)
    parser.add_argument('--encoder-sweeps', type=int, default=6)
    parser.add_argument('--encoder-preconditioner', choices=['jacobi', 'separable'], default='separable')
    parser.add_argument('--bank')
    parser.add_argument('--wo-bank')
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps({'args': vars(args), 'python': sys.executable, 'torch': torch.__version__}), flush=True)
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
