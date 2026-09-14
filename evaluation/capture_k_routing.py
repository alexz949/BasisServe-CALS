"""Capture dense teacher activations with streamed weights and window sharding."""
import argparse
import inspect
import math
from pathlib import Path
import shlex
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, save_tensors, sha256
from basisserve.core.query_position_sampling import candidate_positions
from evaluation.k_routing_config import routing_config


def load_module(module, prefix, model_path, index, source_keys):
    state = {}
    for name in module.state_dict():
        key = source_keys[prefix + name]
        assert key in index, key
        with safe_open(str(model_path / index[key]), framework='pt', device='cpu') as data:
            state[name] = data.get_tensor(key).to(device='cuda', dtype=torch.bfloat16)
    loaded = module.load_state_dict(state, assign=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert all(p.device.type == 'cuda' for p in module.parameters())


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rope', choices=('native', 'yarn2'), required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=4)
    parser.add_argument('--fit-count', type=int, default=64)
    parser.add_argument('--diagnostic-count', type=int, default=16)
    parser.add_argument('--layers', type=str)
    parser.add_argument('--compare-capture', type=Path)
    parser.add_argument('--native-audit', type=Path)
    parser.add_argument('--full-smoke', type=Path)
    args = parser.parse_args()
    configure()
    identity = read_json(args.identity)
    window_manifest = read_json(args.windows.with_name('manifest.json'))
    assert identity['status'] == window_manifest['status'] == 'complete'
    assert window_manifest['model_config_sha256'] == identity['model_config_sha256']
    assert window_manifest['sha256'] == sha256(args.windows)
    assert 0 < args.fit_count <= 64 and 0 < args.diagnostic_count <= 16
    assert 0 <= args.shard_index < args.num_shards
    all_ids = list(range(args.fit_count)) + list(range(64, 64 + args.diagnostic_count))
    window_ids = all_ids[args.shard_index::args.num_shards]
    assert window_ids
    windows = load_file(str(args.windows))['input_ids'][window_ids]
    count, length = windows.shape
    path = Path(identity['model'])
    assert sha256(path / 'config.json') == identity['model_config_sha256']
    config = routing_config(identity, rope=args.rope, sequence_length=length)
    native_hybrid = config.model_type == 'nemotron_h'
    config._attn_implementation = 'sdpa'
    targets = [int(x) for x in args.layers.split(',')] if args.layers else identity['attention_layers']
    assert targets and len(targets) == len(set(targets))
    assert set(targets) <= set(identity['attention_layers'])
    runtime_config = config.to_dict()
    if native_hybrid:
        # The native unbounded time-step limit is metadata, not a tensor.
        # Preserve its meaning in strict JSON without changing the model config.
        runtime_config['time_step_limit'] = [str(x) if math.isinf(x) else x
            for x in config.time_step_limit]
    spec = dict(identity_sha256=sha256(args.identity), windows_sha256=sha256(args.windows),
        runtime_config=runtime_config, rope=args.rope, sequence_length=length,
        fit_ids=list(range(args.fit_count)), diagnostic_ids=list(range(64, 64 + args.diagnostic_count)),
        teacher='native dense BF16 SDPA', num_shards=args.num_shards,
        source_sha256=sha256(__file__), config_helper_sha256=sha256(ROOT / 'evaluation/k_routing_config.py'))
    index = read_json(path / 'model.safetensors.index.json')['weight_map']
    if native_hybrid:
        import causal_conv1d
        import mamba_ssm
        from transformers.models.nemotron_h import modeling_nemotron_h as native
        assert args.native_audit is not None and args.full_smoke is not None
        audit, smoke = read_json(args.native_audit), read_json(args.full_smoke)
        assert audit['status'] == smoke['status'] == 'complete'
        assert audit['config_sha256'] == identity['model_config_sha256']
        assert audit['index_sha256'] == sha256(path / 'model.safetensors.index.json')
        assert audit['implementation_sha256'] == sha256(inspect.getfile(native.NemotronHForCausalLM))
        assert smoke['audit_sha256'] == sha256(args.native_audit) and smoke['full_model_tested']
        assert smoke['verified_tensor_count'] == audit['tensor_count']
        assert identity['attention_layers'] == [i for i, kind in enumerate(config.layers_block_type)
            if kind == 'full_attention']
        source_keys = {target: source for source, target in audit['checkpoint_to_native_keys'].items()}
        assert len(source_keys) == len(index)
        spec.update(native_audit_sha256=sha256(args.native_audit), full_smoke_sha256=sha256(args.full_smoke))
        spec['implementations'] = {}
        for name, package in (('mamba2_chunk_scan', 'mamba_ssm'),
                ('mamba2_selective_state_update', 'mamba_ssm'),
                ('causal_conv1d_fn', 'causal_conv1d'), ('causal_conv1d_update', 'causal_conv1d')):
            implementation = inspect.getclosurevars(getattr(native, name)).nonlocals['implementation']
            assert implementation.__module__.startswith(package)
            spec['implementations'][name] = implementation.__module__ + '.' + implementation.__name__
    else:
        assert args.native_audit is None and args.full_smoke is None
        source_keys = {key: key for key in index}
    if config.model_type == 'llama':
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as Rotary, apply_rotary_pos_emb
    elif config.model_type == 'qwen3':
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding as Rotary, apply_rotary_pos_emb
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16,
            trust_remote_code=False, attn_implementation='sdpa').eval()
    embedding_name = 'embeddings' if native_hybrid else 'embed_tokens'
    embedding = getattr(model.model, embedding_name)
    load_module(embedding, f'model.{embedding_name}.', path, index, source_keys)
    hidden = torch.empty(count, length, config.hidden_size, dtype=torch.bfloat16)
    for i, ids in enumerate(windows):
        hidden[i].copy_(embedding(ids.long().cuda()).cpu())
    setattr(model.model, embedding_name, None)
    del embedding
    # Match the existing dense teacher's CPU buffer initialization before
    # transfer; GPU initialization changes low bits of long-position phases.
    positions = torch.arange(length, device='cuda')[None]
    forward_kwargs = dict(attention_mask=None, position_ids=positions, use_cache=False)
    if not native_hybrid:
        rotary = Rotary(config, device='cpu').cuda()
        cos, sin = rotary(torch.empty(1, device='cuda', dtype=torch.float32), positions)
        forward_kwargs['position_embeddings'] = cos.bfloat16(), sin.bfloat16()
    hq, hkv = config.num_attention_heads, config.num_key_value_heads
    dim = identity['head_dim']
    grid = candidate_positions(length)
    current = {}
    for layer in range(max(targets) + 1):
        module = model.model.layers[layer]
        load_module(module, f'model.layers.{layer}.', path, index, source_keys)
        capture = args.output / f'layer_{layer:03d}' / f'shard_{args.shard_index}.safetensors'
        needs = layer in targets
        reuse = needs and capture.with_suffix('.json').exists()
        if needs and not reuse:
            rows = torch.empty(count, length, hkv, 2 * dim, dtype=torch.bfloat16)
            prekeys = torch.empty(count, length, hkv, dim, dtype=torch.bfloat16)
            queries = torch.empty(count, len(grid), hq, dim, dtype=torch.bfloat16)

        def collect(attention, positional, kwargs):
            x = kwargs['hidden_states']
            q = attention.q_proj(x).view(1, length, hq, dim)
            k = attention.k_proj(x).view(1, length, hkv, dim)
            if config.model_type == 'qwen3':
                q, k = attention.q_norm(q), attention.k_norm(k)
            i = current['window']
            prekeys[i].copy_(k[0].cpu())
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            if not native_hybrid:
                q, k = apply_rotary_pos_emb(q, k, *kwargs['position_embeddings'])
            v = attention.v_proj(x).view(1, length, hkv, dim)
            rows[i].copy_(torch.cat((v[0], k[0].transpose(0, 1)), -1).cpu())
            queries[i].copy_(q[0, :, grid].transpose(0, 1).cpu())

        attention = module.mixer if native_hybrid else module.self_attn
        handle = attention.register_forward_pre_hook(collect, with_kwargs=True) if needs and not reuse else None
        for i in range(count):
            current['window'] = i
            with torch.cuda.device(next(module.parameters()).device):
                output = module(hidden[i:i + 1].cuda(), **forward_kwargs)
            value = output[0] if isinstance(output, tuple) else output
            assert value.shape == hidden[i:i + 1].shape and torch.isfinite(value).all()
            hidden[i].copy_(value[0].cpu())
            del output, value
        if handle is not None:
            handle.remove()
        model.model.layers[layer] = None
        del module, attention
        if needs:
            if reuse:
                record = read_json(capture.with_suffix('.json'))
                assert record['protocol'] == spec and record['window_ids'] == window_ids
                assert record['sha256'] == sha256(capture)
            else:
                tensors = dict(rows=rows, pre_rope_keys=prekeys, candidate_queries=queries)
                if args.compare_capture is not None:
                    assert len(targets) == 1 and args.num_shards == 1
                    reference = load_file(str(args.compare_capture))
                    assert set(tensors) == set(reference)
                    matched = True
                    for name in tensors:
                        same = torch.equal(tensors[name], reference[name])
                        delta = tensors[name].float() - reference[name].float()
                        print('REFERENCE COMPARISON', name, 'equal', same,
                            'max_abs', float(delta.abs().max()),
                            'relative_rmse', float(delta.square().sum().sqrt()
                                / reference[name].float().square().sum().sqrt()), flush=True)
                        matched = matched and same
                        del delta
                    assert matched
                    print('BITWISE REFERENCE MATCH', layer, flush=True)
                    del reference
                save_tensors(capture, tensors)
                write_json(capture.with_suffix('.json'), dict(status='complete', protocol=spec,
                    layer=layer, window_ids=window_ids, candidate_positions=grid,
                    sha256=sha256(capture), command=shlex.join(sys.argv), python=sys.executable))
                del tensors, rows, prekeys, queries
        print('DENSE LAYER COMPLETE', layer, 'windows', window_ids, 'capture', needs, flush=True)
    write_json(args.output / f'complete_{args.shard_index}.json',
        dict(status='complete', protocol=spec, layers=targets, window_ids=window_ids))


if __name__ == '__main__':
    main()
