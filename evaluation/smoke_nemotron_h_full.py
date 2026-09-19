"""Validate native multi-GPU loading and cached decode of the pinned dense model."""
import argparse
import inspect
from pathlib import Path
import shlex
import sys
import time

import torch
import causal_conv1d
import mamba_ssm
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.nemotron_h import modeling_nemotron_h as native

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, save_tensors, sha256
from evaluation.nemotron_h_runtime import install_mamba_device_guards


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--block-smoke', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-relative-rmse-limit', type=float, default=0.05)
    args = parser.parse_args()
    configure()
    torch.manual_seed(20260828)
    audit, block_smoke = read_json(args.audit), read_json(args.block_smoke)
    assert audit['status'] == block_smoke['status'] == 'complete'
    assert block_smoke['audit_sha256'] == sha256(args.audit)
    path = Path(audit['model'])
    assert sha256(path / 'config.json') == audit['config_sha256']
    assert sha256(path / 'model.safetensors.index.json') == audit['index_sha256']
    assert sha256(inspect.getfile(native.NemotronHForCausalLM)) == audit['implementation_sha256']
    for name, expected in block_smoke['implementations'].items():
        implementation = inspect.getclosurevars(getattr(native, name)).nonlocals['implementation']
        assert implementation.__module__ + '.' + implementation.__name__ == expected
    assert torch.cuda.device_count() == 4
    max_memory = {i: torch.cuda.get_device_properties(i).total_memory - 12 * 2**30 for i in range(4)}
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=False,
        local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map='balanced', max_memory=max_memory).eval()
    assert type(model) is native.NemotronHForCausalLM
    assert all(p.device.type == 'cuda' for p in model.parameters())
    assert len({p.device.index for p in model.parameters()}) == 4
    guarded_layers = install_mamba_device_guards(model)
    print('LOADED', model.hf_device_map, time.monotonic() - started, flush=True)

    # Compare every loaded tensor, including the small Mamba parameters that
    # must not be reinitialized by post-load model initialization.
    index = read_json(path / 'model.safetensors.index.json')['weight_map']
    state = model.state_dict()
    assert set(state) == set(audit['checkpoint_to_native_keys'].values())
    checked = 0
    for filename in sorted(set(index.values())):
        with safe_open(str(path / filename), framework='pt', device='cpu') as data:
            for source in data.keys():
                target = audit['checkpoint_to_native_keys'][source]
                value = state[target]
                expected = data.get_tensor(source).to(dtype=value.dtype)
                assert torch.equal(value.cpu(), expected), target
                assert torch.isfinite(value).all(), target
                checked += 1
                del expected
        print('WEIGHTS VERIFIED', checked, '/', len(state), flush=True)
    del state
    phase = {}
    layer_checks = []

    def check_layer(module, positional, kwargs, output):
        value = output[0] if isinstance(output, tuple) else output
        source = positional[0] if positional else kwargs['hidden_states']
        finite = bool(torch.isfinite(value).all())
        record = dict(**phase, layer=module.layer_idx, kind=module.block_type,
            input_finite=bool(torch.isfinite(source).all()), output_finite=finite,
            input_max_abs=float(source.abs().max()), device=str(value.device))
        if finite:
            record['output_max_abs'] = float(value.abs().max())
        layer_checks.append(record)
        if not finite:
            path = args.output.with_name(f'nonfinite_{phase["name"]}_{phase["tokens"]}_layer_{module.layer_idx:03d}')
            save_tensors(path.with_suffix('.safetensors'), dict(hidden_states=source.detach().cpu()))
            write_json(path.with_suffix('.json'), dict(status='nonfinite', checks=layer_checks,
                audit_sha256=sha256(args.audit), source_sha256=sha256(__file__),
                input_sha256=sha256(path.with_suffix('.safetensors'))))
            print('FIRST NONFINITE LAYER', record, flush=True)
        assert finite, record

    for layer in model.model.layers:
        layer.register_forward_hook(check_layer, with_kwargs=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False, local_files_only=True)
    seed = tokenizer('The river passes through the city. Read the passage carefully. ',
        add_special_tokens=False)['input_ids']
    device = model.get_input_embeddings().weight.device
    reports = []
    for length in (129, 2049):
        ids = torch.tensor((seed * ((length + len(seed) - 1) // len(seed)))[:length],
            device=device)[None]
        for i in range(4):
            torch.cuda.reset_peak_memory_stats(i)
        start = time.monotonic()
        phase.update(name='full', tokens=length)
        full = model(ids, use_cache=False, logits_to_keep=1).logits.cpu()
        print('FULL LOGITS FINITE', length, bool(torch.isfinite(full).all()), flush=True)
        phase.update(name='prefix', tokens=length - 1)
        prefix = model(ids[:, :-1], use_cache=True, logits_to_keep=1)
        print('PREFIX LOGITS FINITE', length - 1, bool(torch.isfinite(prefix.logits).all()), flush=True)
        phase.update(name='cached', tokens=1)
        cached = model(ids[:, -1:], past_key_values=prefix.past_key_values,
            use_cache=True, logits_to_keep=1)
        last = cached.logits.cpu()
        assert torch.isfinite(full).all() and torch.isfinite(last).all()
        relative_rmse = float((full - last).square().sum().sqrt() / full.square().sum().sqrt())
        record = dict(tokens=length, relative_rmse=relative_rmse,
            max_absolute_error=float((full - last).abs().max()),
            full_argmax=int(full.argmax(-1)), cached_argmax=int(last.argmax(-1)),
            peak_gib=[torch.cuda.max_memory_allocated(i) / 2**30 for i in range(4)],
            seconds=time.monotonic() - start)
        reports.append(record)
        print('CACHE COMPARISON', record, flush=True)
        assert relative_rmse < args.cache_relative_rmse_limit, record
        del full, last, prefix, cached, ids
    write_json(args.output, dict(status='complete', reports=reports, layer_checks=layer_checks,
        verified_tensor_count=checked, device_map=model.hf_device_map,
        command=shlex.join(sys.argv), python=sys.executable,
        audit_sha256=sha256(args.audit), block_smoke_sha256=sha256(args.block_smoke),
        source_sha256=sha256(__file__), full_model_tested=True,
        guarded_mamba_layers=guarded_layers,
        cache_relative_rmse_limit=args.cache_relative_rmse_limit,
        device_guard_sha256=sha256(ROOT / 'evaluation/nemotron_h_runtime.py'),
        long_context_quality_tested=False, original_remote_implementation_compared=False,
        seconds=time.monotonic() - started))


if __name__ == '__main__':
    main()
