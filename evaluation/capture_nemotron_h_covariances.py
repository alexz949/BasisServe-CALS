"""Collect dense-teacher attention and Mamba Wo covariances in one model pass."""
import argparse
import inspect
from pathlib import Path
import shlex
import sys
import time

import torch
import causal_conv1d
import mamba_ssm
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.models.nemotron_h import modeling_nemotron_h as native

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, save_tensors, sha256
from evaluation.capture_attention_o_proj_covariances import _StreamingCovarianceCapture, FORMAT
from basisserve.core.nemotron_h_c1 import discover_nemotron_h_c1_targets, projection_for_target
from evaluation.nemotron_h_runtime import install_mamba_device_guards


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--full-smoke', type=Path, required=True)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fit-windows', type=int, default=256)
    parser.add_argument('--heldout-windows', type=int, default=64)
    parser.add_argument('--sequence-length', type=int, default=2048)
    parser.add_argument('--layers', type=str)
    args = parser.parse_args()
    configure()
    audit, smoke = read_json(args.audit), read_json(args.full_smoke)
    assert audit['status'] == smoke['status'] == 'complete'
    assert smoke['audit_sha256'] == sha256(args.audit) and smoke['full_model_tested']
    assert smoke['verified_tensor_count'] == audit['tensor_count']
    assert smoke['device_guard_sha256'] == sha256(ROOT / 'evaluation/nemotron_h_runtime.py')
    model_path = Path(audit['model'])
    assert sha256(model_path / 'config.json') == audit['config_sha256']
    assert sha256(model_path / 'model.safetensors.index.json') == audit['index_sha256']
    assert sha256(inspect.getfile(native.NemotronHForCausalLM)) == audit['implementation_sha256']
    windows_manifest = read_json(args.windows.with_name('manifest.json'))
    assert windows_manifest['status'] == 'complete'
    assert windows_manifest['protocol']['audit_sha256'] == sha256(args.audit)
    assert windows_manifest['sha256'] == sha256(args.windows)
    assert 0 < args.fit_windows <= 256 and 0 < args.heldout_windows <= 64
    assert 0 < args.sequence_length <= 2048
    ids = list(range(args.fit_windows)) + list(range(256, 256 + args.heldout_windows))
    implementations = {}
    for name, package in [('mamba2_chunk_scan', 'mamba_ssm'),
            ('mamba2_selective_state_update', 'mamba_ssm'),
            ('causal_conv1d_fn', 'causal_conv1d'), ('causal_conv1d_update', 'causal_conv1d')]:
        implementation = inspect.getclosurevars(getattr(native, name)).nonlocals['implementation']
        assert implementation.__module__.startswith(package)
        implementations[name] = implementation.__module__ + '.' + implementation.__name__
    protocol = dict(audit_sha256=sha256(args.audit), full_smoke_sha256=sha256(args.full_smoke),
        windows_sha256=sha256(args.windows), window_ids=ids, sequence_length=args.sequence_length,
        selected_layers=args.layers, source_sha256=sha256(__file__),
        device_guard_sha256=smoke['device_guard_sha256'],
        capture_source_sha256=sha256(inspect.getfile(_StreamingCovarianceCapture)),
        implementations=implementations, teacher='native dense BF16 SDPA', covariance_dtype='float32')
    completed = set()
    for kind in ('full_attention', 'linear_attention'):
        directory = args.output / kind
        manifest_path = directory / 'manifest.json'
        if manifest_path.exists():
            record = read_json(manifest_path)
            assert record['status'] == 'complete' and record['protocol'] == protocol
            for artifact in record['artifacts'].values():
                assert sha256(directory / artifact['file']) == artifact['sha256']
            completed.add(kind)
    if len(completed) == 2:
        print('ALL COVARIANCES VERIFIED COMPLETE', flush=True)
        return
    windows = load_file(str(args.windows))['input_ids'][ids, :args.sequence_length]
    assert torch.cuda.device_count() == 4
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=False,
        local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map='balanced', max_memory={i: torch.cuda.get_device_properties(i).total_memory
            - 12*2**30 for i in range(4)}).eval()
    assert type(model) is native.NemotronHForCausalLM
    assert all(p.device.type == 'cuda' for p in model.parameters())
    assert install_mamba_device_guards(model) == smoke['guarded_mamba_layers']
    targets = discover_nemotron_h_c1_targets(model)
    config = model.config
    head_dim = int(getattr(config, 'attention_head_dim', 0)
        or getattr(config, 'head_dim', 0)
        or config.hidden_size // config.num_attention_heads)
    if args.layers is not None:
        selected = {int(i) for i in args.layers.split(',')}
        assert selected <= {t.layer_index for t in targets}
        targets = tuple(t for t in targets if t.layer_index in selected)
    groups = {}
    for kind in ('full_attention', 'linear_attention'):
        if kind in completed:
            continue
        chosen = [t for t in targets if t.layer_kind == kind]
        assert chosen
        assert len({t.input_width for t in chosen}) == 1
        modules = {t.layer_index: projection_for_target(model, t) for t in chosen}
        groups[kind] = _StreamingCovarianceCapture(modules, chosen[0].input_width, torch.float32)
    print('CAPTURE TARGETS', {kind: list(c.modules) for kind, c in groups.items()}, flush=True)
    input_device = model.get_input_embeddings().weight.device
    for split, indices in [('fit', range(args.fit_windows)),
            ('heldout', range(args.fit_windows, len(windows)))]:
        for offset in indices:
            tokens = windows[offset:offset + 1].long().to(input_device)
            for capture in groups.values():
                capture.begin(split, rows=args.sequence_length)
            output = model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
            assert torch.isfinite(output.last_hidden_state).all()
            del output
            for capture in groups.values():
                capture.finish()
            print('DENSE COVARIANCE', split, 'window', ids[offset], flush=True)
        for capture in groups.values():
            capture.normalize_and_offload(split, expected_rows=len(indices)*args.sequence_length)
    for capture in groups.values():
        capture.close()
    for kind, capture in groups.items():
        directory = args.output / kind
        artifacts = {}
        for layer, projection in capture.modules.items():
            tensors = dict(fit_covariance=capture.sums['fit'].pop(layer),
                heldout_covariance=capture.sums['heldout'].pop(layer),
                weight=projection.weight.detach().cpu().contiguous())
            assert all(torch.isfinite(value).all() for value in tensors.values())
            path = directory / f'layer_{layer:03d}.safetensors'
            save_tensors(path, tensors)
            artifacts[str(layer)] = dict(file=path.name, sha256=sha256(path),
                fit_covariance_shape=list(tensors['fit_covariance'].shape),
                heldout_covariance_shape=list(tensors['heldout_covariance'].shape),
                weight_shape=list(tensors['weight'].shape), covariance_dtype='torch.float32',
                weight_dtype=str(tensors['weight'].dtype))
            del tensors
            print('SAVED', kind, layer, flush=True)
        write_json(directory / 'manifest.json', dict(status='complete',
            format=FORMAT if kind=='full_attention' else 'basisserve.nemotron_h.wo_covariances.v1',
            protocol=protocol,
            layer_kind=kind, layers=list(capture.modules), artifacts=artifacts,
            model=dict(path=str(model_path), config_sha256=audit['config_sha256'], model_type='nemotron_h',
                attention_type='gqa', hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads, head_dim=head_dim),
            calibration=dict(storage='normalized_covariance_sufficient_statistics',
                window_count=len(ids), fit_windows=args.fit_windows, heldout_windows=args.heldout_windows,
                sequence_length=args.sequence_length, positions_per_window=args.sequence_length,
                fit_rows=args.fit_windows*args.sequence_length,
                heldout_rows=args.heldout_windows*args.sequence_length,
                rows_per_layer=len(ids)*args.sequence_length, window_ids=ids,
                windows_sha256=sha256(args.windows)),
            audit_sha256=sha256(args.audit), full_smoke_sha256=sha256(args.full_smoke),
            source_sha256=sha256(__file__), capture_source_sha256=sha256(inspect.getfile(_StreamingCovarianceCapture)),
            command=shlex.join(sys.argv), python=sys.executable, device_map=model.hf_device_map,
            dense_teacher=True, seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
