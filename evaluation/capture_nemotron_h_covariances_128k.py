"""Collect dense-teacher attention and Mamba Wo covariances at 128K in one single-GPU model pass.

Same artifact and manifest layout as ``capture_nemotron_h_covariances.py`` (attention ``o_proj`` input and
Mamba ``out_proj`` input, normalized FP32 sufficient statistics), but for the 128K calibration protocol:
window ids come from the window bank manifest (``fit_ids`` + ``validation_ids``), the model lives on one
GPU, and the Mamba chunk scan runs through the Triton drop-in because this host has no ``mamba_ssm``.
"""
import argparse
import inspect
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.models.nemotron_h import modeling_nemotron_h as native

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, save_tensors, sha256
from evaluation.capture_attention_o_proj_covariances import _StreamingCovarianceCapture, FORMAT
from basisserve.core.nemotron_h_c1 import discover_nemotron_h_c1_targets, projection_for_target
from evaluation import nemotron_h_triton_mamba as triton_mamba


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--windows', type=Path, required=True, help='window bank directory (windows.safetensors + manifest.json)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', type=str)
    args = parser.parse_args()
    configure()
    audit = read_json(args.audit)
    assert audit['status'] == 'complete'
    model_path = Path(audit['model'])
    assert sha256(model_path / 'config.json') == audit['config_sha256']
    assert sha256(model_path / 'model.safetensors.index.json') == audit['index_sha256']
    assert sha256(inspect.getfile(native.NemotronHForCausalLM)) == audit['implementation_sha256']
    windows_manifest = read_json(args.windows / 'manifest.json')
    windows_path = args.windows / 'windows.safetensors'
    assert windows_manifest['status'] == 'complete' and windows_manifest['sha256'] == sha256(windows_path)
    assert windows_manifest['model_config_sha256'] == audit['config_sha256']
    fit_ids, validation_ids = windows_manifest['fit_ids'], windows_manifest['validation_ids']
    ids = fit_ids + validation_ids
    assert ids == list(range(len(ids))) and fit_ids and validation_ids
    sequence_length = int(windows_manifest['shape'][1])
    triton_mamba.install()
    implementations = dict(
        mamba2_chunk_scan='vllm.model_executor.layers.mamba.ops.ssd_combined.mamba_chunk_scan_combined_varlen (Triton, via evaluation/nemotron_h_triton_mamba.py)',
        mamba2_selective_state_update='transformers torch path', causal_conv1d_fn='transformers torch path',
        causal_conv1d_update='transformers torch path')
    protocol = dict(audit_sha256=sha256(args.audit), windows_sha256=sha256(windows_path),
        windows_manifest_sha256=sha256(args.windows / 'manifest.json'), window_ids=ids,
        sequence_length=sequence_length, selected_layers=args.layers, source_sha256=sha256(__file__),
        triton_dropin_sha256=sha256(ROOT / 'evaluation/nemotron_h_triton_mamba.py'),
        capture_source_sha256=sha256(inspect.getfile(_StreamingCovarianceCapture)),
        implementations=implementations, teacher='native dense BF16 SDPA, single GPU', covariance_dtype='float32',
        dt_limit=list(map(str, triton_mamba.DT_LIMIT)), dt_limit_note='transformers time_step_min clamp reset to the original (0, inf) on every Mamba mixer')
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
    windows = load_file(str(windows_path))['input_ids'][ids]
    assert windows.shape == (len(ids), sequence_length)
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=False, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    assert type(model) is native.NemotronHForCausalLM
    triton_mamba.restore_dt_limit(model)
    targets = discover_nemotron_h_c1_targets(model)
    config = model.config
    head_dim = int(getattr(config, 'attention_head_dim', 0) or getattr(config, 'head_dim', 0)
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
    for split, indices in [('fit', range(len(fit_ids))), ('heldout', range(len(fit_ids), len(ids)))]:
        for offset in indices:
            tokens = windows[offset:offset + 1].long().cuda()
            for capture in groups.values():
                capture.begin(split, rows=sequence_length)
            output = model.model(input_ids=tokens, use_cache=False)
            assert torch.isfinite(output.last_hidden_state).all()
            del output
            for capture in groups.values():
                capture.finish()
            print('DENSE COVARIANCE', split, 'window', ids[offset], f'{time.monotonic() - started:.0f}s', flush=True)
        for capture in groups.values():
            capture.normalize_and_offload(split, expected_rows=len(indices) * sequence_length)
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
            format=FORMAT if kind == 'full_attention' else 'basisserve.nemotron_h.wo_covariances.v1',
            protocol=protocol, layer_kind=kind, layers=list(capture.modules), artifacts=artifacts,
            model=dict(path=str(model_path), config_sha256=audit['config_sha256'], model_type='nemotron_h',
                attention_type='gqa', hidden_size=config.hidden_size, num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads, num_key_value_heads=config.num_key_value_heads,
                head_dim=head_dim),
            calibration=dict(storage='normalized_covariance_sufficient_statistics', window_count=len(ids),
                fit_windows=len(fit_ids), heldout_windows=len(validation_ids), sequence_length=sequence_length,
                positions_per_window=sequence_length, fit_rows=len(fit_ids) * sequence_length,
                heldout_rows=len(validation_ids) * sequence_length, rows_per_layer=len(ids) * sequence_length,
                window_ids=ids, windows_sha256=sha256(windows_path), windows_manifest=str(args.windows / 'manifest.json'),
                windows_seed=windows_manifest.get('seed'), windows_condition=windows_manifest.get('condition')),
            audit_sha256=sha256(args.audit), source_sha256=sha256(__file__),
            capture_source_sha256=sha256(inspect.getfile(_StreamingCovarianceCapture)),
            command=shlex.join(sys.argv), python=sys.executable, device='cuda:0 single GPU',
            dense_teacher=True, seconds=time.monotonic() - started))


if __name__ == '__main__':
    main()
