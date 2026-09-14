"""Replay a recorded nonfinite Mamba block with independent scan implementations."""
import argparse
from pathlib import Path
import sys

import torch
import causal_conv1d
import mamba_ssm
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.models.nemotron_h import modeling_nemotron_h as native
from mamba_ssm.ops.triton.ssd_combined import ssd_selective_scan

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, save_tensors, sha256


def stats(value):
    finite = torch.isfinite(value)
    result = dict(shape=list(value.shape), dtype=str(value.dtype),
        finite=bool(finite.all()), nonfinite=int((~finite).sum()))
    if finite.any():
        result.update(minimum=float(value[finite].min()), maximum=float(value[finite].max()))
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit', 'failure', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    configure()
    audit, failure = read_json(args.audit), read_json(args.failure)
    assert failure['audit_sha256'] == sha256(args.audit)
    assert failure['input_sha256'] == sha256(args.failure.with_suffix('.safetensors'))
    failed = failure['checks'][-1]
    assert failed['name'] == 'full' and failed['kind'] == 'linear_attention'
    layer = failed['layer']
    model_path = Path(audit['model'])
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    config._attn_implementation = 'sdpa'
    with torch.device('meta'):
        block = native.NemotronHBlock(config, layer)
    index = read_json(model_path / 'model.safetensors.index.json')['weight_map']
    state = {}
    prefix = f'model.layers.{layer}.'
    for source, target in audit['checkpoint_to_native_keys'].items():
        if target.startswith(prefix):
            with safe_open(str(model_path / index[source]), framework='pt', device='cpu') as handle:
                state[target[len(prefix):]] = handle.get_tensor(source).to(device='cuda', dtype=torch.bfloat16)
    loaded = block.load_state_dict(state, assign=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    block.eval()
    del state
    x = load_file(str(args.failure.with_suffix('.safetensors')))['hidden_states'].cuda()
    assert torch.isfinite(x).all()
    original = native.mamba2_chunk_scan
    reports, outputs = {}, {}
    for mode in ('native', 'fp32_chunk', 'selective_scan'):
        report = dict(modules={}, scans=[])

        def scan(hidden, dt, A, B, C, **kwargs):
            record = dict(inputs={name: stats(value) for name, value in
                dict(hidden=hidden, dt=dt, A=A, B=B, C=C, D=kwargs['D'], dt_bias=kwargs['dt_bias']).items()})
            if mode == 'native':
                result = original(hidden, dt, A, B, C, **kwargs)
            else:
                work = {k: v.float() if torch.is_tensor(v) else v for k, v in kwargs.items()}
                if mode == 'fp32_chunk':
                    result = original(hidden.float(), dt.float(), A.float(), B.float(), C.float(), **work)
                else:
                    assert not work['return_final_states'] and work['initial_states'] is None
                    work.pop('return_final_states')
                    work.pop('initial_states')
                    work.pop('chunk_size')
                    result = ssd_selective_scan(hidden.float(), dt.float(), A.float(), B.float(), C.float(), **work)
            record['output'] = stats(result)
            report['scans'].append(record)
            print('SCAN', mode, record, flush=True)
            return result

        native.mamba2_chunk_scan = scan
        handles = []
        for name, module in block.named_modules():
            if name:
                def hook(module, inputs, output, label=name):
                    if torch.is_tensor(output):
                        report['modules'][label] = stats(output)
                handles.append(module.register_forward_hook(hook))
        output = block(x)
        for handle in handles:
            handle.remove()
        report['output'] = stats(output)
        print('BLOCK', mode, report, flush=True)
        reports[mode] = report
        outputs[mode] = output.cpu()
        del output
    native.mamba2_chunk_scan = original
    for mode in ('native', 'fp32_chunk'):
        if reports[mode]['output']['finite'] and reports['selective_scan']['output']['finite']:
            reference = outputs['selective_scan'].float()
            reports[mode]['relative_rmse_vs_selective_scan'] = float(
                (outputs[mode].float() - reference).norm() / reference.norm())
    save_tensors(args.output.with_suffix('.safetensors'), outputs)
    write_json(args.output, dict(status='complete', layer=layer, reports=reports,
        audit_sha256=sha256(args.audit), failure_sha256=sha256(args.failure),
        source_sha256=sha256(__file__), python=sys.executable,
        outputs_sha256=sha256(args.output.with_suffix('.safetensors')),
        diagnostic_only=True, full_model_fix_validated=False))


if __name__ == '__main__':
    main()
