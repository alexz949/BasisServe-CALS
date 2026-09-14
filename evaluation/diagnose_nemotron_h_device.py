"""Compare native Mamba replay on the default and a non-default CUDA device."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, save_tensors, sha256
from evaluation.diagnose_nemotron_h_nonfinite import stats


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit', 'failure', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    configure()
    assert torch.cuda.device_count() == 2
    audit, failure = read_json(args.audit), read_json(args.failure)
    assert failure['audit_sha256'] == sha256(args.audit)
    assert failure['input_sha256'] == sha256(args.failure.with_suffix('.safetensors'))
    layer = failure['checks'][-1]['layer']
    path = Path(audit['model'])
    config = AutoConfig.from_pretrained(path, trust_remote_code=False, local_files_only=True)
    config._attn_implementation = 'sdpa'
    index = read_json(path / 'model.safetensors.index.json')['weight_map']
    with torch.device('meta'):
        block = native.NemotronHBlock(config, layer)
    prefix = f'model.layers.{layer}.'
    state = {}
    for source, target in audit['checkpoint_to_native_keys'].items():
        if target.startswith(prefix):
            with safe_open(str(path / index[source]), framework='pt', device='cpu') as handle:
                state[target[len(prefix):]] = handle.get_tensor(source).to('cuda:0', dtype=torch.bfloat16)
    loaded = block.load_state_dict(state, assign=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    block.eval()
    del state
    source = load_file(str(args.failure.with_suffix('.safetensors')))['hidden_states']
    outputs, reports = {}, {}
    # No per-module hooks or intermediate synchronizations: preserve the
    # native asynchronous execution that failed inside the dispatched model.
    for name, tensor_device, current_device in (
        ('gpu0_current0', 0, 0), ('gpu1_current0', 1, 0), ('gpu1_current1', 1, 1)):
        torch.cuda.set_device(0)
        block.to(f'cuda:{tensor_device}')
        x = source.to(f'cuda:{tensor_device}')
        torch.cuda.synchronize(tensor_device)
        with torch.cuda.device(current_device):
            output = block(x)
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)
            outputs[name] = output.cpu()
        report = stats(outputs[name])
        reports[name] = report
        if report['finite']:
            reference = outputs['gpu0_current0'].float()
            report['relative_rmse_vs_gpu0'] = float((outputs[name].float() - reference).norm() / reference.norm())
        print('DEVICE REPLAY', name, report, flush=True)
        del x, output
    save_tensors(args.output.with_suffix('.safetensors'), outputs)
    write_json(args.output, dict(status='complete', reports=reports, layer=layer,
        audit_sha256=sha256(args.audit), failure_sha256=sha256(args.failure),
        source_sha256=sha256(__file__), python=sys.executable,
        outputs_sha256=sha256(args.output.with_suffix('.safetensors')),
        diagnostic_only=True, full_model_fix_validated=False))


if __name__ == '__main__':
    main()
