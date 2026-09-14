"""Real-model BF16 compact-V execution checks with native output projections."""

import argparse
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file

from basisserve.core.qwen35_gated_v_runtime import GatedVRuntime
from evaluation.qwen35_hybrid_common import atomic_save, full_layers, load_model, sha256


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokens = load_file(str(args.calibration/'windows.safetensors'))['input_ids'][:1].cuda()
    assert tokens.shape == (1, 32768)
    model = load_model(str(args.model), 'cuda:0')
    projections = {name: module.weight for name, module in model.named_modules()
        if name.endswith(('.o_proj', '.out_proj'))}
    assert len(projections) == 32
    before = {name: (weight.data_ptr(), weight._version) for name, weight in projections.items()}
    native_full = model.model(tokens[:, :64], use_cache=False).last_hidden_state[:, -1:]
    native_prefix = model.model(tokens[:, :63], use_cache=True)
    native_last = model.model(tokens[:, 63:64], past_key_values=native_prefix.past_key_values,
        use_cache=True).last_hidden_state
    native_error = native_last.float()-native_full.float()
    native_max = native_error.abs().max().item()
    native_relative = (native_error.norm()/native_full.float().norm()).item()
    print(json.dumps(dict(native_cached_max_abs_error=native_max,
        native_cached_relative_l2=native_relative)), flush=True)
    del native_full, native_prefix, native_last, native_error
    bank = {}
    for layer in full_layers(model):
        encoder = torch.linalg.qr(torch.randn(4, 256, 192, device='cuda:0')).Q.bfloat16()
        bank[layer] = dict(E_V=encoder, R_V=encoder.mT.contiguous())
    started = time.monotonic()
    with GatedVRuntime(model, bank):
        reference = model.model(tokens[:, :64], use_cache=False).last_hidden_state
        prefix = model.model(tokens[:, :63], use_cache=True)
        cache = prefix.past_key_values
        last = model.model(tokens[:, 63:64], past_key_values=cache, use_cache=True).last_hidden_state
        error = (last.float()-reference[:, -1:].float()).abs()
        relative = (error.norm()/reference[:, -1:].float().norm()).item()
        print(json.dumps(dict(compact_cached_max_abs_error=error.max().item(),
            compact_cached_relative_l2=relative)), flush=True)
        assert relative <= 0.03 and relative <= 2*native_relative
        for layer in bank:
            assert cache.layers[layer].values.shape == (1, 4, 64, 192)
            assert cache.layers[layer].keys.shape == (1, 4, 64, 256)
        print(json.dumps(dict(cached_decode_passed=True, max_abs_error=error.max().item())), flush=True)
        del prefix, cache, last, reference
        torch.cuda.reset_peak_memory_stats()
        output = model.model(tokens, use_cache=False).last_hidden_state
        assert output.shape == (1, 32768, 4096) and torch.isfinite(output).all()
        assert all(module.last_attention_backend == 'flash' for module in full_layers(model).values())
    after = {name: (module.weight.data_ptr(), module.weight._version)
        for name, module in model.named_modules() if name in projections}
    assert after == before
    report = dict(status='complete', purpose='execution only; synthetic orthogonal V192 factors',
        prefill_backend='flash',
        wo_compression=False, unchanged_output_projections=len(before), length=32768,
        cached_decode_max_abs_error=error.max().item(), cached_decode_relative_l2=relative,
        native_cached_max_abs_error=native_max, native_cached_relative_l2=native_relative,
        criterion='relative L2 <= 0.03 and <= 2x native BF16 segmented-forward difference',
        peak_gib=torch.cuda.max_memory_allocated()/2**30, seconds=time.monotonic()-started,
        windows_sha256=sha256(args.calibration/'windows.safetensors'),
        model_config_sha256=sha256(args.model/'config.json'),
        runtime_sha256=sha256(Path('basisserve/core/qwen35_gated_v_runtime.py')))
    atomic_save(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
