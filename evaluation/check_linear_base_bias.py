"""Layer-0 K Base ablation: freeze linear weights and fit only an intercept."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from evaluation.v96kl_common import (
    MODEL, CHECKPOINT, CALIBRATION, BANK, checkpoint_manifest, configure,
    read_json, sha256, write_json, save_tensors,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (
    _rotary_embeddings, _pre_rope_rows, _post_rope_rows, _value_codes, _apply_base,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--linear-bank', type=Path, default=Path('results/checkpoints/v96kl_linear_b16r16'))
    parser.add_argument('--output', type=Path, default=Path('results/evaluation/linear_base_bias'))
    args = parser.parse_args()
    assert not (args.output / 'result.json').exists()
    configure()
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    manifest = checkpoint_manifest()
    wm = read_json(CALIBRATION / 'manifest.json')
    assert wm['sha256'] == sha256(CALIBRATION / 'windows.safetensors')
    assert wm['fit_ids'] == list(range(64)) and wm['validation_ids'] == list(range(64, 80))
    records, factors, hashes = {}, {}, {}
    for name, root in [('affine', BANK), ('linear', args.linear_bank)]:
        record = read_json(root / 'layer_000.json')
        assert record['status'] == 'complete' and record['layer'] == 0
        assert record['protocol']['windows_sha256'] == wm['sha256']
        assert record['protocol']['checkpoint_sha256'] == sha256(CHECKPOINT / 'manifest.json')
        hashes[name] = sha256(root / 'layer_000.safetensors')
        assert record['sha256'] == hashes[name]
        payload = load_file(str(root / 'layer_000.safetensors'))
        records[name] = record
        factors[name] = tuple(payload[f'base_{k}_b16'].cuda() for k in ('left', 'right', 'bias'))
    assert records['linear']['protocol']['base_fit_bias'] is False
    assert torch.count_nonzero(factors['linear'][2]) == 0
    positions = records['affine']['query_selection']['selected_positions']
    assert positions == records['linear']['query_selection']['selected_positions']
    windows = load_file(str(CALIBRATION / 'windows.safetensors'))['input_ids']
    assert windows.shape == (80, 32768)
    encoder = load_file(str(CHECKPOINT / manifest['layers'][0]['file']))['value_coordinate_encoders'].float().cuda()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').eval()
    embedding = model.model.embed_tokens.cuda()
    layer = model.model.layers[0].cuda()
    cos, sin = _rotary_embeddings(MODEL, sequence=32768, device=torch.device('cuda'))
    rows = torch.empty(80, 32768, 8, 256, dtype=torch.bfloat16)
    # Layer 0's K/V depend only on embeddings and input normalization, so no
    # attention/MLP execution is required to reproduce its calibration operands.
    for i in range(80):
        x = layer.input_layernorm(embedding(windows[i:i+1].long().cuda()))
        attn = layer.self_attn
        k = attn.k_norm(attn.k_proj(x).view(1, 32768, 8, 128)).transpose(1, 2)
        _, post = apply_rotary_pos_emb(k, k, cos.bfloat16(), sin.bfloat16())
        v = attn.v_proj(x).view(1, 32768, 8, 128)
        rows[i].copy_(torch.cat([v[0], post[0].transpose(0, 1)], -1).cpu())
        if (i+1) % 8 == 0:
            print(json.dumps({'capture': i+1, 'total': 80}), flush=True)
    embedding.cpu()
    layer.cpu()
    del x, k, post, v, model
    torch.cuda.empty_cache()
    input_sum = torch.zeros(8, encoder.shape[-1], dtype=torch.float64, device='cuda')
    target_sum = torch.zeros(8, 128, dtype=torch.float64, device='cuda')
    for i in range(64):
        current = rows[i].cuda().float()
        codes = _value_codes(current[..., :128], encoder)
        pre = _pre_rope_rows(current[..., 128:], cos, sin)
        input_sum += codes.sum(0).double()
        target_sum += pre.sum(0).double()
    left, right, _ = factors['linear']
    count = 64 * 32768
    bias = (target_sum/count - torch.einsum('gv,gvr,grd->gd', input_sum/count,
                                          left.double(), right.double())).float()
    factors['linear_plus_bias'] = (left, right, bias)
    totals = {split: {name: {str(p): {'error': 0., 'energy': 0.} for p in positions}
                      for name in factors} for split in ('fit', 'validation')}
    for i in range(80):
        split = 'fit' if i < 64 else 'validation'
        current = rows[i].cuda().float()
        exact = current[..., 128:]
        codes = _value_codes(current[..., :128], encoder)
        for name, factor in factors.items():
            predicted = _post_rope_rows(_apply_base(codes, factor), cos, sin)
            for p in positions:
                row = totals[split][name][str(p)]
                row['error'] += float((exact[:p+1]-predicted[:p+1]).square().sum())
                row['energy'] += float(exact[:p+1].square().sum())
        if (i+1) % 8 == 0:
            print(json.dumps({'evaluate': i+1, 'total': 80}), flush=True)
    for split, arms in totals.items():
        for name, values in arms.items():
            for p, row in values.items():
                row['relative_mse'] = row['error']/row['energy']
                if name in records:
                    reference = records[name]['reconstruction'][split][p]['16']
                    assert abs(row['relative_mse']-reference['post_rope_relative_mse']) <= 1e-7 + 1e-5*reference['post_rope_relative_mse']
                    assert abs(row['energy']/reference['exact_key_squared_energy']-1) < 1e-5
    for name, root in [('affine', BANK), ('linear', args.linear_bank)]:
        assert hashes[name] == sha256(root / 'layer_000.safetensors')
    # Save Base only: old residual factors are not valid for this changed Base.
    save_tensors(args.output / 'base_only.safetensors', {
        'base_left_b16': left.cpu(), 'base_right_b16': right.cpu(), 'base_bias_b16': bias.cpu()})
    result = dict(status='complete_and_audited', layer=0, fit_bias_windows=list(range(64)),
        validation_windows=list(range(64,80)), frozen_linear_weights=True,
        original_arms_reproduced=True, residual_refitted=False, command=sys.argv,
        python=sys.executable, gpu=torch.cuda.get_device_name(0), source_hashes=hashes,
        windows_sha256=wm['sha256'], query_positions=positions, metrics=totals,
        elapsed_seconds=time.monotonic()-started)
    write_json(args.output / 'result.json', result)
    print(json.dumps({'status': result['status'], 'last_prefix': {
        split: {name: values[str(max(positions))] for name, values in arms.items()}
        for split, arms in totals.items()}}), flush=True)


if __name__ == '__main__':
    main()
