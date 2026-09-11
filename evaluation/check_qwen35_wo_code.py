"""Independent matrix and covariance checks of exported Wo artifacts."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from basisserve.core.qwen35_gdn_private_ag_runtime import Qwen35PrivateAGOutput
from evaluation.qwen35_hybrid_common import atomic_save, sha256


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20260909)
    root = Path('results/q35_hybrid')
    path = root / 'wo_dense/wo_bank.pt'
    bank = torch.load(path, weights_only=True, map_location='cpu')
    records = {r['layer_index']: r for family in ('gdn', 'full_attention') for r in bank[family]['layers']}
    index = json.loads((root / 'model/model.safetensors.index.json').read_text())['weight_map']
    rows = []
    for i in (0, 3, 15, 31):
        record = records[i]
        moment = torch.load(root / f'wo_dense_moments/heldout_l{i:02d}.pt', weights_only=True, map_location='cpu')
        family = 'linear_attn.out_proj' if record['layer_type'] == 'gdn' else 'self_attn.o_proj'
        key = f'model.language_model.layers.{i}.{family}.weight'
        with safe_open(root / 'model' / index[key], framework='pt', device='cpu') as f:
            original = f.get_tensor(key)
        assert torch.equal(original, moment['weight'])
        e = record['private_encoders'].to(a.device, torch.float64)
        d = record['joint_decoder_weight'].to(a.device, torch.float64)
        rank = e.shape[-1]
        # Independent construction: each contiguous input block contributes
        # x_s @ E_s @ decoder_s.T to the same output coordinates.
        folded = torch.cat([e[s] @ d[:, s*rank:(s+1)*rank].T for s in range(len(e))], 0).T
        w = original.to(a.device, torch.float64)
        c = moment['second_moment'].to(a.device, torch.float64)
        delta = folded - w
        mse = float(((delta @ c) * delta).sum() / ((w @ c) * w).sum())
        saved = record['metrics']['quantized_heldout_relative_output_mse']
        assert abs(mse - saved) <= 1e-8 * max(1.0, abs(saved))
        x = torch.randn(17, w.shape[1], device=a.device, dtype=torch.float64)
        operator = Qwen35PrivateAGOutput(e, d)
        expected = x @ folded.T
        actual = operator(x)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        # BF16 runtime includes an intermediate code rounding absent from the
        # exported-factor covariance metric. Quantify that difference explicitly.
        xb = x.to(torch.bfloat16)
        runtime = Qwen35PrivateAGOutput(e.to(torch.bfloat16), d.to(torch.bfloat16))(xb)
        reference = xb.double() @ folded.T
        rounding = float((runtime.double() - reference).square().sum() / reference.square().sum())
        row = {'layer': i, 'type': record['layer_type'], 'original_weight_exact_match': True,
               'saved_bf16_factor_heldout_relative_mse': saved, 'independent_heldout_relative_mse': mse,
               'fp64_runtime_fold_max_abs_difference': float((actual-expected).abs().max()),
               'bf16_runtime_rounding_relative_mse_on_random_inputs': rounding}
        rows.append(row)
        print(json.dumps(row), flush=True)
    atomic_save(a.output, {'wo_bank_sha256': sha256(path), 'rows': rows,
                          'scope': 'Four exported layers, saved heldout covariance and synthetic inputs; not whole-model generation validation.'})


if __name__ == '__main__':
    main()
