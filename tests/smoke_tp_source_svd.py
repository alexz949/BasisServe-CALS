"""Synthetic Qwen SDPA integration smoke; not pretrained-model or WT2 quality."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basisserve.core.tp_source_svd import fit_layer
from basisserve.core.tp_source_wo_c1 import TPSourceWOLayout
from evaluation.run_tp_source_svd import FORMAT, install, write_json
from evaluation.eval_attention_o_proj_collective_ppl import _eval_ppl_fp32_loss


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve previous smoke'
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(2718)
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=96,
                        num_hidden_layers=2, num_attention_heads=16,
                        num_key_value_heads=4, head_dim=4, max_position_embeddings=128)
    config._attn_implementation = 'sdpa'
    model = Qwen3ForCausalLM(config).to(device=args.device, dtype=torch.bfloat16).eval()
    originals = [layer.self_attn.o_proj.weight.detach().cpu().clone() for layer in model.model.layers]
    captured = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        def hook(module, inputs, index=index):
            captured[index] = inputs[0].detach().reshape(-1, 64).double()
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(hook))
    calibration = torch.randint(0, 128, (5, 32), device=args.device)
    model(input_ids=calibration, use_cache=False)
    for handle in handles:
        handle.remove()
    output_config = {'layers': [0, 1], 'tp_size': 8,
                     'model_identity': {'repo': 'synthetic/random-Qwen3-tiny', 'revision': 'seed2718'}}
    metrics = {'status': 'running', 'scope': __doc__, 'command': shlex.join(sys.argv),
               'environment': os.environ.get('CONDA_DEFAULT_ENV'), 'layers': [], 'arms': {}}
    write_json(args.output / 'smoke.json', metrics)
    started = time.perf_counter()
    for index, layer in enumerate(model.model.layers):
        x = captured[index]
        fit_x, held_x = x[:128], x[128:]
        for rank in (4, 6):
            layout = TPSourceWOLayout(64, 64, 8, rank)
            factors, audit = fit_layer(layer.self_attn.o_proj.weight.double(),
                                      fit_x.T @ fit_x / len(fit_x), held_x.T @ held_x / len(held_x),
                                      layout, fit_rows=len(fit_x), heldout_rows=len(held_x))
            path = args.output / 'factors' / f'r{rank}' / f'layer_{index:03d}.safetensors'
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file({k: v.cpu().contiguous() for k, v in factors.items()}, str(path),
                      metadata={'format': FORMAT, 'layout': json.dumps(asdict(layout)),
                                'model_identity': json.dumps(output_config['model_identity']), 'layer': str(index)})
            metrics['layers'].append({'layer': index, 'rank': rank, **audit})
    tokens = torch.randint(0, 128, (103,), dtype=torch.long)
    for rank in (None, 4, 6, None):
        install(model, originals, args.output, rank, output_config)
        result = _eval_ppl_fp32_loss(model, None, dataset='synthetic_random_tokens', split='smoke',
                                    seqlen=32, batch_size=1, max_samples=None,
                                    max_tokens=None, input_ids=tokens)
        arm = 'dense' if rank is None else f'r{rank}'
        if arm in metrics['arms']:
            assert result == metrics['arms'][arm], 'Dense restoration changed evaluation'
        metrics['arms'][arm] = result
    assert all(r['tokens'] == 93 and r['discarded_tail_tokens'] == 7 for r in metrics['arms'].values())
    metrics.update(status='complete', elapsed_seconds=time.perf_counter() - started,
                   dense_restore_verified=True, real_model_ppl=False)
    write_json(args.output / 'smoke.json', metrics)
    print(json.dumps({'status': 'complete', 'real_model_ppl': False,
                      'layers': len(metrics['layers']), 'arms': list(metrics['arms']),
                      'dense_restore_verified': True}), flush=True)


if __name__ == '__main__':
    main()
