"""Matched GQA-shared V96 independent versus joint objective quality ablation."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basisserve.core.gqa_vo_svdllm import GQAVOLayout
from basisserve.core.joint_aa_gqa_o import (
    factorize_joint_aa_gqa_o, materialize_structured_encoder,
    resolve_head_to_kv_group,
)
from basisserve.core.tp_source_svd import output_losses
from basisserve.core.tp_source_wo_c1 import TPSourceWOLayout
from evaluation.run_tp_source_svd import dense_weight, validate_inputs, write_json


def independent_covariance(covariance, source_width):
    assert covariance.shape[0] == covariance.shape[1]
    assert covariance.shape[0] % source_width == 0
    result = torch.zeros_like(covariance)
    for start in range(0, covariance.shape[0], source_width):
        section = slice(start, start + source_width)
        result[section, section] = covariance[section, section]
    return result


def fit_pair(weight, covariance, layout, rows, iterations):
    mapping = resolve_head_to_kv_group(layout)
    width = layout.query_heads_per_kv_group * layout.head_dim
    local_cov = independent_covariance(covariance, width)
    loss_layout = TPSourceWOLayout(layout.query_width, layout.hidden_size,
                                  layout.num_key_value_heads,
                                  layout.query_heads_per_kv_group * layout.rank)
    answers = {}
    for arm, metric in [('independent', local_cov), ('joint', covariance)]:
        candidates = []
        for init in ('pooled_activation_svd', 'random_orthogonal'):
            fit = factorize_joint_aa_gqa_o(
                layout=layout, o_weight_pt=weight, covariance=metric, init=init,
                outer_iters=iterations, max_backtracks=20, relative_improve_tol=1e-7,
                patience=3, decoder_ridge=0, covariance_ridge=0,
                seed=1234, work_dtype=torch.float64, work_device=weight.device,
                output_dtype=torch.float64,
            )
            assert all(item.ridge == 0 for item in fit.iteration_history)
            assert fit.decoder_diagnostics.ridge == 0
            encoder = fit.E_unique.to(weight.device)
            decoder = fit.D_row.to(weight.device)
            structured = materialize_structured_encoder(encoder, mapping)
            replacement = (structured @ decoder).T.contiguous()
            losses = output_losses(weight, replacement, covariance, loss_layout, rows)
            score = losses['local_per_row' if arm == 'independent' else 'final_per_row']
            assert abs(score - fit.final_metrics.data_loss) <= 1e-8 * max(1, abs(score))
            candidates.append((score, encoder, decoder, replacement, {
                'initialization': init, 'objective': score,
                'initial_objective': fit.init_metrics.data_loss,
                'history': [asdict(item) for item in fit.iteration_history],
                'budget_exhausted': fit.iteration_history[-1].iteration == iterations,
                'losses_fp64': losses,
            }))
        best = min(candidates, key=lambda item: item[0])
        exported = best[3].bfloat16()
        assert torch.isfinite(exported).all()
        answers[arm] = ({'encoder_fp64': best[1].cpu(), 'decoder_fp64': best[2].cpu(),
                         'materialized_weight_bf16': exported.cpu()}, {
            'selected_initialization': best[4]['initialization'],
            'starts': [item[4] for item in candidates],
            'losses_bf16': output_losses(weight, exported.double(), covariance, loss_layout, rows),
            'global_optimum_certified': False,
        })
    return answers


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.eval_attention_o_proj_collective_ppl import _eval_ppl_fp32_loss
    from scripts.eval_svdllm_safetensors_ppl_accelerate import _token_ids, WIKITEXT_REVISION

    _, manifest, identity = validate_inputs(args.model, args.covariance, 'formal')
    layers = [0] if args.phase == 'smoke' else list(range(36))
    iterations = 2 if args.phase == 'smoke' else 50
    layout = GQAVOLayout(4096, 32, 8, 128, 96)
    config_path = args.output / 'config.json'
    assert not config_path.exists(), 'Preserve completed or partial artifacts'
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        'phase': args.phase, 'model_identity': identity, 'layout': asdict(layout),
        'layers': layers, 'iterations': iterations, 'command': shlex.join(sys.argv),
        'environment': os.environ.get('CONDA_DEFAULT_ENV'),
        'calibration': manifest['calibration'], 'covariance': str(args.covariance),
        'ridge': 0, 'seed': 1234, 'initializations': ['pooled_activation_svd', 'random_orthogonal'],
        'quality_execution': 'FP64 structured E@D materialized once as BF16 o_proj',
        'scope': 'shared V96 representable structure; not compressed-cache runtime or speed',
        'global_optimum_certified': False,
    }
    write_json(config_path, config)
    metrics = {}
    for layer in layers:
        tensor = load_file(str(args.covariance / manifest['artifacts'][str(layer)]['file']))
        assert torch.equal(tensor['weight'], dense_weight(args.model, layer))
        answers = fit_pair(tensor['weight'].to('cuda', torch.float64),
                           tensor['fit_covariance'].to('cuda', torch.float64),
                           layout, manifest['calibration']['fit_rows'], iterations)
        metrics[str(layer)] = {}
        for arm, (factors, audit) in answers.items():
            folder = args.output / arm
            folder.mkdir(exist_ok=True)
            save_file(factors, str(folder / f'layer_{layer:03d}.safetensors'),
                      metadata={'model_identity': json.dumps(identity), 'layout': json.dumps(asdict(layout)),
                                'arm': arm, 'layer': str(layer)})
            metrics[str(layer)][arm] = audit
        write_json(args.output / 'fit.json', metrics)
        print(f'FIT layer={layer} complete', flush=True)
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = _token_ids(tokenizer, 'wikitext2', 'test', None).long().flatten()
    save_file({'input_ids': tokens}, str(args.output / 'ppl_tokens.safetensors'))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True).to('cuda').eval()
    originals = [layer.self_attn.o_proj.weight.detach().cpu().clone() for layer in model.model.layers]
    results = {'dataset_revision': WIKITEXT_REVISION, 'arms': {}}
    with torch.inference_mode():
        for arm in ('dense', 'independent', 'joint'):
            for module, original in zip(model.model.layers, originals):
                module.self_attn.o_proj.weight.copy_(original)
            if arm != 'dense':
                for layer in layers:
                    value = load_file(str(args.output / arm / f'layer_{layer:03d}.safetensors'))
                    model.model.layers[layer].self_attn.o_proj.weight.copy_(value['materialized_weight_bf16'])
            result = _eval_ppl_fp32_loss(
                model, tokenizer, dataset='wikitext2', split='test', seqlen=2048,
                batch_size=1, max_samples=2 if args.phase == 'smoke' else None,
                max_tokens=None, input_ids=tokens)
            results['arms'][arm] = result
            write_json(args.output / 'ppl.json', results)
            print(json.dumps({'arm': arm, **result}), flush=True)
    lines = ['# V96 Objective Ablation', '', f'Phase: {args.phase}. Environment: basis.',
             'Same shared V96 encoder structure, two starts and optimizer budget in both arms.',
             'Independent retains full within-source covariance; joint includes cross-source terms.',
             'Iterative fits, not certified global optima. No held-out calibration set.',
             'Materialized BF16 o_proj quality execution, not a compressed KV runtime benchmark.',
             'Smoke compresses only layer 0 and uses two WT2 windows; not formal quality.', '',
             '| Arm | PPL | Scored tokens |', '| --- | ---: | ---: |']
    for arm, result in results['arms'].items():
        lines.append(f"| {arm} | {result['ppl']:.8f} | {result['tokens']} |")
    lines += ['', 'Command:', '```bash', shlex.join(sys.argv), '```',
              'See fit.json for per-start convergence history and both local/full-output losses.']
    (args.output / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'formal'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--covariance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    success = False
    write_json(args.output / 'status.json', {'status': 'running'})
    try:
        run(args)
        success = True
    finally:
        write_json(args.output / 'status.json', {'status': 'complete' if success else 'failed'})


if __name__ == '__main__':
    main()
