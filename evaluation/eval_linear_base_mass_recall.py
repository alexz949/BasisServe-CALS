"""Compare layer-0 Base-only routing with frozen affine/linear/bias factors."""
import argparse
import math
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from evaluation.v96kl_common import (
    MODEL, CHECKPOINT, CALIBRATION, BANK, configure, checkpoint_manifest,
    read_json, write_json, sha256,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (
    _rotary_embeddings, _value_codes, _apply_base, _post_rope_rows,
)
from evaluation.eval_qwen3_8b_v80_base16_page32_sink_diagnostics import (
    _group_page_scores, _fixed_budget_page_mask, _token_mask,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--linear-bank', type=Path, default=Path('results/checkpoints/v96kl_linear_b16r16'))
    parser.add_argument('--bias-result', type=Path, default=Path('results/evaluation/linear_base_bias'))
    parser.add_argument('--token-budgets', type=int, nargs='+', default=[1024, 2048, 4096])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert all(b > 32 and b % 32 == 0 for b in args.token_budgets)
    configure()
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    manifest = checkpoint_manifest()
    wm = read_json(CALIBRATION / 'manifest.json')
    bias_record = read_json(args.bias_result / 'result.json')
    assert wm['sha256'] == sha256(CALIBRATION / 'windows.safetensors') == bias_record['windows_sha256']
    assert wm['validation_ids'] == list(range(64, 80))
    positions = bias_record['query_positions']
    factors, hashes = {}, {}
    paths = dict(affine=BANK / 'layer_000.safetensors',
                 linear=args.linear_bank / 'layer_000.safetensors',
                 linear_plus_bias=args.bias_result / 'base_only.safetensors')
    for name, path in paths.items():
        hashes[name] = sha256(path)
        if name != 'linear_plus_bias':
            assert hashes[name] == bias_record['source_hashes'][name]
        tensors = load_file(str(path))
        factors[name] = tuple(tensors[f'base_{k}_b16'].cuda() for k in ('left', 'right', 'bias'))
    assert all(torch.equal(factors['linear'][i], factors['linear_plus_bias'][i]) for i in (0, 1))
    assert torch.count_nonzero(factors['linear'][2]) == 0
    encoder = load_file(str(CHECKPOINT / manifest['layers'][0]['file']))['value_coordinate_encoders'].float().cuda()
    windows = load_file(str(CALIBRATION / 'windows.safetensors'))['input_ids']
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').eval()
    embedding = model.model.embed_tokens.cuda()
    layer = model.model.layers[0].cuda()
    attn = layer.self_attn
    cos, sin = _rotary_embeddings(MODEL, sequence=32768, device=torch.device('cuda'))
    rows = []
    for doc in wm['validation_ids']:
        x = layer.input_layernorm(embedding(windows[doc:doc+1].long().cuda()))
        q = attn.q_norm(attn.q_proj(x).view(1, 32768, 32, 128)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(1, 32768, 8, 128)).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos.bfloat16(), sin.bfloat16())
        codes = _value_codes(attn.v_proj(x).view(32768, 8, 128).float(), encoder)
        predictions = {name: _post_rope_rows(_apply_base(codes, f), cos, sin) for name, f in factors.items()}
        for p in positions:
            query = q[0, :, p].float().reshape(8, 4, 128)
            exact = torch.einsum('ghd,gtd->ght', query, k[0, :, :p+1].float()).reshape(32, p+1) / math.sqrt(128)
            probabilities = exact.softmax(-1)
            non_sink = exact[:, 32:].softmax(-1)
            scores = {'exact_score_reference': exact}
            scores.update({name: torch.einsum('ghd,tgd->ght', query, pred[:p+1]).reshape(32, p+1) / math.sqrt(128)
                           for name, pred in predictions.items()})
            for name, score in scores.items():
                group_scores = _group_page_scores(score, num_kv_heads=8, page_size=32, excluded_prefix_pages=1)
                for budget in args.token_budgets:
                    pages = _fixed_budget_page_mask(group_scores, page_budget=budget//32, forced_prefix_pages=1)
                    mask = _token_mask(pages, page_size=32, tokens=p+1).repeat_interleave(4, 0)
                    mass = (probabilities * mask).sum(-1)
                    nonsink_mass = (non_sink * mask[:, 32:]).sum(-1)
                    assert torch.isfinite(mass).all() and torch.isfinite(nonsink_mass).all()
                    rows.append(dict(document=doc, position=p, arm=name, token_budget=budget,
                                     mass=mass.tolist(), non_sink_mass=nonsink_mass.tolist(),
                                     selected_tokens_mean=float(mask.sum(-1).float().mean())))
        print(f'validation document {doc-63}/16 complete', flush=True)
    summary = {}
    for scope in ('all_queries', 'longest_prefix'):
        summary[scope] = {}
        for name in scores:
            summary[scope][name] = {}
            for budget in args.token_budgets:
                subset = [r for r in rows if r['arm'] == name and r['token_budget'] == budget
                          and (scope == 'all_queries' or r['position'] == max(positions))]
                metrics = {}
                for key in ('mass', 'non_sink_mass'):
                    values = torch.tensor([v for r in subset for v in r[key]], dtype=torch.float64)
                    metrics[key] = dict(mean=float(values.mean()), minimum=float(values.min()),
                                        p05=float(values.quantile(.05)), count=values.numel())
                summary[scope][name][str(budget)] = metrics
    assert all(sha256(path) == hashes[name] for name, path in paths.items())
    write_json(args.output, dict(status='complete', layer=0, base_only=True, residual_refitted=False,
        validation_ids=wm['validation_ids'], query_positions=positions, page_size=32, pinned_prefix_pages=1,
        selection='non-sink conditional per-head page mass, max across GQA heads, fixed page budget including sink',
        reference='exact scores with the same GQA group-max selection; not a per-head optimal upper bound',
        windows_sha256=wm['sha256'], factor_hashes=hashes, command=sys.argv,
        python=sys.executable, gpu=torch.cuda.get_device_name(0), elapsed_seconds=time.monotonic()-started,
        summary=summary, observations=rows))
    print(summary['longest_prefix'], flush=True)


if __name__ == '__main__':
    main()
