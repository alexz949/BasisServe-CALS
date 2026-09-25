"""Qwen3.5-9B 128K K-routing fitter on the deployed V192 + GDN-Wo75 model (Score-MSE or Page-Fisher residual objective).

Two window-sharded replays, no raw capture. The uniform V192 bank (GatedVAttention) and the GDN Wo75 Private-AllGather
bank are installed before every replay so the router sees the deployed Q/K/V distribution; the dense v_proj weights are
kept only for the Base moments, which map V codes to pre-RoPE K. Qwen3.5 rotates the first 64 of 256 key dimensions
(partial_rotary_factor 0.25, interleaved mRoPE reduced to text positions); the Base prediction is rotated with the same
prefix rule before the residual K - Base(V) enters the unweighted causal score Grams.

Stages: moments (per window shard) -> merge (sum shards, select query positions) -> base -> fisher (per window shard) -> fit.
The base and fit stages are the unchanged implementations from evaluation/fit_k_routing_streaming.py.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from basisserve.core.c1_v_k_index import apply_rotary
from basisserve.core.gqa_joint_routing_payload_s80_fisher import pack_symmetric_fisher_grams
from basisserve.core.query_position_sampling import candidate_positions, select_stratified_query_positions
from basisserve.core.qwen35_gated_v_runtime import GatedVAttention, GatedVRuntime
from basisserve.core.qwen35_gdn_private_ag_runtime import Qwen35PrivateAGRuntime, load_qwen35_gdn_private_ag_factors
from evaluation.fit_k_routing_streaming import fit_bases, fit_residuals, layer_file, restore_base, save_record, verified
from evaluation.qwen35_hybrid_common import load_bank, load_model, verify_model_identity
from evaluation.fit_qwen3_8b_q8_fisher_residual import build_multi_query_statistics
from evaluation.streaming_k_statistics import RawBaseMoments, packed_fisher
from evaluation.v96kl_common import configure, read_json, sha256

PAGE_SIZE = 32
RECENT_TOKENS = 64
ROTARY_DIM = 64
SOURCES = ('evaluation/fit_qwen35_128k_k_routing.py', 'evaluation/fit_k_routing_streaming.py',
           'evaluation/streaming_k_statistics.py', 'basisserve/core/qwen35_gated_v_runtime.py',
           'basisserve/core/qwen35_gdn_private_ag_runtime.py', 'basisserve/core/qwen35_k_routing_runtime.py',
           'basisserve/core/c1_v_k_index.py', 'basisserve/core/c1_v_conditional_k_router.py',
           'basisserve/core/query_position_sampling.py')


def score_mse_statistics(q, v, k_post, positions, encoder, base_payload, cos, sin, *, excluded_prefix_tokens,
                         excluded_recent_tokens, heads, groups, dim):
    """Unweighted Gram of the Base residual K - RoPE(Base(V)) over the routable causal prefix of every selected query,
    packed in the page-Fisher payload layout so the same ALS solver fits the residual factors."""
    codes = torch.einsum('bngd,gdr->bgnr', v.float(), encoder.to(device=v.device, dtype=torch.float32))
    base = restore_base(base_payload)[base_payload['left'].shape[-1]]
    left = torch.stack([m.left for m in base]).to(device=v.device, dtype=torch.float32)
    right = torch.stack([m.right for m in base]).to(device=v.device, dtype=torch.float32)
    bias = torch.stack([m.bias for m in base]).to(device=v.device, dtype=torch.float32)
    base_pre = torch.einsum('bgnr,grk,gkd->bgnd', codes, left, right) + bias[None, :, None, :]
    predicted = apply_rotary(base_pre, cos, sin)
    residual = (k_post.float() - predicted)[0]
    running = torch.zeros(groups, dim, dim, device=residual.device, dtype=torch.float32)
    cursor = excluded_prefix_tokens
    by = {}
    for position in sorted(int(x) for x in positions):
        stop = max(position + 1 - excluded_recent_tokens, cursor)
        if stop > cursor:
            rows = residual[:, cursor:stop]
            running.add_(torch.einsum('gtd,gte->gde', rows, rows))
            cursor = stop
        by[position] = running.clone()
    grams = torch.stack([by[int(x)] for x in positions], 1).index_select(
        0, torch.arange(heads, device=residual.device) // (heads // groups))
    queries = q[0, :, positions].float()
    energy = 0.5 * dim ** -1 * float(torch.einsum('hqd,hqde,hqe->', queries, grams, queries))
    payload = dict(queries=queries.cpu(), packed=pack_symmetric_fisher_grams(grams).cpu(),
                   teacher_energy=torch.tensor(energy, dtype=torch.float64))
    return payload, dict(objective='score_mse', score_energy=energy, queries=len(positions),
                         base_rotation='prefix RoPE on the first 64 dims via apply_rotary; fp32 residual')


def shard_windows(ids, shard, shards):
    return [int(x) for x in ids[shard::shards]]


def part_file(root, layer, shard):
    return root/'moments_parts'/f'layer_{layer:03d}_s{shard}.safetensors'


def load_deployed_model(identity, layers):
    model = load_model(identity['model'], 'cuda:0')
    bank = load_bank(identity['v_bank'])
    assert bank['status'] == 'complete' and bank['method'] == 'uniform' and bank['nominal_v_rank'] == 192
    verify_model_identity(identity['model'], bank['model_identity'])
    gdn = load_qwen35_gdn_private_ag_factors(identity['gdn_bank'])
    assert gdn['status'] == 'complete' and len(gdn['layers']) == 24
    dense_value_weights = {layer: model.model.layers[layer].self_attn.v_proj.weight.detach().clone() for layer in layers}
    GatedVRuntime(model, bank['layers']).install()
    Qwen35PrivateAGRuntime(model, gdn).install()
    model.eval()
    for layer in layers:
        assert isinstance(model.model.layers[layer].self_attn, GatedVAttention)
    return model, dense_value_weights


@torch.inference_mode()
def replay(args, identity, windows, layers, protocol, *, fisher=False):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    model, dense_value_weights = load_deployed_model(identity, layers)
    config = getattr(model.config, 'text_config', model.config)
    heads, groups, dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    assert (heads, groups, dim) == (identity['heads'], identity['hkv'], identity['head_dim'])
    grid = candidate_positions(args.sequence_length, excluded_query_prefix=protocol['query_grid_prefix'])
    window_ids = dict(fit=shard_windows(protocol['fit_ids'], args.window_shard, args.window_shards),
                      heldout=shard_windows(protocol['diagnostic_ids'], args.window_shard, args.window_shards))
    print(dict(stage='fisher' if fisher else 'moments', shard=args.window_shard, shards=args.window_shards,
               windows=window_ids, layers=layers, heads=heads, groups=groups, dim=dim, grid=len(grid)), flush=True)
    moments, candidates, base_payloads, selections = {}, {}, {}, {}
    active, handles = {}, []
    for layer in layers:
        module = model.model.layers[layer].self_attn
        if fisher:
            base_payloads[layer], meta = verified(layer_file(args.output, 'base', layer))
            assert meta['protocol'] == protocol and meta['identity_sha256'] == sha256(args.identity)
            _, selection_meta = verified(layer_file(args.moments_root or args.output, 'moments', layer))
            selections[layer] = selection_meta['selections']
        else:
            moments[layer] = {s: RawBaseMoments(groups, dim) for s in ('fit', 'heldout')}
            candidates[layer] = torch.empty(len(window_ids['fit']), len(grid), heads, dim, dtype=torch.bfloat16)

        def capture(attention, positional, kwargs, layer=layer):
            split, index, slot = active['split'], active['index'], active['slot']
            x = kwargs['hidden_states']
            n = x.shape[1]
            native = attention.native
            q, _ = native.q_proj(x).view(1, n, -1, 2 * dim).chunk(2, dim=-1)
            q = native.q_norm(q).transpose(1, 2)
            k = native.k_norm(native.k_proj(x).view(1, n, groups, dim))
            v = F.linear(x, dense_value_weights[layer]).view(1, n, groups, dim)
            cos, sin = kwargs['position_embeddings']
            assert cos.ndim == 3 and tuple(cos.shape) == (1, n, ROTARY_DIM) and cos.shape == sin.shape
            q, k_post = apply_rotary_pos_emb(q, k.transpose(1, 2), cos, sin)
            if not fisher:
                moments[layer][split].update(v[0], k[0], args.chunk_rows)
                if split == 'fit':
                    candidates[layer][slot].copy_(q[0, :, grid].transpose(0, 1).cpu())
                return
            positions = selections[layer][split]['selected_positions']
            if args.objective == 'page_fisher':
                rows = torch.cat((v, k_post.transpose(1, 2)), -1)
                stats, diagnostics = build_multi_query_statistics(q[:, :, positions].transpose(1, 2), rows,
                    query_positions=positions, cos=cos, sin=sin, value_encoder=base_payloads[layer]['encoder'],
                    base_maps=restore_base(base_payloads[layer]), page_size=args.page_size,
                    excluded_prefix_pages=protocol['excluded_prefix_pages'],
                    excluded_recent_tokens=protocol['excluded_recent_tokens'], device=torch.device('cuda'))
                payload = packed_fisher(stats[args.base_rank])
            else:
                payload, diagnostics = score_mse_statistics(q, v, k_post, positions, base_payloads[layer]['encoder'],
                    base_payloads[layer], cos, sin, excluded_prefix_tokens=args.page_size * protocol['excluded_prefix_pages'],
                    excluded_recent_tokens=protocol['excluded_recent_tokens'], heads=heads, groups=groups, dim=dim)
            if args.smoke:
                # The prefix rotation of the Base prediction must match the model's own partial RoPE rule.
                probe = torch.randn(1, groups, n, dim, device=x.device, dtype=torch.float32)
                _, rotated = apply_rotary_pos_emb(probe, probe, cos.float(), sin.float())
                torch.testing.assert_close(apply_rotary(probe, cos, sin), rotated, rtol=1e-4, atol=1e-4)
            path = args.output/'fisher'/f'l{layer:03d}'/f'w{index:03d}.safetensors'
            save_record(path, payload, dict(protocol=protocol, layer=layer, window_id=index, split=split,
                base_sha256=sha256(layer_file(args.output, 'base', layer)), diagnostics=diagnostics,
                storage='symmetric upper triangle FP32'))
        handles.append(module.register_forward_pre_hook(capture, with_kwargs=True))
    for split in ('fit', 'heldout'):
        for slot, index in enumerate(window_ids[split]):
            active.update(split=split, index=index, slot=slot)
            started = time.monotonic()
            output = model.model(windows[index:index + 1].long().cuda(), use_cache=False)
            assert torch.isfinite(output.last_hidden_state).all()
            del output
            print(dict(stage='fisher' if fisher else 'moments', window=index, split=split, layers=layers,
                       seconds=round(time.monotonic() - started, 1),
                       peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)), flush=True)
    for handle in handles:
        handle.remove()
    if not fisher:
        for layer in layers:
            tensors = {f'{s}_{name}': value for s, m in moments[layer].items() for name, value in m.tensors().items()}
            tensors['fit_candidates'] = candidates[layer]
            save_record(part_file(args.output, layer, args.window_shard), tensors,
                dict(protocol=protocol, layer=layer, window_shard=args.window_shard, window_shards=args.window_shards,
                     fit_windows=window_ids['fit'], heldout_windows=window_ids['heldout'], candidate_positions=grid))
    del model
    gc.collect()
    torch.cuda.empty_cache()


def merge_moments(args, layers, protocol):
    grid = candidate_positions(args.sequence_length, excluded_query_prefix=protocol['query_grid_prefix'])
    for layer in layers:
        sums, slots = {}, {}
        for shard in range(args.window_shards):
            payload, meta = verified(part_file(args.output, layer, shard))
            assert meta['protocol'] == protocol and meta['layer'] == layer and meta['window_shards'] == args.window_shards
            assert meta['candidate_positions'] == grid
            for split in ('fit', 'heldout'):
                for name in ('count', 'sum_v', 'sum_k', 'vv', 'vk', 'kk'):
                    key = f'{split}_{name}'
                    sums[key] = payload[key] if key not in sums else sums[key] + payload[key]
            for slot, window in enumerate(meta['fit_windows']):
                slots[int(window)] = payload['fit_candidates'][slot]
        assert sorted(slots) == list(protocol['fit_ids'])
        assert int(sums['fit_count']) == len(protocol['fit_ids']) * args.sequence_length
        candidates = torch.stack([slots[w] for w in protocol['fit_ids']])
        selected, _, _ = select_stratified_query_positions(candidates, grid, context_length=args.sequence_length,
                                                          num_bins=4, queries_per_bin=16)
        save_record(layer_file(args.output, 'moments', layer), sums,
            dict(protocol=protocol, layer=layer, selections=dict(fit=selected), selection_fit_only=True,
                 window_shards=args.window_shards))
        print(dict(stage='merge', layer=layer, fit_rows=int(sums['fit_count']),
                   queries=len(selected['selected_positions'])), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=('moments', 'merge', 'base', 'fisher', 'fit'))
    for name in ('identity', 'windows', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--layers', required=True)
    p.add_argument('--moments-root', type=Path)
    p.add_argument('--base-rank', type=int, default=16)
    p.add_argument('--residual-rank', type=int, default=16)
    p.add_argument('--sequence-length', type=int, default=131072)
    p.add_argument('--fit-count', type=int, default=32)
    p.add_argument('--chunk-rows', type=int, default=2048)
    p.add_argument('--sweeps', type=int, default=40)
    p.add_argument('--pcg-iterations', type=int, default=100)
    p.add_argument('--window-shard', type=int, default=0)
    p.add_argument('--window-shards', type=int, default=1)
    p.add_argument('--objective', choices=('score_mse', 'page_fisher'), default='score_mse')
    p.add_argument('--page-size', type=int, default=PAGE_SIZE, choices=(1, 2, 4, 8, 16, 32))
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    configure()
    args.dense_v = False
    args.diagnostic_count = 0
    assert 0 <= args.window_shard < args.window_shards
    assert 0 < args.base_rank <= 128 and 0 < args.residual_rank <= 128
    identity = read_json(args.identity)
    assert identity['status'] == 'complete'
    assert sha256(Path(identity['model'])/'config.json') == identity['model_config_sha256']
    assert sha256(identity['v_bank']) == identity['v_bank_sha256'] and sha256(identity['gdn_bank']) == identity['gdn_bank_sha256']
    windows = load_file(str(args.windows))['input_ids']
    manifest = read_json(args.windows.with_name('manifest.json'))
    assert manifest['status'] == 'complete' and manifest['sha256'] == sha256(args.windows)
    assert manifest['model_config_sha256'] == identity['model_config_sha256']
    fit_ids = [int(x) for x in manifest['fit_ids']]
    assert fit_ids == list(range(len(fit_ids)))   # validation windows (if any) are never read by this fitter
    assert windows.shape[0] >= len(fit_ids) and 4096 <= args.sequence_length <= windows.shape[1]   # extra validation windows are ignored
    assert 0 < args.fit_count <= len(fit_ids)
    if not args.smoke:
        assert windows.shape[1] == args.sequence_length and args.fit_count == len(fit_ids)
        assert args.sweeps == 40 and args.pcg_iterations == 100
    windows = windows[:, :args.sequence_length]
    layers = [int(x) for x in args.layers.split(',')]
    assert len(set(layers)) == len(layers) and set(layers) <= set(identity['attention_layers'])
    definition = {'score_mse': 'unweighted causal residual QK score squared error over the routable prefix '
                               '(no pinned sink; recent window excluded)',
                  'page_fisher': 'softmax page-Fisher weighted residual objective over causal historical pages '
                                 '(no pinned sink; recent window excluded)'}[args.objective]
    protocol = dict(format=f'basisserve.qwen35.k_router.streaming.128k.{args.objective}.v1',
        model_config_sha256=identity['model_config_sha256'], model_revision=identity['model_revision'],
        v_bank_sha256=identity['v_bank_sha256'], gdn_bank_sha256=identity['gdn_bank_sha256'],
        windows_sha256=sha256(args.windows), windows_manifest_sha256=sha256(args.windows.with_name('manifest.json')),
        sequence_length=args.sequence_length, fit_ids=fit_ids[:args.fit_count], diagnostic_ids=[],
        fit_queries=64, diagnostic_queries=0, validation_scope='in-sample: no diagnostic windows',
        base_rank=args.base_rank, residual_rank=args.residual_rank, page_size=args.page_size, objective=args.objective,
        objective_definition=definition,
        excluded_prefix_pages=0, excluded_recent_tokens=RECENT_TOKENS, query_grid_prefix=RECENT_TOKENS,
        deployment_page_budget_tokens=2048, maximum_deployment_support_tokens=2048,
        sink_tokens_within_page_budget=0, recent_tokens_within_page_budget=RECENT_TOKENS,
        rotary_dim=ROTARY_DIM, rope='native partial rotary (factor 0.25, interleaved mRoPE on text positions)',
        chunk_rows=args.chunk_rows, smoke=args.smoke,
        teacher='deployed Qwen3.5-9B: uniform V192 (GatedVAttention, flash prefill with padded V) and GDN Wo75 '
                'Private-AllGather installed before the replay; dense v_proj kept only for Base moments; '
                'BF16; model.model; use_cache=False',
        base_objective='affine pre-RoPE closed-form RRR from deployed-trajectory dense V codes',
        base_moments='FP64 raw dense-V / pre-RoPE-K moments summed over window shards; encoder transform at Base fit',
        sweeps=args.sweeps, pcg_iterations=args.pcg_iterations,
        source_sha256={name: sha256(Path(name)) for name in SOURCES})
    if args.stage == 'moments':
        replay(args, identity, windows, layers, protocol)
    elif args.stage == 'merge':
        merge_moments(args, layers, protocol)
    elif args.stage == 'base':
        fit_bases(args, identity, layers, protocol)
    elif args.stage == 'fisher':
        replay(args, identity, windows, layers, protocol, fisher=True)
    else:
        fit_residuals(args, identity, layers, protocol)
    print(json.dumps(dict(stage=args.stage, status='complete', layers=layers)), flush=True)


if __name__ == '__main__':
    main()
