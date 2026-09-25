"""Window-sharded variant of fit_k_routing_streaming: one replay per calibration window covers EVERY attention layer.

The layer-sharded streaming fitter replays all windows once per layer shard, so eight shards repeat the same 128K forward
eight times. Here each shard replays only its own windows (fit_ids[shard::shards]) with hooks on all layers, writes
per-layer partial moments (moments stage) or per-layer per-window Fisher records (fisher stage), and a `merge` stage sums
the partial moments and selects the Fisher query positions exactly as the streaming fitter does. Base solving, residual
ALS and every record layout are reused from fit_k_routing_streaming, so the resulting banks are drop-in for the evaluators.

Stages: moments --window-shard s --window-shards N | merge | base | fisher --window-shard s --window-shards N | fit
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from basisserve.checkpoint.c1_attention_layers import c1_attention_layers
from basisserve.core.query_position_sampling import candidate_positions, select_stratified_query_positions
from evaluation.v96kl_common import configure, read_json, sha256
from evaluation.k_routing_config import residual_fisher_support, routing_config, routing_position_embeddings
from evaluation.streaming_k_statistics import RawBaseMoments, packed_fisher
from evaluation.fit_qwen3_8b_q8_fisher_residual import build_multi_query_statistics
from evaluation import fit_k_routing_streaming as S

MOMENT_KEYS = ('count', 'sum_v', 'sum_k', 'vv', 'vk', 'kk')
ADOPT_IGNORED = ('sharding', 'source_sha256')


def protocol_matches(recorded, protocol, adopt):
    """Strict equality, or (when adopting moments/base records written by the layer-sharded streaming fitter) equality of
    everything except the sharding descriptor and the source hashes, which differ between the two fitters by construction."""
    if recorded == protocol:
        return True
    if not adopt:
        return False
    strip = lambda d: {k: v for k, v in d.items() if k not in ADOPT_IGNORED}
    return strip(recorded) == strip(protocol)


def shard_windows(ids, shard, shards):
    return [int(x) for x in ids[shard::shards]]


def part_file(root, layer, shard):
    return root / 'moments_parts' / f'layer_{layer:03d}_s{shard}.safetensors'


@torch.inference_mode()
def replay(args, identity, windows, layers, protocol, *, fisher=False):
    config = routing_config(identity, rope=args.rope, sequence_length=args.sequence_length)
    if config.model_type == 'nemotron_h':
        from evaluation import nemotron_h_triton_mamba as triton_mamba
        triton_mamba.install()
    model = AutoModelForCausalLM.from_pretrained(identity['model'], config=config, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True, trust_remote_code=False).eval().cuda()
    assert model.config.model_type in ('llama', 'qwen3', 'nemotron_h')
    if model.config.model_type == 'nemotron_h':
        triton_mamba.restore_dt_limit(model)
    dense_value_weights = {i: m.v_proj.weight.detach().clone() for i, m in c1_attention_layers(model)}
    if args.teacher == 'deployed':
        S.install_deployed_teacher(model, args, identity)
    attention_modules = dict(c1_attention_layers(model))
    rotary = config.model_type != 'nemotron_h'
    if config.model_type == 'llama':
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    elif config.model_type == 'qwen3':
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    config = model.config
    heads, groups = config.num_attention_heads, config.num_key_value_heads
    dim = identity['head_dim']
    grid = candidate_positions(args.sequence_length, excluded_query_prefix=protocol['query_grid_prefix'])
    stats_cos, stats_sin = routing_position_embeddings(config, args.sequence_length, torch.device('cuda'))
    window_ids = dict(fit=shard_windows(protocol['fit_ids'], args.window_shard, args.window_shards),
                      heldout=shard_windows(protocol['diagnostic_ids'], args.window_shard, args.window_shards))
    print(dict(stage='fisher' if fisher else 'moments', shard=args.window_shard, shards=args.window_shards, windows=window_ids,
               layers=layers, teacher=args.teacher, grid=len(grid)), flush=True)
    moments, post_moments, candidates, base_payloads, selections = {}, {}, {}, {}, {}
    active, handles = {}, []
    for layer in layers:
        module = attention_modules[layer]
        if fisher:
            base_payloads[layer], meta = S.verified(S.layer_file(args.output, 'base', layer))
            assert protocol_matches(meta['protocol'], protocol, args.adopt_streaming_records) and meta['identity_sha256'] == sha256(args.identity)
            _, selection_meta = S.verified(S.layer_file(args.output, 'moments', layer))
            selections[layer] = selection_meta['selections']
        else:
            moments[layer] = {s: RawBaseMoments(groups, dim) for s in ('fit', 'heldout')}
            post_moments[layer] = {s: RawBaseMoments(groups, dim) for s in ('fit', 'heldout')}
            candidates[layer] = torch.empty(len(window_ids['fit']), len(grid), heads, dim, dtype=torch.bfloat16)

        def capture(attention, positional, kwargs, layer=layer):
            split, index, slot = active['split'], active['index'], active['slot']
            x = kwargs['hidden_states'] if 'hidden_states' in kwargs else positional[0]
            n = x.shape[1]
            k = attention.k_proj(x).view(1, n, groups, dim)
            v = torch.nn.functional.linear(x, dense_value_weights[layer]).view(1, n, groups, dim)
            q = attention.q_proj(x).view(1, n, heads, dim)
            if config.model_type == 'qwen3':
                q, k = attention.q_norm(q), attention.k_norm(k)
            if rotary:
                q, k_post = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), *kwargs['position_embeddings'])
            else:
                q, k_post = q.transpose(1, 2), k.transpose(1, 2)
            if not fisher:
                moments[layer][split].update(v[0], k[0], args.chunk_rows)
                post_moments[layer][split].update(v[0], k_post.transpose(1, 2)[0], args.chunk_rows)
                if split == 'fit':
                    candidates[layer][slot].copy_(q[0, :, grid].transpose(0, 1).cpu())
                return
            positions = selections[layer][split]['selected_positions']
            path = args.output / 'fisher' / f'l{layer:03d}' / f'w{index:03d}.safetensors'
            base_sha = sha256(S.layer_file(args.output, 'base', layer))
            if args.objective == 'score_mse':
                payload, diagnostics = S.score_mse_statistics(q, v, k_post, positions, base_payloads[layer]['encoder'], base_payloads[layer],
                    stats_cos, stats_sin, excluded_prefix_tokens=args.page_size * protocol['excluded_prefix_pages'],
                    excluded_recent_tokens=protocol['excluded_recent_tokens'], heads=heads, groups=groups, dim=dim)
                S.save_record(path, payload, dict(protocol=protocol, layer=layer, window_id=index, split=split, base_sha256=base_sha,
                                                  diagnostics=diagnostics, storage='symmetric upper triangle FP32'))
                return
            rows = torch.cat((v, k_post.transpose(1, 2)), -1)
            stats, diagnostics = build_multi_query_statistics(q[:, :, positions].transpose(1, 2), rows,
                query_positions=positions, cos=stats_cos, sin=stats_sin, value_encoder=base_payloads[layer]['encoder'],
                base_maps=S.restore_base(base_payloads[layer]), page_size=args.page_size,
                excluded_prefix_pages=protocol['excluded_prefix_pages'], excluded_recent_tokens=protocol['excluded_recent_tokens'],
                device=torch.device('cuda'))
            S.save_record(path, packed_fisher(stats[args.base_rank]), dict(protocol=protocol, layer=layer, window_id=index, split=split,
                base_sha256=base_sha, diagnostics=diagnostics, storage='symmetric upper triangle FP32'))
        handles.append(module.register_forward_pre_hook(capture, with_kwargs=True))
    for split in ('fit', 'heldout'):
        for slot, index in enumerate(window_ids[split]):
            active.update(split=split, index=index, slot=slot)
            started = time.monotonic()
            # One window's hidden states exist at a time; never allocate [N,T,H].
            output = model.model(windows[index:index + 1].long().cuda(), use_cache=False)
            assert torch.isfinite(output.last_hidden_state).all()
            del output
            print(dict(stage='fisher' if fisher else 'moments', window=index, split=split, layers=len(layers),
                       seconds=round(time.monotonic() - started, 1), peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)), flush=True)
    for handle in handles:
        handle.remove()
    if not fisher:
        for layer in layers:
            tensors = {f'{s}_{name}': value for s, m in moments[layer].items() for name, value in m.tensors().items()}
            tensors.update({f'{s}_post_{name}': value for s, m in post_moments[layer].items() for name, value in m.tensors().items()})
            tensors['fit_candidates'] = candidates[layer]
            S.save_record(part_file(args.output, layer, args.window_shard), tensors,
                dict(protocol=protocol, layer=layer, window_shard=args.window_shard, window_shards=args.window_shards,
                     fit_windows=window_ids['fit'], heldout_windows=window_ids['heldout'], candidate_positions=grid))
    del model
    gc.collect()
    torch.cuda.empty_cache()


def fit_residuals(args, identity, layers, protocol):
    """fit_k_routing_streaming.fit_residuals with the base record accepted from either fitter (see protocol_matches)."""
    from evaluation.streaming_k_statistics import load_fisher_windows
    from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _fit_residual_grid
    for layer in layers:
        base, meta = S.verified(S.layer_file(args.output, 'base', layer))
        assert protocol_matches(meta['protocol'], protocol, args.adopt_streaming_records) and meta['identity_sha256'] == sha256(args.identity)
        groups, dim = base['encoder'].shape[:2]
        if args.residual_rank == 0:
            # Base-only routing (Section 4 ablation): the sidecar is the predicted post-RoPE K alone, so there is no residual
            # code to fit and no Fisher statistics are read; zero-width residual tensors keep the bank layout of the runtime.
            heads = identity['hq']
            tensors = {f'base_{name}_b{args.base_rank}': base[name].float() for name in ('left', 'right', 'bias')}
            tensors.update({f'residual_encoder_b{args.base_rank}_r0': torch.zeros(groups, dim, 0), f'residual_query_b{args.base_rank}_r0': torch.zeros(heads, dim, 0)})
            losses = {f'b{args.base_rank}_r0': dict(base_rank=args.base_rank, residual_rank=0, sweeps=[], fisher_artifacts_read=[],
                      note='base-only routing: no residual fit; base metrics in the base record')}
            S.save_record(S.layer_file(args.output, f'ours_b{args.base_rank}r0', layer), tensors,
                dict(protocol=protocol, layer=layer, v_rank=base['encoder'].shape[-1], identity_sha256=sha256(args.identity), losses=losses,
                     base_sha256=sha256(S.layer_file(args.output, 'base', layer)), sweeps=0, pcg_iterations=0,
                     base_record_adopted_from_streaming_fitter=bool(args.adopt_streaming_records and meta['protocol'] != protocol)))
            print(dict(stage='fit complete', layer=layer, losses=losses), flush=True)
            continue
        stats = {}
        for split, indices in (('fit', protocol['fit_ids']), ('heldout', protocol['diagnostic_ids'])):
            if not indices:
                continue
            payloads = []
            for index in indices:
                path = args.output / 'fisher' / f'l{layer:03d}' / f'w{index:03d}.safetensors'
                payload, record = S.verified(path)
                assert record['protocol'] == protocol and record['split'] == split and record['window_id'] == index
                assert record['base_sha256'] == sha256(S.layer_file(args.output, 'base', layer))
                payloads.append(payload)
            heads = payloads[0]['queries'].shape[0]
            stats[split] = {args.base_rank: load_fisher_windows(payloads, heads, groups, dim)}
            del payloads
        factors, losses = _fit_residual_grid(stats['fit'], stats.get('heldout', stats['fit']), residual_ranks=(args.residual_rank,),
            sweeps=args.sweeps, relative_damping=1e-5, iterative_tolerance=1e-5, iterative_max_iterations=args.pcg_iterations, device=torch.device('cuda'))
        tensors = {f'base_{name}_b{args.base_rank}': base[name].float() for name in ('left', 'right', 'bias')}
        tensors.update({f'residual_encoder_b{args.base_rank}_r{args.residual_rank}': factors[(args.base_rank, args.residual_rank)][0].cpu().float(),
                        f'residual_query_b{args.base_rank}_r{args.residual_rank}': factors[(args.base_rank, args.residual_rank)][1].cpu().float()})
        assert all(torch.isfinite(t).all() for t in tensors.values())
        S.save_record(S.layer_file(args.output, f'ours_b{args.base_rank}r{args.residual_rank}', layer), tensors,
            dict(protocol=protocol, layer=layer, v_rank=base['encoder'].shape[-1], identity_sha256=sha256(args.identity), losses=losses,
                 base_sha256=sha256(S.layer_file(args.output, 'base', layer)), sweeps=args.sweeps, pcg_iterations=args.pcg_iterations,
                 base_record_adopted_from_streaming_fitter=bool(args.adopt_streaming_records and meta['protocol'] != protocol)))
        del stats, factors
        gc.collect()
        torch.cuda.empty_cache()
        print(dict(stage='fit complete', layer=layer, losses=losses), flush=True)


def merge_moments(args, layers, protocol):
    grid = candidate_positions(args.sequence_length, excluded_query_prefix=protocol['query_grid_prefix'])
    for layer in layers:
        sums, slots = {}, {}
        for shard in range(args.window_shards):
            payload, meta = S.verified(part_file(args.output, layer, shard))
            assert meta['protocol'] == protocol and meta['layer'] == layer and meta['window_shards'] == args.window_shards
            assert meta['candidate_positions'] == grid
            for split in ('fit', 'heldout'):
                for prefix in (f'{split}_', f'{split}_post_'):
                    for name in MOMENT_KEYS:
                        key = prefix + name
                        sums[key] = payload[key] if key not in sums else sums[key] + payload[key]
            for slot, window in enumerate(meta['fit_windows']):
                slots[int(window)] = payload['fit_candidates'][slot]
        assert sorted(slots) == list(protocol['fit_ids'])
        assert int(sums['fit_count']) == len(protocol['fit_ids']) * args.sequence_length
        candidates = torch.stack([slots[w] for w in protocol['fit_ids']])
        selections = {}
        for split, qpb in [('fit', 16)] + ([('heldout', 8)] if protocol['diagnostic_ids'] else []):
            selected, _, _ = select_stratified_query_positions(candidates, grid, context_length=args.sequence_length,
                                                              num_bins=4, queries_per_bin=qpb)
            selections[split] = selected
        S.save_record(S.layer_file(args.output, 'moments', layer), sums,
            dict(protocol=protocol, layer=layer, selections=selections, selection_fit_only=True, window_shards=args.window_shards))
        print(dict(stage='merge', layer=layer, fit_rows=int(sums['fit_count']), queries=len(selections['fit']['selected_positions'])), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=('moments', 'merge', 'base', 'fisher', 'fit'))
    for name in ('identity', 'windows', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--layers', help='comma list; default every attention layer of the identity (fit stage: the layers this process fits)')
    p.add_argument('--window-shard', type=int, default=0)
    p.add_argument('--window-shards', type=int, default=8)
    p.add_argument('--base-rank', type=int, default=16)
    p.add_argument('--page-size', type=int, default=32, choices=(1, 2, 4, 8, 16, 32), help='routing page size of the Fisher statistics and the deployed selector')
    p.add_argument('--pinned-pages', type=int, default=1, choices=(0, 1), help='sink pages pinned by the deployed selector (excluded from the routable prefix); 0 = no sink')
    p.add_argument('--objective', choices=('page_fisher', 'score_mse'), default='page_fisher')
    p.add_argument('--residual-rank', type=int, default=16)
    p.add_argument('--sequence-length', type=int, default=65536)
    p.add_argument('--fit-count', type=int, default=64)
    p.add_argument('--diagnostic-count', type=int, default=0)
    p.add_argument('--chunk-rows', type=int, default=2048)
    p.add_argument('--sweeps', type=int, default=40)
    p.add_argument('--pcg-iterations', type=int, default=100)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--teacher', choices=('dense', 'deployed'), default='deployed')
    p.add_argument('--dense-v', action='store_true')
    p.add_argument('--rope', choices=('native', 'yarn2', 'yarn4'), required=True)
    p.add_argument('--native-audit', type=Path, help='Nemotron-H native audit (deployed teacher installs Wo)')
    p.add_argument('--wo-bank', type=Path, help='Nemotron-H Mamba Wo factor bank (deployed teacher)')
    p.add_argument('--adopt-streaming-records', action='store_true', help='fisher/fit: accept moments and base records written by the layer-sharded fit_k_routing_streaming run in the same --output')
    args = p.parse_args()
    configure()
    args.moments_root = None          # fit_bases / fit_residuals read the moments and fisher records under --output
    args.collect_covariance = False
    assert not args.dense_v or args.teacher == 'dense'
    assert 0 <= args.base_rank <= 128 and 0 <= args.residual_rank <= 128
    assert 0 <= args.window_shard < args.window_shards
    identity = read_json(args.identity)
    assert identity['status'] == 'complete'
    assert sha256(Path(identity['model']) / 'config.json') == identity['model_config_sha256']
    windows = load_file(str(args.windows))['input_ids']
    wm = read_json(args.windows.with_name('manifest.json'))
    assert wm['sha256'] == sha256(args.windows) and wm['model_config_sha256'] == identity['model_config_sha256']
    available_fit_ids = list(map(int, wm['fit_ids']))
    available_diagnostic_ids = list(map(int, wm['validation_ids']))
    assert available_fit_ids == list(range(len(available_fit_ids)))
    heldout_start = len(available_fit_ids)
    expected_windows = heldout_start + len(available_diagnostic_ids)
    assert available_diagnostic_ids == list(range(heldout_start, expected_windows))
    assert windows.shape[0] == expected_windows and 4096 <= args.sequence_length <= windows.shape[1]
    assert 0 < args.fit_count <= len(available_fit_ids)
    assert 0 <= args.diagnostic_count <= len(available_diagnostic_ids)
    if not args.smoke:
        assert not wm.get('test_only', False)
        assert windows.shape[1] == args.sequence_length
        assert args.fit_count == len(available_fit_ids)
        assert args.diagnostic_count in (0, len(available_diagnostic_ids))
        assert args.sweeps == 40 and args.pcg_iterations == 100
    assert args.chunk_rows > 0
    windows = windows[:, :args.sequence_length]
    layers = [int(x) for x in args.layers.split(',')] if args.layers else list(identity['attention_layers'])
    assert len(set(layers)) == len(layers) and set(layers) <= set(identity['attention_layers'])
    runtime = routing_config(identity, rope=args.rope, sequence_length=args.sequence_length)
    runtime_config = runtime.to_dict()
    if runtime.model_type == 'nemotron_h':
        runtime_config['time_step_limit'] = [str(x) if math.isinf(x) else x for x in runtime.time_step_limit]
    fisher_support = residual_fisher_support(runtime.model_type)
    fisher_support['excluded_prefix_pages'] = args.pinned_pages
    sink_tokens = args.page_size * args.pinned_pages
    diagnostic_ids = available_diagnostic_ids[:args.diagnostic_count]
    teacher = ('deployed model: identity checkpoint (compressed V and decoder' +
               (', folded Mamba Wo' if runtime.model_type == 'nemotron_h' else '') +
               ') installed before the replay; dense v_proj kept only for Base moments; BF16, use_cache=False'
               if args.teacher == 'deployed' else 'native dense BF16 SDPA, model.model, use_cache=False')
    protocol = dict(format='basisserve.k_router.streaming.v2', model_config_sha256=identity['model_config_sha256'],
        windows_sha256=sha256(args.windows), windows_manifest_sha256=sha256(args.windows.with_name('manifest.json')),
        sequence_length=args.sequence_length, fit_ids=available_fit_ids[:args.fit_count],
        diagnostic_ids=diagnostic_ids, fit_queries=64, diagnostic_queries=32 if diagnostic_ids else 0,
        validation_scope='held-out diagnostic windows' if diagnostic_ids else 'in-sample: no diagnostic windows',
        base_rank=args.base_rank, residual_rank=args.residual_rank, page_size=args.page_size, objective=args.objective,
        objective_definition=('softmax page-Fisher weighted residual objective' if args.objective == 'page_fisher' else
                              'unweighted causal residual QK score squared error over the routable prefix (sink page and recent window excluded)'),
        **fisher_support, query_grid_prefix=sink_tokens + fisher_support['excluded_recent_tokens'],
        deployment_page_budget_tokens=2048, maximum_deployment_support_tokens=2048,
        sink_tokens_within_page_budget=sink_tokens, recent_tokens_within_page_budget=fisher_support['excluded_recent_tokens'],
        chunk_rows=args.chunk_rows, smoke=args.smoke, teacher=teacher,
        deployed_checkpoint_manifest_sha256=identity['manifest_sha256'] if args.teacher == 'deployed' else None,
        rope=args.rope, runtime_config=runtime_config, dense_v=args.dense_v,
        base_moments='FP64 raw pre-RoPE and post-RoPE moments with identical rows and encoder transform; no bitwise claim versus projected FP32 accumulation',
        covariance='not collected',
        sharding=dict(kind='window', window_shards=args.window_shards, replay='every attention layer per window; partial moments merged in FP64'),
        source_sha256={name: sha256(Path(name)) for name in ('evaluation/fit_k_routing_windowed.py', 'evaluation/fit_k_routing_streaming.py',
            'evaluation/eval_k_routing_ruler.py', 'evaluation/k_routing_config.py',
            'evaluation/streaming_k_statistics.py', 'evaluation/fit_qwen3_8b_q8_fisher_residual.py',
            'basisserve/core/query_position_sampling.py', 'basisserve/core/c1_v_conditional_k_router.py')})
    if runtime.model_type == 'nemotron_h':
        protocol['mamba_scan'] = 'vLLM Triton chunk scan via evaluation/nemotron_h_triton_mamba.py; dt clamp reset to (0, inf)'
        protocol['source_sha256']['evaluation/nemotron_h_triton_mamba.py'] = sha256(Path('evaluation/nemotron_h_triton_mamba.py'))
        if args.teacher == 'deployed':
            assert args.native_audit is not None and args.wo_bank is not None
            protocol['native_audit_sha256'] = sha256(args.native_audit)
            protocol['wo_bank_layers'] = {p.stem: sha256(p) for p in sorted(args.wo_bank.glob('layer_*.safetensors'))}
    # Records compare the protocol after a JSON round trip, so normalize tuples and floats once here.
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    if args.stage == 'moments':
        replay(args, identity, windows, layers, protocol)
    elif args.stage == 'merge':
        merge_moments(args, layers, protocol)
    elif args.stage == 'base':
        S.fit_bases(args, identity, layers, protocol)
    elif args.stage == 'fisher':
        replay(args, identity, windows, layers, protocol, fisher=True)
    else:
        fit_residuals(args, identity, layers, protocol)


if __name__ == '__main__':
    main()
