"""Layer-sharded Base16/Page-Fisher R16 fitting for frozen allocated C1 payloads."""
import argparse
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, save_tensors, sha256
from basisserve.core.query_position_sampling import candidate_positions, select_stratified_query_positions
from basisserve.core.c1_v_conditional_k_router import fit_affine_reduced_rank_map
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _fit_residual_grid, _value_codes, _apply_base, _stack_base_map, _post_rope_rows
from evaluation.fit_qwen3_8b_q8_fisher_residual import build_multi_query_statistics


def fit_base(rows, keys, encoder, fit_count, device):
    groups, dim, rank = encoder.shape
    sums = torch.zeros(groups, rank, dtype=torch.float64)
    targets = torch.zeros(groups, dim, dtype=torch.float64)
    gram = torch.zeros(groups, rank, rank, dtype=torch.float64)
    cross = torch.zeros(groups, rank, dim, dtype=torch.float64)
    encoder = encoder.to(device).float()
    for i in range(fit_count):
        z = _value_codes(rows[i, ..., :dim].to(device).float(), encoder).transpose(0, 1)
        k = keys[i].to(device).float().transpose(0, 1)
        sums += z.sum(1).double().cpu()
        targets += k.sum(1).double().cpu()
        gram += torch.bmm(z.mT, z).double().cpu()
        cross += torch.bmm(z.mT, k).double().cpu()
    return {16: tuple(fit_affine_reduced_rank_map(row_count=fit_count * rows.shape[1],
        input_sum=sums[g], target_sum=targets[g], input_gram=gram[g],
        input_target_gram=cross[g], rank=16, fit_bias=True) for g in range(groups))}


def base_metrics(rows, keys, encoder, bases, start, stop, device):
    dim = encoder.shape[1]
    factors = _stack_base_map(bases[16], device=device)
    error = energy = 0.
    for i in range(start, stop):
        z = _value_codes(rows[i, ..., :dim].to(device).float(), encoder.to(device).float())
        k = keys[i].to(device).float()
        prediction = _apply_base(z, factors)
        error += float((prediction-k).double().square().sum())
        energy += float(k.double().square().sum())
    assert energy > 0
    return dict(relative_mse=error/energy, squared_error=error, key_energy=energy)


def fit_layer(args, layer, rows, prekeys, queries, encoder, cos, sin, spec):
    started = time.monotonic()
    print('FIT START',layer,'verifying capture and selecting queries',flush=True)
    device = torch.device('cuda')
    dim, rank = encoder.shape[1:]
    grid = candidate_positions(args.sequence_length)
    capture = args.output / 'calibration' / f'layer_{layer:03d}.safetensors'
    capture_spec = read_json(capture.with_suffix('.json'))
    assert capture_spec['status']=='complete' and capture_spec['protocol']==spec
    assert capture_spec['layer']==layer and capture_spec['candidate_positions']==grid
    selection_fit, _, _ = select_stratified_query_positions(queries[:args.fit_count].cuda(), grid,
        context_length=args.sequence_length, num_bins=4, queries_per_bin=16)
    selection_diag, _, _ = select_stratified_query_positions(queries[:args.fit_count].cuda(), grid,
        context_length=args.sequence_length, num_bins=4, queries_per_bin=8)
    query_path = args.output / 'qgram' / f'layer_{layer:03d}.safetensors'
    selected = {}
    for split, selection, start, stop in (
        ('fit', selection_fit, 0, args.fit_count),
        ('diagnostic', selection_diag, args.fit_count, args.fit_count+args.diagnostic_count)):
        selected[split] = queries[start:stop, [grid.index(p) for p in selection['selected_positions']]].contiguous()
    save_tensors(query_path, selected)
    query_spec = dict(status='complete', layer=layer, selections=dict(fit=selection_fit, diagnostic=selection_diag),
        selection_window_ids=list(range(args.fit_count)), diagnostic_used_for_selection=False,
        capture_manifest_sha256=sha256(capture.with_suffix('.json')), sha256=sha256(query_path))
    write_json(query_path.with_suffix('.json'), query_spec)
    bases = fit_base(rows, prekeys, encoder, args.fit_count, device)
    reconstruction = {
        'fit': base_metrics(rows, prekeys, encoder, bases, 0, args.fit_count, device),
        'diagnostic': base_metrics(rows, prekeys, encoder, bases, args.fit_count,
                                   args.fit_count+args.diagnostic_count, device)}
    stats = {}
    for split, selection, start, stop in (
        ('fit', selection_fit, 0, args.fit_count),
        ('diagnostic', selection_diag, args.fit_count, args.fit_count+args.diagnostic_count)):
        stats[split], reconstruction[split]['causal_queries'] = build_multi_query_statistics(
            selected[split], rows[start:stop], query_positions=selection['selected_positions'],
            cos=cos, sin=sin, value_encoder=encoder, base_maps=bases, page_size=32,
            excluded_prefix_pages=1, device=device)
    factors, losses = _fit_residual_grid(stats['fit'], stats['diagnostic'], residual_ranks=(16,),
        sweeps=40, relative_damping=1e-5, iterative_tolerance=1e-5,
        iterative_max_iterations=100, device=device)
    tensors = {f'base_{name}_b16': torch.stack([getattr(m, name) for m in bases[16]]).float()
               for name in ('left', 'right', 'bias')}
    tensors.update(residual_encoder_b16_r16=factors[(16,16)][0], residual_query_b16_r16=factors[(16,16)][1])
    hkv, hq = rows.shape[2], queries.shape[2]
    shapes = [(hkv,rank,16), (hkv,16,dim), (hkv,dim), (hkv,dim,16), (hq,dim,16)]
    for t, shape in zip(tensors.values(), shapes, strict=True):
        assert t.shape == shape and t.dtype == torch.float32 and torch.isfinite(t).all()
    # A small held-out reconstruction check of both base and query-specific residual decoders.
    z = _value_codes(rows[args.fit_count,:256,:,:dim].cuda().float(), encoder.cuda().float())
    base = _post_rope_rows(_apply_base(z, _stack_base_map(bases[16],device=device)), cos[:,:256], sin[:,:256])
    exact = rows[args.fit_count,:256,:,dim:].cuda().float()
    residual_code = torch.einsum('tgd,gdr->tgr', exact-base, tensors['residual_encoder_b16_r16'].cuda())
    mapping = torch.arange(hq,device=device)//(hq//hkv)
    proxy = base[:,mapping] + torch.einsum('thr,hdr->thd', residual_code[:,mapping], tensors['residual_query_b16_r16'].cuda())
    assert torch.isfinite(proxy).all()
    smoke = dict(tokens=256, diagnostic_window=64,
        base_relative_mse=float((base-exact).square().sum()/exact.square().sum()),
        residual_relative_mse=float((proxy-exact[:,mapping]).square().sum()/exact[:,mapping].square().sum()))
    path = args.output / 'ours_b16r16' / f'layer_{layer:03d}.safetensors'
    save_tensors(path, tensors)
    write_json(path.with_suffix('.json'), dict(status='complete', layer=layer, protocol=spec,
        v_rank=rank, sha256=sha256(path), capture_manifest_sha256=sha256(capture.with_suffix('.json')),
        query_manifest_sha256=sha256(query_path.with_suffix('.json')), reconstruction=reconstruction,
        losses=losses, numerical_smoke=smoke, seconds=time.monotonic()-started,
        command=shlex.join(sys.argv), python=sys.executable, gpu=torch.cuda.get_device_name(0)))
    print('FIT COMPLETE',layer,losses,smoke,flush=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--windows', type=Path, required=True)
    p.add_argument('--sequence-length', type=int, default=32768)
    p.add_argument('--fit-count', type=int, default=64)
    p.add_argument('--diagnostic-count', type=int, default=16)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=4)
    p.add_argument('--layers', type=str)
    args = p.parse_args()
    configure()
    identity = read_json(args.identity)
    assert identity['status'] == 'complete' and identity['mean_rank'] == 96
    checkpoint, model_path = Path(identity['checkpoint']), Path(identity['model'])
    manifest = read_json(checkpoint / 'manifest.json')
    assert sha256(checkpoint/'manifest.json') == identity['manifest_sha256']
    wm = read_json(args.windows.with_name('manifest.json'))
    assert wm['status'] == 'complete' and wm['sha256'] == sha256(args.windows)
    assert wm['model_config_sha256'] == identity['model_config_sha256']
    windows = load_file(str(args.windows))['input_ids']
    assert windows.shape[1] == args.sequence_length
    assert wm['fit_ids'] == list(range(64)) and wm['validation_ids'] == list(range(64,80))
    assert 0 < args.fit_count <= 64 and 0 < args.diagnostic_count <= 16
    windows = torch.cat((windows[:args.fit_count], windows[64:64+args.diagnostic_count]))
    layers = identity['attention_layers']
    targets = [int(x) for x in args.layers.split(',')] if args.layers else layers[args.shard_index::args.num_shards]
    assert targets and set(targets) <= set(layers)
    spec = dict(identity_sha256=sha256(args.identity), windows_sha256=sha256(args.windows),
        model_config_sha256=identity['model_config_sha256'], v96_manifest_sha256=identity['manifest_sha256'],
        layer_ranks=identity['layer_ranks'], fit_ids=list(range(args.fit_count)),
        diagnostic_ids=list(range(64,64+args.diagnostic_count)), sequence_length=args.sequence_length,
        fit_queries=64, diagnostic_queries=32, selection='fit-only stratified Query-Gram',
        base_rank=16, base_objective='affine pre-RoPE K MSE closed-form RRR from frozen V latent',
        residual_rank=16, residual_objective='causal non-sink Page-Fisher relative to frozen Base16',
        page_size=32, excluded_prefix_pages=1, bcd_sweeps=40, pcg_damping=1e-5,
        pcg_tolerance=1e-5, pcg_iterations=100, endpoint='fixed final sweep',
        intended_routing_budget=2048, factor_dtype='float32', teacher='native dense BF16 SDPA',
        source_sha256={name:sha256(ROOT/name) for name in ('evaluation/fit_k_routing.py',
            'basisserve/core/c1_v_conditional_k_router.py','basisserve/core/query_position_sampling.py',
            'evaluation/fit_qwen3_8b_q8_fisher_residual.py',
            'evaluation/eval_qwen3_8b_v80_conditional_residual_router.py')})
    write_json(args.output/'manifests'/f'protocol_shard_{args.shard_index}.json',spec)
    pending = []
    for layer in targets:
        path = args.output/'ours_b16r16'/f'layer_{layer:03d}.json'
        if path.exists():
            saved = read_json(path)
            assert saved['status']=='complete' and saved['protocol']==spec
            assert saved['sha256']==sha256(path.with_suffix('.safetensors'))
        else:
            pending.append(layer)
    if not pending:
        print('All requested layers verified complete',flush=True)
        return
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).eval()
    assert model.config.model_type in ('llama','qwen3')
    if model.config.model_type == 'llama':
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    else:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    hidden_size = model.config.hidden_size
    hq, hkv = model.config.num_attention_heads, model.config.num_key_value_heads
    dim = model.config.head_dim if hasattr(model.config,'head_dim') else hidden_size//hq
    n = args.sequence_length
    count = len(windows)
    hidden = torch.empty(count,n,hidden_size,dtype=torch.bfloat16)
    for i in range(count):
        hidden[i].copy_(model.model.embed_tokens(windows[i].long()))
    rotary = model.model.rotary_emb.cuda()
    positions = torch.arange(n,device='cuda')[None]
    cos,sin = rotary(torch.empty(1,device='cuda',dtype=torch.float32),positions)
    forward_rope = (cos.bfloat16(),sin.bfloat16())
    grid = candidate_positions(n)
    current = {}
    for layer,module in enumerate(model.model.layers):
        if layer > max(pending):
            break
        capture_path = args.output/'calibration'/f'layer_{layer:03d}.safetensors'
        needs = layer in pending
        reuse = needs and capture_path.with_suffix('.json').exists()
        if needs and not reuse:
            rows = torch.empty(count,n,hkv,2*dim,dtype=torch.bfloat16)
            prekeys = torch.empty(count,n,hkv,dim,dtype=torch.bfloat16)
            queries = torch.empty(count,len(grid),hq,dim,dtype=torch.bfloat16)
        module.cuda()
        print('DENSE LAYER START',layer,'capture',needs,'reuse',reuse,flush=True)
        def collect(attention, positional, kwargs):
            x = kwargs['hidden_states']
            q = attention.q_proj(x).view(1,n,hq,dim)
            k = attention.k_proj(x).view(1,n,hkv,dim)
            if model.config.model_type == 'qwen3':
                q, k = attention.q_norm(q), attention.k_norm(k)
            i = current['window']
            prekeys[i].copy_(k[0].cpu())
            q,k = apply_rotary_pos_emb(q.transpose(1,2),k.transpose(1,2),*kwargs['position_embeddings'])
            v = attention.v_proj(x).view(1,n,hkv,dim)
            rows[i].copy_(torch.cat((v[0],k[0].transpose(0,1)),-1).cpu())
            queries[i].copy_(q[0,:,grid].transpose(0,1).cpu())
        handle = module.self_attn.register_forward_pre_hook(collect,with_kwargs=True) if needs and not reuse else None
        for i in range(count):
            current['window']=i
            output = module(hidden[i:i+1].cuda(),attention_mask=None,position_ids=positions,
                position_embeddings=forward_rope,use_cache=False)
            value = output[0] if isinstance(output,tuple) else output
            assert value.shape == hidden[i:i+1].shape and torch.isfinite(value).all()
            hidden[i].copy_(value[0].cpu())
            if (i+1)%8 == 0:
                print('DENSE CAPTURE',layer,i+1,'/',count,'fit',needs,flush=True)
        if handle is not None:
            handle.remove()
        module.cpu()
        if needs:
            if reuse:
                saved = read_json(capture_path.with_suffix('.json'))
                assert saved['protocol']==spec and saved['sha256']==sha256(capture_path)
                captured = load_file(str(capture_path))
                rows,prekeys,queries = captured['rows'],captured['pre_rope_keys'],captured['candidate_queries']
                del captured
            record = manifest['layers'][layer]
            assert record['ranks'][0] == identity['layer_ranks'][layer]
            assert sha256(checkpoint/record['file']) == record['sha256']
            encoder = load_file(str(checkpoint/record['file']))['value_coordinate_encoders']
            if not reuse:
                save_tensors(capture_path, dict(rows=rows, pre_rope_keys=prekeys, candidate_queries=queries))
                write_json(capture_path.with_suffix('.json'), dict(status='complete', protocol=spec, layer=layer,
                    sha256=sha256(capture_path), row_layout='raw V concatenated with exact post-RoPE K',
                    pre_rope_keys='after architecture-specific normalization; before actual model RoPE',
                    candidate_positions=grid, window_ids=list(range(args.fit_count+args.diagnostic_count))))
            fit_layer(args,layer,rows,prekeys,queries,encoder,cos,sin,spec)
            del rows,prekeys,queries
    write_json(args.output/'manifests'/('layers_'+ '_'.join(map(str,targets))+'.json'),
               dict(status='complete',layers=targets,protocol=spec))


if __name__ == '__main__':
    main()
