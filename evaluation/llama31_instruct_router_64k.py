"""Fit-only Q64 Query-Gram selection, closed-form Base16, Page-Fisher R8."""

import argparse
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.llama31_instruct_c1_64k import (
    MODEL, RUN, FIT, DIAGNOSTIC, LENGTH, configure, install_chunked_mlp,
)
from evaluation.v96kl_common import sha256, read_json, write_json
from basisserve.core.query_position_sampling import candidate_positions, select_stratified_query_positions
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _fit_base_maps, _fit_residual_grid
from evaluation.fit_qwen3_8b_q8_fisher_residual import build_multi_query_statistics


def fit_layer(args, layer, rows, queries, c1, cos, sin, protocol):
    started = time.monotonic()
    grid = candidate_positions(LENGTH)
    selection, whitening, grams = select_stratified_query_positions(
        queries[:FIT].cuda(), grid, context_length=LENGTH, num_bins=8, queries_per_bin=8)
    positions = selection['selected_positions']
    assert len(positions) == 64
    assert all(sum(p//8192 == b for p in positions) == 8 for b in range(8))
    selected = queries[:, [grid.index(p) for p in positions]].contiguous()
    source = c1['artifacts'][str(layer)]
    c1_path = args.run/'c1'/source['file']
    assert sha256(c1_path) == source['sha256']
    encoder = load_file(str(c1_path))['value_coordinate_encoders']
    assert encoder.shape == (8,128,96)
    maps = _fit_base_maps(rows[:FIT], value_encoder=encoder, base_ranks=(16,),
        cos=cos, sin=sin, device=torch.device('cuda'), fit_bias=True)
    statistics, reconstruction = {}, {}
    for split, start, stop in [('fit',0,FIT), ('diagnostic',FIT,FIT+DIAGNOSTIC)]:
        statistics[split], reconstruction[split] = build_multi_query_statistics(
            selected[start:stop], rows[start:stop], query_positions=positions,
            value_encoder=encoder, base_maps=maps, cos=cos, sin=sin,
            page_size=32, excluded_prefix_pages=1, device=torch.device('cuda'))
    factors, losses = _fit_residual_grid(statistics['fit'], statistics['diagnostic'],
        residual_ranks=(8,), sweeps=40, relative_damping=1e-5,
        iterative_tolerance=1e-5, iterative_max_iterations=100, device=torch.device('cuda'))
    tensors = {f'base_{name}_b16': torch.stack([getattr(m,name) for m in maps[16]]).float()
               for name in ('left','right','bias')}
    tensors.update(residual_encoder_b16_r8=factors[(16,8)][0],
                   residual_query_b16_r8=factors[(16,8)][1])
    assert all(torch.isfinite(t).all() for t in tensors.values())
    directory = args.run/'router'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'layer_{layer:03d}.safetensors'
    assert not path.exists()
    save_file({k:v.cpu().contiguous() for k,v in tensors.items()}, str(path))
    selection_path = directory/f'query_{layer:03d}.safetensors'
    save_file(dict(whitening=whitening.cpu(), **{f'gram_{i}':g.cpu() for i,g in enumerate(grams)}),
              str(selection_path))
    write_json(path.with_suffix('.json'), dict(status='complete', layer=layer,
        protocol=protocol, sha256=sha256(path), query_selection=selection,
        selection_statistics_sha256=sha256(selection_path), losses=losses,
        reconstruction=reconstruction, seconds=time.monotonic()-started,
        command=shlex.join(sys.argv), python=sys.executable,
        peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    print('router complete', layer, 'losses', losses, flush=True)


@torch.inference_mode()
def run(args):
    c1 = read_json(args.run/'c1/results.json')
    assert c1['status'] == 'complete'
    config = c1['fit_config']
    assert config['cache_rank_per_head'] == 96 and config['fit_windows'] == FIT
    assert config['positions_per_window'] == LENGTH
    assert config['model_config_sha256'] == sha256(args.model/'config.json')
    window_path = args.run/'windows/windows.safetensors'
    window_record = read_json(window_path.parent/'manifest.json')
    assert sha256(window_path) == window_record['sha256']
    windows = load_file(str(window_path))['input_ids']
    assert windows.shape == (FIT+DIAGNOSTIC,LENGTH)
    protocol = dict(model=str(args.model), model_config_sha256=sha256(args.model/'config.json'),
        c1_sha256=sha256(args.run/'c1/results.json'), windows_sha256=sha256(window_path),
        fit_windows=FIT, diagnostic_windows=DIAGNOSTIC, sequence_length=LENGTH,
        value_rank=96, base_rank=16, residual_rank=8, query_count=64,
        query_selection='fit-only head-whitened Query-Gram; 8 bins of 8192, 8 pivots/bin',
        candidate_stride=64, base_objective='closed-form affine pre-RoPE K MSE RRR',
        residual_objective='causal non-sink exact-teacher Page-Fisher',
        page_size=32, excluded_prefix_pages=1, validation_selects_factors=False,
        bcd_sweeps=40, relative_damping=1e-5, pcg_tolerance=1e-5, pcg_iterations=100,
        teacher='original dense BF16 model; full causal SDPA flash; MLP chunk1024',
        source_sha256={p:sha256(ROOT/p) for p in (
            'evaluation/llama31_instruct_router_64k.py',
            'evaluation/llama31_instruct_c1_64k.py',
            'basisserve/core/query_position_sampling.py',
            'basisserve/core/c1_v_conditional_k_router.py',
            'evaluation/fit_qwen3_8b_q8_fisher_residual.py',
            'evaluation/eval_qwen3_8b_v80_conditional_residual_router.py')})
    assigned = list(range(args.shard*8, (args.shard+1)*8))
    completed = set()
    for layer in assigned:
        record_path = args.run/'router'/f'layer_{layer:03d}.json'
        if record_path.exists():
            record = read_json(record_path)
            assert record['status'] == 'complete' and record['protocol'] == protocol
            assert sha256(record_path.with_suffix('.safetensors')) == record['sha256']
            completed.add(layer)
    if len(completed) == len(assigned):
        print('router shard verified', args.shard, flush=True)
        return
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').eval()
    install_chunked_mlp(model)
    hidden = torch.empty(FIT+DIAGNOSTIC,LENGTH,4096,dtype=torch.bfloat16)
    for document in range(FIT+DIAGNOSTIC):
        hidden[document].copy_(model.embed_tokens(windows[document].long()))
    model.rotary_emb.cuda()
    position_ids = torch.arange(LENGTH, device='cuda')[None]
    cos, sin = model.rotary_emb(torch.empty(1,device='cuda',dtype=torch.float32), position_ids)
    forward_rope = (cos.bfloat16(), sin.bfloat16())
    grid = candidate_positions(LENGTH)
    for layer_index, layer in enumerate(model.layers):
        if layer_index > assigned[-1]:
            break
        needed = layer_index in assigned and layer_index not in completed
        rows = torch.empty(FIT+DIAGNOSTIC,LENGTH,8,256,dtype=torch.bfloat16) if needed else None
        queries = torch.empty(FIT+DIAGNOSTIC,len(grid),32,128,dtype=torch.bfloat16) if needed else None
        active = {}
        def collect(attention, positional, kwargs):
            x = kwargs['hidden_states']
            q = attention.q_proj(x).view(1,LENGTH,32,128).transpose(1,2)
            k = attention.k_proj(x).view(1,LENGTH,8,128).transpose(1,2)
            v = attention.v_proj(x).view(1,LENGTH,8,128)
            q, k = apply_rotary_pos_emb(q,k,*kwargs['position_embeddings'])
            document = active['document']
            rows[document].copy_(torch.cat((v[0],k[0].transpose(0,1)),dim=-1).cpu())
            queries[document].copy_(q[0,:,grid].transpose(0,1).cpu())
        layer.cuda()
        handle = layer.self_attn.register_forward_pre_hook(collect,with_kwargs=True) if needed else None
        for document in range(FIT+DIAGNOSTIC):
            active['document'] = document
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = layer(hidden[document:document+1].cuda(), attention_mask=None,
                    position_ids=position_ids, position_embeddings=forward_rope, use_cache=False)
            hidden[document].copy_(output[0].cpu())
            del output
            if (document+1)%8 == 0:
                print(f'shard={args.shard} layer={layer_index} forward={document+1}/80 fit={needed}',flush=True)
        if handle is not None:
            handle.remove()
        layer.cpu()
        torch.cuda.empty_cache()
        if needed:
            fit_layer(args,layer_index,rows,queries,c1,cos,sin,protocol)
        del rows, queries
    write_json(args.run/'router'/f'shard_{args.shard}.json',
        dict(status='complete', layers=assigned, protocol=protocol))


def merge(args):
    records = [read_json(args.run/'router'/f'layer_{layer:03d}.json') for layer in range(32)]
    protocol = records[0]['protocol']
    artifacts = {}
    for layer,record in enumerate(records):
        assert record['status'] == 'complete' and record['layer'] == layer
        assert record['protocol'] == protocol
        path = args.run/'router'/f'layer_{layer:03d}.safetensors'
        assert sha256(path) == record['sha256']
        assert len(record['query_selection']['selected_positions']) == 64
        artifacts[str(layer)] = dict(file=path.name,sha256=record['sha256'])
    write_json(args.run/'router/manifest.json',dict(status='complete',protocol=protocol,artifacts=artifacts))
    print('verified all 32 Base16/R8 Q64 layers',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--stage',choices=('fit','merge'),default='fit')
    args = parser.parse_args()
    assert 0 <= args.shard < 4
    configure()
    (run if args.stage == 'fit' else merge)(args)


if __name__ == '__main__':
    main()
