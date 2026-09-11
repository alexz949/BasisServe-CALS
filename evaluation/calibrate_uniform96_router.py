"""Layerwise in-memory dense-teacher calibration of Base16 and Fisher R16."""
import argparse
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers.masking_utils import create_causal_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.v96kl_common import (
    MODEL, CHECKPOINT, CALIBRATION, BANK, configure,
    read_json, write_json, save_tensors, sha256, code_hashes,
)
from evaluation.uniform96_common import checkpoint_manifest, paths
from basisserve.core.query_position_sampling import candidate_positions, select_stratified_query_positions
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _fit_base_maps, _fit_residual_grid, _rotary_embeddings
from evaluation.fit_qwen3_8b_q8_fisher_residual import build_multi_query_statistics


def fit_layer(args, layer, rows, queries, manifest, cos, sin, protocol):
    started=time.monotonic()
    grid=candidate_positions(32768)
    selection,_,_=select_stratified_query_positions(queries[:64].cuda(),grid,
        context_length=32768,num_bins=4,queries_per_bin=8)
    positions=selection['selected_positions']
    selected_queries=queries[:,[grid.index(p) for p in positions]].contiguous()
    encoder=load_file(str(args.checkpoint/manifest['layers'][layer]['file']))['value_coordinate_encoders']
    bases=_fit_base_maps(rows[:64],value_encoder=encoder,base_ranks=(16,),cos=cos,sin=sin,device=torch.device('cuda'))
    stats,diagnostics={},{}
    for name,start,stop in [('fit',0,64),('validation',64,80)]:
        stats[name],diagnostics[name]=build_multi_query_statistics(selected_queries[start:stop],rows[start:stop],
            query_positions=positions,cos=cos,sin=sin,value_encoder=encoder,base_maps=bases,
            page_size=32,excluded_prefix_pages=1,device=torch.device('cuda'))
    factors,losses=_fit_residual_grid(stats['fit'],stats['validation'],residual_ranks=(16,),
        sweeps=40,relative_damping=1e-5,iterative_tolerance=1e-5,iterative_max_iterations=100,
        device=torch.device('cuda'))
    tensors={f'base_{name}_b16':torch.stack([getattr(m,name) for m in bases[16]]).float()
             for name in ('left','right','bias')}
    tensors.update(residual_encoder_b16_r16=factors[(16,16)][0],residual_query_b16_r16=factors[(16,16)][1])
    assert all(torch.isfinite(t).all() for t in tensors.values())
    path=args.bank/f'layer_{layer:03d}.safetensors'
    if path.exists():
        prior=load_file(str(path))
        assert all(torch.equal(prior[n],t) for n,t in tensors.items())
    else: save_tensors(path,tensors)
    write_json(path.with_suffix('.json'),dict(status='complete',layer=layer,protocol=protocol,sha256=sha256(path),
        query_selection=selection,losses=losses,reconstruction=diagnostics,elapsed_seconds=time.monotonic()-started))
    print('fit complete layer',layer,'losses',losses,'seconds',time.monotonic()-started,flush=True)


@torch.inference_mode()
def fit(args):
    manifest=checkpoint_manifest(args.checkpoint,args.model)
    window_path=args.calibration/'windows.safetensors'
    window_manifest=read_json(args.calibration/'manifest.json')
    assert window_manifest['status']=='complete' and window_manifest['sha256']==sha256(window_path)
    assert window_manifest['model_config_sha256']==manifest['model']['config_sha256']
    assert window_manifest['fit_ids']==list(range(64)) and window_manifest['validation_ids']==list(range(64,80))
    assert len({r['document_sha256'] for r in window_manifest['records']})==640
    windows=load_file(str(window_path))['input_ids']
    assert windows.shape==(80,32768)
    source=code_hashes(['evaluation/calibrate_uniform96_router.py','evaluation/uniform96_common.py','evaluation/v96kl_common.py',
        'basisserve/core/query_position_sampling.py','basisserve/core/c1_v_conditional_k_router.py',
        'evaluation/fit_qwen3_8b_q8_fisher_residual.py',
        'evaluation/eval_qwen3_8b_v80_conditional_residual_router.py',
        'basisserve/core/gqa_joint_routing_payload_s80_ablation.py',
        'basisserve/core/gqa_joint_routing_payload_s80_fisher.py'])
    protocol=dict(checkpoint_sha256=sha256(args.checkpoint/'manifest.json'),
        windows_sha256=sha256(window_path),fit_windows=64,validation_windows=16,
        sequence_length=32768,base_rank=16,residual_rank=16,query_count=32,
        query_selection='fit-only per-head uncentered whitened Gram; 4 strata x 8 pivots',
        base_objective='closed-form affine pre-RoPE K MSE RRR',
        residual_objective='separate causal exact-teacher non-sink Page-Fisher',
        teacher='dense BF16 SDPA, causal complete windows; layerwise execution',
        fp32_matmul='TF32 enabled, matching the original Base/Fisher fitting entry points',
        storage='current-layer operands and hidden states in host RAM; no raw activation files',
        sweeps=40,relative_damping=1e-5,pcg_tolerance=1e-5,pcg_iterations=100,
        page_size=32,excluded_prefix_pages=1,validation_selects_factors=False,code_sha256=source)
    completed=set()
    for layer in range(len(manifest['layers'])):
        path=args.bank/f'layer_{layer:03d}.json'
        if path.exists():
            record=read_json(path)
            assert record['status']=='complete' and record['protocol']==protocol
            assert record['sha256']==sha256(path.with_suffix('.safetensors'))
            completed.add(layer)
    if len(completed)==len(manifest['layers']):
        print('All layers verified complete',flush=True)
        return
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
        local_files_only=True,attn_implementation='sdpa').eval()
    # One 20 GiB host buffer is updated in place after each layer/document.
    hidden=torch.empty(80,32768,4096,dtype=torch.bfloat16)
    for document in range(80):
        hidden[document].copy_(model.model.embed_tokens(windows[document].long()))
    rotary=model.model.rotary_emb.to('cuda')
    cos,sin=rotary(torch.empty(1,device='cuda',dtype=torch.float32),torch.arange(32768,device='cuda')[None])
    forward_rope=(cos.bfloat16(),sin.bfloat16())
    position_ids=torch.arange(32768,device='cuda')[None]
    causal_mask=create_causal_mask(config=model.config,input_embeds=hidden[:1].cuda(),
        attention_mask=None,cache_position=position_ids[0],past_key_values=None,position_ids=position_ids)
    grid=candidate_positions(32768)
    current={}
    for layer,module in enumerate(model.model.layers):
        if args.stop_after_layer is not None and layer>args.stop_after_layer: break
        started=time.monotonic()
        module.to('cuda')
        needs_fit=layer not in completed and layer % args.num_shards == args.shard_index
        # Raw V/K for one layer: 10 GiB, candidate Q: 320 MiB. Both are released per layer.
        rows=torch.empty(80,32768,8,256,dtype=torch.bfloat16) if needs_fit else None
        queries=torch.empty(80,len(grid),32,128,dtype=torch.bfloat16) if needs_fit else None
        def collect(attention,positional,kwargs):
            x=kwargs['hidden_states'];n=x.shape[1]
            q=attention.q_proj(x).view(1,n,32,128)
            k=attention.k_proj(x).view(1,n,8,128)
            if args.model_family=='qwen3':
                q=attention.q_norm(q);k=attention.k_norm(k)
            q=q.transpose(1,2);k=k.transpose(1,2)
            v=attention.v_proj(x).view(1,n,8,128)
            q,k=apply_rotary_pos_emb(q,k,*kwargs['position_embeddings'])
            document=current['document']
            rows[document].copy_(torch.cat((v[0],k[0].transpose(0,1)),dim=-1).cpu())
            queries[document].copy_(q[0,:,grid].transpose(0,1).cpu())
        handle=module.self_attn.register_forward_pre_hook(collect,with_kwargs=True) if needs_fit else None
        for document in range(80):
            current['document']=document
            output=module(hidden[document:document+1].cuda(),attention_mask=causal_mask,position_ids=position_ids,
                          position_embeddings=forward_rope,use_cache=False)
            hidden[document].copy_(output[0].cpu())
            del output
            if (document+1)%8==0:
                print('layer',layer,'forward',document+1,'/80','refit',needs_fit,flush=True)
        if handle is not None: handle.remove()
        module.cpu()
        if needs_fit: fit_layer(args,layer,rows,queries,manifest,cos,sin,protocol)
        del rows,queries
        print('layer complete',layer,'seconds',time.monotonic()-started,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-family',choices=['qwen3','llama31'],required=True)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=2)
    p.add_argument('--model',type=Path)
    p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--calibration',type=Path,default=None)
    p.add_argument('--bank',type=Path,default=None)
    p.add_argument('--stop-after-layer',type=int,help='Fit through this layer with full calibration; subsequent run reuses factors')
    args=p.parse_args();configure()
    assert 0<=args.shard_index<args.num_shards
    for name,value in paths(args.model_family).items():
        if name in ['model','checkpoint','calibration','bank'] and getattr(args,name) is None: setattr(args,name,value)
    torch.backends.cuda.matmul.allow_tf32=True
    fit(args)


if __name__=='__main__': main()
