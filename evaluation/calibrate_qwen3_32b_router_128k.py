"""Qwen3-32B Base16/Fisher-R16 router calibration on 32 x 128K C4 windows."""
import argparse
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers.masking_utils import create_causal_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.v96kl_common import (
    MODEL, CHECKPOINT, CALIBRATION, BANK, configure,
    read_json, write_json, save_tensors, sha256, code_hashes,
)
from evaluation.uniform96_common import paths
from basisserve.core.query_position_sampling import candidate_positions, select_stratified_query_positions
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _fit_base_maps, _fit_residual_grid, _rotary_embeddings
from evaluation.streaming_k_statistics import packed_fisher
from evaluation.fit_qwen3_8b_q8_fisher_residual import build_multi_query_statistics


SEQUENCE_LENGTH = 131072
FIT_WINDOWS = 32
DISTINCT_DOCUMENTS = 1024
YARN_FACTOR = 4.0
UNIFORM_VALUE_RANK = 96


def checkpoint_manifest(checkpoint, model):
    """Read the 128K uniform-V96 manifest schema and verify every layer artifact."""
    record = read_json(checkpoint / 'manifest.json')
    assert record['status'] == 'complete'
    compression = record['compression']
    assert compression['uniform_v_rank_per_physical_head'] == UNIFORM_VALUE_RANK
    assert record['model']['config_sha256'] == sha256(model / 'config.json')
    audit = read_json(checkpoint / 'AUDIT.json')
    assert audit['status'] == 'complete'
    assert audit['model_config_sha256'] == record['model']['config_sha256']
    assert audit['model_revision'] == record['model']['revision']
    layers = []
    for index in range(audit['layers']):
        name = f'layer_{index:03d}.safetensors'
        assert sha256(checkpoint / name) == audit['artifact_sha256'][name]
        layers.append(dict(layer=index, file=name,
                           ranks=[UNIFORM_VALUE_RANK] * compression['physical_kv_heads']))
    return dict(model=record['model'], layers=layers)


def save_layer_statistics(root, layer, statistics, bases, encoder, selection, protocol):
    """Persist the sufficient statistics this layer's residual fit consumes.

    The teacher replay is by far the most expensive part of this calibration and
    it does not depend on the residual rank, sweep count or PCG budget. Writing
    the packed page-Fisher Grams plus the fitted base maps lets a later refit run
    _fit_residual_grid straight off disk, the way fit_k_routing_streaming.py does.
    """
    root.mkdir(parents=True, exist_ok=True)
    # build_multi_query_statistics returns one routing object per base rank.
    routing = statistics[16]
    payload = packed_fisher(routing)
    payload.update(encoder=encoder.cpu())
    payload.update({f'base_{name}': torch.stack([getattr(m, name) for m in bases[16]]).cpu()
                    for name in ('left', 'right', 'bias')})
    path = root / f'layer_{layer:03d}.safetensors'
    save_tensors(path, payload)
    write_json(path.with_suffix('.json'), dict(status='complete', layer=layer, protocol=protocol,
        sha256=sha256(path), query_selection=selection, base_rank=16,
        heads=int(routing.queries_by_head.shape[0]),
        examples=int(routing.queries_by_head.shape[1]),
        key_dim=int(routing.key_dim), scaling=float(routing.scaling),
        teacher_fisher_energy=float(routing.teacher_fisher_energy)))
    print('statistics saved layer', layer, 'packed', tuple(payload['packed'].shape), flush=True)


def fit_layer(args, layer, rows, queries, manifest, cos, sin, protocol):
    started=time.monotonic()
    grid=candidate_positions(SEQUENCE_LENGTH)
    selection,_,_=select_stratified_query_positions(queries[:FIT_WINDOWS].cuda(),grid,
        context_length=SEQUENCE_LENGTH,num_bins=4,queries_per_bin=8)
    positions=selection['selected_positions']
    selected_queries=queries[:,[grid.index(p) for p in positions]].contiguous()
    encoder=load_file(str(args.checkpoint/manifest['layers'][layer]['file']))['value_coordinate_encoders']
    bases=_fit_base_maps(rows[:FIT_WINDOWS],value_encoder=encoder,base_ranks=(16,),cos=cos,sin=sin,device=torch.device('cuda'))
    # All FIT_WINDOWS windows fit. The bank carries no held-out windows, so the
    # validation statistics are the same object: the reported validation NMSE is
    # in-sample by construction, and protocol['validation_scope'] records that.
    shared,shared_diagnostics=build_multi_query_statistics(selected_queries[:FIT_WINDOWS],rows[:FIT_WINDOWS],
        query_positions=positions,cos=cos,sin=sin,value_encoder=encoder,base_maps=bases,
        page_size=32,excluded_prefix_pages=1,device=torch.device('cuda'))
    stats={'fit':shared,'validation':shared}
    diagnostics={'fit':shared_diagnostics,'validation':shared_diagnostics}
    if args.statistics is not None:
        save_layer_statistics(args.statistics,layer,shared,bases,encoder,selection,protocol)
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
    assert window_manifest['fit_ids']==list(range(FIT_WINDOWS)) and window_manifest['validation_ids']==[]
    assert len({r['document_sha256'] for r in window_manifest['records']})==DISTINCT_DOCUMENTS
    windows=load_file(str(window_path))['input_ids']
    assert windows.shape==(FIT_WINDOWS,SEQUENCE_LENGTH)
    source=code_hashes(['evaluation/calibrate_qwen3_32b_router_128k.py','evaluation/uniform96_common.py','evaluation/v96kl_common.py',
        'basisserve/core/query_position_sampling.py','basisserve/core/c1_v_conditional_k_router.py',
        'evaluation/fit_qwen3_8b_q8_fisher_residual.py',
        'evaluation/eval_qwen3_8b_v80_conditional_residual_router.py',
        'basisserve/core/gqa_joint_routing_payload_s80_ablation.py',
        'basisserve/core/gqa_joint_routing_payload_s80_fisher.py'])
    protocol=dict(checkpoint_sha256=sha256(args.checkpoint/'manifest.json'),
        windows_sha256=sha256(window_path),fit_windows=FIT_WINDOWS,validation_windows=FIT_WINDOWS,
        validation_scope='in-sample: the 32x128K bank carries no held-out windows, so the reported '
                         'validation NMSE equals the fit NMSE and is not a generalisation claim',
        sequence_length=SEQUENCE_LENGTH,yarn_factor=YARN_FACTOR,
        base_rank=16,residual_rank=16,query_count=32,
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
    # The V96 checkpoint was calibrated at 128K under static YaRN 4.0; the teacher
    # must use the same rope or the collected Q/K leave the served distribution.
    config=AutoConfig.from_pretrained(args.model,local_files_only=True)
    native_maximum=config.max_position_embeddings
    config.rope_parameters={**dict(config.rope_parameters),'rope_type':'yarn',
        'factor':YARN_FACTOR,'original_max_position_embeddings':native_maximum}
    config.max_position_embeddings=int(round(native_maximum*YARN_FACTOR))
    assert config.max_position_embeddings>=SEQUENCE_LENGTH
    query_heads,kv_heads,head_dim=config.num_attention_heads,config.num_key_value_heads,config.head_dim
    model=AutoModelForCausalLM.from_pretrained(args.model,config=config,torch_dtype=torch.bfloat16,
        local_files_only=True,attn_implementation='sdpa').eval()
    # One 20 GiB host buffer is updated in place after each layer/document.
    hidden=torch.empty(FIT_WINDOWS,SEQUENCE_LENGTH,config.hidden_size,dtype=torch.bfloat16)
    for document in range(FIT_WINDOWS):
        hidden[document].copy_(model.model.embed_tokens(windows[document].long()))
    rotary=model.model.rotary_emb.to('cuda')
    cos,sin=rotary(torch.empty(1,device='cuda',dtype=torch.float32),torch.arange(SEQUENCE_LENGTH,device='cuda')[None])
    forward_rope=(cos.bfloat16(),sin.bfloat16())
    position_ids=torch.arange(SEQUENCE_LENGTH,device='cuda')[None]
    causal_mask=create_causal_mask(config=model.config,inputs_embeds=hidden[:1].cuda(),
        attention_mask=None,past_key_values=None,position_ids=position_ids)
    grid=candidate_positions(SEQUENCE_LENGTH)
    current={}
    for layer,module in enumerate(model.model.layers):
        if args.stop_after_layer is not None and layer>args.stop_after_layer: break
        started=time.monotonic()
        module.to('cuda')
        needs_fit=layer not in completed and layer % args.num_shards == args.shard_index
        # Raw V/K for one layer: 10 GiB, candidate Q: 320 MiB. Both are released per layer.
        rows=torch.empty(FIT_WINDOWS,SEQUENCE_LENGTH,kv_heads,2*head_dim,dtype=torch.bfloat16) if needs_fit else None
        queries=torch.empty(FIT_WINDOWS,len(grid),query_heads,head_dim,dtype=torch.bfloat16) if needs_fit else None
        def collect(attention,positional,kwargs):
            x=kwargs['hidden_states'];n=x.shape[1]
            q=attention.q_proj(x).view(1,n,query_heads,head_dim)
            k=attention.k_proj(x).view(1,n,kv_heads,head_dim)
            if args.model_family=='qwen3':
                q=attention.q_norm(q);k=attention.k_norm(k)
            q=q.transpose(1,2);k=k.transpose(1,2)
            v=attention.v_proj(x).view(1,n,kv_heads,head_dim)
            q,k=apply_rotary_pos_emb(q,k,*kwargs['position_embeddings'])
            document=current['document']
            rows[document].copy_(torch.cat((v[0],k[0].transpose(0,1)),dim=-1).cpu())
            queries[document].copy_(q[0,:,grid].transpose(0,1).cpu())
        handle=module.self_attn.register_forward_pre_hook(collect,with_kwargs=True) if needs_fit else None
        for document in range(FIT_WINDOWS):
            current['document']=document
            output=module(hidden[document:document+1].cuda(),attention_mask=causal_mask,position_ids=position_ids,
                          position_embeddings=forward_rope,use_cache=False)
            hidden[document].copy_(output[0].cpu())
            del output
            if (document+1)%8==0:
                print('layer',layer,'forward',document+1,'/',FIT_WINDOWS,'refit',needs_fit,flush=True)
        if handle is not None: handle.remove()
        # Each decoder layer is visited exactly once, so release its weights instead
        # of copying them back to host RAM. This frees the GPU before fitting and
        # keeps resident host memory flat when several shards run side by side.
        module.to('meta')
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
    p.add_argument('--statistics',type=Path,default=None,
        help='Directory for per-layer packed Fisher statistics and base maps, reusable by a later refit')
    p.add_argument('--stop-after-layer',type=int,help='Fit through this layer with full calibration; subsequent run reuses factors')
    args=p.parse_args();configure()
    assert 0<=args.shard_index<args.num_shards
    for name,value in paths(args.model_family).items():
        if name in ['model','checkpoint','calibration','bank'] and getattr(args,name) is None: setattr(args,name,value)
    torch.backends.cuda.matmul.allow_tf32=True
    fit(args)


if __name__=='__main__': main()
