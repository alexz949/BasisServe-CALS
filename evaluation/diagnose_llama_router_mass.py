"""Matched attention mass and output errors on held-out native dense replay."""
import argparse
from pathlib import Path
import shlex
import sys
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from evaluation.fit_k_routing_streaming import verified
from evaluation.llama_sink_recent_routing import page_support
from evaluation.v96kl_common import configure, read_json, write_json, sha256


@torch.inference_mode()
def window_mass(value, key, queries, positions, cos, sin, banks, budget=1024, output_weight=None):
    batch,kv_heads,length,dim=key.shape
    heads=queries.shape[1]; groups=heads//kv_heads
    assert batch==1 and heads%kv_heads==0 and queries.shape==(1,heads,len(positions),dim)
    assert len(positions)>0 and min(positions)>=32 and max(positions)<length
    sidecars={}
    for rank,t in banks.items():
        sidecars[rank]=build_conditional_routing_sidecar(value,key,
            base_left=t[f'base_left_b{rank}'],base_right=t[f'base_right_b{rank}'],
            base_bias=t[f'base_bias_b{rank}'],residual_encoder=t[f'residual_encoder_b{rank}_r16'],cos=cos,sin=sin)
    totals={name:dict(attention_mass=0.,non_sink_attention_mass=0.) for name in ['exact_k']+[f'b{r}r16' for r in banks]}
    if output_weight is not None:
        assert output_weight.ndim==2 and output_weight.shape[1]==heads*dim
        output_weight=output_weight.float()
        values=value.float()
        for report in totals.values():
            report.update(output_squared_error=0.,output_energy=0.,wo_squared_error=0.,wo_energy=0.)
    for offset,position in enumerate(positions):
        q=queries[:,:,offset].float()
        logits=(q.reshape(batch,kv_heads,groups,dim)@key[:,:,:position+1].float().transpose(-1,-2))*dim**-.5
        probabilities=logits.softmax(-1)
        if output_weight is not None:
            dense_output=probabilities@values[:,:,:position+1]
            dense_wo=dense_output.reshape(batch,-1)@output_weight.T
        non_sink_logits=logits.clone();non_sink_logits[...,:32]=-torch.inf
        non_sink=non_sink_logits.softmax(-1)
        for rank in [None]+list(banks):
            name='exact_k' if rank is None else f'b{rank}r16'
            if rank is None:
                scores=logits
            else:
                code=torch.cat((q,torch.einsum('bhd,hdr->bhr',q,banks[rank][f'residual_query_b{rank}_r16'].float())),-1)
                scores=(code.reshape(batch,kv_heads,groups,-1)@sidecars[rank][:,:,:position+1].float().transpose(-1,-2))*dim**-.5
            ids,valid=page_support(scores,budget=budget)
            assert (valid.sum(-1)<=budget).all()
            ids=ids.clamp_max(position)[:,:,None].expand(batch,kv_heads,groups,-1)
            valid=valid[:,:,None].expand_as(ids)
            for metric,probs in [('attention_mass',probabilities),('non_sink_attention_mass',non_sink)]:
                mass=probs.gather(-1,ids).masked_fill(~valid,0).sum(-1)
                assert torch.isfinite(mass).all() and mass.min()>=0 and mass.max()<=1.00001
                totals[name][metric]+=float(mass.mean())/len(positions)
            if output_weight is not None:
                selected_logits=logits.gather(-1,ids).masked_fill(~valid,-torch.inf)
                selected_values=values.gather(2,ids[:,:,0,:,None].expand(batch,kv_heads,ids.shape[-1],dim))
                sparse_output=selected_logits.softmax(-1)@selected_values
                difference=sparse_output-dense_output
                wo_difference=difference.reshape(batch,-1)@output_weight.T
                for metric,tensor in [('output_squared_error',difference),('output_energy',dense_output),
                        ('wo_squared_error',wo_difference),('wo_energy',dense_wo)]:
                    assert torch.isfinite(tensor).all()
                    totals[name][metric]+=float(tensor.double().square().sum())
    return totals


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=['smoke','evaluate','summarize'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--base-ranks',type=int,nargs='+',default=[4,16])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--output-errors',action='store_true')
    p.add_argument('--query-ablation',action='store_true')
    args=p.parse_args();configure();r=args.root;out=args.output
    assert len(set(args.base_ranks))==len(args.base_ranks) and set(args.base_ranks)<={0,4,16}
    arms=[f'b{rank}r16' for rank in args.base_ranks]+['exact_k']
    metric_names=('attention_mass','non_sink_attention_mass')
    if args.query_ablation:
        assert args.base_ranks==[4] and not args.output_errors
        from evaluation.query_ablation_metrics import ARMS,load_queries,query_metrics
        arms=ARMS
        metric_names+=('page_recall','routed_page_recall')
    assert 0<=args.shard_index<4
    identity=read_json(r/'manifests/v128.json')
    wm=read_json(r/'calibration/manifest.json')
    assert wm['sha256']==sha256(r/'calibration/windows.safetensors')
    assert wm['validation_ids']==list(range(64,80))
    assert identity['layer_ranks']==[128]*32
    banks,positions,hashes={},{},{}
    for layer in range(32):
        banks[layer]={};hashes[str(layer)]={}
        moment_path=r/'moments'/f'layer_{layer:03d}.safetensors'
        selection=read_json(moment_path.with_suffix('.json'))
        assert selection['protocol']['windows_sha256']==wm['sha256'] and selection['selection_fit_only']
        positions[layer]=selection['selections']['heldout']['selected_positions']
        assert len(positions[layer])==32
        for rank in args.base_ranks:
            base_root=r if rank==16 else r/f'b{rank}r16'
            bankroot=base_root/f'ours_b{rank}r16'
            path=bankroot/f'layer_{layer:03d}.safetensors'
            tensors,meta=verified(path)
            spec=meta['protocol']
            assert not spec['smoke'] and spec['base_rank']==rank and spec['residual_rank']==16
            assert spec['windows_sha256']==wm['sha256'] and spec['sequence_length']==65536
            assert spec['fit_ids']==list(range(64)) and spec['diagnostic_ids']==list(range(64,80))
            assert meta['identity_sha256']==sha256(r/'manifests/v128.json')
            bm=read_json(base_root/'base'/f'layer_{layer:03d}.json')
            assert bm['moments_sha256']==selection['sha256']
            if rank==0:
                assert tensors['base_left_b0'].shape[-1]==0 and torch.count_nonzero(tensors['base_bias_b0'])==0
            banks[layer][rank]=tensors;hashes[str(layer)][str(rank)]=meta['sha256']
    protocol=dict(model=identity['model'],identity_sha256=sha256(r/'manifests/v128.json'),
        windows_sha256=wm['sha256'],sequence_length=65536,diagnostic_ids=list(range(64,80)),
        queries_per_window=32,positions=positions,budget=1024,sink_tokens=32,recent_tokens=64,page_size=32,
        exact_reference='Exact-K, identical Page32 GQA max and sink/recent support',
        arithmetic='BF16 dense activations and runtime sidecars; FP32 query projection, scores and softmax',
        aggregation='equal mean over query heads, selected queries and 16 windows; then equal layer mean',
        bank_sha256=hashes,source_sha256={n:sha256(Path(n)) for n in [__file__,
            'evaluation/llama_sink_recent_routing.py','basisserve/core/c1_conditional_page_attention.py',
            'basisserve/core/c1_v_conditional_k_router.py']})
    protocol['query_ablation']=args.query_ablation
    if args.query_ablation:
        protocol['query_controls']='Base4-only, original affine bias; correct q; same-head same-position q from next held-out window modulo 16; per-head mean over all 64 fit windows and Q64; post-RoPE predicted K norm; random historical pages with same sink/recent; Exact K. Page recall is overlap with Exact-K support; routed_page_recall excludes pinned sink and recent tokens. Empty Exact-K routed support is excluded from routed page recall; eligible query/KV-head pairs weighted equally per window. Random seed 42+layer*100000+heldout_index*100+query_index.'
        protocol['source_sha256']['evaluation/query_ablation_metrics.py']=sha256(Path('evaluation/query_ablation_metrics.py'))
    protocol['output_errors']=args.output_errors
    if args.output_errors:
        protocol['output_error_protocol']='FP32 exact causal QK softmax times original BF16 V cast to FP32; sparse exact QK renormalized over selected valid tokens; original Wo FP32, no bias; sums of squared errors / sums of dense squared norms over all query heads and windows per layer; summary equal layer mean plus pooled energy ratio; teacher-forced local error, no cross-layer error propagation'
    # JSON object keys are strings after serialization.
    protocol['positions']={str(k):v for k,v in positions.items()}
    if args.stage=='summarize':
        reports=[]
        for layer in range(32):
            d=read_json(out/f'layer_{layer:03d}.json')
            assert d['status']=='complete' and d['protocol']==protocol and d['layer']==layer
            assert [w['window'] for w in d['windows']]==list(range(64,80))
            reports.append(d)
        means={arm:{metric:sum(d['means'][arm][metric] for d in reports)/32
            for metric in metric_names} for arm in arms}
        pooled={}
        if args.output_errors:
            for arm in arms:
                for metric in ('output_rel_mse','wo_rel_mse'):
                    means[arm][metric]=sum(d['means'][arm][metric] for d in reports)/32
                pooled[arm]={f'{prefix}_rel_mse':sum(d['error_sums'][arm][f'{prefix}_squared_error'] for d in reports)/sum(d['error_sums'][arm][f'{prefix}_energy'] for d in reports) for prefix in ('output','wo')}
        write_json(out/'summary.json',dict(status='complete',protocol=protocol,means=means,pooled_output_errors=pooled,
            layers=[dict(layer=d['layer'],means=d['means']) for d in reports]))
        print(means,flush=True);return
    smoke_path=out/('query_smoke.json' if args.query_ablation else 'smoke.json')
    if args.stage=='evaluate':
        smoke=read_json(smoke_path);assert smoke['status']=='complete' and smoke['protocol']==protocol
    layers=[0] if args.stage=='smoke' else list(range(args.shard_index*8,(args.shard_index+1)*8))
    selected_windows=[64] if args.stage=='smoke' else list(range(64,80))
    pools={layer:load_queries(r,layer,wm['sha256']) for layer in layers} if args.query_ablation else {}
    model=AutoModelForCausalLM.from_pretrained(identity['model'],dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).eval().cuda()
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    windows=load_file(str(r/'calibration/windows.safetensors'))['input_ids']
    assert tuple(windows.shape)==(80,65536)
    records={l:[] for l in layers};handles=[];active={}
    for layer in layers:
        banks[layer]={rank:{k:v.cuda() for k,v in tensors.items()} for rank,tensors in banks[layer].items()}
        def capture(module,positional,kwargs,layer=layer):
            x=kwargs['hidden_states'];n=x.shape[1]
            q=module.q_proj(x).view(1,n,32,128).transpose(1,2)
            k=module.k_proj(x).view(1,n,8,128).transpose(1,2)
            v=module.v_proj(x).view(1,n,8,128).transpose(1,2)
            cos,sin=kwargs['position_embeddings'];q,k=apply_rotary_pos_emb(q,k,cos,sin)
            if args.query_ablation:
                metrics=query_metrics(v,k,q[:,:,positions[layer]],positions[layer],cos,sin,banks[layer][4],*pools[layer],active['window'],layer)
            else:
                metrics=window_mass(v,k,q[:,:,positions[layer]],positions[layer],cos,sin,banks[layer],
                    output_weight=module.o_proj.weight if args.output_errors else None)
            records[layer].append(dict(window=active['window'],metrics=metrics))
            print(dict(layer=layer,window=active['window'],metrics=metrics),flush=True)
        handles.append(model.model.layers[layer].self_attn.register_forward_pre_hook(capture,with_kwargs=True))
    for window in selected_windows:
        active['window']=window
        result=model.model(windows[window:window+1].long().cuda(),use_cache=False)
        assert torch.isfinite(result.last_hidden_state).all()
        del result
    for h in handles:h.remove()
    for layer in layers:
        means={arm:{metric:sum(w['metrics'][arm][metric] for w in records[layer])/len(selected_windows)
            for metric in metric_names} for arm in arms}
        error_sums={}
        if args.output_errors:
            for arm in arms:
                error_sums[arm]={metric:sum(w['metrics'][arm][metric] for w in records[layer]) for metric in ('output_squared_error','output_energy','wo_squared_error','wo_energy')}
                for prefix in ('output','wo'):
                    assert error_sums[arm][f'{prefix}_energy']>0
                    means[arm][f'{prefix}_rel_mse']=error_sums[arm][f'{prefix}_squared_error']/error_sums[arm][f'{prefix}_energy']
        path=smoke_path if args.stage=='smoke' else out/f'layer_{layer:03d}.json'
        write_json(path,dict(status='complete',layer=layer,protocol=protocol,means=means,
            windows=records[layer],error_sums=error_sums,command=shlex.join(sys.argv),python=sys.executable))


if __name__=='__main__':main()
