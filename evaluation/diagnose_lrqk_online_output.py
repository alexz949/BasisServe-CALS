"""Continuous official LRQK state versus fitted routers on the same dense teacher tail."""
import argparse
from pathlib import Path
import shlex
import sys
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from basisserve.core.c1_lrqk import LRQKConfig
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from evaluation.official_lrqk_state import OfficialLRQKState
from evaluation.llama_sink_recent_routing import page_support
from evaluation.fit_k_routing_streaming import verified
from evaluation.v96kl_common import configure,read_json,write_json,sha256

ARMS=('b0r16','b4r16','b16r16','b0r20','exact_k','lrqk')
PAIRS=((0,16),(4,16),(16,16),(0,20))


@torch.inference_mode()
def online_errors(q,k,v,cos,sin,wo,banks,steps,budget,lrqk_topk):
    batch,heads,length,dim=q.shape;kv=k.shape[1];groups=heads//kv;prefix=65280
    assert batch==1 and length==65536 and 0<steps<=256
    torch.manual_seed(0)
    state=OfficialLRQKState(q[:,:,:prefix],k[:,:,:prefix],LRQKConfig(topk=lrqk_topk))
    state.bind_values(v[:,:,:prefix])
    sidecars={}
    for b,r in PAIRS:
        tag=f'b{b}r{r}';f=banks[tag]
        sidecars[tag]=build_conditional_routing_sidecar(v,k,base_left=f[f'base_left_b{b}'],
            base_right=f[f'base_right_b{b}'],base_bias=f[f'base_bias_b{b}'],
            residual_encoder=f[f'residual_encoder_b{b}_r{r}'],cos=cos,sin=sin)
    metrics={a:{m:0. for m in ('output_squared_error','output_energy','wo_squared_error','wo_energy')} for a in ARMS}
    wo=wo.float();vf=v.float()
    for position in range(prefix,prefix+steps):
        query=q[:,:,position:position+1];n=position+1
        logits=(query[:,:,0].float().reshape(batch,kv,groups,dim)@k[:,:,:n].float().transpose(-1,-2))*dim**-.5
        dense=logits.softmax(-1)@vf[:,:,:n]
        dense_wo=dense.reshape(batch,-1)@wo.T
        for b,r in (*PAIRS,(None,None)):
            tag='exact_k' if b is None else f'b{b}r{r}'
            if b is None:scores=logits
            else:
                f=banks[tag];qf=query[:,:,0].float()
                code=torch.cat((qf,torch.einsum('bhd,hdr->bhr',qf,f[f'residual_query_b{b}_r{r}'].float())),-1)
                scores=(code.reshape(batch,kv,groups,-1)@sidecars[tag][:,:,:n].float().transpose(-1,-2))*dim**-.5
            ids,valid=page_support(scores,budget=budget)
            assert (valid.sum(-1)<=budget).all()
            gather=ids.clamp_max(position)[:,:,None].expand(batch,kv,groups,-1)
            selected_logits=logits.gather(-1,gather).masked_fill(~valid[:,:,None],-torch.inf)
            selected_v=vf.gather(2,ids.clamp_max(position)[...,None].expand(batch,kv,ids.shape[-1],dim))
            predicted=selected_logits.softmax(-1)@selected_v
            difference=predicted-dense;wo_difference=difference.reshape(batch,-1)@wo.T
            for name,t in [('output_squared_error',difference),('output_energy',dense),('wo_squared_error',wo_difference),('wo_energy',dense_wo)]:
                assert torch.isfinite(t).all()
                metrics[tag][name]+=float(t.double().square().sum())
        # One state persists through every token; no reset or refit between queries.
        selected_k,selected_v=state.cache.decode(query,k[:,:,position:position+1],v[:,:,position:position+1])
        assert selected_k.shape==selected_v.shape==(1,heads,lrqk_topk+64,dim)
        assert torch.isfinite(selected_k).all() and torch.isfinite(selected_v).all()
        selected_logits=(query.float()@selected_k.float().transpose(-1,-2))*dim**-.5
        predicted=(selected_logits.softmax(-1)@selected_v.float()).reshape(batch,kv,groups,dim)
        difference=predicted-dense;wo_difference=difference.reshape(batch,-1)@wo.T
        for name,t in [('output_squared_error',difference),('output_energy',dense),('wo_squared_error',wo_difference),('wo_energy',dense_wo)]:
            assert torch.isfinite(t).all()
            metrics['lrqk'][name]+=float(t.double().square().sum())
        if (position-prefix+1)%32==0:print('ONLINE_STEP',position-prefix+1,flush=True)
    return metrics


def aggregate(windows):
    sums={a:{m:sum(w['metrics'][a][m] for w in windows) for m in ('output_squared_error','output_energy','wo_squared_error','wo_energy')} for a in ARMS}
    means={a:{f'{p}_rel_mse':sums[a][f'{p}_squared_error']/sums[a][f'{p}_energy'] for p in ('output','wo')} for a in ARMS}
    return sums,means


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['smoke','evaluate','summarize'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--budget',type=int,choices=[1024,2048],required=True)
    p.add_argument('--lrqk-topk',type=int,choices=[1024,2048],required=True)
    args=p.parse_args();configure();root=args.root;out=root/f'online_output{args.budget}_lrqk{args.lrqk_topk}';assert 0<=args.shard_index<4
    identity=read_json(root/'manifests/v128.json');wm=read_json(root/'calibration/manifest.json')
    assert wm['sha256']==sha256(root/'calibration/windows.safetensors')
    spec=dict(identity_sha256=sha256(root/'manifests/v128.json'),windows_sha256=wm['sha256'],
        windows=list(range(64,80)),sequence_length=65536,prefill_length=65280,online_decode_tokens=256,
        teacher='Native dense BF16 SDPA replay; fixed teacher QKV, layer-local output errors',
        lrqk=dict(implementation='official LightAttentionIndicesFactory',rank=32,topk=args.lrqk_topk,lite=64,iterations=[2,2],tolerance=.01,
            seed=0,state='one prefill, then 256 consecutive decode updates without resetting; native cache semantics unchanged'),
        ours=dict(token_budget=args.budget,sink=32,recent=64,page=32),
        arithmetic='FP32 exact selected QK softmax times original BF16 V cast to FP32; original Wo FP32; official LRQK selected K/V used unchanged',
        source_sha256={name:sha256(Path(name)) for name in [__file__,'evaluation/official_lrqk_state.py',
            'external/LRQK/lrqk_attention.py','evaluation/llama_sink_recent_routing.py','basisserve/core/c1_v_conditional_k_router.py']})
    if args.stage=='summarize':
        reports=[read_json(out/f'layer_{l:03d}.json') for l in range(32)]
        for l,d in enumerate(reports):
            assert d['status']=='complete' and d['protocol']==spec and d['layer']==l
            assert [w['window'] for w in d['windows']]==list(range(64,80))
            if args.budget==2048:
                old=read_json(root/'online_output1024'/f'layer_{l:03d}.json')
                for current,reference in zip(d['windows'],old['windows']):
                    for field in ('output_energy','wo_energy'):
                        a=current['metrics']['exact_k'][field];b=reference['metrics']['exact_k'][field]
                        assert abs(a-b)<=1e-5*max(abs(b),1e-12), (l,current['window'],field,a,b)
        means={a:{m:sum(d['means'][a][m] for d in reports)/32 for m in ('output_rel_mse','wo_rel_mse')} for a in ARMS}
        pooled={a:{f'{p}_rel_mse':sum(d['error_sums'][a][f'{p}_squared_error'] for d in reports)/sum(d['error_sums'][a][f'{p}_energy'] for d in reports) for p in ('output','wo')} for a in ARMS}
        write_json(out/'summary.json',dict(status='complete',protocol=spec,means=means,pooled=pooled,
            layers=[dict(layer=d['layer'],means=d['means']) for d in reports]));print(means,flush=True);return
    if args.stage=='evaluate':
        smoke=read_json(out/'smoke.json');assert smoke['status']=='complete' and smoke['protocol']==spec
        assert read_json(root/'b0r20/fit_audit.json')['status']=='complete'
    layers=[0] if args.stage=='smoke' else list(range(args.shard_index*8,(args.shard_index+1)*8))
    banks={};hashes={}
    for layer in layers:
        banks[layer]={};hashes[layer]={}
        for b,r in PAIRS:
            tag=f'b{b}r{r}';folder=root if tag=='b16r16' else root/tag
            path=folder/f'ours_{tag}'/f'layer_{layer:03d}.safetensors';f,meta=verified(path)
            assert meta['identity_sha256']==spec['identity_sha256'] and meta['protocol']['base_rank']==b and meta['protocol']['residual_rank']==r
            assert not meta['protocol']['smoke'] and meta['sweeps']==40 and meta['pcg_iterations']==100
            banks[layer][tag]={k:v.cuda() for k,v in f.items()};hashes[layer][tag]=meta['sha256']
    model=AutoModelForCausalLM.from_pretrained(identity['model'],dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).eval().cuda()
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    windows=load_file(str(root/'calibration/windows.safetensors'))['input_ids'];assert tuple(windows.shape)==(80,65536)
    records={l:[] for l in layers};active={};handles=[];steps=256
    for layer in layers:
        def capture(module,positional,kwargs,layer=layer):
            x=kwargs['hidden_states'];n=x.shape[1]
            q=module.q_proj(x).view(1,n,32,128).transpose(1,2)
            k=module.k_proj(x).view(1,n,8,128).transpose(1,2)
            v=module.v_proj(x).view(1,n,8,128).transpose(1,2)
            cos,sin=kwargs['position_embeddings'];q,k=apply_rotary_pos_emb(q,k,cos,sin)
            metrics=online_errors(q,k,v,cos,sin,module.o_proj.weight,banks[layer],steps,args.budget,args.lrqk_topk)
            records[layer].append(dict(window=active['window'],steps=steps,metrics=metrics))
            print(dict(layer=layer,window=active['window'],means=aggregate(records[layer])[1]),flush=True)
        handles.append(model.model.layers[layer].self_attn.register_forward_pre_hook(capture,with_kwargs=True))
    for window in ([64] if args.stage=='smoke' else range(64,80)):
        active['window']=window;result=model.model(windows[window:window+1].long().cuda(),use_cache=False)
        assert torch.isfinite(result.last_hidden_state).all();del result
    for h in handles:h.remove()
    for layer in layers:
        sums,means=aggregate(records[layer]);path=out/'smoke.json' if args.stage=='smoke' else out/f'layer_{layer:03d}.json'
        write_json(path,dict(status='complete',layer=layer,protocol=spec,bank_sha256=hashes[layer],windows=records[layer],
            error_sums=sums,means=means,command=shlex.join(sys.argv),python=sys.executable))


if __name__=='__main__':main()
