"""Prompt-local identity/diagonal/full residual-core diagnostic on frozen C1 trajectories."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import eval_longbench_c1_v96_router as source
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256,write_json
from evaluation.eval_longbench_lrqk_fp16 import v100_prefill,kernel_check
from basisserve.checkpoint import gqa_vo_qwen3 as attention_impl
from basisserve.checkpoint.gqa_vo_qwen3 import install_qwen3_gqa_vo_als_export
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar,residual_page_fisher_gram
from basisserve.core.c1_prompt_core import core_statistics,fit_cores,quadratic_loss
from basisserve.core.c1_conditional_page_attention import _selected_pages

INDICES=[0,1,32,33,64,65,96,97,128,129,160,161]


def positions(length):
    start=min(length-49,max(2048,length//4))
    points=torch.linspace(start,length-1,48).long().tolist()
    fit=[x for i,x in enumerate(points) if i%3!=1]
    diagnostic=[x for i,x in enumerate(points) if i%3==1]
    assert len(set(points))==48 and not set(fit)&set(diagnostic)
    return fit,diagnostic


def selection(scores):
    shaped=scores[None,:,None,:]
    ids,valid=_selected_pages(shaped,torch.ones_like(shaped,dtype=torch.bool),kv_heads=1,
        page_size=32,page_budget=64,pinned_prefix_pages=1)
    assert valid.all()
    return ids.flatten()


@torch.inference_mode()
def diagnose_layer(q,k,cv,cos,sin,bank,fit,diagnostic):
    q=q[0].float(); k=k.float()
    bank={n:t.to(q.device) for n,t in bank.items()}
    side=build_conditional_routing_sidecar(cv.float(),k,
        base_left=bank['base_left_b16'],base_right=bank['base_right_b16'],base_bias=bank['base_bias_b16'],
        residual_encoder=bank['residual_encoder_b16_r8'],cos=cos.float(),sin=sin.float())[0]
    k=k[0]; scale=128**-.5
    metrics={n:dict(train_loss=0.,diagnostic_loss=0.,mass_recall=0.,non_sink_mass_recall=0.,page_overlap=0.)
             for n in ('identity','diagonal','full')}
    saved={n:[] for n in metrics}; displacements={n:[] for n in metrics}
    solves=[]; started=time.monotonic()
    for g in range(8):
        e=bank['residual_encoder_b16_r8'][g]; u=bank['residual_query_b16_r8'][g*4:(g+1)*4]
        residual=k[g]-side[g,:,:128]
        h=torch.zeros(4,64,64,device=q.device,dtype=torch.float64)
        b=torch.zeros(4,64,device=q.device,dtype=torch.float64)
        c=torch.zeros(4,device=q.device,dtype=torch.float64)
        for pos in fit:
            query=q[g*4:(g+1)*4,pos]
            gram,_=residual_page_fisher_gram(query,k[g,:pos+1],residual[:pos+1],
                scaling=scale,page_size=32,excluded_prefix_pages=1)
            hh,bb,cc=core_statistics(query*scale,u,e,gram)
            h+=hh.double()/len(fit); b+=bb.double()/len(fit); c+=cc.double()/len(fit)
        before=time.monotonic(); cores,lam=fit_cores(h,b,relative_ridge=1e-3)
        torch.cuda.synchronize(); solves.append(time.monotonic()-before)
        for name,s in cores.items():
            metrics[name]['train_loss']+=float(quadratic_loss(h,b,c,s).sum())/32
            saved[name].append(s.cpu()); displacements[name].append(float((s-torch.eye(8,device=s.device)).norm(dim=(-2,-1)).mean()))
        for pos in diagnostic:
            query=q[g*4:(g+1)*4,pos]*scale
            exact=query@k[g,:pos+1].mT
            exact_ids=selection(exact)
            prob=exact.softmax(-1)
            prob=torch.nn.functional.pad(prob,(0,(-prob.shape[-1])%32))
            mass=prob.reshape(4,-1,32).sum(-1)
            gram,_=residual_page_fisher_gram(query/scale,k[g,:pos+1],residual[:pos+1],
                scaling=scale,page_size=32,excluded_prefix_pages=1)
            hh,bb,cc=core_statistics(query,u,e,gram)
            a=torch.einsum('hd,hdr->hr',query,u)
            base=query@side[g,:pos+1,:128].mT
            for name,s in cores.items():
                code=torch.einsum('hi,hij->hj',a,s.float())
                scores=base+code@side[g,:pos+1,128:].mT
                assert torch.isfinite(scores).all()
                ids=selection(scores)
                norm=8*len(diagnostic)
                metrics[name]['diagnostic_loss']+=float(quadratic_loss(hh,bb,cc,s).mean())/norm
                metrics[name]['mass_recall']+=float(mass[:,ids].sum(-1).mean())/norm
                nonsink=ids[ids!=0]
                metrics[name]['non_sink_mass_recall']+=float((mass[:,nonsink].sum(-1)/(1-mass[:,0]).clamp_min(1e-20)).mean())/norm
                metrics[name]['page_overlap']+=float((ids[:,None]==exact_ids[None,:]).any(-1).float().mean())/norm
    tensors={n:torch.cat(ts).contiguous() for n,ts in saved.items()}
    return dict(metrics=metrics,core_distance_to_identity={n:sum(v)/8 for n,v in displacements.items()},
        diagnostic_seconds=time.monotonic()-started,solve_seconds=sum(solves)),tensors


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','evaluate','summarize'),required=True)
    p.add_argument('--shard-index',type=int,default=0)
    for name,path in {'data-dir':'results/datasets/longbench_c1_32k','c1-results':'results/evaluation/longbench_c1_32k',
        'dense-results':'results/evaluation/longbench_dense_32k','full-results':'results/evaluation/longbench_c1_v96',
        'c1-checkpoint':'results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6',
        'bank':'results/checkpoints/c1_v96_b16r8_qgram','output-dir':'results/evaluation/c1_prompt_core_v100'}.items():
        p.add_argument('--'+name,type=Path,default=ROOT/path)
    args=p.parse_args(); args.num_shards=4
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    rows,tokens,_,_,_,bank,provenance,_=source.inputs(args)
    settings=dict(format='basisserve.prompt_core.v1',source=provenance,samples=INDICES,
        fit_queries=32,diagnostic_queries=16,relative_ridge=1e-3,
        sampling='48 uniform positions from min(T-49,max(2048,T/4)) to T-1; every third middle point diagnostic',
        trajectory='frozen full-K C1-V96 FP16 V100 memory-efficient SDPA prefill; post-RoPE Q/K; no benchmark answers or generated queries',
        objective='non-sink Page-Fisher with exact K teacher; fixed FP32 Base/E/U and sidecar; FP64 normal solve',
        scope='offline prompt-local diagnostic on completed prompts, not causal adaptation during prefill or end-to-end accuracy',
        code_sha256={n:sha256(ROOT/n) for n in ('evaluation/diagnose_c1_prompt_core.py','basisserve/core/c1_prompt_core.py',
            'evaluation/eval_longbench_lrqk_fp16.py')})
    if args.stage=='summarize':
        records=[]
        for i in INDICES:
            r=json.loads((args.output_dir/'evaluate'/f'sample_{i:03d}.json').read_text())
            assert r['status']=='complete' and r['protocol']==settings and len(r['layers'])==36
            assert r['cores_sha256']==sha256(args.output_dir/'evaluate'/f'sample_{i:03d}.safetensors')
            records.append(r)
        aggregate={n:{m:sum(r['layers'][str(l)]['metrics'][n][m] for r in records for l in range(36))/(12*36)
                      for m in ('train_loss','diagnostic_loss','mass_recall','non_sink_mass_recall','page_overlap')}
                   for n in ('identity','diagonal','full')}
        write_json(args.output_dir/'result.json',dict(status='complete',protocol=settings,aggregate=aggregate,records=records))
        lines=['# Prompt-specific fixed-subspace residual core diagnostic','',
            '12 fixed LongBench prompts, all36 layers;32 fit Q and16 disjoint diagnostic Q per prompt.',
            'Frozen full-K C1-V96 FP16 V100 memory-efficient SDPA prefill. All routing diagnostic arms FP32; FP64 core solve.',
            'Page32/B2048, pinned page0, non-sink Page-Fisher. Not generation accuracy.','',
            '```json',json.dumps(aggregate,indent=2),'```','']
        (args.output_dir/'summary.md').write_text('\n'.join(lines))
        print(json.dumps(aggregate),flush=True); return
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0)
    if args.stage=='smoke': print('prefill_kernel_relative_error',kernel_check(),flush=True)
    attention_impl.compressed_v_prefill_attention=v100_prefill
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float16,local_files_only=True,
        low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
    install_qwen3_gqa_vo_als_export(model,args.c1_checkpoint,attention_backend='triton'); model.eval()
    selected=[0] if args.stage=='smoke' else INDICES[args.shard_index::4]
    for index in selected:
        path=args.output_dir/args.stage/f'sample_{index:03d}.json'
        if path.exists():
            old=json.loads(path.read_text()); assert old['status']=='complete' and old['protocol']==settings
            continue
        row=rows[index]; fit,diag=positions(row['prompt_tokens']); results={}; cores={}; handles=[]
        chosen=[0,15,33,35] if args.stage=='smoke' else list(range(36))
        def hook(layer):
            def observe(module,a,kw):
                hidden=kw['hidden_states']; cos,sin=kw['position_embeddings']; b,t,_=hidden.shape
                q=module.q_norm(module.q_proj(hidden).view(b,t,32,128)).transpose(1,2)
                k=module.k_norm(module.k_proj(hidden).view(b,t,8,128)).transpose(1,2)
                q,k=apply_rotary_pos_emb(q,k,cos,sin)
                cv=module.v_proj(hidden).view(b,t,8,96).transpose(1,2)
                result,values=diagnose_layer(q,k,cv,cos,sin,bank[layer],fit,diag)
                results[str(layer)]=result
                cores.update({f'l{layer}_{n}':v for n,v in values.items()})
                print(f'sample={index} layer={layer} metrics={result["metrics"]}',flush=True)
            return observe
        for l in chosen: handles.append(model.model.layers[l].self_attn.register_forward_pre_hook(hook(l),with_kwargs=True))
        out=model(input_ids=tokens[f'sample_{index:03d}'].long()[None].cuda(),
            past_key_values=RoutingDynamicCache(),use_cache=True,logits_to_keep=1)
        assert torch.isfinite(out.logits).all() and len(results)==len(chosen)
        for handle in handles: handle.remove()
        del out
        path.parent.mkdir(parents=True,exist_ok=True)
        save_file(cores,str(path.with_suffix('.safetensors')))
        write_json(path,dict(status='complete',protocol=settings,sample=row,fit_positions=fit,diagnostic_positions=diag,
            layers=results,cores_sha256=sha256(path.with_suffix('.safetensors'))))


if __name__=='__main__': main()
