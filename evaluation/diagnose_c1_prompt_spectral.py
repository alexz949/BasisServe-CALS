"""Three hard prompts: closed-form shared-query-metric residual rank8."""
import argparse
import time
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import eval_longbench_c1_v96_router as source
from evaluation.diagnose_c1_prompt_core import selection
from evaluation.eval_longbench_lrqk_fp16 import v100_prefill,kernel_check
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256,write_json
from basisserve.checkpoint import gqa_vo_qwen3 as attention
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar,residual_page_fisher_gram
from basisserve.core.gqa_joint_routing_payload_s80_fisher import S80CompactSoftmaxFisherRouting,compact_softmax_fisher_loss
from basisserve.core.c1_residual_spectral import spectral_factors

LAYERS=(15,19,24,33)
NAMES=('offline','closed_form','exact')


def query_positions(t):
    start=min(t-289,max(2048,t//4))
    p=torch.linspace(start,t-1,288).long().tolist()
    fit=[x for i,x in enumerate(p) if i%9!=4]
    diagnostic=[x for i,x in enumerate(p) if i%9==4]
    assert len(set(p))==288 and len(fit)==256 and len(diagnostic)==32
    return fit,diagnostic


def page_details(scores):
    p=torch.nn.functional.pad(scores,(0,(-scores.shape[-1])%32),value=-torch.inf)
    lse=p.reshape(4,-1,32).logsumexp(-1); lse[:,0]=-torch.inf
    priority=lse.softmax(-1).amax(0)
    priority[0]=-torch.inf
    ordered=priority.sort().values
    lo=len(priority)-torch.searchsorted(ordered,priority,right=True)+1
    hi=len(priority)-torch.searchsorted(ordered,priority,right=False)
    ids=selection(scores); mask=torch.zeros_like(priority,dtype=torch.bool); mask[ids]=True
    return priority,lo,hi,mask


@torch.inference_mode()
def layer_diagnostic(q,k,cv,cos,sin,bank,fit,diagnostic,groups):
    q=q[0].float(); k=k.float(); bank={n:t.to(q.device) for n,t in bank.items()}
    side=build_conditional_routing_sidecar(cv.float(),k,base_left=bank['base_left_b16'],
        base_right=bank['base_right_b16'],base_bias=bank['base_bias_b16'],
        residual_encoder=bank['residual_encoder_b16_r8'],cos=cos.float(),sin=sin.float())[0]
    k=k[0]; scale=128**-.5; tensors={}; reports={}; misses=[]
    for g in groups:
        residual=k[g]-side[g,:,:128]; e=bank['residual_encoder_b16_r8'][g:g+1]
        u=bank['residual_query_b16_r8'][g*4:g*4+4]
        grams=[]; energy=0.
        for pos in fit:
            gram,eng=residual_page_fisher_gram(q[g*4:g*4+4,pos],k[g,:pos+1],residual[:pos+1],
                scaling=scale,page_size=32,excluded_prefix_pages=1)
            grams.append(gram); energy+=eng
        queries=q[g*4:g*4+4,fit]
        stats=S80CompactSoftmaxFisherRouting(queries,torch.stack(grams,1),
            torch.zeros(4,device=q.device,dtype=torch.long),0,128,scale,energy)
        # Fisher statistics above are diagnostic only, not inputs to the spectral fit.
        torch.cuda.synchronize(); begin=time.perf_counter()
        covariance=residual.mT@residual/residual.shape[0]
        pooled=queries.reshape(-1,128)
        query_covariance=pooled.mT@pooled/pooled.shape[0]
        torch.cuda.synchronize(); moments_seconds=time.perf_counter()-begin
        begin=time.perf_counter()
        ee,uu,spectral=spectral_factors(covariance,query_covariance,rank=8,relative_ridge=1e-5)
        torch.cuda.synchronize(); spectral_seconds=time.perf_counter()-begin
        factors={'offline':(e,u),'closed_form':(ee[None],uu[None].expand(4,-1,-1))}
        def loss(pair):
            return compact_softmax_fisher_loss(stats,routing_payload_encoders=pair[0],routing_query_factors=pair[1])
        metrics={n:dict(mass=0.,non_sink_mass=0.,page_overlap=0.,fisher_loss=0.) for n in NAMES}
        for name,(ee,uu) in factors.items():
            assert torch.isfinite(ee).all() and torch.isfinite(uu).all()
            tensors[f'g{g}_{name}_e']=ee.cpu().contiguous(); tensors[f'g{g}_{name}_u']=uu.cpu().contiguous()
        for pos in diagnostic:
            query=q[g*4:g*4+4,pos]*scale
            exact=query@k[g,:pos+1].mT; base=query@side[g,:pos+1,:128].mT
            scores={'exact':exact}
            for name,(ee,uu) in factors.items():
                scores[name]=base+torch.einsum('hd,hdr->hr',query,uu)@(residual[:pos+1]@ee[0]).mT
            details={n:page_details(s) for n,s in scores.items()}
            prob=torch.nn.functional.pad(exact.softmax(-1),(0,(-exact.shape[-1])%32))
            mass=prob.reshape(4,-1,32).sum(-1)
            gram,_=residual_page_fisher_gram(query/scale,k[g,:pos+1],residual[:pos+1],
                scaling=scale,page_size=32,excluded_prefix_pages=1)
            for name,(priority,lo,hi,mask) in details.items():
                assert torch.isfinite(scores[name]).all()
                metrics[name]['mass']+=float(mass[:,mask].sum(-1).mean())/len(diagnostic)
                non=mask.clone(); non[0]=False
                metrics[name]['non_sink_mass']+=float((mass[:,non].sum(-1)/(1-mass[:,0]).clamp_min(1e-20)).mean())/len(diagnostic)
                metrics[name]['page_overlap']+=float((mask&details['exact'][3]).sum()/details['exact'][3].sum())/len(diagnostic)
                if name!='exact':
                    ee,uu=factors[name]; error=torch.einsum('hd,hdr,kr->hk',query,uu,ee[0])-query
                    metrics[name]['fisher_loss']+=float(torch.einsum('hd,hdk,hk->h',error,gram,error).mean())/len(diagnostic)
                tensors[f'g{g}_q{pos}_{name}_priority']=priority.cpu()
                tensors[f'g{g}_q{pos}_{name}_rank_min']=lo.cpu()
                tensors[f'g{g}_q{pos}_{name}_rank_max']=hi.cpu()
                tensors[f'g{g}_q{pos}_{name}_selected']=mask.cpu()
            tensors[f'g{g}_q{pos}_teacher_mass']=mass.cpu()
            missed=details['exact'][3]&~details['offline'][3]; missed[0]=False
            pages=missed.nonzero().flatten()
            top=pages[mass.mean(0)[pages].argsort(descending=True)[:5]]
            for page in top.tolist():
                misses.append(dict(group=g,query=pos,page=page,teacher_mass=float(mass[:,page].mean()),
                    ranks={n:[int(d[1][page]),int(d[2][page])] for n,d in details.items()},
                    selected={n:bool(d[3][page]) for n,d in details.items()}))
        reports[str(g)]=dict(metrics=metrics,fit_losses={n:loss(p) for n,p in factors.items()},
            query_rank=int(torch.linalg.matrix_rank(queries).min()),
            covariance_seconds=moments_seconds,spectral_seconds=spectral_seconds,
            ridge=float(spectral['ridge']),spectrum=spectral['spectrum'].cpu().tolist())
        print(f'group={g} metrics={metrics}',flush=True)
    return dict(groups=reports,high_mass_missed_pages=misses),tensors


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','evaluate','summarize'),required=True)
    p.add_argument('--shard-index',type=int,default=0)
    for name,path in {'data-dir':'results/datasets/longbench_c1_32k','c1-results':'results/evaluation/longbench_c1_32k',
        'dense-results':'results/evaluation/longbench_dense_32k','full-results':'results/evaluation/longbench_c1_v96',
        'c1-checkpoint':'results/checkpoints/qwen3_8b_c1_v96_32f4h_s32768_als6',
        'bank':'results/checkpoints/c1_v96_b16r8_qgram','output-dir':'results/evaluation/c1_prompt_spectral'}.items():
        p.add_argument('--'+name,type=Path,default=ROOT/path)
    args=p.parse_args(); args.num_shards=4
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    rows,tokens,_,_,_,bank,provenance,_=source.inputs(args)
    selection_path=ROOT/'results/evaluation/c1_prompt_core_v100/result.json'
    prior=json.loads(selection_path.read_text())
    ranking=sorted((sum(r['layers'][str(l)]['metrics']['identity']['non_sink_mass_recall'] for l in LAYERS)/4,
        r['sample']['index']) for r in prior['records'])
    indices=[i for _,i in ranking[:3]]; assert indices==[160,128,161]
    settings=dict(format='basisserve.prompt_spectral.v1',source=provenance,indices=indices,layers=LAYERS,
        selection_sha256=sha256(selection_path),selection_ranking=ranking,fit_queries=256,diagnostic_queries=32,
        objective='shared pooled-query-metric raw residual score approximation; Page-Fisher used only for diagnosis',
        solver='two128x128 eigendecompositions; relative query ridge1e-5; uncentered normalized moments',
        selection='no hyperparameter or initialization selection; shared U across four GQA query heads',
        sampling='288 uniform positions from min(T-289,max(2048,T/4)), i%9==4 diagnostic',
        precision='FP16 C1 full-K V100 memory-efficient prefill; FP32 routing and spectral fit',
        caveat='selected hard-case diagnostic; not optimal for Page-Fisher, causal per-query metric or page selection; no generation accuracy',
        code_sha256={n:sha256(ROOT/n) for n in ('evaluation/diagnose_c1_prompt_spectral.py',
            'basisserve/core/c1_residual_spectral.py','basisserve/core/gqa_joint_routing_payload_s80_fisher.py')})
    # JSON normalization preserves equality for tuples in saved provenance.
    settings=json.loads(json.dumps(settings))
    if args.stage=='summarize':
        records=[]
        for index in indices:
            path=args.output_dir/'evaluate'/f'sample_{index:03d}.json'; r=json.loads(path.read_text())
            assert r['status']=='complete' and r['protocol']==settings and len(r['layers'])==4
            assert r['tensor_sha256']==sha256(path.with_suffix('.safetensors'))
            records.append(r)
        groups=[g for r in records for l in r['layers'].values() for g in l['groups'].values()]
        assert len(groups)==96
        aggregate={n:{m:sum(g['metrics'][n][m] for g in groups)/len(groups)
                      for m in ('mass','non_sink_mass','page_overlap','fisher_loss')} for n in NAMES}
        write_json(args.output_dir/'result.json',dict(status='complete',protocol=settings,aggregate=aggregate,records=records))
        (args.output_dir/'summary.md').write_text('# Closed-form prompt residual diagnostic\n\nThree hard prompts, four layers; shared-metric spectral fit, not a page-selection ceiling or generation benchmark.\n\n```json\n'+json.dumps(aggregate,indent=2)+'\n```\n')
        print(json.dumps(aggregate),flush=True); return
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0)
    if args.stage=='smoke': print('kernel_error',kernel_check(),flush=True)
    attention.compressed_v_prefill_attention=v100_prefill
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float16,local_files_only=True,
        low_cpu_mem_usage=True,attn_implementation='sdpa').cuda().eval()
    attention.install_qwen3_gqa_vo_als_export(model,args.c1_checkpoint,attention_backend='triton'); model.eval()
    index=indices[0] if args.stage=='smoke' else indices[args.shard_index]
    row=rows[index]; fit,diag=query_positions(row['prompt_tokens']); results={}; tensors={}; handles=[]
    layers=LAYERS[:1] if args.stage=='smoke' else LAYERS
    def hook(layer):
        def observe(module,a,kw):
            hidden=kw['hidden_states']; cos,sin=kw['position_embeddings']; b,t,_=hidden.shape
            q=module.q_norm(module.q_proj(hidden).view(b,t,32,128)).transpose(1,2)
            k=module.k_norm(module.k_proj(hidden).view(b,t,8,128)).transpose(1,2)
            q,k=apply_rotary_pos_emb(q,k,cos,sin)
            cv=module.v_proj(hidden).view(b,t,8,96).transpose(1,2)
            r,ts=layer_diagnostic(q,k,cv,cos,sin,bank[layer],fit,diag,[0] if args.stage=='smoke' else range(8))
            results[str(layer)]=r; tensors.update({f'l{layer}_{n}':v for n,v in ts.items()})
            print(f'sample={index} layer={layer} complete',flush=True)
        return observe
    for l in layers: handles.append(model.model.layers[l].self_attn.register_forward_pre_hook(hook(l),with_kwargs=True))
    out=model(input_ids=tokens[f'sample_{index:03d}'].long()[None].cuda(),
        past_key_values=RoutingDynamicCache(),use_cache=True,logits_to_keep=1)
    assert torch.isfinite(out.logits).all() and len(results)==len(layers)
    for h in handles: h.remove()
    path=args.output_dir/args.stage/f'sample_{index:03d}.json'; path.parent.mkdir(parents=True,exist_ok=True)
    save_file(tensors,str(path.with_suffix('.safetensors')))
    write_json(path,dict(status='complete',protocol=settings,sample=row,fit_positions=fit,diagnostic_positions=diag,
        layers=results,tensor_sha256=sha256(path.with_suffix('.safetensors'))))


if __name__=='__main__': main()
