"""Matched continuous-state A/B and uninstrumented decode for local kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys
import torch
import benchmarks.system.native_basis_cache as basis
import basisserve.kernels.mapped_host_paged_attention as mapped
from benchmarks.system.local_kernel_candidates import compile_slots, compile_router
from benchmarks.system.validate_local_kernels import workspace
from benchmarks.system.profile_router_phases import timing
from basisserve.kernels.gqa_slot_attention import gqa_slot_attention
from basisserve.kernels.slot_indexed_attention import slot_indexed_attention
from basisserve.kernels.mapped_host_paged_attention import mapped_host_device_pointer, select_fixed_group_max_pages_cuda


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['trace','baseline','gqa','vector','combined','router','optimized','deployed'],required=True)
    p.add_argument('--rank',type=int,choices=[8,16],default=16);p.add_argument('--length',type=int,default=65536)
    p.add_argument('--warps',type=int,choices=[4,8],default=4);p.add_argument('--smoke',action='store_true')
    p.add_argument('--repeat',type=int,default=0);a=p.parse_args()
    root=RESULT_ROOT
    run=root/f'{a.mode}_b{a.rank}_w{a.warps}_r{a.repeat}{"_smoke" if a.smoke else ""}';run.mkdir(parents=True,exist_ok=True)
    if a.mode=='deployed':
        from torch.utils.cpp_extension import load
        scalar=load(name='basis_persistent_key_slots',sources=['basisserve/kernels/csrc/persistent_key_slots.cu'],
            extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-std=c++17'])
    else:scalar=compile_slots(root/'source',False)
    vector=compile_slots(root/'source',True) if a.mode in ['trace','vector','combined','optimized'] else None
    router=compile_router(root/'source',a.rank) if a.mode in ['trace','router','optimized'] else None
    reference_router=compile_router(root/'source',a.rank,reference=True) if a.mode!='deployed' else None
    original_loader=mapped._load_extension
    def extension(value_dim=80,queries_per_kv=4,page_size=32,base_rank=16,residual_rank=8):
        if a.mode!='deployed' and (value_dim,queries_per_kv,page_size,base_rank,residual_rank)==(80,4,32,a.rank,a.rank):
            return router if a.mode in ['router','optimized'] else reference_router
        return original_loader(value_dim,queries_per_kv,page_size,base_rank,residual_rank)
    mapped._load_extension=extension
    caches={};samples=[];router_steps={};router_samples=[]
    decode_inputs=[];finite=[]
    original_class=basis.basis_llama_class
    def recorded_class(native,width):
        cls=original_class(native,width);inference=cls.inference
        def recorded(self,input_ids,position_ids):
            if input_ids.shape[1]==1:decode_inputs.append(input_ids.clone())
            out=inference(self,input_ids,position_ids)
            if input_ids.shape[1]==1:finite.append(torch.isfinite(out).all())
            return out
        cls.inference=recorded
        return cls
    basis.basis_llama_class=recorded_class
    original_router=basis.conditional_router_page_lse
    def route(q,b,res,**kw):
        if router is None or a.mode in ['router','optimized']:return original_router(q,b,res,**kw)
        identity=kw['base_right'].data_ptr();step=router_steps.get(identity,0);router_steps[identity]=step+1
        output=torch.empty(q.shape[0],b.shape[1],4,(b.shape[2]+31)//32,device=q.device)
        code=torch.empty(q.shape[0],b.shape[1],4,a.rank,device=q.device,dtype=q.dtype)
        call=(q,b,res,kw['base_right'],kw['base_bias'],kw['residual_query'],kw['rope_cos'],kw['rope_sin'],code,output,kw['scale'],False)
        baseline=original_router(q,b,res,**kw)
        if step==0 or 10<=step<20:
            router.conditional_router_page_lse(*call)
            delta=(output-baseline).abs()
            oldids=select_fixed_group_max_pages_cuda(baseline,pages_per_kv_head=62,pinned_prefix_pages=1,force_current_page=False)
            newids=select_fixed_group_max_pages_cuda(output,pages_per_kv_head=62,pinned_prefix_pages=1,force_current_page=False)
            overlap=(oldids[...,None]==newids[...,None,:]).any(-1).float().mean()
            row=dict(layer=list(router_steps).index(identity),step=step,max_abs=float(delta.max()),score_rmse=float(delta.square().mean().sqrt()),page_overlap=float(overlap))
            if step==0 or step in [10,19]:
                reference_output=torch.empty_like(output)
                reference_call=call[:9]+(reference_output,)+call[10:]
                row['original']=timing(lambda:reference_router.conditional_router_page_lse(*reference_call),3)
                row['candidate']=timing(lambda:router.conditional_router_page_lse(*call),3)
            router_samples.append(row)
        return baseline
    basis.conditional_router_page_lse=route
    original_attention=basis.mapped_host_paged_attention
    def attention(host,q,v,ids,**kw):
        identity=host.data_ptr();b,h,n=ids.shape
        if identity not in caches:
            caches[identity]=dict(key=torch.empty(b,h,n,128,device=q.device,dtype=q.dtype),
                resident=torch.full_like(ids,-1),lookup=torch.full(v.shape[:3],-1,device=q.device,dtype=torch.int32),
                slots=torch.empty_like(ids),missing=torch.empty_like(ids,dtype=torch.int32),
                counts=torch.empty(101,b,h,2,device=q.device,dtype=torch.int32),step=0,
                pointer=mapped_host_device_pointer(host),workspace=workspace(q,v.shape[-1]),alt=workspace(q,v.shape[-1]))
        c=caches[identity];step=c['step'];c['step']+=1
        fetch=vector if a.mode in ['vector','combined','optimized'] else scalar
        fetch.refresh(c['pointer'],c['key'],ids,c['resident'],c['lookup'],c['slots'],c['missing'],c['counts'][step],True)
        def base():return slot_indexed_attention(q,c['key'],v,ids,c['slots'],c['workspace'],scale=kw['scale'])
        def grouped(w):return gqa_slot_attention(q,c['key'],v,ids,c['slots'],c['alt'],scale=kw['scale'],num_warps=w)
        out=grouped(a.warps) if a.mode in ['gqa','combined'] else base()
        if a.smoke:
            expected=host.gather(2,ids.clamp_min(0)[...,None].expand(-1,-1,-1,128).cpu()).cuda().masked_fill((ids<0)[...,None],0)
            actual=c['key'].gather(2,c['slots'].clamp_min(0)[...,None].expand(-1,-1,-1,128)).masked_fill((ids<0)[...,None],0)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            reference=original_attention(host,q,v,ids,**kw)
            torch.testing.assert_close(out,reference,rtol=.003,atol=.003)
        if TRACE_SLOT_OPERATORS and a.mode=='trace' and (step==0 or 10<=step<20):
            reference=out.clone();before=c['key'].clone()
            vector.fetch_only(c['pointer'],c['key'],c['missing'],c['slots'],v.shape[2])
            assert torch.equal(c['key'].view(torch.int16),before.view(torch.int16))
            errors={}
            for warps in [4,8]:
                candidate=grouped(warps)
                torch.testing.assert_close(candidate,reference,rtol=.003,atol=.003)
                errors[str(warps)]=float((candidate-reference).abs().max())
            counts=c['counts'][step].cpu()
            times={
                'attention_old':timing(base,3),
                'attention_gqa4':timing(lambda:grouped(4),3),
                'attention_gqa8':timing(lambda:grouped(8),3),
                'fetch_scalar':timing(lambda:scalar.fetch_only(c['pointer'],c['key'],c['missing'],c['slots'],v.shape[2]),3),
                'fetch_vector':timing(lambda:vector.fetch_only(c['pointer'],c['key'],c['missing'],c['slots'],v.shape[2]),3)}
            samples.append(dict(layer=list(caches).index(identity),step=step,hits=int(counts[...,0].sum()),valid=int(counts[...,1].sum()),max_abs=errors,timing=times))
            out=base()
            if len(samples)%32==0:print('TRACE',len(samples),'step',step,flush=True)
        return out
    basis.mapped_host_paged_attention=attention
    sys.argv=['bench_shadow_native','--method',f'basis{a.rank}','--length',str(a.length),'--batch','1','--output',str(run)]
    sys.argv.extend(NATIVE_EXTRA_ARGS)
    if a.smoke:sys.argv.append('--smoke')
    runpy.run_module('benchmarks.system.bench_shadow_native',run_name='__main__')
    length=8192 if a.smoke else a.length;steps=8 if a.smoke else 100
    result=json.loads((run/f'basis{a.rank}_t{length}_b1{"_smoke" if a.smoke else ""}'/'result.json').read_text())
    assert bool(torch.stack(finite).all())
    raw=result['raw_inference_ms'];counts=torch.stack([c['counts'][:steps].sum((1,2)).cpu() for c in caches.values()])
    steady=[row for row in samples if row['step']>=10] or samples
    totals={name:sum(row['timing'][name]['median_ms'] for row in steady)/len({row['step'] for row in steady}) for name in steady[0]['timing']} if steady else {}
    rt=[r for r in router_samples if 'original' in r and r['step']>=10] or [r for r in router_samples if 'original' in r]
    router_totals={name:sum(r[name]['median_ms'] for r in rt)/len({r['step'] for r in rt}) for name in ['original','candidate']} if rt else {}
    report=dict(status='complete',mode=a.mode,rank=a.rank,length=length,warps=a.warps,environment='basis',
        command=f'python -m benchmarks.system.bench_local_kernels --mode {a.mode} --rank {a.rank} --length {a.length} --warps {a.warps} --repeat {a.repeat}'+(' --smoke' if a.smoke else ''),
        hit_fraction=float(counts[...,0].sum()/counts[...,1].sum()),operator_ms_per_step=totals,router_ms_per_step=router_totals,
        native_wall_ms=result['native_benchmark_ms_per_step'],cuda_median_after10_ms=statistics.median(raw[10:]) if len(raw)>10 else None,
        cuda_mean_after10_ms=statistics.mean(raw[10:]) if len(raw)>10 else None,
        decode_input_ids=torch.cat(decode_inputs).cpu().tolist(),all_decode_logits_finite=True,
        samples=samples,router_samples=router_samples,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('basisserve/kernels/gqa_slot_attention.py'),Path('benchmarks/system/local_kernel_candidates.py')]},
        scope='Trace always serves original router/attention; both candidates see identical real queries, IDs and evolving slot state. Fetch replay uses actual miss lists without rerunning the planner; repeated microbench reads can be cache-warm. Trace native timings include probes and are NOT serving performance. Non-trace modes perform normal 100-step continuous decode with no A/B probes.')
    (run/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SUMMARY',{k:v for k,v in report.items() if k not in ['samples','router_samples','sources']},flush=True)


RESULT_ROOT=Path('results/system_benchmarks/local_kernels')
TRACE_SLOT_OPERATORS=True
NATIVE_EXTRA_ARGS=[]

if __name__=='__main__':main()
