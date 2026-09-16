"""Three-way real-query quality and full routing latency, with incremental metadata."""
import argparse
import json
from pathlib import Path
import sys
import torch
import benchmarks.system.bench_local_kernels as harness
import benchmarks.system.native_basis_cache as basis
from benchmarks.system.two_stage_router import Metadata,candidates,compile_fine
from benchmarks.system.profile_router_phases import timing
from basisserve.kernels.mapped_host_paged_attention import select_fixed_group_max_pages_cuda as select


@torch.inference_mode()
def main():
    global candidates
    torch.backends.cuda.matmul.allow_tf32=False
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['trace','full','coarse','two'],required=True)
    p.add_argument('--check-select',action='store_true');p.add_argument('--fused-select',action='store_true');p.add_argument('--fused-append',action='store_true');p.add_argument('--window',type=int,default=64);p.add_argument('--smoke',action='store_true');p.add_argument('--repeat',type=int,default=0)
    a=p.parse_args();root=Path('results/system_benchmarks/two_stage');root.mkdir(parents=True,exist_ok=True)
    selector_checks=[]
    if a.fused_select:
        from benchmarks.system.fused_candidates import candidates as fused_candidates
        if a.check_select:
            reference_candidates=candidates
            def checked_candidates(u,n):
                ids=fused_candidates(u,n);old=reference_candidates(u,n)
                same=torch.equal(ids,old)
                if not same:
                    from benchmarks.system.two_stage_router import _group
                    import triton as tr
                    b,h,_,p=u.shape;g=torch.empty(b,h,p,device=u.device)
                    _group[(b*h,)](u,g,p,n//32,tr.next_power_of_2(p),num_warps=8)
                    torch.testing.assert_close(g.gather(-1,ids).sort(-1).values,g.gather(-1,old).sort(-1).values,rtol=0,atol=0)
                selector_checks.append(same)
                return ids
            candidates=checked_candidates
        else:candidates=fused_candidates
    full,fine=compile_fine(root/'source');metadata={};active={};records=[];pending={};step_by_layer={};updates=[]
    original_append=basis.append_mapped_host_key
    def append(host,k,*,start):
        if a.mode!='full':
            identity=host.data_ptr()
            if start==0:
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
                metadata[identity]=Metadata(k,host.shape[2]);end.record();updates.append((begin,end))
            elif not a.fused_append:metadata[identity].advance(k)
            active['meta']=metadata[identity]
        return original_append(host,k,start=start)
    basis.append_mapped_host_key=append
    if a.fused_append:
        assert a.mode=='two'
        from benchmarks.system.fused_append import append as fused_append
        original_codes=basis.BasisCache.append_codes
        def append_codes(self,k,v,layer):
            if k.shape[2]>1:return original_codes(self,k,v,layer)
            return fused_append(self,k,v,layer,active['meta'])
        basis.BasisCache.append_codes=append_codes
    def choose(logs):return select(logs,pages_per_kv_head=62,pinned_prefix_pages=1,force_current_page=False)
    def route(q,b,res,**kw):
        n=b.shape[2];np=(n+31)//32
        code=torch.empty(q.shape[0],b.shape[1],4,16,device=q.device,dtype=q.dtype)
        args=(q,b,res,kw['base_right'],kw['base_bias'],kw['residual_query'],kw['rope_cos'],kw['rope_sin'],code)
        def full_route():
            scores=torch.empty(q.shape[0],b.shape[1],4,np,device=q.device)
            full.conditional_router_page_lse(*args,scores,kw['scale'],False)
            return scores,choose(scores)
        if a.mode=='full':logs,ids=full_route()
        else:
            meta=active['meta'];assert meta.n==n
            def coarse_route():
                u=meta.scores(q)
                return u,choose(u[...,:np])
            def two_route():
                u=meta.scores(q);c=candidates(u,n)
                scores=torch.empty(q.shape[0],b.shape[1],4,c.shape[-1],device=q.device)
                fine.conditional_router_page_lse(*args,scores,kw['scale'],False,c)
                return scores,c.gather(-1,choose(scores)),c,u
            if a.mode=='coarse':logs,ids=coarse_route()
            elif a.mode=='two':logs,ids,_,_=two_route()
            else:
                logs,ids=full_route()
                identity=kw['base_right'].data_ptr();step=step_by_layer.get(identity,0);step_by_layer[identity]=step+1
                if step==0 or 10<=step<20:
                    fine_logs,two_ids,c,u=two_route();coarse_ids=choose(u[...,:np])
                    reference=logs.gather(-1,c.clamp_max(np-1)[:,:,None].expand(-1,-1,4,-1)).masked_fill(c[:,:,None]>=np,-torch.inf)
                    torch.testing.assert_close(fine_logs,reference,rtol=0,atol=0)
                    adaptive=ids[...,1:]
                    hit=(adaptive[...,None]==c[...,None,:]).any(-1)
                    # Forced sink/recent-intersecting pages are excluded from candidate recall.
                    eligible=adaptive<n//32
                    coverage=(hit&eligible).sum(-1)/eligible.sum(-1)
                    mass=logs.clone();mass[...,0]=-torch.inf
                    # E is the non-forced historical page set used by the coarse selector.
                    mass[...,n//32:]=-torch.inf
                    prob=mass.softmax(-1)
                    retained=prob.gather(-1,c.clamp_max(np-1)[:,:,None].expand(-1,-1,4,-1)).masked_fill(c[:,:,None]>=np,0).sum(-1)
                    row=dict(layer=list(step_by_layer).index(identity),step=step,window=a.window,
                        candidate_recall=coverage.cpu().tolist(),lost_reference_mass=(1-retained).clamp_min(0).cpu().tolist(),
                        fine_max_abs=float((fine_logs-reference).nan_to_num().abs().max()),
                        coarse_overlap=(ids[...,None]==coarse_ids[...,None,:]).any(-1).float().mean(-1).cpu().tolist(),
                        two_overlap=(ids[...,None]==two_ids[...,None,:]).any(-1).float().mean(-1).cpu().tolist())
                    if step==0 or step in [10,19]:
                        row['routing_ms']={name:timing(fn,3)['median_ms'] for name,fn in [('full',full_route),('coarse',coarse_route),('two',two_route)]}
                    pending['diag']=(row,dict(full=ids,coarse=coarse_ids,two=two_ids),q,n)
        pending['ids']=ids
        return logs
    basis.conditional_router_page_lse=route
    basis.select_fixed_group_max_pages_cuda=lambda logs,**kw:pending['ids']
    original_attention=basis.BasisCache.attention
    def checked_attention(self,q,k,v,layer,positions):
        out=original_attention(self,q,k,v,layer,positions)
        if a.mode=='trace' and q.shape[2]==1 and 'diag' in pending:
            row,methods,query,n=pending.pop('diag');end=n+64
            keys=self.keys[layer][:,:,:end].cuda().float()
            values=self.values[layer][:,:,:end].float()
            s=query.reshape(query.shape[0],8,4,128).float()@keys.transpose(-1,-2)*128**-.5
            prob=s.softmax(-1);dense=prob@values;den=float(dense.square().sum())
            outputs={};row['output']={}
            for name,pages in methods.items():
                ids=(pages[...,None]*32+torch.arange(32,device=q.device)).flatten(-2)
                ids=ids.masked_fill(ids>=n,-1)
                ids=torch.cat((ids,torch.arange(n,end,device=q.device).expand(q.shape[0],8,-1)),-1)
                ps=prob.gather(-1,ids.clamp_min(0)[:,:,None].expand(-1,-1,4,-1)).masked_fill(ids[:,:,None]<0,0)
                val=values.gather(2,ids.clamp_min(0)[...,None].expand(-1,-1,-1,128))
                outputs[name]=(ps/ps.sum(-1,keepdim=True))@val
                adaptive=(ids>=32)&(ids<n)
                nonforced=(ps*adaptive[:,:,None]).sum(-1)/prob[...,32:n].sum(-1)
                numerator=float((outputs[name]-dense).square().sum())
                row['output'][name]=dict(error_squared=numerator,dense_squared=den,relative_mse=numerator/den,
                    mass=float(ps.sum(-1).mean()),non_sink_recent_mass=float(nonforced.mean()))
            for name in ['coarse','two']:
                row['output'][name]['delta_vs_full_mse']=float((outputs[name]-outputs['full']).square().sum()/outputs['full'].square().sum())
            records.append(row)
            if len(records)%32==0:print('DIAGNOSTIC',len(records),'step',row['step'],flush=True)
        return out
    basis.BasisCache.attention=checked_attention
    harness.RESULT_ROOT=root/f'{a.mode}{"_fused" if a.fused_append else ""}{"_select" if a.fused_select else ""}_w{a.window}_r{a.repeat}';harness.TRACE_SLOT_OPERATORS=False
    harness.NATIVE_EXTRA_ARGS=['--window',str(a.window)]
    harness.compile_router=lambda root,rank,reference=False:full
    original_slots=harness.compile_slots
    harness.compile_slots=lambda folder,vector:original_slots(root/'source/slots',True)
    sys.argv=['bench_two_stage','--mode','optimized','--rank','16','--length','65536']
    if a.smoke:sys.argv.append('--smoke')
    harness.main()
    torch.cuda.synchronize()
    summary=dict(selector_checks=len(selector_checks),selector_identical=sum(selector_checks),status='complete',mode=a.mode,window=a.window,records=records,
        summary_prefill_ms=sum(x.elapsed_time(y) for x,y in updates),
        metadata_bytes=sum(m.minimum.numel()*4+m.ring.numel()*2 for m in metadata.values()),
        command=f'python -m benchmarks.system.bench_two_stage --mode {a.mode} --window {a.window} --repeat {a.repeat}'+(' --smoke' if a.smoke else ''))
    (harness.RESULT_ROOT/'diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('COMPLETE',a.mode,a.window,'samples',len(records),flush=True)

if __name__=='__main__':main()
