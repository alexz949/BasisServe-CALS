"""Nested CUDA-event attribution on the actual continuous two-stage decode path."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import torch
from torch.utils.cpp_extension import load
import benchmarks.system.bench_two_stage as bench
import benchmarks.system.bench_local_kernels as harness
import benchmarks.system.native_basis_cache as basis
from benchmarks.system.local_kernel_candidates import write_source


class Recorder:
    def __init__(self):
        self.active=False;self.step=-1;self.rows=[];self.weights={}
        # Preallocate to keep event-object creation out of measured intervals.
        self.pool=[torch.cuda.Event(enable_timing=True) for _ in range(30000)]
        self.cursor=0
    def call(self,name,fn,*args,**kw):
        if not self.active:return fn(*args,**kw)
        begin,end=self.pool[self.cursor:self.cursor+2];self.cursor+=2
        begin.record();out=fn(*args,**kw);end.record()
        self.rows.append((self.step,name,begin,end));return out


def slots_with_plan(root):
    text=Path('basisserve/kernels/csrc/persistent_key_slots.cu').read_text()
    left=text.index('void refresh(');end=text.index('\nPYBIND11_MODULE',left)
    plan=text[left:end].replace('void refresh(','void plan_only(',1)
    cut=plan.index('  fetch_missing<<<')
    plan=plan[:cut]+'  assert(cudaGetLastError()==cudaSuccess);\n}\n'
    fetch='''
void fetch_only(int64_t pointer,const at::Tensor& cache,const at::Tensor& missing,const at::Tensor& slots,int capacity){
  c10::cuda::CUDAGuard guard(cache.device());
  int rows=cache.size(0)*cache.size(1),budget=cache.size(2);
  fetch_missing<<<(rows*budget*16+255)/256,256,0,c10::cuda::getCurrentCUDAStream().stream()>>>(
    reinterpret_cast<const __nv_bfloat16*>(pointer),reinterpret_cast<__nv_bfloat16*>(cache.data_ptr<at::BFloat16>()),
    missing.data_ptr<int>(),slots.data_ptr<int64_t>(),capacity,budget,rows);
  assert(cudaGetLastError()==cudaSuccess);
}
'''
    text=text[:end]+plan+fetch+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("refresh",&refresh);m.def("plan_only",&plan_only);m.def("fetch_only",&fetch_only);}\n'
    path=root/'slots_profile.cu';write_source(path,text);digest=hashlib.sha256(text.encode()).hexdigest()[:10]
    return load(name=f'profile_slots_{digest}',sources=[str(path)],extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-std=c++17'])


def main():
    root=Path('results/system_benchmarks/two_stage');timer=Recorder()
    original_factory=basis.basis_llama_class
    def factory(native,width):
        parent=original_factory(native,width)
        class ProfileLlama(parent):
            def inference(self,input_ids,position_ids):
                if input_ids.shape[1]==1:
                    timer.step+=1;timer.active=10<=timer.step<20
                result=timer.call('inference_total',super().inference,input_ids,position_ids)
                timer.active=False
                return result
            def pre_attention_compute(self,hidden,buffer,*args):
                timer.weights[buffer.wqkv.data_ptr()]='qkv_linear'
                return timer.call('pre_attention_total',super().pre_attention_compute,hidden,buffer,*args)
            def apply_rotary_pos_emb(self,*args):
                return timer.call('query_key_rope',super().apply_rotary_pos_emb,*args)
            def post_attention_compute(self,out,residual,buffer):
                timer.weights[buffer.wo.data_ptr()]='wo_linear'
                timer.weights[buffer.gate_up_proj.data_ptr()]='mlp_gate_up_linear'
                timer.weights[buffer.down_proj.data_ptr()]='mlp_down_linear'
                return timer.call('post_attention_total',super().post_attention_compute,out,residual,buffer)
        return ProfileLlama
    basis.basis_llama_class=factory
    linear=torch.nn.functional.linear
    def measured_linear(x,w,bias=None):
        name=timer.weights.get(w.data_ptr())
        return timer.call(name,linear,x,w,bias) if name else linear(x,w,bias)
    torch.nn.functional.linear=measured_linear
    def wrap(owner,name,tag):
        fn=getattr(owner,name)
        setattr(owner,name,lambda *args,**kw:timer.call(tag,fn,*args,**kw))
    wrap(basis.BasisCache,'attention','cache_attention_total')
    wrap(basis,'append_mapped_host_key','append_cpu_key')
    wrap(bench.Metadata,'advance','metadata_update')
    wrap(bench.Metadata,'scores','coarse_scoring')
    wrap(bench,'candidates','select_512')
    wrap(bench,'select','final_selection')
    wrap(harness,'slot_indexed_attention','sparse_attention')
    build=bench.compile_fine
    class RouterProxy:
        def __init__(self,module,tag):self.module=module;self.tag=tag
        def conditional_router_page_lse(self,*args):
            return timer.call(self.tag,self.module.conditional_router_page_lse,*args)
        def __getattr__(self,name):return getattr(self.module,name)
    def routers(folder):
        full,fine=build(folder)
        return RouterProxy(full,'full_router'),RouterProxy(fine,'fine_router_including_query_projection')
    bench.compile_fine=routers
    slots=slots_with_plan(root/'profile_source')
    class SlotsProxy:
        def refresh(self,pointer,cache,ids,resident,lookup,selected,missing,counts,reuse):
            if not timer.active:return slots.refresh(pointer,cache,ids,resident,lookup,selected,missing,counts,reuse)
            timer.call('slot_planner',slots.plan_only,pointer,cache,ids,resident,lookup,selected,missing,counts,reuse)
            timer.call('fetch_missing_key',slots.fetch_only,pointer,cache,missing,selected,lookup.size(2))
    harness.compile_slots=lambda folder,vector:SlotsProxy()
    sys.argv=['profile_two_stage','--mode','two','--window','64','--repeat','2']
    bench.main();torch.cuda.synchronize()
    totals=defaultdict(lambda:defaultdict(float))
    for step,name,begin,end in timer.rows:totals[step][name]+=begin.elapsed_time(end)
    exclusive=[]
    for step,t in sorted(totals.items()):
        row={name:t[name] for name in ['qkv_linear','query_key_rope','wo_linear','mlp_gate_up_linear','mlp_down_linear','append_cpu_key','metadata_update','coarse_scoring','select_512','fine_router_including_query_projection','final_selection','slot_planner','fetch_missing_key','sparse_attention']}
        row['pre_norm_and_dispatch']=t['pre_attention_total']-t['qkv_linear']
        row['post_norm_silu_residual_dispatch']=t['post_attention_total']-t['wo_linear']-t['mlp_gate_up_linear']-t['mlp_down_linear']
        children=['append_cpu_key','metadata_update','coarse_scoring','select_512','fine_router_including_query_projection','final_selection','slot_planner','fetch_missing_key','sparse_attention']
        row['cache_append_codes_support_and_dispatch']=t['cache_attention_total']-sum(t[x] for x in children)
        row['outside_layers_and_dispatch']=t['inference_total']-t['pre_attention_total']-t['post_attention_total']-t['query_key_rope']-t['cache_attention_total']
        assert abs(sum(row.values())-t['inference_total'])<1e-4
        exclusive.append(dict(step=step,total_ms=t['inference_total'],phases_ms=row))
    mean={name:statistics.mean(r['phases_ms'][name] for r in exclusive) for name in exclusive[0]['phases_ms']}
    unprofiled=[json.loads((root/f'two_w64_r{i}/optimized_b16_w4_r0/summary.json').read_text()) for i in [0,1]]
    profiled=json.loads((root/'two_w64_r2/optimized_b16_w4_r0/summary.json').read_text())
    result=dict(status='complete',command='python -m benchmarks.system.profile_two_stage',environment='basis',
        mean_phases_ms=mean,profiled_mean_total_ms=statistics.mean(r['total_ms'] for r in exclusive),
        unprofiled_mean_steady_median_ms=statistics.mean(r['cuda_median_after10_ms'] for r in unprofiled),
        same_recorded_token_sequence=profiled['decode_input_ids']==unprofiled[0]['decode_input_ids'],
        nested_event_totals={str(k):dict(v) for k,v in totals.items()},steps=exclusive,
        scope='CUDA events in actual continuous decode steps 10..19, summed across all 32 layers. Stage spans include launch gaps and instrumentation. Residual categories are subtractions of nested intervals from this same run; no mixed-run subtraction or zero-router lower-bound claim.')
    (root/'breakdown.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Two-stage decode breakdown','',result['scope'],'',f'Instrumented mean: {result["profiled_mean_total_ms"]:.4f} ms; uninstrumented reference mean steady median: {result["unprofiled_mean_steady_median_ms"]:.4f} ms. Different sampling statistics; their difference is a diagnostic estimate of perturbation, not a rigorous correction.','',f'Generated sequence unchanged: {result["same_recorded_token_sequence"]}.','', '| Phase | ms/step | Share of instrumented total |','|---|---|---|']
    lines += [f'| {k} | {v:.4f} | {100*v/result["profiled_mean_total_ms"]:.2f}% |' for k,v in mean.items()]
    lines+=['','Timing does not include prefill. Configuration: basis environment, single L40S, TP1, Llama-3.1-8B, 64K, B16R16, 512 candidates, final hard2048 including sink32 and recent64.','', '`python -m benchmarks.system.profile_two_stage`','']
    (root/'BREAKDOWN.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
