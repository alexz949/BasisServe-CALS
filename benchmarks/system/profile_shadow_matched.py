"""Measure request wall time and optional decode stages in the matched runtime."""
import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import runpy
import statistics
import sys
import time
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',required=True,choices=['full','shadowkv_cpu','basis16','basis8'])
    parser.add_argument('--stages',action='store_true')
    a=parser.parse_args()
    root=Path('/home/zhangal/BasisServe-CALS-runs/shadow_native')
    upstream=root/'ShadowKV';sys.path.insert(0,str(upstream));sys.path.insert(0,str(root/'deps'))
    spec=importlib.machinery.ModuleSpec('models',loader=None,is_package=True)
    package=importlib.util.module_from_spec(spec);package.__path__=[str(upstream/'models')];sys.modules['models']=package
    binary=Path(importlib.util.find_spec('vllm').origin).parent/'_C.abi3.so'
    torch.ops.load_library(str(binary))
    from models.llama import Llama
    from models.kv_cache import ShadowKVCache_CPU
    import models.base as base
    import benchmarks.system.native_basis_cache as basis
    state=dict(step=0,active=False,warmup=0.0);events={};traffic={}

    def timed(owner,name,label):
        original=getattr(owner,name)
        def wrapped(*args,**kwargs):
            if not state['active']:return original(*args,**kwargs)
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record();out=original(*args,**kwargs);end.record()
            events.setdefault(label,[]).append((begin,end))
            if label=='shadow_fetch_V':
                traffic.setdefault('shadow_cached_chunks',[]).append(args[0].cnts.clone())
            if label=='basis_fetch_K_attention':
                traffic.setdefault('basis_selected_tokens',[]).append((args[3]>=0).sum(-1))
            return out
        setattr(owner,name,wrapped)

    if a.stages:
        for name,label in [('conditional_router_page_lse','basis_route'),('select_fixed_group_max_pages_cuda','basis_select'),
                           ('append_mapped_host_key','basis_append_K'),('mapped_host_paged_attention','basis_fetch_K_attention')]:
            timed(basis,name,label)
        timed(basis.BasisCache,'attention','basis_cache_attention_total')
        for name,label in [('get_retrieval_position_ids','shadow_route'),('get_value_cache','shadow_fetch_V'),('get_key_cache','shadow_reconstruct_K')]:
            timed(ShadowKVCache_CPU,name,label)
        timed(base,'flash_attn_with_kvcache','shadow_or_dense_attention')

    inference=Llama.inference;generate=Llama.batch_generate;warmup=Llama.warmup
    def measured_inference(self,input_ids,position_ids):
        decode=input_ids.shape[1]==1
        if decode:
            state['step']+=1
            if state['step']==101:
                torch.cuda.synchronize();state['request_end']=time.perf_counter()
        state['active']=a.stages and decode and 11<=state['step']<=20
        out=inference(self,input_ids,position_ids)
        state['active']=False
        if decode:state.setdefault('finite',[]).append(torch.isfinite(out).all())
        return out
    def measured_warmup(self):
        torch.cuda.synchronize();start=time.perf_counter();out=warmup(self);torch.cuda.synchronize()
        state['warmup']+=time.perf_counter()-start
        return out
    def measured_generate(self,*args,**kwargs):
        torch.cuda.synchronize();state['request_start']=time.perf_counter()
        return generate(self,*args,**kwargs)
    Llama.inference=measured_inference;Llama.warmup=measured_warmup;Llama.batch_generate=measured_generate
    output=Path('results/system_benchmarks/shadow_profile' if a.stages else 'results/system_benchmarks/shadow_e2e')
    sys.argv=['bench_shadow_native','--method',a.method,'--length','65536','--batch','1','--output',str(output)]
    runpy.run_module('benchmarks.system.bench_shadow_native',run_name='__main__')
    torch.cuda.synchronize();assert state['step']==101 and bool(torch.stack(state['finite']).all())
    stage_rows={}
    for label,pairs in events.items():
        raw=[begin.elapsed_time(end) for begin,end in pairs]
        stage_rows[label]=dict(calls=len(raw),raw_ms=raw,mean_call_ms=statistics.mean(raw),ms_per_model_step=sum(raw)/10)
    traffic_report={}
    if 'shadow_cached_chunks' in traffic:
        counts=torch.stack(traffic['shadow_cached_chunks']).float().cpu()
        traffic_report['shadow_cached_chunks_mean_per_kv_head']=float(counts.mean())
        traffic_report['shadow_new_chunks_mean_per_kv_head']=float((256-counts).mean())
        traffic_report['shadow_logical_H2D_V_bytes_per_decode']=float((256-counts).sum()*8*128*2/10)
    if 'basis_selected_tokens' in traffic:
        counts=torch.stack(traffic['basis_selected_tokens']).float().cpu()
        traffic_report['basis_logical_host_K_bytes_per_decode']=float(counts.sum()*128*2/10)
    report=dict(traffic=traffic_report,method=a.method,length=65536,batch=1,decode_steps=100,
        request_wall_seconds=state['request_end']-state['request_start'],synthetic_warmup_seconds=state['warmup'],
        request_without_synthetic_warmup_seconds=state['request_end']-state['request_start']-state['warmup'],
        stages=stage_rows,all_decode_logits_finite=True,
        scope='Model already loaded; starts before prefill, ends after100 decode iterations and sampling; excludes native extra final inference and cleanup. Stage profiling is instrumented, nested and overlapping across streams: do not add stage totals.',
        instrumented=a.stages)
    (output/f'{a.method}_t65536_b1/request.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='stages'},flush=True)


if __name__=='__main__':main()
