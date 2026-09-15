"""Router microbenchmarks. Each method keeps its native support semantics."""
import argparse
import csv
import json
import os
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from benchmarks.system.basis_router import BasisRouter
from benchmarks.system.common import metadata,save
from benchmarks.system.timing import measure,measure_host_sequence


def load_other(method,output,length,batch,smoke=False):
    from benchmarks.system.other_routers import LokiRouter,ShadowRouter
    folder=output/('capture_smoke' if smoke else 'capture')/f't{length}'
    keys={'loki':['k'],'shadow':['k','pre_k'],'lrqk':['k','raw_v']}[method]
    collected={key:[] for key in ['q',*keys]}
    for sample in range(batch):
        with safe_open(str(folder/f'sample{sample}.safetensors'),framework='pt',device='cpu') as f:
            collected['q'].append(f.get_tensor('q').cuda() if method=='lrqk' else f.get_slice('q')[:,:,-1:,:].cuda())
            for key in keys:collected[key].append(f.get_tensor(key).cuda())
            if sample==0:cos=f.get_tensor('cos').cuda();sin=f.get_tensor('sin').cuda()
    data={key:torch.cat(value,0) for key,value in collected.items()}
    del collected
    if method=='loki':
        basis=load_file(str(output/'loki/layer_003.safetensors'),device=f'cuda:{torch.cuda.current_device()}')['key_projector']
        return LokiRouter(data['q'],data['k'],basis)
    if method=='lrqk':
        from benchmarks.system.lrqk_router import LRQKRouter
        return LRQKRouter(data['q'],data['k'],data['raw_v'])
    return ShadowRouter(data['q'],data['pre_k'],data['k'],cos,sin)


def benchmark_other(a,rank):
    from benchmarks.system.other_routers import shadow_provenance
    dependency=shadow_provenance() if a.method=='shadow' else {}
    if a.method=='lrqk':
        from benchmarks.system.lrqk_router import provenance
        dependency=provenance()
    lengths=[a.length] if a.length is not None else [4096] if a.smoke else [[16384,32768,65536,131072][rank]]
    for length in lengths:
        for batch in ([a.batch] if a.batch is not None else [1] if a.smoke else [1,4,8]):
            print(dict(method=a.method,length=length,batch=batch,phase='build real routing state'),flush=True)
            before=metadata()
            router=load_other(a.method,a.output,length,batch,a.smoke)
            audit=router.validate()
            settings=dict(warmup=5 if a.smoke else 100,iterations=10 if a.smoke else 500)
            update_timer=measure_host_sequence if getattr(router,'native_dynamic_preprocessing',False) else measure
            pre=update_timer(router.preprocess,**settings) if router.has_preprocessing else None
            scan=measure(router.scan,**settings);total=update_timer(router.full,**settings)
            record=dict(method=a.method,model='Llama-3.1-8B base',layer=3,v_rank=96,
                dtype='bfloat16',length=length,batch=batch,query_heads=32,kv_heads=8,head_dim=128,
                state=router.state_stats(),audit=audit,query_preprocessing=pre,
                route_to_ids=scan,total_with_preprocessing=total,measured_dram_bytes=None,
                preprocessing_absent=not router.has_preprocessing)
            name=f'router_{a.method}_{"smoke_" if a.smoke else ""}t{length}_b{batch}'
            save(a.output/f'{name}.json',dict(metadata=before,dependency=dependency,records=[record]))
            print(dict(method=a.method,length=length,batch=batch,p50_us=scan['p50_us'],reference_ids_equal=audit['reference_ids_equal']),flush=True)
            del router;torch.cuda.empty_cache()


def load_basis(root,output,length,batch,smoke=False,budget=2048):
    folder=output/('capture_smoke' if smoke else 'capture')/f't{length}'
    collected={k:[] for k in ['q','k','value_transformed']}
    for sample in range(batch):
        with safe_open(str(folder/f'sample{sample}.safetensors'),framework='pt',device='cpu') as f:
            collected['q'].append(f.get_slice('q')[:,:,-1:,:].cuda())
            for key in ['k','value_transformed']:collected[key].append(f.get_tensor(key).cuda())
            if sample==0:cos=f.get_tensor('cos').cuda();sin=f.get_tensor('sin').cuda()
    data={k:torch.cat(v,0) for k,v in collected.items()}
    factors=load_file(str(root/'ours_b16r16/layer_003.safetensors'),device=f'cuda:{torch.cuda.current_device()}')
    return BasisRouter(data['q'],data['k'],data['value_transformed'][...,:16],data['value_transformed'][...,16:],factors,cos,sin,budget=budget)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--method',choices=['basis','loki','shadow','lrqk'],default='basis')
    p.add_argument('--length',type=int,choices=[4096,16384,32768,65536,131072])
    p.add_argument('--batch',type=int,choices=[1,4,8])
    p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'));p.add_argument('--smoke',action='store_true')
    a=p.parse_args();rank=int(os.environ.get('LOCAL_RANK',0))
    assert not a.smoke or (a.length in (None,4096) and a.batch in (None,1))
    if a.method=='lrqk':
        from benchmarks.system.numa_memory import bind_host_allocations,slurm_gpu_numa
        hardware=json.loads((a.output/'hardware.json').read_text())
        bind_host_allocations(slurm_gpu_numa(hardware,rank)['numa_node'])
    torch.cuda.set_device(rank);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    if a.method!='basis':
        benchmark_other(a,rank)
        return
    lengths=[a.length] if a.length is not None else [4096] if a.smoke else [[16384,32768,65536,131072][rank]]
    records=[]
    for length in lengths:
        for batch in ([a.batch] if a.batch is not None else [1] if a.smoke else [1,4,8]):
            router=load_basis(a.root,a.output,length,batch,a.smoke)
            audit=router.validate()
            pre=measure(router.preprocess,warmup=5 if a.smoke else 100,iterations=10 if a.smoke else 500)
            scan=measure(router.scan,warmup=5 if a.smoke else 100,iterations=10 if a.smoke else 500)
            total=measure(router.full,warmup=5 if a.smoke else 100,iterations=10 if a.smoke else 500)
            record=dict(method='BasisKV B16R16',layer=3,v_rank=96,dtype='bfloat16',length=length,batch=batch,
                        budget=2048,page_size=32,sink=32,recent=64,logical_routing_width=32,
                        routing_cache_bytes=router.base.numel()*2+router.residual.numel()*2,
                        estimated_coordinate_scan_bytes=batch*8*(length-64)*32*2,
                        measured_dram_bytes=None,audit=audit,query_preprocessing=pre,route_to_ids=scan,total_with_preprocessing=total)
            records.append(record);print(dict(length=length,batch=batch,router_p50_us=scan['p50_us'],audit=audit),flush=True)
            del router;torch.cuda.empty_cache()
    name='router_basis_smoke' if a.smoke else f'router_basis_rank{rank}'
    save(a.output/f'{name}.json',dict(metadata=metadata(),records=records,
        status='basis_only; ShadowKV/Loki/LRQK integration pending'))
    with (a.output/f'{name}.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['method','length','batch','query_p50_us','route_p50_us','route_p95_us','total_p50_us'])
        writer.writeheader()
        for r in records:writer.writerow(dict(method=r['method'],length=r['length'],batch=r['batch'],query_p50_us=r['query_preprocessing']['p50_us'],
            route_p50_us=r['route_to_ids']['p50_us'],route_p95_us=r['route_to_ids']['p95_us'],total_p50_us=r['total_with_preprocessing']['p50_us']))

if __name__=='__main__':main()
