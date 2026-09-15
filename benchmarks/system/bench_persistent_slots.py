"""Persistent K slots and direct GPU V reads in the matched Llama runtime."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import torch
from torch.utils.cpp_extension import load
import benchmarks.system.native_basis_cache as basis
from basisserve.kernels.mapped_host_paged_attention import mapped_host_device_pointer
from basisserve.kernels.slot_indexed_attention import slot_indexed_attention


def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,choices=[8,16],required=True)
    p.add_argument('--mode',choices=['reuse','reload'],required=True);p.add_argument('--length',type=int,default=65536)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    ext=load(name='basis_persistent_key_slots',sources=['basisserve/kernels/csrc/persistent_key_slots.cu'],
        extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-std=c++17'])
    caches={};original=basis.mapped_host_paged_attention
    def attention(host,q,value,ids,**kw):
        identity=host.data_ptr();b,h,s=ids.shape;heads=q.shape[1];rank=value.shape[-1]
        if identity not in caches:
            caches[identity]=dict(key=torch.empty(b,h,s,128,device='cuda',dtype=torch.bfloat16),
                resident=torch.full_like(ids,-1),lookup=torch.full(value.shape[:3],-1,device='cuda',dtype=torch.int32),
                slots=torch.empty_like(ids),missing=torch.empty_like(ids,dtype=torch.int32),
                counts=torch.empty(101,b,h,2,device='cuda',dtype=torch.int32),step=0,
                pointer=mapped_host_device_pointer(host),workspace=(
                    torch.empty(b,heads,16,rank,device='cuda',dtype=torch.float32),
                    torch.empty(b,heads,16,device='cuda',dtype=torch.float32),
                    torch.empty(b,heads,1,rank,device='cuda',dtype=torch.bfloat16)))
        c=caches[identity];step=c['step'];assert step<101
        ext.refresh(c['pointer'],c['key'],ids,c['resident'],c['lookup'],c['slots'],c['missing'],c['counts'][step],a.mode=='reuse')
        out=slot_indexed_attention(q,c['key'],value,ids,c['slots'],c['workspace'],scale=kw['scale'])
        if a.smoke:
            safe=c['slots'].clamp_min(0)[...,None].expand(-1,-1,-1,128)
            actual=c['key'].gather(2,safe).masked_fill((ids<0)[...,None],0)
            expected=host.gather(2,ids.clamp_min(0)[...,None].expand(-1,-1,-1,128).cpu()).cuda().masked_fill((ids<0)[...,None],0)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            reference=original(host,q,value,ids,**kw)
            torch.testing.assert_close(out,reference,rtol=.003,atol=.003)
            assert bool(torch.isfinite(out).all())
        c['step']+=1
        return out
    basis.mapped_host_paged_attention=attention
    root=Path('results/system_benchmarks/persistent_slots')/a.mode
    sys.argv=['bench_shadow_native','--method',f'basis{a.rank}','--length',str(a.length),'--batch','1','--output',str(root)]
    if a.smoke:sys.argv.append('--smoke')
    runpy.run_module('benchmarks.system.bench_shadow_native',run_name='__main__')
    rows=[];measured=8 if a.smoke else 100
    for c in caches.values():
        counts=c['counts'][:measured].cpu()
        persistent=sum(c[k].numel()*c[k].element_size() for k in ['key','resident','lookup','slots','missing','counts'])
        persistent+=sum(x.numel()*x.element_size() for x in c['workspace'])
        rows.append(dict(hits_by_step=counts[...,0].sum((1,2)).tolist(),valid_by_step=counts[...,1].sum((1,2)).tolist(),cache_bytes=persistent))
    hits=sum(sum(r['hits_by_step']) for r in rows);valid=sum(sum(r['valid_by_step']) for r in rows)
    report=dict(method=f'basis{a.rank}',mode=a.mode,length=8192 if a.smoke else a.length,smoke=a.smoke,
        hit_fraction=hits/valid,logical_host_K_bytes_per_step=(valid-hits)*128*2/measured,
        cache_bytes=sum(r['cache_bytes'] for r in rows),layers=rows,
        scope='One persistent GPU K slot array per layer; only misses copied from CPU. No copying cache hits and no V staging. Separate K-slot/V-token indexed Triton attention. Reload uses identical kernels with hits disabled. Cache bytes include slot metadata and attention workspace.')
    folder=root/f'basis{a.rank}_t{report["length"]}_b1{"_smoke" if a.smoke else ""}'
    (folder/'cache.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='layers'},flush=True)


if __name__=='__main__':main()
