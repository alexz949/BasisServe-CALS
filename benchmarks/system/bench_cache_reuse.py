"""Cache reuse ablation, preserving each method's selected support."""
import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import torch
from torch.utils.cpp_extension import load
import benchmarks.system.native_basis_cache as basis
from basisserve.kernels.mapped_host_paged_attention import mapped_host_device_pointer,gpu_paged_attention


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['basis16','basis8','shadowkv_cpu'],required=True)
    p.add_argument('--mode',choices=['native','reuse','reload'],required=True)
    p.add_argument('--length',type=int,default=65536);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    assert a.method!='shadowkv_cpu' or a.mode in ['native','reload']
    root=Path('results/system_benchmarks/cache_reuse')/a.mode
    caches={};shadow_counts=[]
    if a.method.startswith('basis') and a.mode!='native':
        ext=load(name='basis_selected_key_reuse',sources=['basisserve/kernels/csrc/selected_key_reuse.cu'],
            extra_cuda_cflags=['-O3','-std=c++17'],extra_cflags=['-O3','-std=c++17'])
        def fetch(host,q,value,ids,**kw):
            key=host.data_ptr();b,h,s=ids.shape
            if key not in caches:
                shape=(b,h,s,128)
                caches[key]=dict(old=torch.empty(shape,device='cuda',dtype=torch.bfloat16),
                    new=torch.empty(shape,device='cuda',dtype=torch.bfloat16),v=torch.empty(shape,device='cuda',dtype=torch.bfloat16),
                    ids=torch.full_like(ids,-1),slots=torch.full(value.shape[:3],-1,device='cuda',dtype=torch.int32),
                    counts=torch.zeros(101,b,h,device='cuda',dtype=torch.int32),step=0,pointer=mapped_host_device_pointer(host),
                    indices=torch.arange(s,device='cuda').expand(b,h,-1).contiguous(),valid=[])
            c=caches[key];step=c['step'];assert step<101
            ext.refresh(c['pointer'],c['old'],c['new'],value,c['v'],c['ids'],ids,c['slots'],c['counts'][step],a.mode=='reuse')
            compact=c['indices'].masked_fill(ids<0,-1)
            out=gpu_paged_attention(c['new'],q,c['v'],compact,sequence_length=s,page_size=1,
                splits=kw['splits'],workspace=kw['workspace'],scale=kw['scale'])
            if a.smoke:
                index=ids.clamp_min(0)[...,None].expand(-1,-1,-1,128)
                expected=host.gather(2,index.cpu()).cuda().masked_fill((ids<0)[...,None],0)
                torch.testing.assert_close(c['new'],expected,rtol=0,atol=0)
                reference=original(host,q,value,ids,**kw)
                torch.testing.assert_close(out,reference,rtol=.003,atol=.003)
            c['old'],c['new']=c['new'],c['old'];c['ids']=ids;c['step']+=1
            c['valid'].append((ids>=0).sum())
            return out
        original=basis.mapped_host_paged_attention;basis.mapped_host_paged_attention=fetch
    if a.method=='shadowkv_cpu':
        upstream=Path('/home/zhangal/BasisServe-CALS-runs/shadow_native/ShadowKV')
        sys.path.insert(0,str(upstream));sys.path.insert(0,str(upstream.parent/'deps'))
        spec=importlib.machinery.ModuleSpec('models',loader=None,is_package=True)
        package=importlib.util.module_from_spec(spec);package.__path__=[str(upstream/'models')];sys.modules['models']=package
        torch.ops.load_library(str(Path(importlib.util.find_spec('vllm').origin).parent/'_C.abi3.so'))
        from models.kv_cache import ShadowKVCache_CPU
        route=ShadowKVCache_CPU.get_retrieval_position_ids
        def retrieval(self,layer_idx,query_states):
            reference=None
            if a.mode=='reload':
                if a.smoke:reference=route(self,layer_idx,query_states).clone().sort(-1).values
                self.position_ids[layer_idx].fill_(-1)
            out=route(self,layer_idx,query_states)
            if a.smoke and a.mode=='reload':
                assert bool((self.cnts==0).all())
                assert torch.equal(out.sort(-1).values,reference)
            shadow_counts.append(self.cnts.clone())
            return out
        ShadowKVCache_CPU.get_retrieval_position_ids=retrieval
    sys.argv=['bench_shadow_native','--method',a.method,'--length',str(a.length),'--batch','1','--output',str(root)]
    if a.smoke:sys.argv.append('--smoke')
    runpy.run_module('benchmarks.system.bench_shadow_native',run_name='__main__')
    rows=[]
    for c in caches.values():
        count=c['step'];hits=c['counts'][:count].cpu();valid=torch.stack(c['valid']).cpu()
        rows.append(dict(steps=count,hits_by_step=hits.sum((1,2)).tolist(),valid_by_step=valid.tolist(),
            cache_bytes=sum(c[k].numel()*c[k].element_size() for k in ['old','new','v','ids','slots','counts','indices'])))
    folder=root/f'{a.method}_t{8192 if a.smoke else a.length}_b1{"_smoke" if a.smoke else ""}'

    report=dict(method=a.method,mode=a.mode,layers=rows,
        shadow_cached_chunks=[x.cpu().tolist() for x in shadow_counts],
        scope='Basis reuse and reload share the same GPU staging and attention kernels; reuse copies hits from previous GPU K. Shadow reload invalidates previous routed chunk IDs before native routing, retaining outlier/local caches. No refit or budget change.')
    (folder/'cache.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
