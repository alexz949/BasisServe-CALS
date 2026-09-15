"""Native router support through one deduplicating exact-K/split-V backend."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from basisserve.core.c1_k_offload import PinnedCPUExactKeyPageStore,PreparedQueryKeyFetch
from basisserve.kernels.indexed_sparse_decode_attention import gqa_indexed_sparse_decode_attention_triton
from benchmarks.system.bench_router import load_basis,load_other
from benchmarks.system.common import metadata,save
from benchmarks.system.numa_memory import bind_host_allocations,audit_host_tensor,slurm_gpu_numa
from benchmarks.system.other_routers import support_stats,shadow_provenance
from benchmarks.system.timing import measure,measure_host_sequence


class CommonAttention:
    def __init__(self,router,host_key,prefix,tail):
        self.router=router;self.prefix=prefix;self.tail=tail
        b,h,t,d=host_key.shape;g=router.q.shape[1]//h
        router.full()
        native=router.ids
        if native.ndim==3:native=native[:,:,None].expand(-1,-1,g,-1)
        self.ids=native.contiguous().clone()
        self.fetch=PreparedQueryKeyFetch(PinnedCPUExactKeyPageStore(host_key),groups=g,
            tokens_per_query=self.ids.shape[-1],device=router.q.device)
        self.output=torch.empty(b,h*g,1,prefix.shape[-1]+tail.shape[-1],device=router.q.device,dtype=router.q.dtype)
        self.ids_view=self.ids.view(b,h*g,-1)
        self.rows_view=self.fetch.inverse_ids.view(b,h*g,-1)

    def copy_support(self):
        source=self.router.ids
        self.ids.copy_(source[:,:,None] if source.ndim==3 else source)

    def attention(self):
        return gqa_indexed_sparse_decode_attention_triton(self.router.q,self.fetch.destination,self.tail,
            self.ids_view,selected_key_rows=self.rows_view,value_prefix=self.prefix,output=self.output,scale=128**-.5)

    def validate(self,host_key):
        self.fetch(self.ids);observed=self.attention().clone()
        b,h,g,n=self.ids.shape;d=host_key.shape[-1]
        cpu_ids=self.ids.cpu()
        # Reference uploads selected K only, never the full host cache.
        selected=host_key[:,:,None].expand(b,h,g,-1,d).gather(3,cpu_ids.clamp_min(0)[...,None].expand(b,h,g,n,d)).cuda().float()
        logits=(self.router.q.reshape(b,h,g,1,d).float()*selected).sum(-1)*128**-.5
        mask=self.ids>=0;weights=logits.masked_fill(~mask,-torch.inf).softmax(-1)
        expected=[]
        for value in [self.prefix,self.tail]:
            width=value.shape[-1]
            picked=value[:,:,None].expand(b,h,g,-1,width).gather(3,self.ids.clamp_min(0)[...,None].expand(b,h,g,n,width)).float()
            expected.append((weights[...,None]*picked).sum(-2))
        expected=torch.cat(expected,-1).reshape_as(observed)
        torch.testing.assert_close(observed.float(),expected,rtol=.015,atol=.004)
        flat=host_key.reshape(-1,d)
        n_unique=self.fetch.actual_count
        assert torch.equal(self.fetch.destination[:n_unique].cpu(),flat[self.fetch.host_indices[:n_unique]])
        return dict(exact_key_equal=True,independent_query_support_reference=True,
            rel_mse=float((observed.float()-expected).square().sum()/expected.square().sum()),
            persistent_gpu_key_staging_bytes=self.fetch.destination.numel()*2,
            full_host_key_bytes=host_key.numel()*2,**support_stats(self.ids,host_key.shape[2]))


def graph(function):
    for _ in range(5):function()
    torch.cuda.synchronize()
    result=torch.cuda.CUDAGraph()
    with torch.cuda.graph(result):function()
    return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'))
    p.add_argument('--method',choices=['basis','loki','shadow','lrqk'],required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    hardware=json.loads((a.output/'hardware.json').read_text())
    topology=slurm_gpu_numa(hardware)
    node=topology['numa_node'];library=bind_host_allocations(node)
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    meta=metadata();length=4096 if a.smoke else 65536
    router=load_basis(a.root,a.output,length,1,a.smoke) if a.method=='basis' else load_other(a.method,a.output,length,1,a.smoke)
    router_audit=router.validate()
    folder=a.output/('capture_smoke' if a.smoke else 'capture')/f't{length}'
    with safe_open(str(folder/'sample0.safetensors'),framework='pt',device='cpu') as f:
        source=f.get_tensor('k');host=torch.empty(source.shape,dtype=source.dtype,pin_memory=True);host.copy_(source)
        if a.method=='basis':prefix,tail=router.base,router.tail
        else:
            value=f.get_tensor('value_transformed')
            prefix=value[...,:16].contiguous().cuda();tail=value[...,16:].contiguous().cuda()
    backend=CommonAttention(router,host,prefix,tail)
    audit=backend.validate(host)
    numa=[audit_host_tensor(t,node,library) for t in [host,backend.fetch.host_count,backend.fetch.host_indices,backend.fetch.staging]]
    settings=dict(warmup=5 if a.smoke else 100,iterations=10 if a.smoke else 500)
    dynamic=getattr(router,'native_dynamic_preprocessing',False)
    update_timer=measure_host_sequence if dynamic else measure
    pre=update_timer(router.preprocess,**settings) if getattr(router,'has_preprocessing',True) else None
    route=measure(router.scan,**settings)
    route_graph=None if dynamic else graph(router.full)
    backend.copy_support();backend.fetch(backend.ids)
    attention=measure(backend.attention,**settings)
    attention_graph=graph(backend.attention)
    fetch=measure_host_sequence(lambda:backend.fetch(backend.ids),**settings)
    def total():
        if dynamic:router.full()
        else:route_graph.replay()
        backend.copy_support();backend.fetch(backend.ids);attention_graph.replay()
    combined=measure_host_sequence(total,**settings)
    dependency=shadow_provenance() if a.method=='shadow' else {}
    if a.method=='lrqk':
        from benchmarks.system.lrqk_router import provenance
        dependency=provenance()
    save(a.output/f'common_{a.method}{"_smoke" if a.smoke else ""}.json',dict(metadata=meta,topology=topology,
        dependency=dependency,native_dynamic_preprocessing=dynamic,
        method=a.method,label='native routing + common exact-K backend; not full native serving system',
        model='Llama-3.1-8B base',layer=3,batch=1,length=length,v_rank=96,dtype='bfloat16',
        query_preprocessing=pre,route_to_ids=route,fetch=fetch,exact_qk_softmax_pv_fused=attention,
        total_with_query_preprocessing=combined,traffic=backend.fetch.traffic(),audit=audit,router_audit=router_audit,numa=numa,
        timing_note='Fused exact QK/softmax/PV reported together; staged fetch includes GPU deduplication and host dependencies. LRQK native eager updates include temporary allocation and convergence synchronization; not allocator-isolated.'))
    print(dict(method=a.method,length=length,total_p50_us=combined['p50_us'],traffic=backend.fetch.traffic()),flush=True)


if __name__=='__main__':main()
