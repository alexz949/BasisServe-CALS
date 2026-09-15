"""Official LRQK's final routing point after a continuous 64-token teacher tail."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from evaluation.official_lrqk_state import OfficialLRQKState,upstream
from benchmarks.system.other_routers import support_stats
from benchmarks.system.common import command


class LRQKRouter:
    has_preprocessing=True
    native_dynamic_preprocessing=True

    @torch.inference_mode()
    def __init__(self,q,k,value,steps=64):
        self.batch,self.heads,self.length,_=k.shape
        assert q.shape[1]==self.heads*4 and self.length>2048+steps+64
        torch.manual_seed(0)
        prefix=self.length-steps
        self.state=OfficialLRQKState(q[:,:,:prefix],k[:,:,:prefix],SimpleNamespace(topk=2048))
        self.state.bind_values(value[:,:,:prefix])
        cache=self.state.cache
        self.native_scan=cache.compute_hit_indices
        native_update=upstream.cast_lrqk_decode
        self.capture=False
        def observe_update(**kwargs):
            if self.capture:
                self.update_inputs={name:item.clone() if isinstance(item,torch.Tensor) else item for name,item in kwargs.items()}
            result=tuple(native_update(**kwargs))
            if self.capture:self.update_reference=tuple(x.clone() for x in result)
            return result
        def observe_scan(query_states,key_states):
            result=self.native_scan(query_states=query_states,key_states=key_states)
            if self.capture:
                self.query_code=query_states.clone()
                self.codes=key_states
                self.reference_hits=result.clone()
            return result
        upstream.cast_lrqk_decode=observe_update
        cache.compute_hit_indices=observe_scan
        for position in range(prefix,self.length):
            self.capture=position==self.length-1
            cache.decode(q[:,:,position:position+1],k[:,:,position:position+1],value[:,:,position:position+1])
        upstream.cast_lrqk_decode=native_update
        cache.compute_hit_indices=self.native_scan
        self.native_update=native_update
        self.q=q[:,:,-1:].contiguous()
        self.lite_ids=torch.tensor(cache.lite_indices,device=q.device,dtype=torch.int64).view(1,1,64).expand(self.batch,self.heads*4,-1)
        self.native_ids=torch.cat((cache.hit_indices[...,0],self.lite_ids),-1).reshape(self.batch,self.heads,4,-1).clone()
        # Verify physical slot order against original exact K before discarding
        # the setup input. The native cache reorders slots when it reuses hits.
        gathered=k[:,:,None].expand(self.batch,self.heads,4,self.length,128).gather(3,
            self.native_ids[...,None].expand(*self.native_ids.shape,128))
        assert torch.equal(gathered.reshape_as(cache.Kgpu),cache.Kgpu)
        assert all(bool(torch.isfinite(t).all()) for t in [self.query_code,self.codes,cache.B_Q,cache.B_K])
        self.steps=steps
        self.scan()

    def preprocess(self):
        self.update_output=tuple(self.native_update(**self.update_inputs))
        self.query_code=self.update_output[2]

    def scan(self):
        hits=self.native_scan(query_states=self.query_code,key_states=self.codes)
        self.ids=torch.cat((hits[...,0],self.lite_ids),-1).reshape(self.batch,self.heads,4,-1)
        return self.ids

    def full(self):self.preprocess();return self.scan()

    def validate(self):
        from benchmarks.system.numa_memory import audit_host_tensor,bind_host_allocations,slurm_gpu_numa
        import os
        self.full()
        for actual,reference in zip(self.update_output,self.update_reference):
            torch.testing.assert_close(actual,reference,rtol=0,atol=0)
        assert torch.equal(self.ids.sort(-1).values,self.native_ids.sort(-1).values)
        # A FP32 accumulation near a BF16 midpoint can round to the adjacent
        # BF16 score and change a top-k tie. Check numeric error separately
        # from exact native-precision selection; do not change native routing.
        fp32=self.codes.float()@self.query_code.float().transpose(-1,-2)
        scores=self.codes@self.query_code.transpose(-1,-2)
        magnitude=self.codes.float().abs()@self.query_code.float().abs().transpose(-1,-2)
        u=torch.finfo(torch.float32).eps
        gamma=2*self.codes.shape[-1]*u/(1-self.codes.shape[-1]*u)
        bound=torch.finfo(scores.dtype).eps*fp32.abs()+gamma*magnitude+torch.finfo(torch.float32).tiny
        assert bool(((scores.float()-fp32).abs()<=bound).all())
        hits=scores.topk(2048,dim=2).indices[...,0]
        reference=torch.cat((hits,self.lite_ids),-1).reshape_as(self.ids)
        assert torch.equal(reference.sort(-1).values,self.ids.sort(-1).values)
        fp32_hits=fp32.to(scores.dtype).topk(2048,dim=2).indices[...,0]
        fp32_ids_equal=torch.equal(fp32_hits.sort(-1).values,hits.sort(-1).values)
        hardware=json.loads(Path('results/system_benchmarks/l40s/hardware.json').read_text())
        node=slurm_gpu_numa(hardware,int(os.environ.get('LOCAL_RANK',0)))['numa_node']
        host_audit=audit_host_tensor(self.state.cache.KVcpu.data,node,bind_host_allocations(node))
        return dict(reference_ids_equal=True,native_online_update_bitwise_equal=True,host_cache_audit=host_audit,
            independent_fp32_score_error_bound_passed=True,fp32_rounded_topk_ids_equal=fp32_ids_equal,
            native_exact_key_cache_verified=True,continuous_decode_steps=self.steps,**support_stats(self.ids,self.length))

    def state_stats(self):
        cache=self.state.cache
        return dict(rank=32,nominal_budget=2048,lite=64,sink=0,
            routing_read_dimensions_per_token=32*4,
            routing_state_bytes=cache._A_K.data.numel()*cache._A_K.data.element_size()+cache.B_Q.numel()*cache.B_Q.element_size()+cache.B_K.numel()*cache.B_K.element_size()+cache.Kgpu.numel()*cache.Kgpu.element_size(),
            scan_state_bytes=self.codes.numel()*self.codes.element_size(),
            active_exact_key_bytes_for_query_update=cache.Kgpu.numel()*cache.Kgpu.element_size(),
            benchmark_update_snapshot_bytes=sum(x.numel()*x.element_size() for x in self.update_inputs.values() if isinstance(x,torch.Tensor)),
            state_dtype=str(cache.A_K.dtype),solve_dtype='float32',max_iter=[2,2],tolerance=[.01,.01],
            continuous_decode_steps=self.steps,scanned_tokens=self.codes.shape[2],
            selection='official independent query-head topk plus native lite slot IDs; native slot permutation verified',
            query_update_measurement='Native eager update of a frozen last-step input snapshot; allocations and convergence host synchronization are included and not isolated')


def provenance():
    root=Path('external/LRQK');path=root/'lrqk_attention.py'
    return dict(commit=command(['git','-C',str(root),'rev-parse','HEAD']),
        dirty=command(['git','-C',str(root),'diff']),
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()})
