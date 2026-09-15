"""Exercise cold fill, full hits, evictions, invalid slots and repeated reuse."""
import json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from basisserve.kernels.mapped_host_paged_attention import mapped_host_bf16_empty,append_mapped_host_key,mapped_host_device_pointer
from basisserve.kernels.slot_indexed_attention import slot_indexed_attention


@torch.inference_mode()
def main():
    torch.manual_seed(19);torch.set_num_threads(2)
    ext=load(name='basis_persistent_key_slots',sources=['basisserve/kernels/csrc/persistent_key_slots.cu'],
        extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','-std=c++17'])
    b,h,capacity,budget=1,2,8192,2048
    key=torch.randn(b,h,capacity,128,device='cuda',dtype=torch.bfloat16)
    value=torch.randn_like(key);q=torch.randn(b,h*4,1,128,device='cuda',dtype=torch.bfloat16)
    host=mapped_host_bf16_empty(batch=b,kv_heads=h,capacity=capacity);append_mapped_host_key(host,key,start=0)
    pointer=mapped_host_device_pointer(host);cached=torch.empty(b,h,budget,128,device='cuda',dtype=torch.bfloat16)
    resident=torch.full((b,h,budget),-1,device='cuda',dtype=torch.long)
    lookup=torch.full((b,h,capacity),-1,device='cuda',dtype=torch.int32)
    slots=torch.empty_like(resident);missing=torch.empty_like(resident,dtype=torch.int32)
    counts=torch.empty(b,h,2,device='cuda',dtype=torch.int32)
    workspace=(torch.empty(b,h*4,16,128,device='cuda'),torch.empty(b,h*4,16,device='cuda'),torch.empty_like(q))
    sequences=[torch.arange(budget,device='cuda'),torch.arange(budget-1,-1,-1,device='cuda'),
        torch.arange(1024,1024+budget,device='cuda')]+[torch.randperm(capacity,device='cuda')[:budget] for _ in range(10)]
    sequences.append(sequences[-1].clone());rows=[]
    for step,sequence in enumerate(sequences):
        ids=sequence.expand(b,h,-1).clone()
        if step in [3,7]:ids[:,:,-31:]=-1
        reuse=step!=8
        previous=lookup.gather(2,ids.clamp_min(0)).long()
        hits=(ids>=0)&(previous>=0)&(resident.gather(2,previous.clamp_min(0))==ids)&reuse
        ext.refresh(pointer,cached,ids,resident,lookup,slots,missing,counts,reuse)
        assert torch.equal(counts[...,0],hits.sum(-1).int())
        assert torch.equal(counts[...,1],(ids>=0).sum(-1).int())
        assert torch.equal(slots[hits],previous[hits])
        actual=cached.gather(2,slots.clamp_min(0)[...,None].expand(-1,-1,-1,128)).masked_fill((ids<0)[...,None],0)
        expected=key.gather(2,ids.clamp_min(0)[...,None].expand(-1,-1,-1,128)).masked_fill((ids<0)[...,None],0)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        out=slot_indexed_attention(q,cached,value,ids,slots,workspace,scale=128**-.5)
        v=value.gather(2,ids.clamp_min(0)[...,None].expand(-1,-1,-1,128)).repeat_interleave(4,1).float()
        scores=(q.float()@expected.repeat_interleave(4,1).float().transpose(-1,-2))*128**-.5
        scores=scores.masked_fill((ids<0).repeat_interleave(4,1)[:,:,None],-torch.inf)
        reference=scores.softmax(-1)@v
        torch.testing.assert_close(out.float(),reference,rtol=.003,atol=.003)
        rows.append(dict(step=step,reuse=reuse,hits=int(hits.sum()),valid=int((ids>=0).sum())))
    root=Path('results/system_benchmarks/persistent_slots');root.mkdir(parents=True,exist_ok=True)
    (root/'kernel_validation.json').write_text(json.dumps(dict(status='complete',cases=rows),indent=2)+'\n')
    print('persistent slots validation passed',rows,flush=True)


if __name__=='__main__':main()
