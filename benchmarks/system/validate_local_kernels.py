"""Synthetic correctness checks for local kernel candidates."""
import json
from pathlib import Path
import torch
from basisserve.kernels.gqa_slot_attention import gqa_slot_attention
from basisserve.kernels.slot_indexed_attention import slot_indexed_attention
from basisserve.kernels.mapped_host_paged_attention import mapped_host_bf16_empty, append_mapped_host_key, mapped_host_device_pointer, conditional_router_page_lse
from benchmarks.system.local_kernel_candidates import compile_slots, compile_router


def workspace(q, value_dim):
    b,h=q.shape[:2]
    return (torch.empty(b,h,16,value_dim,device=q.device),torch.empty(b,h,16,device=q.device),
            torch.empty(b,h,1,value_dim,device=q.device,dtype=q.dtype))


@torch.inference_mode()
def main():
    torch.manual_seed(91);torch.set_num_threads(2)
    root=Path('results/system_benchmarks/local_kernels');root.mkdir(parents=True,exist_ok=True)
    scalar=compile_slots(root/'source',False);vector=compile_slots(root/'source',True)
    records=[]
    for width in [80,128]:
        b,h,t,n=1,2,4096,2048
        q=torch.randn(b,h*4,1,128,device='cuda',dtype=torch.bfloat16)
        key=torch.randn(b,h,t,128,device='cuda',dtype=torch.bfloat16)
        value=torch.randn(b,h,t,width,device='cuda',dtype=torch.bfloat16)
        host=mapped_host_bf16_empty(batch=b,kv_heads=h,capacity=t);append_mapped_host_key(host,key,start=0)
        pointer=mapped_host_device_pointer(host)
        states=[]
        for _ in range(2):
            states.append(dict(cache=torch.empty(b,h,n,128,device='cuda',dtype=torch.bfloat16),
                resident=torch.full((b,h,n),-1,device='cuda',dtype=torch.long),lookup=torch.full((b,h,t),-1,device='cuda',dtype=torch.int32),
                slots=torch.empty(b,h,n,device='cuda',dtype=torch.long),missing=torch.empty(b,h,n,device='cuda',dtype=torch.int32),counts=torch.empty(b,h,2,device='cuda',dtype=torch.int32)))
        sequences=[torch.arange(n,device='cuda'),torch.arange(n-1,-1,-1,device='cuda'),torch.arange(1024,3072,device='cuda')]
        sequences += [torch.randperm(t,device='cuda')[:n] for _ in range(5)]
        sequences.append(sequences[-1].clone())
        for step,sequence in enumerate(sequences):
            ids=sequence.expand(b,h,-1).clone()
            if step==3:ids[...,128:]=-1  # Many entirely empty splits, but nonempty support.
            if step==5:ids[...,-31:]=-1
            expected=key.gather(2,ids.clamp_min(0)[...,None].expand(-1,-1,-1,128)).masked_fill((ids<0)[...,None],0)
            # Atomic free-slot assignment can choose different victims. Start both
            # refresh variants from exactly the same resident/lookup state.
            for name in ['cache','resident','lookup']:
                states[1][name].copy_(states[0][name])
            for ext,c in zip([scalar,vector],states):
                ext.refresh(pointer,c['cache'],ids,c['resident'],c['lookup'],c['slots'],c['missing'],c['counts'],step!=6)
                actual=c['cache'].gather(2,c['slots'].clamp_min(0)[...,None].expand(-1,-1,-1,128)).masked_fill((ids<0)[...,None],0)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(states[0]['counts'],states[1]['counts'],rtol=0,atol=0)
            c=states[0];work=workspace(q,width)
            reference=slot_indexed_attention(q,c['cache'],value,ids,c['slots'],work,scale=128**-.5).clone()
            gathered=value.gather(2,ids.clamp_min(0)[...,None].expand(-1,-1,-1,width)).repeat_interleave(4,1).float()
            scores=(q.float()@expected.repeat_interleave(4,1).float().transpose(-1,-2))*128**-.5
            scores=scores.masked_fill((ids<0).repeat_interleave(4,1)[:,:,None],-torch.inf)
            dense=scores.softmax(-1)@gathered
            for warps in [4,8]:
                out=gqa_slot_attention(q,c['cache'],value,ids,c['slots'],work,scale=128**-.5,num_warps=warps)
                torch.testing.assert_close(out,reference,rtol=.003,atol=.003)
                torch.testing.assert_close(out.float(),dense,rtol=.003,atol=.003)
                records.append(dict(kind='fetch_attention',width=width,step=step,warps=warps,max_abs=float((out-reference).abs().max())))
    for rank in [8,16]:
        ext=compile_router(root/'source',rank)
        reference_ext=compile_router(root/'source',rank,reference=True)
        for tokens in [1,31,32,33,257]:
            q=torch.randn(1,8,1,128,device='cuda',dtype=torch.bfloat16)
            base=torch.randn(1,2,tokens,rank,device='cuda',dtype=torch.bfloat16)*.2
            res=torch.randn_like(base)*.2
            right=torch.randn(2,rank,128,device='cuda',dtype=torch.bfloat16)*.2
            bias=torch.randn(2,128,device='cuda',dtype=torch.bfloat16)*.1
            residual_q=torch.randn(8,128,rank,device='cuda',dtype=torch.bfloat16)*.2
            angle=torch.randn(tokens,64,device='cuda');cos=angle.cos().bfloat16();sin=angle.sin().bfloat16()
            reference=torch.empty(1,2,4,(tokens+31)//32,device='cuda')
            code=torch.empty(1,2,4,rank,device='cuda',dtype=torch.bfloat16);out=torch.empty_like(reference)
            reference_ext.conditional_router_page_lse(q,base,res,right,bias,residual_q,cos,sin,code,reference,128**-.5,False)
            ext.conditional_router_page_lse(q,base,res,right,bias,residual_q,cos,sin,code,out,128**-.5,False)
            torch.testing.assert_close(out,reference,rtol=.003,atol=.003)
            records.append(dict(kind='router',rank=rank,tokens=tokens,max_abs=float((out-reference).abs().max())))
    (root/'validation.json').write_text(json.dumps(dict(status='complete',cases=records),indent=2)+'\n')
    print('VALIDATED',len(records),'cases',flush=True)


if __name__=='__main__':main()
