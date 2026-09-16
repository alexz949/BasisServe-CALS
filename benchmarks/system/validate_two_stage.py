"""Check incremental historical metadata, original-position fine scan, and selection."""
from pathlib import Path
import json
import torch
from benchmarks.system.two_stage_router import Metadata,candidates,compile_fine


@torch.inference_mode()
def main():
    torch.manual_seed(314);torch.set_num_threads(2)
    root=Path('results/system_benchmarks/two_stage');root.mkdir(parents=True,exist_ok=True)
    full,fine=compile_fine(root/'source');rows=[]
    for length in [127,128,129,8192,65536]:
        b,h=2,2
        k=torch.randn(b,h,length+70,128,device='cuda',dtype=torch.bfloat16)
        meta=Metadata(k[:,:,:length],length+128)
        for step in range(66):
            if step:meta.advance(k[:,:,length+step-1:length+step])
            if step not in [0,1,31,32,33,65]:continue
            n=length+step-64;p=(n+31)//32
            history=torch.nn.functional.pad(k[:,:,:n],(0,0,0,(-n)%32))
            valid=torch.arange(p*32,device='cuda').reshape(p,32)<n
            tiles=history.reshape(b,h,p,32,128).float()
            lo=tiles.masked_fill(~valid[None,None,:,:,None],torch.inf).amin(-2).bfloat16()
            hi=tiles.masked_fill(~valid[None,None,:,:,None],-torch.inf).amax(-2).bfloat16()
            torch.testing.assert_close(meta.minimum[:,:,:p],lo,rtol=0,atol=0)
            torch.testing.assert_close(meta.maximum[:,:,:p],hi,rtol=0,atol=0)
            q=torch.randn(b,h*4,1,128,device='cuda',dtype=torch.bfloat16)
            u=meta.scores(q);ids=candidates(u,n)
            expected=torch.maximum(q.reshape(b,h,4,1,128).float()*lo[:,:,None].float(),q.reshape(b,h,4,1,128).float()*hi[:,:,None].float()).sum(-1)*128**-.5
            expected+=torch.minimum(torch.tensor(32,device='cuda'),n-torch.arange(p,device='cuda')*32).float().log()
            torch.testing.assert_close(u[...,:p],expected,rtol=1e-5,atol=2e-5)
            assert bool((ids[...,1:]>ids[...,:-1]).all())
            assert bool((ids[...,0]==0).all())
            for page in range(n//32,(n+64+31)//32):assert bool((ids==page).any(-1).all())
            if step!=0:continue
            base=torch.randn(b,h,n,16,device='cuda',dtype=q.dtype)*.2
            res=torch.randn_like(base)*.2;right=torch.randn(h,16,128,device='cuda',dtype=q.dtype)*.2
            bias=torch.randn(h,128,device='cuda',dtype=q.dtype)*.1;rq=torch.randn(h*4,128,16,device='cuda',dtype=q.dtype)*.2
            angle=torch.randn(n,64,device='cuda');cos=angle.cos().bfloat16();sin=angle.sin().bfloat16()
            code=torch.empty(b,h,4,16,device='cuda',dtype=q.dtype)
            ref=torch.empty(b,h,4,p,device='cuda');out=torch.empty(b,h,4,ids.shape[-1],device='cuda')
            args=(q,base,res,right,bias,rq,cos,sin,code)
            full.conditional_router_page_lse(*args,ref,128**-.5,False)
            fine.conditional_router_page_lse(*args,out,128**-.5,False,ids)
            selected=ref.gather(-1,ids.clamp_max(p-1)[:,:,None].expand(b,h,4,-1)).masked_fill(ids[:,:,None]>=p,-torch.inf)
            torch.testing.assert_close(out,selected,rtol=.003,atol=.003)
            rows.append(dict(length=length,candidates=ids.shape[-1],fine_max_abs=float((out-selected).nan_to_num().abs().max())))
            print(rows[-1],flush=True)
    (root/'validation.json').write_text(json.dumps(dict(status='complete',metadata_steps_checked=30,rows=rows),indent=2)+'\n')
    print('VALIDATED',flush=True)

if __name__=='__main__':main()
