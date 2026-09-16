"""Independent rounded mathematical reference and current-kernel A/B."""
import json
from pathlib import Path
import torch
from benchmarks.system.register_router import compile_register
from benchmarks.system.profile_router_phases import timing


@torch.inference_mode()
def main():
    torch.manual_seed(93);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path('results/system_benchmarks/register_router');root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for rank in [8,16]:
        modules={w:compile_register(root/'source',rank,w) for w in [0,4,8]}
        for n in [1,31,32,33,63,64,65,127,128,129,257,8192,65536]:
            b,h=(2,3) if n<8192 else (1,8)
            q=torch.randn(b,h*4,1,128,device='cuda',dtype=torch.bfloat16)
            base=torch.randn(b,h,n,rank,device='cuda',dtype=torch.bfloat16)*.2
            res=torch.randn_like(base)*.2
            right=torch.randn(h,rank,128,device='cuda',dtype=torch.bfloat16)*.2
            bias=torch.randn(h,128,device='cuda',dtype=torch.bfloat16)*.1
            rq=torch.randn(h*4,128,rank,device='cuda',dtype=torch.bfloat16)*.2
            angle=torch.randn(n,64,device='cuda');cos=angle.cos().bfloat16();sin=angle.sin().bfloat16()
            code=torch.empty(b,h,4,rank,device='cuda',dtype=torch.bfloat16)
            outputs={w:torch.empty(b,h,4,(n+31)//32,device='cuda') for w in modules}
            def call(w):modules[w].conditional_router_page_lse(q,base,res,right,bias,rq,cos,sin,code,outputs[w],128**-.5,False)
            call(0)
            if n<=257:
                # Independent BF16 boundary reference, including both bias and each RoPE product.
                k=(base.float()@right.float()).bfloat16()
                k=(k.float()+bias[None,:,None].float()).bfloat16().float()
                x=((k[...,:64]*cos.float()).bfloat16().float()-(k[...,64:]*sin.float()).bfloat16().float()).bfloat16().float()
                y=((k[...,64:]*cos.float()).bfloat16().float()+(k[...,:64]*sin.float()).bfloat16().float()).bfloat16().float()
                s=torch.einsum('bhqd,bhtd->bhqt',q.reshape(b,h,4,128).float(),torch.cat([x,y],-1)).bfloat16().float()
                r=torch.einsum('bhqr,bhtr->bhqt',code.float(),res.float()).bfloat16().float()
                s=((s+r).bfloat16().float()*128**-.5).bfloat16().float()
                s=torch.nn.functional.pad(s,(0,(-n)%32),value=-torch.inf).reshape(b,h,4,-1,32).logsumexp(-1)
            for w in [4,8]:
                call(w)
                assert torch.isfinite(outputs[w]).all()
                torch.testing.assert_close(outputs[w],outputs[0],rtol=.003,atol=.003)
                if n<=257:torch.testing.assert_close(outputs[w],s,rtol=.003,atol=.003)
                row=dict(rank=rank,warps=w,tokens=n,max_abs=float((outputs[w]-outputs[0]).abs().max()))
                if n>=8192:
                    row['baseline']=timing(lambda:call(0),5)
                    row['candidate']=timing(lambda:call(w),5)
                rows.append(row);print(row,flush=True)
        (root/'validation.json').write_text(json.dumps(dict(status='running',cases=rows),indent=2)+'\n')
    (root/'validation.json').write_text(json.dumps(dict(status='complete',cases=rows),indent=2)+'\n')
    print('VALIDATED',len(rows),flush=True)

if __name__=='__main__':main()
