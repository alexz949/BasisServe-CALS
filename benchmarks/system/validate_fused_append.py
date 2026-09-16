"""Check real fitted factors, ring wrap/page boundaries, batches, and append latency."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors.torch import load_file
from benchmarks.system.native_basis_cache import BasisCache
from benchmarks.system.two_stage_router import Metadata
from benchmarks.system.fused_append import append
from benchmarks.system.profile_router_phases import timing


@torch.inference_mode()
def main():
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    root=Path('results/system_benchmarks/fused_append');root.mkdir(parents=True,exist_ok=True)
    bank=Path('/home/zhangal/BasisServe-CALS-runs/llama31_8b_64k_densev/ours_b16r16')
    results=[]
    for layer in range(32):
        b=1 if layer%2==0 else 2;cap=256;start=95
        f={k:v.cuda().bfloat16().contiguous() for k,v in load_file(str(bank/f'layer_{layer:03d}.safetensors')).items()}
        f['encoder']=f['residual_encoder_b16_r16'].contiguous()
        rope=torch.randn(cap,128,device='cuda').bfloat16()
        def state():
            return SimpleNamespace(width=16,length=start,factors=[f],llm=SimpleNamespace(cos_sin_cache=rope),
                cos=rope[:,:64].contiguous(),sin=rope[:,64:].contiguous(),
                values=[torch.zeros(b,8,cap,128,device='cuda',dtype=torch.bfloat16)],
                bases=[torch.zeros(b,8,cap,16,device='cuda',dtype=torch.bfloat16)],
                residuals=[torch.zeros(b,8,cap,16,device='cuda',dtype=torch.bfloat16)])
        ref=state();new=state();initial=torch.randn(b,8,start,128,device='cuda').bfloat16()
        m0=Metadata(initial,cap);m1=Metadata(initial,cap)
        # Non-contiguous QKV views, matching the native model's head layout.
        for step in range(66):
            x=torch.randn(b,1,8,384,device='cuda').bfloat16().transpose(1,2)
            k=x[...,:128];v=x[...,128:256]
            BasisCache.append_codes(ref,k,v,0);m0.advance(k);append(new,k,v,0,m1)
            ref.length+=1;new.length+=1
        p=ref.length
        torch.testing.assert_close(m0.ring,m1.ring,rtol=0,atol=0)
        pages=(m0.n+31)//32
        for name in ['minimum','maximum']:
            torch.testing.assert_close(getattr(m0,name)[:,:,:pages],getattr(m1,name)[:,:,:pages],rtol=0,atol=0)
        metrics={}
        for name in ['values','bases','residuals']:
            a=getattr(ref,name)[0][:,:,start:p].float();c=getattr(new,name)[0][:,:,start:p].float()
            err=(a-c).square().sum()/a.square().sum().clamp_min(1e-30)
            metrics[name]=dict(relative_mse=float(err),max_abs=float((a-c).abs().max()),equal_fraction=float((a==c).float().mean()))
            assert float(err)<1e-4,(layer,name,metrics[name])
        results.append(dict(layer=layer,batch=b,metrics=metrics));print('PASS',layer,metrics,flush=True)
    # CUDA graph timing fixes a single append position for both paths; state is not a cache-hit benchmark.
    def old():
        m0.n=ref.length-64;m0.slot=(ref.length-start)%64
        BasisCache.append_codes(ref,k,v,0);m0.advance(k)
    def fused():
        m1.n=new.length-64;m1.slot=(new.length-start)%64
        append(new,k,v,0,m1)
    latency={name:timing(fn,7) for name,fn in [('reference',old),('fused',fused)]}
    result=dict(status='complete',cases=results,latency=latency,environment='basis')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('LATENCY',latency,flush=True)

if __name__=='__main__':main()
