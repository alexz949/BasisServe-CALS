"""Deterministic stream-order regression for the LRQK prefill cache offload."""
import json,sys,argparse
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from basisserve.core import lrqk_upstream
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
m=lrqk_upstream.upstream();original=m.LightAttentionIndicesFactory.prefill
refs=[]
def delayed_prefill(self,q,k,v):
    self.Kgpu=torch.zeros((16,4,2112,128),device='cuda',dtype=torch.bfloat16)
    self.Vgpu=torch.zeros_like(self.Kgpu)
    self._A_K=SimpleNamespace(data=torch.zeros((16,4,8192,32),device='cuda',dtype=torch.bfloat16))
    refs.extend((self.Kgpu,self.Vgpu,self._A_K.data))
    torch.cuda.synchronize()
    torch.cuda._sleep(200_000_000)
    self.Kgpu.fill_(2);self.Vgpu.fill_(3);self._A_K.data.fill_(4)

def probe(cls):
    obj=cls();m.LightAttentionIndicesFactory.prefill=delayed_prefill
    obj.prefill(torch.empty(1,device='cuda'),None,None)
    torch.cuda.synchronize()
    values=[obj.Kgpu_temp,obj.Vgpu_temp,obj._A_K_data_temp]
    return {name:dict(wrong=int((value!=expected).sum()),total=value.numel(),min=float(value.min()),max=float(value.max())) for name,value,expected in zip(('K','V','A_K'),values,(2,3,4))}
try:
    old=probe(m.LightAttentionIndicesOffloadPrefill)
    new=probe(lrqk_upstream.ordered_offload_prefill()) if hasattr(lrqk_upstream,'ordered_offload_prefill') else None
finally:m.LightAttentionIndicesFactory.prefill=original
result=dict(upstream=old,ordered=new,delay_cycles=200_000_000,
            source_refs_retained=True,
            purpose="Isolate missing producer-to-copy stream ordering; source storage is kept alive in both cases.")
args.output.mkdir(parents=True,exist_ok=True);(args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
