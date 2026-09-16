"""Candidate support, forced pages, ties, boundaries, and operator latency."""
from pathlib import Path
import json
import torch
from benchmarks.system.two_stage_router import candidates as reference,_group
from benchmarks.system.fused_candidates import candidates
from benchmarks.system.profile_router_phases import timing
import triton as tr


@torch.inference_mode()
def main():
    torch.manual_seed(2026);root=Path('results/system_benchmarks/fused_candidates');root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in [8192-64,16384-64,16385-64,32768-64,65536-64,65537-64,65568-64,131072-64]:
        p=(n+64+31)//32
        for kind in ['random','ties','large']:
            u=torch.randn(2,8,4,p,device='cuda')
            if kind=='ties':u.zero_()
            if kind=='large':u=u*100+10000
            u[..., (n+31)//32:]=-torch.inf
            old=reference(u,n);new=candidates(u,n)
            assert new.shape==old.shape and bool((new[...,1:]>new[...,:-1]).all())
            assert bool((new[...,0]==0).all())
            for page in range(n//32,p):assert bool((new==page).any(-1).all())
            if p>512:
                g=torch.empty(2,8,p,device='cuda');_group[(16,)](u,g,p,n//32,tr.next_power_of_2(p),num_warps=8)
                a=g.gather(-1,old).sort(-1).values;b=g.gather(-1,new).sort(-1).values
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            else:torch.testing.assert_close(old,new,rtol=0,atol=0)
            row=dict(historical=n,pages=p,kind=kind,identical_ids=bool(torch.equal(old,new)))
            if kind=='random' and p>=1024:row['timing']={name:timing(fn,7) for name,fn in [('reference',lambda:reference(u,n)),('fused',lambda:candidates(u,n))]}
            rows.append(row);print('PASS',row,flush=True)
    (root/'validation.json').write_text(json.dumps(dict(status='complete',cases=rows),indent=2)+'\n')

if __name__=='__main__':main()
