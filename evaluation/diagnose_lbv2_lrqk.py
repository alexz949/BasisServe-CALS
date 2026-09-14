import runpy
import torch
import basisserve.core.c1_lrqk as lrqk

def inspect_solve(a,b,*,left=True):
    result,info=torch.linalg.solve_ex(a,b,left=left)
    ok=(info==0).all() and torch.isfinite(result).all()
    if not ok:
        torch.save(dict(a=a.cpu(),b=b.cpu(),result=result.cpu(),info=info.cpu(),left=left),'results/k_routing_fit/qwen35_9b/lbv2_lrqk_failure.pt')
        print('FAILED_SOLVE',dict(shape=list(a.shape),a_finite=bool(torch.isfinite(a).all()),b_finite=bool(torch.isfinite(b).all()),info=info.cpu().tolist(),amax=float(a.abs().max()),bmax=float(b.abs().max())),flush=True)
    assert ok
    return result
lrqk._solve=inspect_solve
runpy.run_module('evaluation.eval_qwen35_longbench_v2',run_name='__main__')
