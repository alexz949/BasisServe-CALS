"""Reproduce a failed FP16 prompt without altering LRQK arithmetic."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from basisserve.core import c1_lrqk as core
from evaluation import eval_longbench_lrqk_fp16 as evaluator
from evaluation.fit_qwen3_8b_residual_kl_bank import write_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--sample',required=True,type=int)
    p.add_argument('--arm',required=True,choices=('k1152','k1280'))
    args=p.parse_args()
    destination=ROOT/'results/evaluation/lrqk_fp16_diagnosis'/f'{args.arm}_s{args.sample}'
    inputs=evaluator.inputs
    def selected_inputs(a):
        values=list(inputs(a))
        values[0]=[r for r in values[0] if r['index']==args.sample]
        assert len(values[0])==1
        return tuple(values)
    evaluator.inputs=selected_inputs
    context={}
    initialize=core.LRQKState.__init__
    decode=core.LRQKState.decode
    factors=core.decode_factors
    def init(self,q,k,config,layer=0):
        self.diagnostic_layer=layer
        return initialize(self,q,k,config,layer)
    def step(self,q,k,v,scale):
        context.update(layer=self.diagnostic_layer,step=self.steps+1,length=self.length+1)
        return decode(self,q,k,v,scale)
    def observe(*a,**kw):
        result=factors(*a,**kw)
        finite32=all(bool(torch.isfinite(t).all()) for t in result)
        finite16=all(bool(torch.isfinite(t.to(torch.float16)).all()) for t in result)
        if not finite32 or not finite16:
            metrics={}
            for name,t in zip(('bq','bk','qcode','kcode'),result,strict=True):
                metrics[name]=dict(dtype=str(t.dtype),shape=list(t.shape),
                    nonfinite_fp32=int((~torch.isfinite(t)).sum()),
                    nonfinite_after_fp16=int((~torch.isfinite(t.to(torch.float16))).sum()),
                    abs_max_fp32=float(t.abs().max()),
                    above_fp16_max=int((t.abs()>torch.finfo(torch.float16).max).sum()))
            record=dict(sample=args.sample,arm=args.arm,context=dict(context),
                all_finite_fp32=finite32,all_finite_after_fp16=finite16,metrics=metrics,
                diagnostic_sha256=sha256(Path(__file__)),python=sys.executable)
            write_json(destination/'failure.json',record)
            print(json.dumps(record),flush=True)
        return result
    core.LRQKState.__init__=init
    core.LRQKState.decode=step
    core.decode_factors=observe
    sys.argv=[str(evaluator.__file__),'--model',args.model,'--stage','evaluate',
              '--arm',args.arm,'--output-dir',str(destination)]
    evaluator.main()


if __name__=='__main__':
    main()
