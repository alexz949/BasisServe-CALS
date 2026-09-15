"""Cross-mode prefill and local/offload generation gates for the full model."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from benchmarks.system.common import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'));a=p.parse_args()
    records=[]
    for batch in [1,4]:
        reports={};logits={};tokens={}
        for mode in ['dense','c1','sparse_local','offload']:
            folder=a.output/'e2e_smoke'/f'{mode}_t4096_b{batch}'
            reports[mode]=[json.loads((folder/f'rank{r}.json').read_text()) for r in range(4)]
            assert all(r['status']=='complete' and r['generated_tokens']==8 for r in reports[mode])
            logits[mode]=[load_file(str(folder/f'prefill{r}.safetensors'))['logits'] for r in range(4)]
            tokens[mode]=[load_file(str(folder/f'tokens{r}.safetensors'))['tokens'] for r in range(4)]
            assert all(torch.equal(tokens[mode][0],t) for t in tokens[mode])
        for mode in ['sparse_local','offload']:
            for observed,reference in zip(logits[mode],logits['c1']):
                torch.testing.assert_close(observed,reference,rtol=0,atol=0)
        assert torch.equal(tokens['sparse_local'][0],tokens['offload'][0])
        assert all(r['native_dense_smoke_prefill_rel_mse']<.002 for r in reports['dense'])
        records.append(dict(batch=batch,length=4096,generated_tokens=8,c1_prefill_bitwise_equal=True,
            sparse_local_offload_tokens_equal=True,replica_tokens_equal=True,dense_native_reference=True))
    save(a.output/'e2e_smoke_validation.json',dict(status='complete',records=records,
        scope='Cross-mode cache/backend consistency, not a model quality benchmark'))
    print(records,flush=True)


if __name__=='__main__':main()
