"""Run native ShadowKV and native Dense in fresh single-GPU processes."""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import torch
from safetensors.torch import load_file


def main():
    root=Path('results/system_benchmarks/shadow_native')
    source=Path('benchmarks/system/bench_shadow_native.py')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    def trial(method,length,batch,smoke=False):
        tag=f'{method}_t{length}_b{batch}{"_smoke" if smoke else ""}'
        folder=root/tag;folder.mkdir(parents=True,exist_ok=True)
        report=folder/'outcome.json'
        if report.exists():
            previous=json.loads(report.read_text());assert previous['benchmark_sha256']==digest
            return previous
        cmd=[sys.executable,'-m','benchmarks.system.bench_shadow_native','--method',method,'--length',str(length),'--batch',str(batch)]
        if smoke:cmd.append('--smoke')
        print('COMMAND',' '.join(cmd),flush=True)
        with (folder/'run.log').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        record=dict(method=method,length=length,batch=batch,command=cmd,returncode=result.returncode,benchmark_sha256=digest)
        if result.returncode==0:
            data=json.loads((folder/'result.json').read_text());assert data['status']=='complete'
            record.update(status='complete',native_tokens_s=data['native_benchmark_tokens_per_second'],native_ms_step=data['native_benchmark_ms_per_step'])
        else:
            text=(folder/'run.log').read_text()
            record['status']='gpu_oom' if 'CUDA out of memory' in text or 'torch.OutOfMemoryError' in text else 'error'
            record['log']=str(folder/'run.log')
        report.write_text(json.dumps(record,indent=2)+'\n')
        print(record,flush=True)
        assert record['status']!='error'
        return record
    for method in ['full','shadowkv_cpu']:assert trial(method,8192,1,True)['status']=='complete'
    dense=load_file(str(root/'full_t8192_b1_smoke/prefill_logits.safetensors'))['logits']
    shadow=load_file(str(root/'shadowkv_cpu_t8192_b1_smoke/prefill_logits.safetensors'))['logits']
    torch.testing.assert_close(shadow,dense,rtol=.02,atol=.02)
    (root/'smoke_validation.json').write_text(json.dumps(dict(status='complete',native_dense_shadow_prefill_reference=True,
        rel_mse=float((shadow-dense).square().sum()/dense.square().sum()),benchmark_sha256=digest),indent=2)+'\n')
    records=[]
    for length,batch,method in itertools.product([65536,131072],[1,4],['full','shadowkv_cpu']):
        records.append(trial(method,length,batch))
        (root/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
    (root/'summary.json').write_text(json.dumps(dict(status='complete',records=records,
        scope='Native single-GPU implementation, Dense V and original Wo, rank160/chunk8/budget2048,100 decode steps. Distinct from TP4 BasisKV results.'),indent=2)+'\n')


if __name__=='__main__':main()
