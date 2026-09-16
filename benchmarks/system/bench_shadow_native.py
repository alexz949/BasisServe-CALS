"""Measure the upstream single-GPU ShadowKV CPU-cache serving path."""
import argparse
import hashlib
from importlib.metadata import version
import importlib.util
import importlib.machinery
import json
from pathlib import Path
import statistics
import sys
import time
import torch
from safetensors.torch import load_file,save_file
from benchmarks.system.common import metadata,save,command
from benchmarks.system.numa_memory import bind_host_allocations,audit_host_tensor,slurm_gpu_numa


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['full','shadowkv_cpu','basis16','basis8'],required=True)
    p.add_argument('--length',type=int,default=65536)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--window',type=int,default=64)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/shadow_native'))
    a=p.parse_args()
    root=Path('/home/zhangal/BasisServe-CALS-runs/shadow_native')
    output=a.output
    upstream=root/'ShadowKV';sys.path.insert(0,str(upstream));sys.path.insert(0,str(root/'deps'))
    hardware=json.loads(Path('results/system_benchmarks/l40s/hardware.json').read_text())
    node=slurm_gpu_numa(hardware)['numa_node'];library=bind_host_allocations(node)
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(0)
    # Load only the upstream Llama package, not unrelated GLM/Phi/Qwen models.
    spec=importlib.machinery.ModuleSpec('models',loader=None,is_package=True)
    package=importlib.util.module_from_spec(spec);package.__path__=[str(upstream/'models')]
    sys.modules['models']=package
    vllm_spec=importlib.util.find_spec('vllm')
    vllm_binary=Path(vllm_spec.origin).parent/'_C.abi3.so'
    torch.ops.load_library(str(vllm_binary))
    assert hasattr(torch.ops._C,'rotary_embedding') and hasattr(torch.ops._C,'silu_and_mul')
    from models.llama import Llama
    if a.method.startswith('basis'):
        from benchmarks.system.native_basis_cache import basis_llama_class
        Llama=basis_llama_class(Llama,int(a.method[5:]))
    length=8192 if a.smoke else a.length
    steps=8 if a.smoke else 100
    tag=f'{a.method}_t{length}_b{a.batch}{"_smoke" if a.smoke else ""}'
    folder=output/tag;folder.mkdir(parents=True,exist_ok=True)
    def phase(name):
        (folder/'phase.json').write_text(json.dumps(dict(phase=name,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved()))+'\n')
        print(dict(phase=name,method=a.method,length=length,batch=a.batch),flush=True)
    run_root=Path('/home/zhangal/BasisServe-CALS-runs/llama31_8b_64k')
    identity=json.loads((run_root/'manifests/v96.json').read_text())
    meta=metadata();phase('load')
    llm=Llama(model_name=identity['model'],device='cuda:0',batch_size=a.batch,max_length=length,
        attn_mode=a.method,sparse_budget=2048,rank=160,chunk_size=8,minference=False)
    if a.method.startswith('basis'):llm.kv_cache.validate=a.smoke
    windows=load_file(str(run_root/'calibration/windows.safetensors'))['input_ids']
    pairs=[[64+(a.window-64+i)%16,64+(a.window-64+i+8)%16] for i in range(a.batch)]
    tokens=torch.stack([torch.cat((windows[x],windows[y]))[:length] for x,y in pairs]).long().cuda()
    native_prefill=llm.batch_prefill;native_h2d=llm.kv_cache.H2D;native_inference=llm.inference
    details={};prefill_logits=[];events=[]
    allocated_events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(steps+1)]
    def prefill(ids):
        phase('prefill');torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        logits=native_prefill(ids);torch.cuda.synchronize()
        details['prefill_seconds']=time.perf_counter()-start
        details['prefill_peak_bytes']=torch.cuda.max_memory_allocated()
        assert bool(torch.isfinite(logits).all())
        if a.smoke:prefill_logits.append(logits.cpu())
        return logits
    def h2d():
        phase('cache_placement');torch.cuda.synchronize();start=time.perf_counter();native_h2d();torch.cuda.synchronize()
        details['cache_placement_seconds']=time.perf_counter()-start
        torch.cuda.reset_peak_memory_stats();phase('decode')
    def inference(*args,**kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None)
        if ids.shape[1]!=1:return native_inference(*args,**kwargs)
        begin,end=allocated_events[len(events)];begin.record()
        logits=native_inference(*args,**kwargs);end.record();events.append((begin,end))
        if a.smoke:assert bool(torch.isfinite(logits).all())
        return logits
    llm.batch_prefill=prefill;llm.kv_cache.H2D=h2d;llm.inference=inference
    # Upstream batch_generate includes routing, slot reuse, K reconstruction,
    # overlapped CPU V gather, FlashAttention, projections, MLP and sampling.
    _,throughput=llm.batch_generate(tokens,gen_len=steps,benchmark=True,temperature=.6)
    torch.cuda.synchronize();assert len(events)==steps+1
    raw=[begin.elapsed_time(end) for begin,end in events[:steps]]
    details['decode_peak_bytes']=torch.cuda.max_memory_allocated()
    audits=[]
    if a.method=='shadowkv_cpu':audits=[audit_host_tensor(llm.kv_cache.v_cache_cpu,node,library)]
    if prefill_logits:save_file({'logits':prefill_logits[0]},str(folder/'prefill_logits.safetensors'))
    if a.method.startswith('basis'):audits=[audit_host_tensor(k,node,library) for k in llm.kv_cache.keys]
    source_paths=[Path('benchmarks/system/native_basis_cache.py')]+list((upstream/'models').glob('*.py'))+list((upstream/'kernels').glob('*.cu'))+list((upstream/'kernels').glob('*.h'))
    save(folder/'result.json',dict(status='complete',metadata=meta,method=a.method,length=length,batch=a.batch,
        model=identity['model'],tp=1,value='Dense V128; original Wo; '+('CPU V offload' if a.method=='shadowkv_cpu' else 'GPU V'),
        rank=int(a.method[5:]) if a.method.startswith('basis') else 160,chunk_size=8,budget=2048,decode_steps=steps,
        basis_kernel_verified_layers=sorted(llm.kv_cache.validated) if a.method.startswith('basis') else None,
        basis_truncation='First8 of original B16R16 codes; residual target unchanged; no refit' if a.method=='basis8' else None,
        native_benchmark_tokens_per_second=throughput,native_benchmark_ms_per_step=a.batch*1000/throughput,
        raw_inference_ms=raw,inference_p50_ms=statistics.median(raw),inference_mean_ms=statistics.mean(raw),
        host_buffer_audit=audits,details=details,
        dependency=dict(commit=command(['git','-C',str(upstream),'rev-parse','HEAD']),dirty=command(['git','-C',str(upstream),'diff']),
            versions={p:version(p) for p in ['torch','transformers','tokenizers','huggingface-hub','vllm','flashinfer-python','minference']},
            cutlass_commit=command(['git','-C',str(upstream/'3rdparty/cutlass'),'rev-parse','HEAD']),
            cutlass_dirty=command(['git','-C',str(upstream/'3rdparty/cutlass'),'status','--porcelain']),
            extension_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (upstream/'kernels').glob('*.so')},
            vllm_binary_sha256=hashlib.sha256(vllm_binary.read_bytes()).hexdigest(),
            import_adaptation='Load Llama without unrelated model imports; call installed native vLLM torch.ops._C RoPE/SiLU directly; defer inactive MInference imports. ShadowKV cache and CUDA source unmodified.',
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}),
        scope='Official single-GPU batch_generate, native100 decode iterations plus native final unmeasured inference. CUDA inference events exclude sampling; native wall throughput includes sampling and host dependencies. No TP4 speedup claim.'))
    phase('complete');print(dict(method=a.method,length=length,batch=a.batch,native_tokens_s=throughput),flush=True)


if __name__=='__main__':main()
