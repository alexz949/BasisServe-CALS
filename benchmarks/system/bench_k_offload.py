"""Real selected Page32 K fetch, staged breakdown and separate fused total."""
import argparse
import json
import os
from pathlib import Path
import torch
from safetensors import safe_open
from benchmarks.system.bench_router import load_basis
from benchmarks.system.common import metadata,save
from benchmarks.system.numa_memory import bind_host_allocations,audit_host_tensor
from benchmarks.system.k_fetch import SplitValueKFetch
from benchmarks.system.timing import measure,measure_host_sequence
from basisserve.kernels.mapped_host_paged_attention import mapped_host_bf16_empty


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'));p.add_argument('--smoke',action='store_true')
    a=p.parse_args();rank=int(os.environ.get('LOCAL_RANK',0))
    hardware=json.loads((a.output/'hardware.json').read_text());node=hardware['gpu_numa'][rank]['numa_node']
    library=bind_host_allocations(node);torch.cuda.set_device(rank);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    length=4096 if a.smoke else 65536;budget=2048 if a.smoke else [512,1024,2048,4096][rank]
    router=load_basis(a.root,a.output,length,1,a.smoke,budget);router.validate()
    path=a.output/('capture_smoke' if a.smoke else 'capture')/f't{length}'/'sample0.safetensors'
    host=mapped_host_bf16_empty(batch=1,kv_heads=8,capacity=length)
    with safe_open(str(path),framework='pt',device='cpu') as f:host.copy_(f.get_tensor('k'))
    operator=SplitValueKFetch(router,host)
    numa={name:audit_host_tensor(tensor,node,library) for name,tensor in
          [('source',host),('staging',operator.fetch.staging),('indices',operator.fetch.host_indices)]}
    validation=operator.validate();torch.cuda.empty_cache()
    counts=dict(warmup=5 if a.smoke else 100,iterations=10 if a.smoke else 500)
    record=dict(layer=3,length=length,batch=1,page_size=32,physical_budget=budget,v_rank=96,dtype='bfloat16',
        staged_qk_pv_accumulation='float32, TF32 disabled',logical_unique_k_bytes=8*budget*128*2,
        staged_dma_payload_bytes=operator.fetch.requested_bytes,measured_pcie_bus_bytes=None,
        bus_traffic_note='DMA payload is known; mapped-host hardware bus counters have not yet been profiled',
        host_audit=numa,validation=validation,
        fetch=measure_host_sequence(operator.staged_fetch,**counts),
        exact_qk=measure(operator.exact_qk,**counts),softmax_pv_with_selected_v_gather=measure(operator.softmax_pv,**counts),
        staged_total=measure_host_sequence(operator.staged_attention,**counts),
        mapped_fused_total=measure(operator.mapped_attention,**counts),
        interpretation='Staged breakdown is not attributed to the fused mapped kernel; neither total includes routing',
        actual_tokens=router.budget,actual_pages=router.budget//32)
    name='k_offload_smoke' if a.smoke else f'k_offload_budget{budget}'
    save(a.output/f'{name}.json',dict(metadata=metadata(),record=record))
    print(dict(budget=budget,fetch_p50_us=record['fetch']['p50_us'],staged_p50_us=record['staged_total']['p50_us'],
               mapped_fused_p50_us=record['mapped_fused_total']['p50_us'],validation=validation),flush=True)

if __name__=='__main__':main()
