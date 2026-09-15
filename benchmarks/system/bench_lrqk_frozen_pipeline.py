"""Supplementary fixed-state cuSOLVER timings, with native support unchanged."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from benchmarks.system.bench_router import load_other
from benchmarks.system.bench_lrqk_frozen_update import prepare_frozen_update,measure_replay
from benchmarks.system.bench_common_offload import CommonAttention,graph
from benchmarks.system.common import metadata,save
from benchmarks.system.numa_memory import bind_host_allocations,audit_host_tensor,slurm_gpu_numa
from benchmarks.system.timing import measure,measure_host_sequence


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--length',type=int,default=65536)
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--common',action='store_true')
    args=parser.parse_args();root=Path('results/system_benchmarks/l40s')
    length=4096 if args.smoke else args.length
    node=slurm_gpu_numa(json.loads((root/'hardware.json').read_text()))['numa_node']
    library=bind_host_allocations(node)
    torch.cuda.set_device(0);torch.set_num_threads(2)
    router=load_other('lrqk',root,length,args.batch,smoke=args.smoke)
    native_audit=router.validate();native_ids=router.ids.clone()
    update,output,decisions=prepare_frozen_update(router,cusolver=True)
    def preprocess():
        update.replay()
        router.query_code=output[2]
    router.preprocess=preprocess
    router.full();torch.cuda.synchronize()
    assert torch.equal(router.ids,native_ids)
    settings=dict(warmup=5 if args.smoke else 100,iterations=10 if args.smoke else 500)
    pre=measure_replay(update,**settings)
    route=measure(router.scan,**settings)
    routing_graph,joint_output,joint_decisions=prepare_frozen_update(router,cusolver=True,include_scan=True)
    assert decisions==joint_decisions
    joint=measure_replay(routing_graph,**settings)
    assert torch.equal(router.ids,native_ids)
    common=None
    if args.common:
        assert args.batch==1
        folder=root/('capture_smoke' if args.smoke else 'capture')/f't{length}'
        with safe_open(str(folder/'sample0.safetensors'),framework='pt',device='cpu') as source:
            key=source.get_tensor('k');host=torch.empty(key.shape,dtype=key.dtype,pin_memory=True);host.copy_(key)
            value=source.get_tensor('value_transformed')
            prefix=value[...,:16].contiguous().cuda();tail=value[...,16:].contiguous().cuda()
        backend=CommonAttention(router,host,prefix,tail)
        audit=backend.validate(host)
        numa=[audit_host_tensor(t,node,library) for t in [host,backend.fetch.host_count,backend.fetch.host_indices,backend.fetch.staging]]
        # CommonAttention initialization invokes preprocess and may replace
        # router.ids; restore the graph's persistent selection output first.
        router.query_code=joint_output[2]
        routing_graph,joint_output,joint_decisions=prepare_frozen_update(router,cusolver=True,include_scan=True)
        attention_graph=graph(backend.attention)
        def total():
            routing_graph.replay()
            backend.copy_support();backend.fetch(backend.ids);attention_graph.replay()
        combined=measure_host_sequence(total,**settings)
        assert torch.equal(router.ids,native_ids)
        common=dict(total_with_query_preprocessing=combined,audit=audit,numa=numa,traffic=backend.fetch.traffic())
    save(root/f'lrqk_frozen_pipeline_t{length}_b{args.batch}{"_smoke" if args.smoke else ""}.json',dict(
        metadata=metadata(),length=length,batch=args.batch,native_audit=native_audit,
        native_output_bitwise_equal=True,native_ids_equal=True,branch_decisions=decisions,
        linalg_preference=str(torch.backends.cuda.preferred_linalg_library()),
        query_preprocessing=pre,route_to_ids=route,total_with_preprocessing=joint,common_backend=common,
        scope='Supplementary fixed-state graph after64 native online updates. cuSOLVER solver dispatch; native tensor algebra, precision, outputs and selection retained. CPU convergence decisions recorded before capture; not a general online implementation.'))
    print(dict(length=length,batch=args.batch,query_us=pre['p50_us'],route_us=route['p50_us'],joint_us=joint['p50_us'],
        common_us=None if common is None else common['total_with_query_preprocessing']['p50_us']),flush=True)


if __name__=='__main__':main()
