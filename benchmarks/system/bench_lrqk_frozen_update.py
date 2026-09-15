"""Allocator-excluded replay of the native update at one frozen online state."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from benchmarks.system.bench_router import load_other
from benchmarks.system.common import metadata,save
from benchmarks.system.numa_memory import bind_host_allocations,slurm_gpu_numa


def measure_replay(graph,*,warmup,iterations):
    for _ in range(warmup):graph.replay()
    torch.cuda.synchronize()
    events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(iterations)]
    for begin,end in events:begin.record();graph.replay();end.record()
    torch.cuda.synchronize()
    raw=[begin.elapsed_time(end)*1000 for begin,end in events]
    timing=dict(raw_us=raw,mean_us=statistics.mean(raw),p50_us=statistics.median(raw),
        p95_us=sorted(raw)[max(0,int(.95*len(raw)+.999)-1)],stddev_us=statistics.pstdev(raw),
        warmup=warmup,iterations=iterations,timing='CUDA events around fixed-native-path graph replay; allocations outside timing')
    return timing


@torch.inference_mode()
def prepare_frozen_update(router,*,cusolver,include_scan=False):
    # Capture only a previously executed, fixed-state native path. This helper
    # is not valid for a different input or a changing online convergence path.
    original_bool=torch.Tensor.__bool__
    decisions=[]
    def record_bool(tensor):
        value=original_bool(tensor)
        decisions.append(value)
        return value
    torch.Tensor.__bool__=record_bool
    reference=tuple(router.native_update(**router.update_inputs))
    torch.Tensor.__bool__=original_bool
    assert decisions
    for actual,expected in zip(reference,router.update_reference):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    if cusolver:
        torch.backends.cuda.preferred_linalg_library('cusolver')
        solver_output=tuple(router.native_update(**router.update_inputs))
        for actual,expected in zip(solver_output,reference):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    cursor=0
    def operations():
        output=tuple(router.native_update(**router.update_inputs))
        if include_scan:
            router.query_code=output[2]
            router.scan()
        return output
    def frozen_bool(tensor):
        nonlocal cursor
        assert cursor<len(decisions)
        value=decisions[cursor];cursor+=1
        return value
    graph=torch.cuda.CUDAGraph()
    capture_stream=torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):operations()
    torch.cuda.synchronize()
    torch.Tensor.__bool__=frozen_bool
    # Replacing scalar decisions changes Dynamo guards. Compile this fixed
    # path before capture as well, rather than compiling inside the graph.
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            cursor=0
            operations()
            assert cursor==len(decisions)
    torch.cuda.synchronize()
    cursor=0
    with torch.cuda.graph(graph,stream=capture_stream):
        output=operations()
    torch.Tensor.__bool__=original_bool
    assert cursor==len(decisions)
    for _ in range(3):
        graph.replay();torch.cuda.synchronize()
        for actual,expected in zip(output,reference):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    return graph,output,decisions


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--cusolver',action='store_true',help='Capability probe only: prefer cuSOLVER after recording the native reference')
    parser.add_argument('--length',type=int,default=65536)
    parser.add_argument('--batch',type=int,default=1)
    args=parser.parse_args()
    root=Path('results/system_benchmarks/l40s')
    length=4096 if args.smoke else args.length
    bind_host_allocations(slurm_gpu_numa(json.loads((root/'hardware.json').read_text()),0)['numa_node'])
    torch.cuda.set_device(0);torch.set_num_threads(2)
    router=load_other('lrqk',root,length,args.batch,smoke=args.smoke)
    audit=router.validate()
    graph,output,decisions=prepare_frozen_update(router,cusolver=args.cusolver)
    warmup,iterations=(5,10) if args.smoke else (100,500)
    timing=measure_replay(graph,warmup=warmup,iterations=iterations)
    case='' if args.smoke or (length==65536 and args.batch==1) else f'_t{length}_b{args.batch}'
    save(root/f'lrqk_frozen_update{"_cusolver" if args.cusolver else ""}{case}{"_smoke" if args.smoke else ""}.json',dict(metadata=metadata(),
        length=length,batch=args.batch,audit=audit,query_preprocessing=timing,branch_decisions=decisions,
        native_output_bitwise_equal=True,
        linalg_preference=str(torch.backends.cuda.preferred_linalg_library()),
        scope='Frozen final state after64 official decode updates. Native tensor algebra and precision are retained; cuSOLVER preference, when requested, changes solver dispatch. Recorded scalar branch decisions apply only to this fixed input. Not a general online update implementation or a replacement for native eager measurements.'))
    print(dict(length=length,decisions=decisions,p50_us=timing['p50_us'],native_output_bitwise_equal=True),flush=True)


if __name__=='__main__':main()
