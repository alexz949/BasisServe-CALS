"""Four real layer-3 attention paths; routing included in sparse totals."""
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
from benchmarks.system.bench_common_offload import graph
from basisserve.kernels.mapped_host_paged_attention import mapped_host_bf16_empty,mapped_host_paged_attention,gpu_paged_attention
from basisserve.kernels.compressed_v_decode_attention import c1_dense_gqa_v96_decode_attention_cuda


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'));p.add_argument('--smoke',action='store_true')
    p.add_argument('--staged-only',action='store_true')
    a=p.parse_args();rank=int(os.environ.get('LOCAL_RANK',0))
    hardware=json.loads((a.output/'hardware.json').read_text());node=hardware['gpu_numa'][rank]['numa_node']
    library=bind_host_allocations(node);torch.cuda.set_device(rank);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    length=4096 if a.smoke else [16384,32768,65536,131072][rank]
    counts=dict(warmup=5 if a.smoke else 100,iterations=10 if a.smoke else 500)
    router=load_basis(a.root,a.output,length,1,a.smoke);route_audit=router.validate()
    path=a.output/('capture_smoke' if a.smoke else 'capture')/f't{length}'/'sample0.safetensors'
    host=mapped_host_bf16_empty(batch=1,kv_heads=8,capacity=length)
    with safe_open(str(path),framework='pt',device='cpu') as f:host.copy_(f.get_tensor('k'))
    numa=audit_host_tensor(host,node,library)
    op=SplitValueKFetch(router,host);sparse_audit=op.validate()
    routing_graph=graph(router.full)
    def selected_attention():
        op.exact_qk();return op.softmax_pv()
    attention_graph=graph(selected_attention)
    def staged_total():
        routing_graph.replay();op.staged_fetch();attention_graph.replay()
        return op.output
    reference=op.staged_attention().clone()
    torch.testing.assert_close(staged_total(),reference,rtol=0,atol=0)
    if a.staged_only:
        staged=measure_host_sequence(staged_total,**counts)
        name='sparse_staged_total_smoke' if a.smoke else f'sparse_staged_total_t{length}'
        save(a.output/f'{name}.json',dict(metadata=metadata(),length=length,batch=1,
            staged_sparse_total=staged,route_audit=route_audit,sparse_audit=sparse_audit,
            host_audit=numa,reference_output_bitwise_equal=True,
            scope='Staged total with routing and attention captured separately, preallocated host fetch between replays. Supersedes the earlier eager-routing staged total only; fused and component measurements are unchanged.'))
        print(dict(length=length,staged_total_us=staged['p50_us']),flush=True)
        return
    dense_key=host.cuda();dense_value=torch.cat((router.base,router.tail),-1)
    valid=torch.tensor(length,device='cuda',dtype=torch.int64)
    dense_workspace=torch.empty(32,32,98,device='cuda',dtype=torch.float32)
    dense_output=torch.empty(1,32,1,96,device='cuda',dtype=torch.bfloat16)
    def dense_local():
        return c1_dense_gqa_v96_decode_attention_cuda(router.q,dense_key,dense_value,valid,
            splits=32,workspace=dense_workspace,output=dense_output)
    expected=((router.q.float().reshape(1,8,4,128)@dense_key.float().transpose(-1,-2))*128**-.5).softmax(-1)@dense_value.float()
    expected=expected.reshape_as(dense_output)
    torch.testing.assert_close(dense_local().float(),expected,rtol=.02,atol=.004)
    local=measure(dense_local,**counts)
    def sparse_local():
        router.full();op.update_pages()
        return gpu_paged_attention(dense_key,router.q,router.tail,op.page_ids,sequence_length=length,
            value_prefix=router.base,workspace=op.workspace,output=op.output,splits=32)
    selected_reference=op.staged_attention().clone()
    torch.testing.assert_close(sparse_local(),selected_reference,rtol=.02,atol=.004)
    sparse_local_time=measure(sparse_local,**counts)
    # Full resident K/V exists only for the local arms above.
    del dense_key,dense_value,dense_workspace;torch.cuda.empty_cache()
    all_pages=torch.arange(length//32,device='cuda').expand(1,8,-1).contiguous()
    def dense_offload():
        return mapped_host_paged_attention(host,router.q,router.tail,all_pages,sequence_length=length,
            value_prefix=router.base,workspace=op.workspace,output=op.output,splits=32)
    torch.testing.assert_close(dense_offload().float(),expected,rtol=.02,atol=.004)
    offload=measure(dense_offload,**counts)
    def sparse_offload():
        router.full();return op.mapped_attention()
    sparse_time=measure(sparse_offload,**counts)
    staged=measure_host_sequence(staged_total,**counts)
    breakdown=dict(query_preprocessing=measure(router.preprocess,**counts),route_to_ids=measure(router.scan,**counts),
        staged_fetch=measure_host_sequence(op.staged_fetch,**counts),staged_exact_qk=measure(op.exact_qk,**counts),
        staged_softmax_pv=measure(op.softmax_pv,**counts))
    record=dict(layer=3,length=length,batch=1,query_heads=32,kv_heads=8,qk_dim=128,v_rank=96,
        dtype='bfloat16',budget=2048,page_size=32,base_rank=16,residual_rank=16,sink=32,recent=64,
        dense_local=local,dense_k_offload=offload,sparse_local=sparse_local_time,sparse_k_offload=sparse_time,
        staged_sparse_total=staged,staged_breakdown=breakdown,
        primary_speedup=offload['p50_us']/sparse_time['p50_us'],
        sparse_local_over_dense_local=sparse_local_time['p50_us']/local['p50_us'],
        dense_local_over_sparse_local=local['p50_us']/sparse_local_time['p50_us'],
        dense_k_logical_bytes=host.numel()*2,sparse_k_logical_bytes=op.fetch.requested_bytes,
        measured_pcie_bus_bytes=None,host_audit=numa,route_audit=route_audit,sparse_audit=sparse_audit,
        note='Sparse totals include query preprocessing and routing. Staged components are not assigned to fused kernels. Dense-local uses existing native V96 CUDA GQA kernel.')
    name='sparse_operator_smoke' if a.smoke else f'sparse_operator_t{length}'
    save(a.output/f'{name}.json',dict(metadata=metadata(),record=record))
    print(dict(length=length,dense_local_us=local['p50_us'],dense_offload_us=offload['p50_us'],
        sparse_local_us=sparse_local_time['p50_us'],sparse_offload_us=sparse_time['p50_us'],speedup=record['primary_speedup']),flush=True)

if __name__=='__main__':main()
