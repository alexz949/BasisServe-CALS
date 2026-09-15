"""One isolated full-model Llama-base TP4 run, real KL V ranks and split Base16."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import resource
import statistics
import time
import torch
import torch.distributed as dist
from safetensors.torch import load_file,save_file
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig
from basisserve.core.llama_tp4_c1_system import install
from benchmarks.system.common import metadata,save
from benchmarks.system.collective_audit import CollectiveAudit
from benchmarks.system.numa_memory import bind_host_allocations,audit_host_tensor,slurm_gpu_numa


def choose(logits,vocab_size):
    values,ids=logits.max(-1)
    if logits.shape[-1]!=vocab_size:
        assert logits.shape[-1]*4==vocab_size
        ids=ids+dist.get_rank()*logits.shape[-1]
        all_values=[torch.empty_like(values) for _ in range(4)]
        all_ids=[torch.empty_like(ids) for _ in range(4)]
        dist.all_gather(all_values,values);dist.all_gather(all_ids,ids)
        winner=torch.stack(all_values).argmax(0,keepdim=True)
        ids=torch.stack(all_ids).gather(0,winner).squeeze(0)
    return ids


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'))
    p.add_argument('--mode',choices=['dense','c1','sparse_local','offload'],required=True)
    p.add_argument('--length',type=int,required=True);p.add_argument('--batch',type=int,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--profile',action='store_true')
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK'])
    hardware=json.loads((a.output/'hardware.json').read_text());topology=slurm_gpu_numa(hardware,rank)
    library=bind_host_allocations(topology['numa_node'])
    torch.cuda.set_device(rank);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(minutes=5));assert dist.get_world_size()==4
    identity=json.loads((a.root/'manifests/v96.json').read_text())
    folder=a.output/('e2e_smoke' if a.smoke else 'e2e_profile' if a.profile else 'e2e')/f'{a.mode}_t{a.length}_b{a.batch}'
    folder.mkdir(parents=True,exist_ok=True)
    def phase(name):
        (folder/f'phase{rank}.json').write_text(json.dumps(dict(phase=name,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved()))+'\n')
        print(dict(rank=rank,phase=name,mode=a.mode,length=a.length,batch=a.batch),flush=True)
    phase('load');meta=metadata()
    model=AutoModelForCausalLM.from_pretrained(identity['model'],dtype=torch.bfloat16,attn_implementation='flash_attention_2',
        distributed_config=DistributedConfig(tp_size=4),local_files_only=True).eval()
    windows=load_file(str(a.root/'calibration/windows.safetensors'))['input_ids']
    assert 0<a.batch<=16 and 0<a.length<=131072
    first=windows[64:64+a.batch]
    window_pairs=[[64+i,64+(i+8)%16] for i in range(a.batch)]
    prompt=first if a.length<=65536 else torch.cat((first,windows[[pair[1] for pair in window_pairs]]),dim=1)
    ids=prompt[:,:a.length].long().cuda()
    reference=None
    if a.smoke and a.mode=='dense':
        reference=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.detach().clone()
        assert bool(torch.isfinite(reference).all())
    steps=8 if a.smoke else 256;discard=2 if a.smoke else 32
    phase('allocate caches')
    modules,communicator=install(model,mode=a.mode,batch=a.batch,capacity=a.length+steps,
        root=a.root,basis_root=a.output/'routing_basis',rank=rank,profile=a.profile)
    capacity=a.length+steps
    cos,sin=model.model.rotary_emb(torch.empty(1,1,4096,device='cuda',dtype=torch.bfloat16),torch.arange(capacity,device='cuda')[None])
    for module in modules:
        module.rope_cos=cos[0,:,:64].contiguous();module.rope_sin=sin[0,:,:64].contiguous()
    torch.cuda.synchronize();dist.barrier();torch.cuda.reset_peak_memory_stats();phase('prefill')
    started=time.perf_counter()
    out=model(input_ids=ids,use_cache=False,logits_to_keep=1)
    token=choose(out.logits,model.config.vocab_size)
    torch.cuda.synchronize();ttft=time.perf_counter()-started
    assert bool(torch.isfinite(out.logits).all())
    error=None
    if reference is not None:
        error=float((out.logits.float()-reference.float()).square().sum()/reference.float().square().sum())
        assert error<.002,error
    if a.smoke:
        path=folder/f'prefill{rank}.safetensors';assert not path.exists()
        save_file({'logits':out.logits.cpu().contiguous()},str(path))
    prefill_peak=torch.cuda.max_memory_allocated();prefill_reserved=torch.cuda.max_memory_reserved()
    del ids,out,reference
    host_audit=[]
    if a.mode=='offload':
        for module in modules:host_audit.append(audit_host_tensor(module.host_key,topology['numa_node'],library))
    # One prefill emits the first generated token. Exactly 255 further model
    # forwards produce a 256-token sequence, with 32 early tokens excluded.
    events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(steps-1)]
    positions=[torch.full((1,1),a.length+i,device='cuda',dtype=torch.long) for i in range(steps-1)]
    generated=[token.clone()];finite=[]
    collective_audit=CollectiveAudit()
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();dist.barrier();phase('decode')
    start_wall=time.perf_counter();steady_wall=None
    for i,((begin,end),position) in enumerate(zip(events,positions)):
        if i==0:collective_audit.start()
        if i==discard-1:
            torch.cuda.synchronize();steady_wall=time.perf_counter()
        begin.record()
        out=model(input_ids=token,position_ids=position,use_cache=False,logits_to_keep=1)
        token=choose(out.logits,model.config.vocab_size)
        finite.append(torch.isfinite(out.logits).all())
        generated.append(token.clone());end.record();del out
        if i==0:collective_audit.stop()
    torch.cuda.synchronize();finish=time.perf_counter()
    decode_peak=torch.cuda.max_memory_allocated();decode_reserved=torch.cuda.max_memory_reserved()
    assert bool(torch.stack(finite).all()) and steady_wall is not None
    times=torch.tensor([begin.elapsed_time(end) for begin,end in events],device='cuda',dtype=torch.float64)
    dist.all_reduce(times,op=dist.ReduceOp.MAX)
    steady_seconds=torch.tensor(finish-steady_wall,device='cuda',dtype=torch.float64)
    dist.all_reduce(steady_seconds,op=dist.ReduceOp.MAX)
    raw=times.cpu().tolist();steady=raw[discard-1:]
    support=[]
    logical_host_reads=[]
    for module in modules:
        if a.mode=='dense':support.append([a.length+i+1 for i in range(steps-1)])
        else:
            support.append([x if isinstance(x,int) else x.cpu().tolist() for x in module.selected_counts])
            if a.mode=='offload':
                logical_host_reads.append([x*a.batch*2*128*2 if isinstance(x,int) else int(x.sum())*128*2 for x in module.selected_counts])
    generated=torch.cat(generated,-1).cpu()
    assert generated.shape==(a.batch,steps)
    path=folder/f'tokens{rank}.safetensors';assert not path.exists();save_file({'tokens':generated.contiguous()},str(path))
    native_ag_bytes=0 if a.mode=='dense' else a.batch*2*sum(8*r for r in identity['layer_ranks'])
    save(folder/f'rank{rank}.json',dict(metadata=meta,rank=rank,topology=topology,status='complete',mode=a.mode,
        model=identity['model'],dtype='bfloat16',tp=4,length=a.length,batch=a.batch,generated_tokens=steps,
        input_window_ids=[[pair[0]] if a.length<=65536 else pair for pair in window_pairs],
        input_protocol='Original diagnostic windows; above64K concatenate distinct windows offset by8, matching the128K operator capture for samples0..7. No calibration refit.',
        discard_first_generated_tokens=discard,steady_forward_count=len(steady),ttft_seconds=ttft,
        steady_decode_mean_ms=statistics.mean(steady),steady_decode_p50_ms=statistics.median(steady),
        steady_decode_p95_ms=sorted(steady)[max(0,int(.95*len(steady)+.999)-1)],raw_decode_ms=raw,
        steady_aggregate_tokens_per_second=a.batch*(steps-discard)/float(steady_seconds),
        steady_wall_seconds=float(steady_seconds),decode_total_wall_seconds=finish-start_wall,
        prefill_peak_allocated_bytes=prefill_peak,prefill_peak_reserved_bytes=prefill_reserved,
        decode_peak_allocated_bytes=decode_peak,decode_peak_reserved_bytes=decode_reserved,
        host_key_bytes=sum(m.host_key.numel()*2 for m in modules) if a.mode=='offload' else 0,
        process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        host_buffer_audit=host_audit,actual_selected_tokens_per_layer=support,
        attention_nccl_input_bytes_per_rank_per_decode=a.batch*2*(32*4096 if a.mode=='dense' else sum(8*r for r in identity['layer_ranks'])),
        torch_collectives_per_decode=collective_audit.records,
        total_nccl_input_bytes_per_rank_per_decode=native_ag_bytes+sum(r['input_bytes'] for r in collective_audit.records),
        analytical_nccl_bus_bytes_per_rank_per_decode=native_ag_bytes*3+sum(r['analytical_bus_bytes'] for r in collective_audit.records),
        nccl_scope='Observed first warmup step PyTorch collectives (model and token choice), plus native C1 attention AG; excludes benchmark barriers and metric reductions; bus bytes analytical, not hardware counters',
        logical_unique_host_k_read_bytes_per_rank_per_decode=0 if not logical_host_reads else statistics.mean(sum(row[i] for row in logical_host_reads) for i in range(steps-1)),
        logical_new_k_host_write_bytes_per_rank_per_decode=a.batch*32*2*128*2 if a.mode=='offload' else 0,
        measured_pcie_bus_bytes=None,host_traffic_note='Mapped K reads/writes: logical payload only; no H2D DMA or hardware bus counter is inferred from this value',
        allgather_nccl_version=None if communicator is None else communicator.nccl_version,
        native_dense_smoke_prefill_rel_mse=error,prefill='native FlashAttention, Q tiles2048, bottom-right causal offset',
        value_basis='dense' if a.mode=='dense' else 'E T and T^-1 D; physically separate Base16/tail caches',
        instrumentation='CUDA events, finite-output flags and actual support counts included; NUMA audit outside timing'))
    phase('complete');torch.cuda.empty_cache();dist.barrier()
    if communicator is not None:communicator.close()
    dist.destroy_process_group()


if __name__=='__main__':main()
