"""Real C1 layer ranks: dense output AR, global-coordinate C1 AR, and C1 AG."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file
from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.ragged_allgather import StaticRaggedPlan
from benchmarks.system.common import metadata,save
from evaluation.benchmark_tp4_interconnect import _critical_cuda_timings,_summary


def timed(function,device,smoke):
    warmup,iterations=(5,10) if smoke else (100,500)
    raw=_critical_cuda_timings(function,warmup=warmup,iterations=iterations,device=device)
    return dict(**_summary(raw),raw_ms=raw,stddev_ms=statistics.pstdev(raw),warmup=warmup,iterations=iterations,
        aggregation='per-iteration maximum across all TP ranks',
        timing='CUDA events on the current stream, native launches; no explicit tensor allocation in timed functions')


class OutputBlock:
    def __init__(self,encoder,decoder,wo,batch,rank,communicator):
        r=encoder.shape[-1];device=encoder.device
        self.r=r;self.rank=rank;self.batch=batch
        self.encoder=encoder[rank*2:(rank+1)*2].repeat_interleave(4,0).transpose(1,2).contiguous()
        self.decoder=decoder.reshape(32*r,4096).contiguous()
        self.wo=wo[:,rank*1024:(rank+1)*1024].T.contiguous()
        # Synthetic attention-output vectors; real factors and Wo. This is a block
        # communication microbenchmark, not a model quality measurement.
        self.input=torch.randn(batch,1024,device=device,dtype=torch.bfloat16)
        self.head_input=self.input.view(batch,8,128).permute(1,2,0)
        self.coordinates=torch.empty(8,r,batch,device=device,dtype=torch.bfloat16)
        self.ar=torch.empty(32*r,batch,device=device,dtype=torch.bfloat16)
        self.ar_slot=self.ar[rank*8*r:(rank+1)*8*r]
        self.dense=torch.empty(batch,4096,device=device,dtype=torch.bfloat16)
        self.output=torch.empty_like(self.dense)
        plan=StaticRaggedPlan.from_source_widths((8*r,)*4)
        self.ag=communicator.prepare_uniform(plan,tokens=batch,dtype=torch.bfloat16,backend='uniform_nccl')
        self.ag_slot=self.ag.local_feature_major_view_fast().view(8,r,batch)
        self.ag_slot.zero_()
        self.ag_full=self.ag.gather_inplace_fast()

    def dense_projection(self):torch.mm(self.input,self.wo,out=self.dense)
    def dense_collective(self):dist.all_reduce(self.dense)
    def dense_total(self):self.dense_projection();self.dense_collective()
    def ar_projection(self):
        torch.bmm(self.encoder,self.head_input,out=self.coordinates)
        self.ar.zero_();self.ar_slot.copy_(self.coordinates.view(8*self.r,self.batch))
    def ar_collective(self):dist.all_reduce(self.ar)
    def ar_decode(self):torch.mm(self.ar.T,self.decoder,out=self.output)
    def ar_total(self):self.ar_projection();self.ar_collective();self.ar_decode()
    def ag_projection(self):torch.bmm(self.encoder,self.head_input,out=self.ag_slot)
    def ag_collective(self):self.ag.gather_inplace_fast()
    def ag_decode(self):torch.mm(self.ag_full.T,self.decoder,out=self.output)
    def ag_total(self):self.ag_projection();self.ag_collective();self.ag_decode()

    def validate(self):
        self.ar_total();ar=self.output.clone();coordinates=self.ar.clone()
        self.ag_total()
        assert torch.equal(coordinates,self.ag_full)
        torch.testing.assert_close(ar,self.output,rtol=0,atol=0)
        # Check BF16 encoding against the same operation accumulated in FP32.
        local_reference=torch.bmm(self.encoder.float(),self.head_input.float()).bfloat16()
        torch.testing.assert_close(self.ag_slot,local_reference,rtol=.01,atol=.003)
        self.dense_projection();partials=[torch.empty_like(self.dense) for _ in range(4)]
        dist.all_gather(partials,self.dense)
        self.dense_collective()
        reference=torch.stack(partials).float().sum(0)
        torch.testing.assert_close(self.dense.float(),reference,rtol=.03,atol=.03)
        assert torch.isfinite(self.output).all() and torch.isfinite(self.dense).all()
        return dict(c1_ar_ag_coordinates_bitwise_equal=True,c1_ar_ag_output_bitwise_equal=True,
                    local_encoding_reference=True,dense_ar_reference=True)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'));p.add_argument('--smoke',action='store_true')
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl');assert dist.get_world_size()==4
    assert 'NCCL_ALGO' not in os.environ and 'NCCL_PROTO' not in os.environ
    device=torch.device('cuda',rank);torch.manual_seed(2026+rank)
    identity=json.loads((a.root/'manifests/v96.json').read_text());model=Path(identity['model'])
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    communicator=FeatureRaggedCommunicator.from_distributed(device=device)
    run_metadata=metadata();run_metadata['allgather_nccl_version']=communicator.nccl_version
    for layer in ([3] if a.smoke else range(32)):
        path=a.root/'v96/selected_factors'/f'layer_{layer:03d}.safetensors'
        factors=load_file(str(path),device=str(device))
        r=identity['layer_ranks'][layer]
        encoder=factors['value_coordinate_encoders'].bfloat16();decoder=factors['head_output_decoders'].bfloat16()
        assert encoder.shape==(8,128,r) and decoder.shape==(32,r,4096)
        assert bool((factors['source_ranks']==r).all())
        name=f'model.layers.{layer}.self_attn.o_proj.weight'
        with safe_open(str(model/index[name]),framework='pt',device='cpu') as f:wo=f.get_tensor(name).to(device)
        records=[]
        for batch in ([1,8] if a.smoke else [1,8,32,64,128,256]):
            block=OutputBlock(encoder,decoder,wo,batch,rank,communicator);audit=block.validate()
            for method in ['dense','ar','ag']:
                projection=getattr(block,f'{method}_projection');collective=getattr(block,f'{method}_collective')
                total=getattr(block,f'{method}_total')
                projection_time=timed(projection,device,a.smoke)
                # Avoid repeated in-place reductions overflowing during the isolated
                # collective timing. End-to-end block timing recomputes its inputs.
                if method=='dense':block.dense.zero_()
                elif method=='ar':block.ar.zero_()
                collective_time=timed(collective,device,a.smoke)
                total()
                decode_time=None if method=='dense' else timed(getattr(block,f'{method}_decode'),device,a.smoke)
                total_time=timed(total,device,a.smoke)
                payload=batch*2*({'dense':4096,'ar':32*r,'ag':8*r}[method])
                records.append(dict(method={'dense':'Dense AR','ar':'C1 global-coordinate AR','ag':'C1 LR-AG'}[method],
                    layer=layer,rank_per_kv_head=r,batch=batch,dtype='bfloat16',tp=4,
                    projection=projection_time,collective=collective_time,decoder=decode_time,total=total_time,
                    collective_input_bytes_per_rank=payload,analytical_bus_bytes_per_rank=payload*(3 if method=='ag' else 1.5),
                    measured_bus_bytes=None,audit=audit))
                if rank==0:print(dict(layer=layer,r=r,batch=batch,method=method,total_p50_ms=total_time['p50_ms']),flush=True)
            del block
        if rank==0:
            save(a.output/f'tp_collective_{"smoke_" if a.smoke else ""}layer{layer:03d}.json',dict(metadata=run_metadata,
                factor_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model=str(model),records=records,
                interpretation='C1 AR embeds disjoint per-head coordinates in one global buffer; mathematically identical to C1 AG, not independently fitted global low-rank factors'))
        del factors,encoder,decoder,wo;torch.cuda.empty_cache()
    dist.barrier();communicator.close();torch.cuda.empty_cache();dist.destroy_process_group()


if __name__=='__main__':main()
