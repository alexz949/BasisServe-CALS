"""One isolated full-model TP4 capacity trial; failures are classified by its parent."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig
from basisserve.core.llama_tp4_k_offload import install
from evaluation.v96kl_common import configure,read_json,write_json,sha256

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['dense','offload'],required=True);p.add_argument('--batch',type=int,required=True)
    p.add_argument('--length',type=int,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();configure();rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=3));assert dist.get_world_size()==4
    a.output.mkdir(parents=True,exist_ok=True)
    def phase(name):
        (a.output/f'phase{rank}.json').write_text(json.dumps(dict(phase=name,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved()))+'\n')
        print('PHASE',rank,name,flush=True)
    identity=read_json(a.root/'manifests/v128.json');phase('load')
    model=AutoModelForCausalLM.from_pretrained(identity['model'],dtype=torch.bfloat16,attn_implementation='flash_attention_2',
        distributed_config=DistributedConfig(tp_size=4),local_files_only=True).eval()
    windows=load_file(str(a.root/'calibration/windows.safetensors'))['input_ids']
    ids=torch.stack([torch.cat((windows[i%64],windows[(i+1)%64]))[:a.length] for i in range(a.batch)]).long().cuda()
    def choose(logits):
        values,ids=logits.max(-1)
        if logits.shape[-1]!=model.config.vocab_size:
            assert logits.shape[-1]*4==model.config.vocab_size
            ids=ids+rank*logits.shape[-1]
            all_values=[torch.empty_like(values) for _ in range(4)];all_ids=[torch.empty_like(ids) for _ in range(4)]
            dist.all_gather(all_values,values);dist.all_gather(all_ids,ids)
            winner=torch.stack(all_values).argmax(0,keepdim=True)
            ids=torch.stack(all_ids).gather(0,winner).squeeze(0)
        return ids
    reference=None
    if a.smoke:
        reference=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.detach().clone()
        assert torch.isfinite(reference).all()
    phase('allocate_cache');capacity=a.length+4
    modules=install(model,mode=a.mode,batch=a.batch,capacity=capacity,root=a.root,rank=rank)
    cos,sin=model.model.rotary_emb(torch.empty(1,1,4096,device='cuda',dtype=torch.bfloat16),torch.arange(capacity,device='cuda')[None])
    cos=cos[0,:,:64].contiguous();sin=sin[0,:,:64].contiguous()
    for module in modules:module.rope_cos=cos;module.rope_sin=sin
    phase('prefill');torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    out=model(input_ids=ids,use_cache=False,logits_to_keep=1)
    torch.cuda.synchronize();prefill_seconds=time.monotonic()-started
    assert torch.isfinite(out.logits).all()
    error=None
    if reference is not None:
        # TP reduction and token tiling can change BF16 rounding.
        error=float((out.logits.float()-reference.float()).square().sum()/reference.float().square().sum())
        assert error<0.002,error
    prefill_peak=torch.cuda.max_memory_allocated();prefill_reserved=torch.cuda.max_memory_reserved()
    del reference
    token=choose(out.logits);del out,ids
    phase('decode');torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    for step in range(4):
        out=model(input_ids=token,position_ids=torch.full((1,1),a.length+step,device='cuda'),use_cache=False,logits_to_keep=1)
        assert torch.isfinite(out.logits).all();token=choose(out.logits);del out
    torch.cuda.synchronize()
    result=dict(status='complete',rank=rank,mode=a.mode,batch=a.batch,length=a.length,decode_steps=4,
        prefill_peak_bytes=prefill_peak,prefill_reserved_bytes=prefill_reserved,decode_peak_bytes=torch.cuda.max_memory_allocated(),
        decode_reserved_bytes=torch.cuda.max_memory_reserved(),resident_bytes=torch.cuda.memory_allocated(),
        prefill_seconds=prefill_seconds,decode_seconds=time.monotonic()-started,smoke_prefill_rel_mse=error,
        host_key_bytes=sum(m.host_key.numel()*m.host_key.element_size() for m in modules) if a.mode=='offload' else 0,
        gpu=torch.cuda.get_device_name(rank),total_gpu_bytes=torch.cuda.get_device_properties(rank).total_memory,
        runtime_sha256=sha256(Path('basisserve/core/llama_tp4_k_offload.py')))
    write_json(a.output/f'rank{rank}.json',result);phase('complete')
    # Peaks are already recorded. Return unused allocator blocks before NCCL teardown.
    torch.cuda.empty_cache()
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
