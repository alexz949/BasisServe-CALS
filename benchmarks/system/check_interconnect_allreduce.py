"""All-reduce part of check_interconnect.sh; run under torch.distributed.run (torchrun).

BF16 NCCL all-reduce at the message sizes the TP4 Llama-3.1-8B benchmark issues: 256 KiB / 512 KiB /
1 MiB are one decode all-reduce at batch 32 / 64 / 128 (batch x 4096 hidden x 2 bytes), 64 MiB is one
8192-token prefill chunk. A decode step issues about 64 of them (two per layer, 32 layers).
"""
import json
import os

import torch
import torch.distributed as dist

local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dist.init_process_group("nccl", device_id=torch.device("cuda", local))
rank, world = dist.get_rank(), dist.get_world_size()
results = []
for nbytes in (256 << 10, 512 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20):
    x = torch.zeros(nbytes // 2, dtype=torch.bfloat16, device="cuda")
    iters = 300 if nbytes <= 1 << 20 else 100 if nbytes <= 16 << 20 else 30
    for _ in range(30):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dist.all_reduce(x)
    end.record()
    torch.cuda.synchronize()
    t = torch.tensor([start.elapsed_time(end) * 1000 / iters], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    us = t.item()
    results.append(dict(bytes=nbytes, us=round(us, 1), busbw_GBs=round(nbytes * 2 * (world - 1) / world / (us * 1e-6) / 1e9, 2)))
if rank == 0:
    print("RESULT allreduce", json.dumps(dict(gpus=os.environ.get("CUDA_VISIBLE_DEVICES"), nccl_p2p_disable=os.environ.get("NCCL_P2P_DISABLE"),
                                              nccl=".".join(map(str, torch.cuda.nccl.version())), world=world, results=results)), flush=True)
dist.destroy_process_group()
