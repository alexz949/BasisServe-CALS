"""TP8 latency of the C1 decode collective under CUDA Graph replay, Qwen3-32B shapes.

C1 gathers [512 x T] BF16 feature-major coordinates per rank into [4096 x T]; dense all-reduces the [T x 5120] attention
output. Compared on this node:
  nccl       NCCL AllGather on ordinary buffers (the archived `uniform_nccl` path, NCCL ring LL)
  nccl_symm  NCCL AllGather on symmetric-window buffers (ncclMemAlloc + NCCL_WIN_COLL_SYMMETRIC, NCCL >= 2.27
             one-shot NVLink kernels), allocated through vLLM's NCCL symmetric memory pool
  custom_ag  vLLM custom all-gather (one-shot NVLink IPC, device-side signals) of the [512 x T] block, dim 0
  custom_ar  vLLM custom all-reduce of [T x 5120] (what the dense arm uses on NVLink), for reference

torchrun --nproc-per-node 8 benchmarks/a100/bench_allgather.py --output <json>
"""

import argparse
import json
import os

os.environ.setdefault("VLLM_USE_NCCL_SYMM_MEM", "1")

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_allocator import nccl_symm_mem_context

TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
LOCAL_WIDTH, HIDDEN = 512, 5120


def capture(fn, calls=64):
    """A CUDA graph holding `calls` back-to-back calls (warmed eagerly first)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            fn()
    return graph


def replay_us(graph, calls=64, replays=20):
    """Mean microseconds per call over `replays` replays."""
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return 1000 * start.elapsed_time(end) / (calls * replays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    group = dist.new_group(backend="gloo")
    nccl = PyNcclCommunicator(group, device)
    custom = CustomAllreduce(group, device, max_size=2 * 256 * HIDDEN * 2)
    assert not custom.disabled
    rows = []
    for tokens in TOKENS:
        plain_in = torch.full((LOCAL_WIDTH * tokens,), rank + 1, dtype=torch.bfloat16, device=device)
        plain_out = torch.empty(world * LOCAL_WIDTH * tokens, dtype=torch.bfloat16, device=device)
        with nccl_symm_mem_context(nccl):
            symm_in = torch.full((LOCAL_WIDTH * tokens,), rank + 1, dtype=torch.bfloat16, device=device)
            symm_out = torch.empty(world * LOCAL_WIDTH * tokens, dtype=torch.bfloat16, device=device)
        expected = torch.arange(1, world + 1, dtype=torch.bfloat16, device=device).repeat_interleave(LOCAL_WIDTH * tokens)
        nccl.all_gather(plain_out, plain_in)
        nccl.all_gather(symm_out, symm_in)
        torch.cuda.synchronize()
        assert torch.equal(plain_out, expected) and torch.equal(symm_out, expected)
        dense = torch.full((tokens, HIDDEN), 1.0, dtype=torch.bfloat16, device=device)
        row = dict(tokens=tokens,
                   nccl_us=replay_us(capture(lambda: nccl.all_gather(plain_out, plain_in))),
                   nccl_symm_us=replay_us(capture(lambda: nccl.all_gather(symm_out, symm_in))))
        # vLLM's one-shot NVLink all-gather (IPC staging buffer, device-side signals): [512, T] per rank -> [4096, T].
        local = plain_in.view(LOCAL_WIDTH, tokens)
        assert custom.should_custom_all_gather(local)
        assert torch.equal(custom.custom_all_gather(local).flatten(), expected)
        row["custom_ag_us"] = replay_us(capture(lambda: custom.custom_all_gather(local)))
        # vLLM registers the captured custom all-reduce buffers when its capture context exits.
        with custom.capture():
            graph = capture(lambda: custom.custom_all_reduce(dense))
        row["custom_ar_us"] = replay_us(graph)
        times = [None] * world
        dist.all_gather_object(times, row)
        merged = dict(tokens=tokens, **{key: max(r[key] for r in times) for key in row if key != "tokens"})
        rows.append(merged)
        if rank == 0:
            print(json.dumps(merged), flush=True)
    if rank == 0:
        with open(args.output, "w") as handle:
            json.dump(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, nccl=nccl.nccl_version,
                           calls_per_graph=64, replays=20, metric="max over ranks of mean us per call",
                           rows=rows), handle, indent=2)
    dist.barrier()


if __name__ == "__main__":
    main()
