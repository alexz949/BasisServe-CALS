"""TP4 latency of the attention-output boundary under CUDA Graph replay, Qwen3-32B shapes.

Per layer the dense arm runs o_proj [T x 2048] -> [T x 5120] on every rank and all-reduces the result. C1 gathers
[T x 16r] value coordinates from every rank and applies one replicated decoder GEMM [T x 64r] -> [T x 5120].
Each part is timed alone on the served code path: decode steps gather in place on the symmetric arena, prefill chunks
pack token-major coordinates feature-major and gather through the prepared NCCL plan. Weights are random; attention and the rest of the model are excluded.
"""

import argparse
import json
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.distributed.parallel_state import graph_capture

from basisserve.vllm.prepared_c1_output import PreparedC1Output
from evaluation.validate_gh200_tp4 import tp_group

DECODE_TOKENS = [1, 8, 32, 64, 128, 256]
PREFILL_TOKENS = [4096, 8192]
HIDDEN, LOCAL_HEADS, HEAD_DIM, CALLS, REPLAYS = 5120, 16, 128, 16, 20


def replay_us(device, fn):
    """Mean microseconds per call of `fn`, replayed from one CUDA graph holding CALLS back-to-back calls."""
    # vLLM's capture context registers custom all-reduce buffers when it exits.
    with graph_capture(device) as context:
        with torch.cuda.stream(context.stream):
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=context.stream):
            for _ in range(CALLS):
                fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    end.synchronize()
    return 1000 * start.elapsed_time(end) / (CALLS * REPLAYS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--value-rank', type=int, choices=[64, 96], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda', torch.cuda.current_device())
    group = tp_group(4, rank)
    local_width = LOCAL_HEADS * args.value_rank
    total_width = 4 * local_width
    boundary = PreparedC1Output(group.device_group, device, local_width, max(PREFILL_TOKENS))
    assert boundary.symmetric is not None
    nccl, buffer, source = boundary.symmetric
    decoder = torch.randn(total_width, HIDDEN, device=device, dtype=torch.bfloat16)
    o_proj = torch.randn(HIDDEN, LOCAL_HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16)
    rows = []
    for tokens in DECODE_TOKENS + PREFILL_TOKENS:
        local = torch.randn(tokens, local_width, device=device, dtype=torch.bfloat16)
        heads = torch.randn(tokens, LOCAL_HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16)
        projected = torch.randn(tokens, HIDDEN, device=device, dtype=torch.bfloat16)
        if tokens in DECODE_TOKENS:
            # Decode attention writes straight into this rank's slot of the arena, so there is no pack.
            arena = buffer[:total_width * tokens].view(total_width, tokens)
            slot = arena[source * local_width:(source + 1) * local_width]
            slot.copy_(local.T)

            def pack():
                return slot

            def gather():
                nccl.all_gather(arena.view(-1), slot.reshape(-1))
                return arena
        else:
            arena = torch.randn(total_width, tokens, device=device, dtype=torch.bfloat16)

            # Prefill attention emits token-major [T x 16r]; the gather needs this rank's block feature-major.
            def pack():
                return boundary._prepared(tokens, local.dtype, device).local_feature_major_view_fast().copy_(local.T)

            def gather():
                return boundary._prepared(tokens, local.dtype, device).gather_inplace_fast()
        row = dict(
            c1_pack_us=replay_us(device, pack) if tokens in PREFILL_TOKENS else 0.0,
            c1_gather_us=replay_us(device, gather),
            c1_decoder_gemm_us=replay_us(device, lambda: boundary._decode(arena, decoder)),
            c1_total_us=replay_us(device, lambda: (pack(), boundary._decode(gather(), decoder))[1]),
            dense_o_proj_gemm_us=replay_us(device, lambda: F.linear(heads, o_proj)),
            dense_all_reduce_us=replay_us(device, lambda: group.all_reduce(projected)),
            dense_total_us=replay_us(device, lambda: group.all_reduce(F.linear(heads, o_proj))),
        )
        ranks = [None] * 4
        dist.all_gather_object(ranks, row)
        rows.append(dict(tokens=tokens, path='decode' if tokens in DECODE_TOKENS else 'prefill',
                         **{key: max(r[key] for r in ranks) for key in row}))
        if rank == 0:
            print(json.dumps(rows[-1]), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(
                status='complete' if len(rows) == len(DECODE_TOKENS + PREFILL_TOKENS) else 'running',
                hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(), value_rank=args.value_rank,
                local_width=local_width, calls_per_graph=CALLS, replays=REPLAYS,
                metric='max over ranks of mean microseconds per call',
                collective=boundary.statistics(), rows=rows), indent=2) + '\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
