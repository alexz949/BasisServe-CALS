#!/usr/bin/env python3
"""Measure the GEMM cost of consuming 1/2/4/8 C1 source groups.

All coordinates are resident before timing. This isolates the cost that a
communication-overlap implementation must recover; it does not measure overlap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import statistics
import sys

import torch


def decode_groups(arena, decoder, output, groups):
    width = decoder.shape[0] // groups
    torch.mm(arena[:width].T, decoder[:width], out=output)
    for start in range(width, decoder.shape[0], width):
        output.addmm_(arena[start : start + width].T, decoder[start : start + width])
    return output


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    batches = [int(item) for item in args.batches.split(",")]
    assert all(batch > 0 for batch in batches)
    assert args.warmup > 0 and args.iterations > 0
    torch.manual_seed(71)
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    records = []
    with torch.cuda.stream(stream):
        decoder = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16) / 2048**0.5
        for batch in batches:
            arena = torch.randn(2048, batch, device="cuda", dtype=torch.bfloat16)
            reference = arena.T.float() @ decoder.float()
            for groups in (1, 2, 4, 8):
                output = torch.empty(batch, 4096, device="cuda", dtype=torch.bfloat16)
                for _ in range(args.warmup):
                    decode_groups(arena, decoder, output, groups)
                for mode in ("eager", "cuda_graph"):
                    graph = None
                    if mode == "cuda_graph":
                        stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            decode_groups(arena, decoder, output, groups)
                    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
                    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
                    for start, end in zip(starts, ends, strict=True):
                        start.record(stream)
                        if graph is None:
                            decode_groups(arena, decoder, output, groups)
                        else:
                            graph.replay()
                        end.record(stream)
                    stream.synchronize()
                    samples = sorted(start.elapsed_time(end) * 1000 for start, end in zip(starts, ends, strict=True))
                    error = float((output.float() - reference).norm() / reference.norm())
                    assert error < 0.02
                    row = dict(batch=batch, groups=groups, mode=mode,
                               p50_us=statistics.median(samples),
                               p90_us=samples[min(len(samples)-1, int(len(samples)*0.9))],
                               relative_l2_fp32=error)
                    records.append(row)
                    print(json.dumps(row), flush=True)
                    del graph
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(dict(
        command=shlex.join([sys.executable, *sys.argv]),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
        geometry=dict(tp=8, local_width=256, global_width=2048, hidden=4096),
        caveat="Resident-input GEMM cost only; no communication overlap measured. BF16 output rounds after each partial GEMM.",
        records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
