"""Compare optimized routing with the frozen pre-change source on every TP rank."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
import torch.distributed as dist

from basisserve.kernels import paged_joint_router as current


def capture(function):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        function()
    return graph


def measure_pair(graphs):
    samples = [[], []]
    for repeat in range(6):
        for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            dist.barrier()
            for _ in range(10):
                graphs[index].replay()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(100):
                graphs[index].replay()
            end.record()
            end.synchronize()
            samples[index].append(begin.elapsed_time(end) / 100)
    return dict(baseline_ms=statistics.median(samples[0]), optimized_ms=statistics.median(samples[1]),
                baseline_samples_ms=samples[0], optimized_samples_ms=samples[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    reference = Path("results/system_benchmarks/tp4_router_opt/baseline_source/paged_joint_router.py")
    spec = importlib.util.spec_from_file_location("router_before_padding_fix", reference)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    torch.manual_seed(400 + rank)
    heads = 8 // dist.get_world_size()
    results = []
    for batch, length, capacity in ((1, 4112, 5120), (32, 4112, 5120),
                                    (128, 4112, 5120), (8, 16400, 17408)):
        pages = capacity // 32
        gpu = torch.randn(batch * pages, heads, 32, 128, device="cuda", dtype=torch.bfloat16)
        low = torch.randn(batch * pages, heads, 128, device="cuda") - 2
        high = low + 4
        gpu[..., 112:] = torch.stack((low, high), dim=2).contiguous().view(torch.bfloat16).reshape(
            batch * pages, heads, 32, 16)
        cache = SimpleNamespace(gpu=gpu)
        table = torch.randperm(batch * pages, device="cuda").view(batch, pages).int()
        lengths = torch.full((batch,), length, device="cuda", dtype=torch.int32)
        query = torch.randn(batch, heads * 4, 128, device="cuda", dtype=torch.bfloat16)
        right = (torch.randn(heads, 16, 128, device="cuda") * .1).bfloat16()
        bias = (torch.randn(heads, 128, device="cuda") * .1).bfloat16()
        residual_query = (torch.randn(heads * 4, 128, 16, device="cuda") * .1).bfloat16()
        angles = torch.randn(capacity, 64, device="cuda")
        rope = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()
        routers = [old.PagedJointRouter(batch, heads, capacity, "cuda"),
                   current.PagedJointRouter(batch, heads, capacity, "cuda")]
        graphs = [capture(lambda router=router: router.select(
            query, cache, table, lengths, right, bias, residual_query, rope)) for router in routers]
        for delta in (0, 1, 16, 32, 127):
            query.normal_()
            lengths.fill_(length + delta)
            if batch > 1:
                # Exercise empty padded rows, tail-only rows and partial pages.
                lengths[:3] = torch.tensor([0, 64, 65], device="cuda", dtype=torch.int32)
            for graph in graphs:
                graph.replay()
            for name in ("coarse", "candidates", "fine", "selected_pages", "support"):
                torch.testing.assert_close(getattr(routers[0], name), getattr(routers[1], name),
                                           rtol=0, atol=0)
        lengths.fill_(length)
        for graph in graphs:
            graph.replay()
        router_times = measure_pair(graphs)
        fine_graphs = []
        for module, router, candidates in ((old, routers[0], 512),
                                           (current, routers[1], min(512, pages))):
            fine_graphs.append(capture(lambda module=module, router=router, candidates=candidates:
                module._fine[(candidates, batch * heads)](
                    query, router.query_code, gpu, table, lengths, router.candidates,
                    right, bias, rope, router.fine, heads, *query.stride()[:2], table.stride(0),
                    *gpu.stride()[:3], rope.stride(0), num_warps=4)))
        fine_times = measure_pair(fine_graphs)
        row = dict(batch=batch, length=length, capacity=capacity, exact_parity=True,
                   dynamic_graph_replays=5, router=router_times, fine=fine_times)
        results.append(row)
        print(json.dumps(dict(rank=rank, **row)), flush=True)
        del graphs, fine_graphs, routers, cache, gpu
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(dict(
        rank=rank, tp=dist.get_world_size(), status="complete",
        scope="synthetic single-layer kernel comparison; concurrent ranks; CUDA Graph; not full-model latency",
        baseline_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        optimized_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        cases=results), indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
