"""Real TP8 exact-width NCCL transport plus replicated decoder, eager and graph."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from basisserve.core.latent_a8_tp import LatentTPBoundary
from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.fp8_wire import quantize_e4m3_static
from evaluation.eval_qwen3_kv4_fp8_ppl import write

SOURCES = ("evaluation/benchmark_qwen3_a8_tp8.py", "evaluation/prepare_qwen3_a8_tp8.py",
    "basisserve/core/latent_a8_tp.py", "basisserve/kernels/latent_a8_pack.py",
    "basisserve/kernels/fp8_wire.py", "basisserve/kernels/feature_ragged_allgather.py",
    "basisserve/kernels/ragged_allgather.py", "basisserve/kernels/csrc/feature_ragged_allgather.cpp",
    "basisserve/kernels/csrc/feature_ragged_allgather_pack.cu",
    "basisserve/kernels/csrc/feature_uniform_allgather_ipc.cu",
    "basisserve/kernels/csrc/feature_ragged_allgather_common.h")


def measure(operation, graph_mode, stream, warmup, repeats):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    dist.barrier()
    graph = None
    if graph_mode:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = operation()
        operation = graph.replay
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        dist.barrier()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, stop in zip(starts, stops):
        start.record(stream)
        operation()
        stop.record(stream)
    torch.cuda.synchronize()
    samples = torch.tensor([s.elapsed_time(e) for s,e in zip(starts,stops)], device="cuda", dtype=torch.float64)
    dist.all_reduce(samples, op=dist.ReduceOp.MAX)
    values = samples.cpu().tolist()
    return dict(median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values),
        samples_ms=values)


def summarize(directory, records):
    lines = ["# Real TP8 Latent A8 Transport and Decoder", "",
        "Eight L40S GPUs, basis environment. Actual NCCL transport; replicated global decoder.",
        "Encoder/attention/KV cache execution is excluded. Fixtures come from BF16 encoder + fixed NUQ4 on WT2 train.",
        "This is not full-model E2E, packed KV4 serving, or a throughput measurement.",
        "Same adaptive checkpoint, scale, exact-width transport, samples and timing protocol across arms.", "",
        "Median across trial medians; each sample uses the maximum CUDA-event latency over all eight ranks.",
        "Full pipeline includes pack/quantize, gather, receive conversion and decoder GEMM.", "",
        "| Rank | Layer | Rows | Mode | BF16 ms | Fused A8 ms | Fused W8A8 ms | BF16 / W8A8 | Unfused / fused W8A8 |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    keys = sorted({(r["rank"],r["layer"],r["rows"],r["execution"]) for r in records})
    summary = []
    for key in keys:
        trials = [r for r in records if (r["rank"],r["layer"],r["rows"],r["execution"])==key]
        def median(name):
            return statistics.median(r["timings"][name]["median_ms"] for r in trials)
        b,a,w,old = [median(n) for n in ("bf16_full","a8_full","w8a8_full","unfused_w8a8_full")]
        row = dict(rank=key[0],layer=key[1],rows=key[2],execution=key[3],bf16_ms=b,
            a8_ms=a,w8a8_ms=w,speedup_vs_bf16=b/w,speedup_vs_unfused=old/w)
        summary.append(row)
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {b:.6f} | {a:.6f} | {w:.6f} | {b/w:.3f} | {old/w:.3f} |")
    lines += ["", "A8 kernel correctness is bit-exact against the frozen quantization expression on tested inputs.",
        "Communication payload halving is not a promise of latency halving. No additional decoder output all-reduce is included.",
        "Source widths are adaptive per KV group; ragged cases use exact-width NCCL ring, uniform cases prepared NCCL AllGather.",
        "Separate component timings do not sum exactly to the pipeline due to launch gaps, overlap and measurement boundaries.",
        "All paths use source-local packing, although a BF16 serving attention kernel may write directly into its send slot.",
        "Graph results replay a fixed input/shape, as appropriate for a boundary microbenchmark; not continuous batching.", ""]
    write(directory / "summary.json", summary)
    (directory / "SUMMARY.md").write_text("\n".join(lines))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", required=True, choices=("smoke","formal"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/q3-a8-tp8")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=10))
    assert dist.get_world_size() == 8 and dist.get_rank() == rank
    directory = args.output / args.phase
    directory.mkdir(parents=True, exist_ok=True)
    layers = [18] if args.phase == "smoke" else [0,18,35]
    batches = [1,16] if args.phase == "smoke" else [1,4,16,64,128,256]
    cohorts = [0] if args.phase == "smoke" else [0,1,2]
    warmup, repeats = (2,5) if args.phase == "smoke" else (10,50)
    if rank == 0:
        source = directory / "source"
        source.mkdir(exist_ok=True)
        for name in SOURCES:
            original = ROOT / name
            destination = source / original.name
            if destination.exists():
                assert destination.read_bytes() == original.read_bytes()
            else:
                shutil.copy2(original, destination)
        write(directory / "manifest.json", dict(command=shlex.join(sys.argv),environment="basis",
            torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            tp=8,layers=layers,batches=batches,cohorts=cohorts,warmup=warmup,repeats=repeats,
            scope="real transport plus decoder, not full model",hashes=False))
        (directory / "topology.txt").write_text(subprocess.check_output(["nvidia-smi","topo","-m"], text=True))
    comm = FeatureRaggedCommunicator.from_distributed(device=device)
    stream = torch.cuda.Stream()
    records = []
    for nominal in (64,96):
        fixture = torch.load(args.output / f"r{nominal}.pt", map_location="cpu", weights_only=False)
        assert fixture["rank"] == nominal
        for layer in layers:
            data = fixture["layers"][layer]
            widths = data["widths"]
            decoder, scale = data["decoder"].to(device), data["scale"].to(device)
            assert len(widths)==8 and all(w>0 and w%16==0 for w in widths)
            assert scale.dtype == torch.float32 and torch.isfinite(scale) and scale>0
            for rows in batches:
                full = data["latent"][:rows].to(device).contiguous()
                offset = sum(widths[:rank])
                local = full[:,offset:offset+widths[rank]].contiguous()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    paths = {mode:LatentTPBoundary(comm,widths,rows,decoder,scale,mode)
                        for mode in ("bf16","a8","w8a8")}
                    for mode, path in paths.items():
                        path.pack(local)
                        arena = path.gather()
                        expected = full.T if mode=="bf16" else quantize_e4m3_static(full,scale).view(torch.uint8).T
                        assert torch.equal(arena,expected), (nominal,layer,rows,mode,rank,
                            int((arena!=expected).sum()),
                            (arena!=expected).nonzero()[:4].tolist(),
                            arena[arena!=expected][:4].tolist(),expected[arena!=expected][:4].tolist())
                        fused = path.decode(arena).clone()
                        unfused = path(local,fused=False)
                        torch.testing.assert_close(fused,unfused,rtol=0,atol=0)
                    operations = {}
                    for mode,path in paths.items():
                        operations[f"{mode}_full"] = lambda p=path:p(local)
                        operations[f"{mode}_pack"] = lambda p=path:p.pack(local)
                        operations[f"{mode}_gather"] = path.gather
                        path.pack(local)
                        arena = path.gather()
                        operations[f"{mode}_decode"] = lambda p=path,a=arena:p.decode(a)
                    for mode in ("a8","w8a8"):
                        operations[f"unfused_{mode}_full"] = lambda p=paths[mode]:p(local,fused=False)
                    for cohort in cohorts:
                        names = list(operations)
                        shift = (cohort * 5) % len(names)
                        names = names[shift:] + names[:shift]
                        for execution in ("eager","graph"):
                            timings = {name:measure(operations[name], execution=="graph",stream,warmup,repeats) for name in names}
                            record = dict(rank=nominal,layer=layer,rows=rows,cohort=cohort,
                                execution=execution,widths=widths,timings=timings,
                                backend="uniform_nccl" if len(set(widths))==1 else "exact_width_ring",
                                payload_per_rank=dict(bf16=[2*rows*w for w in widths],a8=[rows*w for w in widths]))
                            if rank==0:
                                records.append(record)
                                write(directory / "progress.json",records)
                                print("CASE",nominal,layer,rows,cohort,execution,
                                    timings["bf16_full"]["median_ms"],timings["w8a8_full"]["median_ms"],flush=True)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
    comm.close()
    if rank==0:
        write(directory / "results.json",dict(status="complete",records=records))
        summarize(directory,records)
        print("COMPLETE",args.phase,flush=True)
    dist.destroy_process_group()


if __name__=="__main__":
    main()
