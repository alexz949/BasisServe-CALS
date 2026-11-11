"""GH200 Triton/FA4 prefill crossover with warm and cache-flushed timing.

Cold-start timing captures a 256 MiB zero fill before external CUDA timing
events around one attention call. The fill and Python dispatch are excluded
from the measured interval. This is an operator diagnostic, not model timing.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import statistics
import sys

import torch
import triton
import vllm
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

from basisserve.kernels.diffkv_prefill import diffkv_prefill
from evaluation.benchmark_fa4_diffkv import check, make_case, reference_rows
from evaluation.benchmark_fa4_prefill_output import measure


def measure_cold(runners, flush, rng):
    graphs = {}
    flush.zero_()
    for name, run in runners.items():
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        # Initialize the event objects before capture.
        start.record()
        end.record()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
            flush.zero_()
            start.record()
            run()
            end.record()
        graph.replay()
        torch.cuda.synchronize()
        initial_ms = start.elapsed_time(end)
        assert math.isfinite(initial_ms) and initial_ms > 0, initial_ms
        graphs[name] = graph, start, end
    samples = {name: [] for name in runners}
    for _ in range(9):
        names = list(runners)
        rng.shuffle(names)
        for name in names:
            graph, start, end = graphs[name]
            graph.replay()
            # Outside the GPU event interval; guarantees latest replay completed.
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            assert math.isfinite(elapsed) and elapsed > 0, elapsed
            samples[name].append(elapsed)
    return {name: dict(samples_ms=values, median_ms=statistics.median(values))
            for name, values in samples.items()}


def run_case(group, query_lengths, kv_lengths, flush, rng):
    data = make_case(query_lengths, kv_lengths, group)
    outputs = dict(triton=data["out"], fa4_auto=torch.empty_like(data["out"]))
    runners = dict(triton=lambda: diffkv_prefill(**data),
                   fa4_auto=lambda: flash_attn_varlen_func(
                       q=data["q"], k=data["k"], v=data["v"], out=outputs["fa4_auto"],
                       cu_seqlens_q=data["cu_seqlens_q"], seqused_k=data["seqused_k"],
                       max_seqlen_q=max(query_lengths), max_seqlen_k=max(kv_lengths),
                       block_table=data["block_table"], softmax_scale=data["softmax_scale"],
                       causal=True, fa_version=4, num_splits=0))
    rows, expected = reference_rows(data, query_lengths, kv_lengths, full=False)
    correctness = {}
    for name, run in runners.items():
        outputs[name].fill_(float("nan"))
        run()
        assert bool(torch.isfinite(outputs[name]).all())
        correctness[name] = check(outputs[name][rows], expected)
    cross_check = check(outputs["fa4_auto"], outputs["triton"])
    warm = measure(runners, rng)
    cold = measure_cold(runners, flush, rng)
    # The flush buffer must not alter inputs or outputs.
    cold_checks = {name: check(output[rows], expected) for name, output in outputs.items()}
    return dict(status="passed", query_heads=2 * group, sequences=len(query_lengths),
                query_lengths=query_lengths, kv_lengths=kv_lengths, reference_rows=len(rows),
                correctness=correctness, all_rows_cross_check=cross_check,
                post_cold_correctness=cold_checks, warm=warm, cold_start_gpu=cold,
                fa4_auto_speedup={
                    "eager_wall": warm["triton"]["eager_wall_ms"]["median"] /
                    warm["fa4_auto"]["eager_wall_ms"]["median"],
                    "warm_graph_gpu": warm["triton"]["graph_gpu_ms"]["median"] /
                    warm["fa4_auto"]["graph_gpu_ms"]["median"],
                    "cold_start_gpu": cold["triton"]["median_ms"] / cold["fa4_auto"]["median_ms"],
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261111)
    rng = random.Random(20261111)
    installed = Path(vllm.__file__).parent / "vllm_flash_attn"
    sources = [Path(__file__), Path("evaluation/benchmark_fa4_diffkv.py"),
               Path("evaluation/benchmark_fa4_prefill_output.py"),
               Path("basisserve/kernels/diffkv_prefill.py"),
               installed / "flash_attn_interface.py", installed / "cute/interface.py",
               installed / "cute/flash_fwd.py", installed / "cute/flash_fwd_sm90.py"]
    result = dict(status="running", command=[sys.executable, *sys.argv],
                  environment=os.environ.get("CONDA_DEFAULT_ENV"),
                  job=os.environ.get("SLURM_JOB_ID"), host=socket.gethostname(),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  triton=triton.__version__, vllm=vllm.__version__, cuda=torch.version.cuda,
                  cutlass_dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  scope="Single-GPU attention only at TP4-local geometry; no model, decoder, AR or AG",
                  timing=dict(warm="7 shuffled rounds: eager wall and CUDA graph GPU ms/call",
                              cold_start="9 shuffled graph replays: 256MiB zero fill then external start event, one attention call, external end event; device synchronize outside interval",
                              cache_flush_bytes=256 * 1024 * 1024,
                              caveat="Cache-flushed operator diagnostic; not a simulation of full-model cache or scheduling"),
                  config=dict(dtype="bfloat16", qk_dim=128, v_dim=64, kv_heads=2,
                              page_size=16, fa_version=4, fa4_num_splits=0, seed=20261111), records=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    cases = [(f"b{batch}_q{qlen}", [qlen] + [1] * (batch - 1),
              [4096] + [4352] * (batch - 1))
             for batch in (1, 128) for qlen in (32, 128, 512, 2048)]
    cases += [("2x4k", [4096, 4096], [4096, 4096]), ("1x8k", [8192], [8192])]
    save()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.int32)
        for model, group in (("llama", 4), ("qwen", 8)):
            for label, query_lengths, kv_lengths in cases:
                record = dict(model=model, case=label, status="running")
                result["records"].append(record)
                save()
                print(f"START {model} {label}", flush=True)
                record.update(run_case(group, query_lengths, kv_lengths, flush, rng))
                save()
                print(json.dumps(record), flush=True)
    result["status"] = "passed"
    save()
    print("PASS FA4 crossover and cold-start comparison", flush=True)


if __name__ == "__main__":
    main()
