"""Single-GH200 prefill attention and local output GEMM microbenchmark.

FA3 uses K128/V128 with the V64 data zero-padded before timing. Its output
weights likewise have zeros in each head's added channels, giving equivalent
synthetic outputs while executing dense-width kernels. Preparation and output
allocation are excluded. There are no model weights or TP collectives.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import socket
import statistics
import sys
import time

import torch
import triton
import vllm
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

from basisserve.kernels.diffkv_prefill import diffkv_prefill
from evaluation.benchmark_fa4_diffkv import check, make_case, reference_rows


def measure(runners, rng):
    """Warm eager wall ms/call and CUDA graph GPU ms/call; inputs stay fixed."""
    graphs = {}
    for name, run in runners.items():
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
            for _ in range(4):
                run()
        graph.replay()
        torch.cuda.synchronize()
        graphs[name] = graph
    samples = {name: dict(eager_wall_ms=[], graph_gpu_ms=[]) for name in runners}
    for _ in range(7):
        names = list(runners)
        rng.shuffle(names)
        for name in names:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                runners[name]()
            torch.cuda.synchronize()
            samples[name]["eager_wall_ms"].append((time.perf_counter() - start) * 100)
            begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            begin.record()
            for _ in range(10):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name]["graph_gpu_ms"].append(begin.elapsed_time(end) / 40)
    return {name: {metric: dict(samples=values, median=statistics.median(values))
                   for metric, values in metrics.items()}
            for name, metrics in samples.items()}


def run_case(group, hidden, query_lengths, kv_lengths, rng):
    data = make_case(query_lengths, kv_lengths, group)
    tokens, heads, _ = data["out"].shape
    dense_cache = torch.zeros(*data["k"].shape[:-1], 256,
                              device="cuda", dtype=torch.bfloat16)
    dense_cache[..., :128].copy_(data["k"])
    dense_cache[..., 128:192].copy_(data["v"])
    outputs = dict(triton_v64=data["out"],
                   fa4_v64_splits1=torch.empty_like(data["out"]),
                   fa4_v64_auto=torch.empty_like(data["out"]),
                   fa3_dense_width128=torch.empty(tokens, heads, 128,
                                                 device="cuda", dtype=torch.bfloat16))

    def flash(name, version, splits):
        dense = version == 3
        return flash_attn_varlen_func(
            q=data["q"], k=dense_cache[..., :128] if dense else data["k"],
            v=dense_cache[..., 128:] if dense else data["v"], out=outputs[name],
            cu_seqlens_q=data["cu_seqlens_q"], seqused_k=data["seqused_k"],
            max_seqlen_q=max(query_lengths), max_seqlen_k=max(kv_lengths),
            block_table=data["block_table"], softmax_scale=data["softmax_scale"],
            causal=True, fa_version=version, num_splits=splits)

    attention = dict(triton_v64=lambda: diffkv_prefill(**data),
                     fa4_v64_splits1=lambda: flash("fa4_v64_splits1", 4, 1),
                     fa4_v64_auto=lambda: flash("fa4_v64_auto", 4, 0),
                     fa3_dense_width128=lambda: flash("fa3_dense_width128", 3, 1))
    rows, expected = reference_rows(data, query_lengths, kv_lengths, full=False)
    correctness = {}
    for name, run in attention.items():
        outputs[name].fill_(float("nan"))
        run()
        assert bool(torch.isfinite(outputs[name]).all())
        correctness[name + "_fp32"] = check(outputs[name][rows, :, :64], expected)
        if name != "triton_v64":
            correctness[name + "_all_rows"] = check(outputs[name][..., :64], outputs["triton_v64"])
    dense_zero_max = outputs["fa3_dense_width128"][..., 64:].abs().max().item()
    assert dense_zero_max == 0

    local_width = heads * 64
    weights = torch.randn(local_width, hidden, device="cuda", dtype=torch.bfloat16)
    weights.div_(local_width ** 0.5)
    dense_weights = torch.zeros(heads, 128, hidden, device="cuda", dtype=torch.bfloat16)
    dense_weights[:, :64].copy_(weights.view(heads, 64, hidden))
    dense_weights = dense_weights.view(heads * 128, hidden)
    pipeline_names = ("triton_v64", "fa4_v64_auto", "fa3_dense_width128")
    pipeline_outputs = {name: torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
                        for name in pipeline_names}

    def gemm(name):
        return torch.mm(outputs[name].flatten(1),
                        dense_weights if name == "fa3_dense_width128" else weights,
                        out=pipeline_outputs[name])

    def pipeline(name):
        attention[name]()
        return gemm(name)

    expected_projected = expected.flatten(1) @ weights.float()
    for name in pipeline_names:
        pipeline(name)
        correctness[name + "_pipeline_fp32"] = check(pipeline_outputs[name][rows], expected_projected)
        if name != "triton_v64":
            correctness[name + "_pipeline_all_rows"] = check(
                pipeline_outputs[name], pipeline_outputs["triton_v64"])
    attention_timings = measure(attention, rng)
    gemm_timings = measure(dict(compact_v64=lambda: gemm("triton_v64"),
                               dense_width128=lambda: gemm("fa3_dense_width128")), rng)
    pipeline_timings = measure({name: lambda name=name: pipeline(name) for name in pipeline_names}, rng)
    return dict(status="passed", query_heads=heads, hidden_size=hidden,
                query_lengths=query_lengths, kv_lengths=kv_lengths,
                reference_rows=len(rows), correctness=correctness,
                dense_extra_channels_max_absolute=dense_zero_max,
                local_gemm_shapes=dict(compact=[tokens, local_width, hidden],
                                       dense_width=[tokens, heads * 128, hidden]),
                attention=attention_timings, local_gemm=gemm_timings,
                attention_plus_local_gemm=pipeline_timings,
                fa4_auto_speedup={
                    metric: {
                        "attention_vs_triton": attention_timings["triton_v64"][metric]["median"] /
                        attention_timings["fa4_v64_auto"][metric]["median"],
                        "attention_vs_fa3_dense_width": attention_timings["fa3_dense_width128"][metric]["median"] /
                        attention_timings["fa4_v64_auto"][metric]["median"],
                        "pipeline_vs_triton": pipeline_timings["triton_v64"][metric]["median"] /
                        pipeline_timings["fa4_v64_auto"][metric]["median"],
                        "pipeline_vs_fa3_dense_width": pipeline_timings["fa3_dense_width128"][metric]["median"] /
                        pipeline_timings["fa4_v64_auto"][metric]["median"],
                    } for metric in ("eager_wall_ms", "graph_gpu_ms")})


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
                  scope="Warm single-GPU operator microbenchmark at TP4-local geometry; synthetic weights; no model, AR or AG",
                  dense_reference="FA3 K128/V128 with V64 zero-padded and each head's decoder padded with zero rows; preparation excluded",
                  timing="7 shuffled rounds; eager wall ms/call and graph GPU ms/call; output allocations excluded",
                  config=dict(dtype="bfloat16", qk_dim=128, v_dim=64, kv_heads=2, page_size=16,
                              fa4_splits=[1, 0], fa3_splits=1, seed=20261111), records=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    cases = [("2x4k", [4096, 4096], [4096, 4096]),
             ("1x8k", [8192], [8192]),
             ("4k_prefix", [4096], [8192]),
             ("mixed", [4096, 4065] + [1] * 31, [4096, 4096] + [4352] * 31),
             ("mixed_b128", [4096, 3970] + [1] * 126, [4096, 4096] + [4352] * 126),
             ("tail_b128", [32] + [1] * 127, [4096] + [4352] * 127)]
    save()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for model, group, hidden in (("llama", 4, 4096), ("qwen", 8, 5120)):
            for label, query_lengths, kv_lengths in cases:
                record = dict(model=model, case=label, status="running")
                result["records"].append(record)
                save()
                print(f"START {model} {label}", flush=True)
                record.update(run_case(group, hidden, query_lengths, kv_lengths, rng))
                save()
                print(json.dumps(record), flush=True)
    result["status"] = "passed"
    save()
    print("PASS FA4 prefill output comparison", flush=True)


if __name__ == "__main__":
    main()
