"""Single-GH200 QK128/V64 comparison: current Triton versus vLLM FA4.

Uses TP4-local heads, packed paged KV, and identical inputs. Eager wall time
includes Python dispatch; CUDA graph event time isolates repeated GPU work.
Neither metric includes model projections, collectives, or scheduling.
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


def make_case(query_lengths, kv_lengths, group):
    page, kv_heads, value_dim = 16, 2, 64
    heads = kv_heads * group
    blocks = [triton.cdiv(n, page) for n in kv_lengths]
    cache = torch.randn(sum(blocks), page, kv_heads, 128 + value_dim,
                        device="cuda", dtype=torch.bfloat16)
    table = torch.zeros(len(blocks), max(blocks), device="cuda", dtype=torch.int32)
    pages = torch.randperm(sum(blocks), device="cuda", dtype=torch.int32)
    offset = 0
    for sequence, count in enumerate(blocks):
        table[sequence, :count] = pages[offset:offset + count]
        offset += count
    starts = [0]
    for length in query_lengths:
        starts.append(starts[-1] + length)
    qkv = torch.randn(starts[-1], heads * 128 + kv_heads * 192,
                      device="cuda", dtype=torch.bfloat16)
    query = qkv[:, :heads * 128].view(-1, heads, 128)
    return dict(q=query, k=cache[..., :128], v=cache[..., 128:],
                out=torch.empty(starts[-1], heads, value_dim,
                                device="cuda", dtype=torch.bfloat16),
                cu_seqlens_q=torch.tensor(starts, device="cuda", dtype=torch.int32),
                seqused_k=torch.tensor(kv_lengths, device="cuda", dtype=torch.int32),
                block_table=table, softmax_scale=128 ** -0.5)


def reference_rows(data, query_lengths, kv_lengths, full):
    """Independent FP32 attention, all short-case rows or sampled long-case rows."""
    indices, answers = [], []
    start = 0
    group = data["q"].shape[1] // 2
    for sequence, (qlen, length) in enumerate(zip(query_lengths, kv_lengths)):
        positions = (list(range(qlen)) if full else sorted({
            0, min(1, qlen - 1), min(15, qlen - 1), min(16, qlen - 1),
            min(127, qlen - 1), min(128, qlen - 1), qlen // 2, qlen - 1}))
        rows = torch.tensor(positions, device="cuda")
        physical = data["block_table"][sequence, :triton.cdiv(length, 16)].long()
        keys = data["k"][physical].flatten(0, 1)[:length].float()
        values = data["v"][physical].flatten(0, 1)[:length].float()
        query = data["q"][start + rows].float()
        answer = torch.empty(len(rows), query.shape[1], 64, device="cuda")
        masked = torch.arange(length, device="cuda")[None, :] > (
            length - qlen + rows[:, None])
        for kv_head in range(2):
            heads = slice(kv_head * group, (kv_head + 1) * group)
            scores = (query[:, heads] @ keys[:, kv_head].T) * data["softmax_scale"]
            scores.masked_fill_(masked[:, None, :], float("-inf"))
            answer[:, heads] = scores.softmax(-1) @ values[:, kv_head]
        indices.append(start + rows)
        answers.append(answer)
        start += qlen
    return torch.cat(indices), torch.cat(answers)


def check(actual, expected):
    difference = actual.float() - expected.float()
    error = (difference.norm() / expected.float().norm()).item()
    maximum = difference.abs().max().item()
    assert bool(torch.isfinite(actual).all()) and error < 0.005, (error, maximum)
    assert torch.allclose(actual.float(), expected.float(), atol=0.008, rtol=0.02)
    return dict(relative_l2=error, max_absolute=maximum)


def measure(runners, outputs, data, query_lengths, kv_lengths, rng):
    graphs = {}
    for name, run in runners.items():
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
            for _ in range(4):
                run()
        graphs[name] = graph
    # Replaying a captured graph must read changed data, including page tables.
    data["q"].normal_()
    data["v"].normal_()
    if len(set(kv_lengths)) == 1:
        data["block_table"].copy_(data["block_table"].flip(0))
    rows, expected = reference_rows(data, query_lengths, kv_lengths, full=False)
    for name, graph in graphs.items():
        graph.replay()
        torch.cuda.synchronize()
        check(outputs[name][rows], expected)
    samples = {name: dict(eager_wall_ms=[], graph_gpu_ms=[]) for name in runners}
    for _ in range(7):
        names = list(runners)
        rng.shuffle(names)
        for name in names:
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(10):
                runners[name]()
            torch.cuda.synchronize()
            samples[name]["eager_wall_ms"].append((time.perf_counter() - started) * 100)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "bench"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261111)
    rng = random.Random(20261111)
    installed = Path(vllm.__file__).parent / "vllm_flash_attn"
    sources = [Path(__file__), Path("basisserve/kernels/diffkv_prefill.py"),
               installed / "flash_attn_interface.py", installed / "cute/interface.py",
               installed / "cute/flash_fwd.py", installed / "cute/flash_fwd_sm90.py"]
    result = dict(status="running", command=[sys.executable, *sys.argv], stage=args.stage,
                  environment=os.environ.get("CONDA_DEFAULT_ENV"),
                  job=os.environ.get("SLURM_JOB_ID"), host=socket.gethostname(),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  triton=triton.__version__, vllm=vllm.__version__, cuda=torch.version.cuda,
                  cutlass_dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  scope="Single GPU attention only; TP4-local heads; no decoder, collectives or model timing",
                  config=dict(dtype="bfloat16", qk_dim=128, v_dim=64, kv_heads=2,
                              page_size=16, fa_version=4, fa4_num_splits=1,
                              triton_head_rows=64, triton_kv_tile=128, triton_warps=4,
                              triton_stages=2), records=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    cases = [("ragged", [129, 17, 1], [146, 51, 257])]
    if args.stage == "bench":
        cases = [("2x4k", [4096, 4096], [4096, 4096]),
                 ("1x8k", [8192], [8192]),
                 ("4k_prefix", [4096], [8192]),
                 ("mixed", [4096, 4065] + [1] * 31, [4096, 4096] + [4352] * 31)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for model, group in (("llama", 4), ("qwen", 8)):
            for label, query_lengths, kv_lengths in cases:
                record = dict(model=model, case=label, status="running", query_heads=2 * group,
                              query_lengths=query_lengths, kv_lengths=kv_lengths)
                result["records"].append(record)
                save()
                print(f"START {model} {label}", flush=True)
                data = make_case(query_lengths, kv_lengths, group)
                outputs = dict(triton=data["out"], fa4=torch.empty_like(data["out"]))
                runners = dict(triton=lambda: diffkv_prefill(**data),
                               fa4=lambda: flash_attn_varlen_func(
                                   q=data["q"], k=data["k"], v=data["v"], out=outputs["fa4"],
                                   cu_seqlens_q=data["cu_seqlens_q"], seqused_k=data["seqused_k"],
                                   max_seqlen_q=max(query_lengths), max_seqlen_k=max(kv_lengths),
                                   block_table=data["block_table"], softmax_scale=data["softmax_scale"],
                                   causal=True, fa_version=4, num_splits=1))
                rows, expected = reference_rows(data, query_lengths, kv_lengths,
                                                full=args.stage == "smoke")
                record["reference_rows"] = len(rows)
                record["correctness"] = {}
                for name, run in runners.items():
                    outputs[name].fill_(float("nan"))
                    run()
                    record["correctness"][name] = check(outputs[name][rows], expected)
                    assert bool(torch.isfinite(outputs[name]).all())
                    save()
                    print(f"CHECK {model} {label} {name} {record['correctness'][name]}", flush=True)
                record["all_rows_cross_check"] = check(outputs["fa4"], outputs["triton"])
                record["timings"] = measure(runners, outputs, data, query_lengths, kv_lengths, rng)
                record["changed_input_graph_replay"] = "passed"
                record["speedup_fa4_over_triton"] = {
                    metric: record["timings"]["triton"][metric]["median"] /
                    record["timings"]["fa4"][metric]["median"]
                    for metric in ("eager_wall_ms", "graph_gpu_ms")}
                record["status"] = "passed"
                save()
                print(json.dumps(record), flush=True)
    result["status"] = "passed"
    save()
    print(f"PASS FA4 DiffKV {args.stage}", flush=True)


if __name__ == "__main__":
    main()
