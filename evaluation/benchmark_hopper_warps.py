"""Single-GH200 DiffKV launch sweep at TP4-local geometry, without model weights.

Correctness uses independently gathered paged KV and FP32 attention. Timings
include the split-decode reduction and use CUDA graphs, matching decode serving.
This measures attention only; it does not measure TP communication or end to end.
"""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import socket
import statistics
import subprocess
import sys

import torch
import triton
import triton.testing

from basisserve.kernels.diffkv_decode import diffkv_decode, _launch_config as decode_config
from basisserve.kernels.diffkv_prefill import diffkv_prefill, _launch_config as prefill_config


def make_case(batch, query_length, length, group, rank, kv_heads=2, decode=False):
    page = 16
    heads = group * kv_heads
    blocks = triton.cdiv(length, page)
    cache = torch.randn(batch * blocks, page, kv_heads, 128 + rank,
                        device="cuda", dtype=torch.bfloat16)
    table = torch.randperm(batch * blocks, device="cuda", dtype=torch.int32).view(batch, blocks)
    qkv = torch.randn(batch * query_length, heads * 128 + kv_heads * (128 + rank),
                      device="cuda", dtype=torch.bfloat16)
    q = qkv[:, :heads * 128].view(-1, heads, 128)
    if decode:
        out = torch.empty(heads * rank, batch, device="cuda", dtype=q.dtype).T.view(batch, heads, rank)
    else:
        out = torch.empty(batch * query_length, heads, rank, device="cuda", dtype=q.dtype)
    data = dict(q=q, k=cache[..., :128], v=cache[..., 128:], out=out,
                cu_seqlens_q=torch.arange(batch + 1, device="cuda", dtype=torch.int32) * query_length,
                seqused_k=torch.full((batch,), length, device="cuda", dtype=torch.int32),
                block_table=table, softmax_scale=128 ** -0.5)
    return data


def reference(data, query_length):
    """Bound FP32 score memory by processing 128 query tokens at a time."""
    answers = torch.empty_like(data["out"], dtype=torch.float32)
    group = data["q"].shape[1] // data["k"].shape[2]
    for sequence in range(len(data["seqused_k"])):
        length = int(data["seqused_k"][sequence])
        pages = data["block_table"][sequence].long()
        keys = data["k"][pages].flatten(0, 1)[:length].float()
        values = data["v"][pages].flatten(0, 1)[:length].float()
        start = sequence * query_length
        for head in range(data["k"].shape[2]):
            heads = slice(head * group, (head + 1) * group)
            for offset in range(0, query_length, 128):
                end = min(query_length, offset + 128)
                query = data["q"][start + offset:start + end, heads].float()
                scores = (query @ keys[:, head].T) * data["softmax_scale"]
                positions = torch.arange(offset, end, device="cuda") + length - query_length
                mask = torch.arange(length, device="cuda")[None, :] > positions[:, None]
                scores.masked_fill_(mask[:, None, :], float("-inf"))
                answers[start + offset:start + end, heads] = scores.softmax(-1) @ values[:, head]
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warps", type=int, nargs="+", default=[4, 8], choices=[2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bench-ms", type=int, default=30)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128, 256])
    parser.add_argument("--modes", nargs="+", choices=["decode", "prefill"], default=["decode", "prefill"])
    parser.add_argument("--completed-results", type=Path,
                        help="Skip fully measured shapes from an earlier allocation; write a separate result file.")
    args = parser.parse_args()
    assert args.repeats > 0 and args.bench_ms > 0
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261109)
    rng = random.Random(20261109)
    props = torch.cuda.get_device_properties(0)
    completed = set()
    if args.completed_results:
        assert args.completed_results.resolve() != args.output.resolve()
        previous = json.loads(args.completed_results.read_text())
        assert all(previous["arguments"][key] == getattr(args, key)
                   for key in ("smoke", "warps", "repeats", "bench_ms", "batches", "modes"))
        completed = {(s["mode"], s["batch"], s["group"], s["value_rank"])
                     for s in previous["summaries"]}
    source_files = [Path(__file__), Path("basisserve/kernels/diffkv_decode.py"),
                    Path("basisserve/kernels/diffkv_prefill.py")]
    if args.completed_results:
        assert all(previous["source_sha256"][str(p)] == hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in source_files[1:])
    result = dict(status="running", command=list(sys.argv),
                  gpu=props.name, sm=90, sm_count=props.multi_processor_count,
                  memory_bytes=props.total_memory, torch=torch.__version__,
                  cuda=torch.version.cuda, triton=triton.__version__,
                  hostname=socket.gethostname(),
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                  git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  scope="single GPU, TP4-local attention, two KV heads, no collective or model timing",
                  arguments=vars(args) | {"output": str(args.output),
                                         "completed_results": str(args.completed_results) if args.completed_results else None},
                  skipped_completed_shapes=sorted(completed), records=[], summaries=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    for mode, group, rank in itertools.product(args.modes, (4, 8), (64, 96)):
        batches = ([1, 8] if args.smoke else args.batches) if mode == "decode" else [1]
        for batch in batches:
            if (mode, batch, group, rank) in completed:
                continue
            query_length = 1 if mode == "decode" else (129 if args.smoke else 4096)
            length = 4097 if mode == "decode" else query_length + (17 if args.smoke else 0)
            data = make_case(batch, query_length, length, group, rank, decode=mode == "decode")
            expected = reference(data, query_length)
            if mode == "decode":
                shape = (batch, group * 2, 16)
                data.update(softmax_segm_output=torch.empty(*shape, triton.next_power_of_2(rank), device="cuda"),
                            softmax_segm_max=torch.empty(shape, device="cuda"),
                            softmax_segm_expsum=torch.empty(shape, device="cuda"))
                default = decode_config(batch, length, 128, group, rank)
                configs = list(itertools.product((1, 16) if args.smoke else (1, 4, 8, 16),
                                                 (64,) if args.smoke else (32, 64, 128),
                                                 args.warps, (2,) if args.smoke else (2, 3)))

                def run(config):
                    segments, tile, warps, stages = config
                    return diffkv_decode(**data, split_threshold=128, max_sequence_length=length,
                                         segments=segments, tile=tile, num_warps=warps, num_stages=stages)
            else:
                default = prefill_config(group, rank, batch, query_length, length)
                configs = list(itertools.product((64,) if args.smoke else (32, 64, 128),
                                                 (64,) if args.smoke else (32, 64, 128),
                                                 args.warps, (2,) if args.smoke else (2, 3)))

                def run(config):
                    bm, bn, warps, stages = config
                    kernel = diffkv_prefill(**data, block_m=bm, block_n=bn,
                                            num_warps=warps, num_stages=stages)
                    return (kernel,)

            configs = list(dict.fromkeys([default, *configs]))
            records = []
            for config in configs:
                data["out"].fill_(float("nan"))
                kernels = run(config)
                actual = data["out"].float()
                error = ((actual - expected).norm() / expected.norm()).item()
                correct = bool(torch.isfinite(actual).all() and
                               torch.allclose(actual, expected, atol=0.008, rtol=0.02) and error < 0.005)
                resources = [dict(name=k.name, registers=k.n_regs, spills=k.n_spills,
                                  shared_bytes=k.metadata.shared, warps=k.metadata.num_warps,
                                  ptx_target=next(line.strip() for line in k.asm["ptx"].splitlines()
                                                  if line.strip().startswith(".target")),
                                  uses_wgmma="wgmma." in k.asm["ptx"],
                                  uses_tma="cp.async.bulk.tensor" in k.asm["ptx"])
                             for k in kernels]
                record = dict(mode=mode, batch=batch, query_length=query_length, kv_length=length,
                              group=group, kv_heads=2, query_heads=group * 2, value_rank=rank,
                              config=list(config), is_default=config == default, correct=correct,
                              config_fields=["segments", "tile", "warps", "stages"] if mode == "decode"
                              else ["block_m", "block_n", "warps", "stages"],
                              relative_l2=error, resources=resources, samples_ms=[])
                records.append(record)
                result["records"].append(record)
                print(json.dumps(record), flush=True)
                save()
            for repeat in range(args.repeats):
                order = list(range(len(configs)))
                rng.shuffle(order)
                for index in order:
                    if records[index]["correct"]:
                        ms = triton.testing.do_bench_cudagraph(lambda: run(configs[index]), rep=args.bench_ms)
                        records[index]["samples_ms"].append(ms)
                        records[index]["median_ms"] = statistics.median(records[index]["samples_ms"])
                save()
            valid = [r for r in records if r["correct"]]
            if valid:
                best = min(valid, key=lambda r: r["median_ms"])
                if not args.smoke:
                    finalists = sorted(valid, key=lambda r: r["median_ms"])[:2]
                    finalists += [r for r in valid if r["is_default"] and r not in finalists]
                    for record in finalists:
                        record["confirm_samples_ms"] = []
                    for repeat in range(5):
                        rng.shuffle(finalists)
                        for record in finalists:
                            ms = triton.testing.do_bench_cudagraph(
                                lambda: run(record["config"]), rep=100)
                            record["confirm_samples_ms"].append(ms)
                            record["confirm_median_ms"] = statistics.median(record["confirm_samples_ms"])
                    best = min(finalists, key=lambda r: r["confirm_median_ms"])
                timing = "median_ms" if args.smoke else "confirm_median_ms"
                baseline = next((r for r in valid if r["is_default"]), None)
                result["summaries"].append(dict(
                    mode=mode, batch=batch, group=group, value_rank=rank,
                    best_config=best["config"], best_ms=best[timing],
                    default_ms=baseline[timing] if baseline else None,
                    speedup=baseline[timing] / best[timing] if baseline else None,
                    best_by_warps={str(w): min((r["median_ms"] for r in valid if r["config"][2] == w),
                                               default=None) for w in args.warps}))
                print("BEST", json.dumps(best), flush=True)
                save()
            del data, expected
    result["status"] = "complete" if all(r["correct"] for r in result["records"]) else "correctness_failed"
    save()
    return int(result["status"] != "complete")


if __name__ == "__main__":
    sys.exit(main())
