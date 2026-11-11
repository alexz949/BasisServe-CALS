"""Native N96 WGMMA correctness and bounded single-GH200 microbenchmark."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import socket
import statistics
import sys

import tilelang
import torch
import triton
import triton.testing

from basisserve.kernels.diffkv_prefill import diffkv_prefill
from basisserve.kernels.prefill_native96 import arguments, compile_prefill
from evaluation.benchmark_hopper_warps import make_case, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261110)
    rng = random.Random(20261110)
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path("basisserve/kernels/prefill_native96.py"),
               Path("basisserve/kernels/diffkv_prefill.py"),
               Path("evaluation/benchmark_hopper_warps.py")]
    result = dict(status="running", command=sys.argv, environment=os.environ.get("CONDA_DEFAULT_ENV"),
                  host=socket.gethostname(), job=os.environ.get("SLURM_JOB_ID"),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  tilelang=tilelang.__version__, triton=triton.__version__,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  scope="Attention only, TP4-local two KV heads, BF16 paged input, warm CUDA graphs; no model or communication",
                  config=dict(query_head_rows=64, kv_tile=128, warps=4, stages=2), records=[])

    def save():
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    for group in (4, 8):
        for batch, qlen, length in ((2, 17, 257), (1, 129, 146), (1, 4096, 4096)):
            label = f"g{group}-b{batch}-q{qlen}-k{length}"
            print(f"START {label}", flush=True)
            data = make_case(batch, qlen, length, group, 96)
            expected = reference(data, qlen)
            runners = {"triton128": lambda: diffkv_prefill(**data)}
            evidence = {}
            for width in (96, 128):
                name = f"tilelang{width}"
                print(f"COMPILE {label} {name}", flush=True)
                kernel = compile_prefill(data, qlen, width)
                stem = args.output / f"{label}-{name}"
                kernel.export_sources(kernel_path=str(stem.with_suffix(".cu")))
                kernel.export_ptx(str(stem.with_suffix(".ptx")))
                ptx = stem.with_suffix(".ptx").read_text()
                shapes = sorted(set(re.findall(r"wgmma\.mma_async[^\s;]+", ptx)))
                assert any(f"m64n{width}k16" in shape for shape in shapes), shapes
                if width == 128:
                    assert not any("m64n96k16" in shape for shape in shapes)
                evidence[name] = shapes
                call_args = arguments(data)
                runners[name] = lambda kernel=kernel, call_args=call_args: kernel(*call_args)
            record = dict(case=label, group=group, batch=batch, query_length=qlen,
                          kv_length=length, instructions=evidence, measurements={})
            for name, run in runners.items():
                data["out"].fill_(float("nan"))
                run()
                torch.cuda.synchronize()
                actual = data["out"].float()
                error = ((actual - expected).norm() / expected.norm()).item()
                maximum = (actual - expected).abs().max().item()
                correct = bool(torch.isfinite(actual).all() and error < 0.005 and
                               torch.allclose(actual, expected, atol=0.008, rtol=0.02))
                record["measurements"][name] = dict(correct=correct, relative_l2=error,
                                                    max_abs=maximum, milliseconds=[])
                print(f"CHECK {label} {name} correct={correct} rel_l2={error:.6f} max_abs={maximum:.6f}", flush=True)
                assert correct
            for repeat in range(5):
                names = list(runners)
                rng.shuffle(names)
                for name in names:
                    elapsed = triton.testing.do_bench_cudagraph(runners[name], rep=30)
                    record["measurements"][name]["milliseconds"].append(elapsed)
            for measurement in record["measurements"].values():
                measurement["median_ms"] = statistics.median(measurement["milliseconds"])
            native = record["measurements"]["tilelang96"]["median_ms"]
            record["speedup_vs_triton128"] = record["measurements"]["triton128"]["median_ms"] / native
            record["speedup_vs_tilelang128"] = record["measurements"]["tilelang128"]["median_ms"] / native
            result["records"].append(record)
            save()
            print(json.dumps(record), flush=True)
    result["status"] = "passed"
    save()
    print("PASS native96 smoke", flush=True)


if __name__ == "__main__":
    main()
