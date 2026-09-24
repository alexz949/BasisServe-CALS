"""Isolated execution-configuration experiments against the frozen TP1 router."""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
from types import SimpleNamespace

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "results/system_benchmarks/tp1_sparse_local/frozen_source"
OUTPUT = ROOT / "results/system_benchmarks/tp1_router_config"
CONFIGS = {"w8u1": (8, 1), "w16u1": (16, 1), "w8u2": (8, 2)}


def build(name):
    warps, unroll = CONFIGS[name]
    folder = OUTPUT / "source" / name
    folder.mkdir(parents=True, exist_ok=True)
    sources = []
    for filename in ("mapped_host_paged_attention.cpp",
                     "mapped_host_paged_attention.cu", "conditional_router_page32.cu"):
        content = (SNAPSHOT / "basisserve/kernels/csrc" / filename).read_text()
        if filename == "conditional_router_page32.cu":
            # Change only the register kernel and its matching launch geometry.
            old = "constexpr int kRegisterWarps = 8;"
            assert content.count(old) == 2
            content = content.replace(old, f"constexpr int kRegisterWarps = {warps};")
            old = "#pragma unroll 1\n  for (int feature_group = 0; feature_group < 16; ++feature_group)"
            assert content.count(old) == 1
            content = content.replace(old, old.replace("unroll 1", f"unroll {unroll}"))
        path = folder / filename
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
        sources.append(str(path))
    return load(
        name=f"tp1_router_config_{name}", sources=sources,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math", "--ptxas-options=-v",
            "-DBASIS_VALUE_DIM=128", "-DBASIS_GQA=4", "-DBASIS_PAGE_SIZE=32",
            "-DBASIS_BASE_RANK=16", "-DBASIS_RESIDUAL_RANK=16",
        ], verbose=True,
    )


def inputs(length, batch=1, heads=8):
    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    angles = torch.randn(length, 64, device="cuda")
    return (
        rand(batch, heads * 4, 1, 128), rand(batch, heads, length, 16) * .2,
        rand(batch, heads, length, 16) * .2, rand(heads, 16, 128) * .2,
        rand(heads, 128) * .1, rand(heads * 4, 128, 16) * .2,
        angles.cos().bfloat16(), angles.sin().bfloat16(),
    )


def call(module, tensors, code, scores):
    module.conditional_router_page_lse(*tensors, code, scores, 128 ** -.5, False)


def timing(fn, iterations):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


@torch.inference_mode()
def micro(args):
    torch.set_num_threads(2)
    torch.manual_seed(127)
    torch.backends.cuda.matmul.allow_tf32 = False
    modules = {name: build(name) for name in CONFIGS}
    checks, rows = [], []
    lengths = [1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 2047, 4096]
    if not args.smoke:
        lengths += [16384, 65536, 131072]
    for seed in (127, 311, 509):
        torch.manual_seed(seed)
        for length in lengths:
            batch, heads = (2, 3) if length < 4096 else (1, 8)
            tensors = inputs(length, batch, heads)
            pages = (length + 31) // 32
            code = torch.empty(batch, heads, 4, 16, device="cuda", dtype=torch.bfloat16)
            scores = {name: torch.empty(batch, heads, 4, pages, device="cuda") for name in modules}
            selected = {name: torch.empty(batch, heads, min(62, pages), device="cuda", dtype=torch.long)
                        for name in modules}
            for name, module in modules.items():
                call(module, tensors, code, scores[name])
                module.select_fixed_group_max_pages(scores[name], selected[name], min(62, pages), 1, False)
            for name in CONFIGS:
                if name == "w8u1":
                    continue
                row = dict(seed=seed, length=length, config=name,
                           max_abs=float((scores[name] - scores["w8u1"]).abs().max()),
                           scores_exact=torch.equal(scores[name], scores["w8u1"]),
                           selected_exact=torch.equal(selected[name], selected["w8u1"]))
                checks.append(row)
                print(json.dumps({"validation": row}), flush=True)
                assert row["scores_exact"] and row["selected_exact"]
            if seed == 127 and (length >= 16384 or args.smoke and length == 4096):
                for name in ("w16u1", "w8u2"):
                    samples = {"w8u1": [], name: []}
                    for repeat in range(3):
                        for config in ("w8u1", name, name, "w8u1"):
                            samples[config].append(timing(
                                lambda: call(modules[config], tensors, code, scores[config]),
                                20 if args.smoke else 200,
                            ))
                    baseline = statistics.median(samples["w8u1"])
                    candidate = statistics.median(samples[name])
                    row = dict(length=length, config=name, baseline_ms=baseline,
                               candidate_ms=candidate, speedup=baseline / candidate,
                               reduction_percent=100 * (1 - candidate / baseline), samples=samples)
                    rows.append(row)
                    print(json.dumps({"timing": row}), flush=True)
    result = dict(status="complete", environment="basis", gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, checks=checks, timings=rows,
                  scope="Synthetic warmed single-layer eager CUDA-event timing, includes query projection. Not model decode.")
    (OUTPUT / ("smoke.json" if args.smoke else "micro.json")).write_text(json.dumps(result, indent=2) + "\n")


def decode(args, remaining):
    sys.path.insert(0, str(SNAPSHOT))
    path = SNAPSHOT / "benchmarks/system/bench_tp1_sparse_full_v6.py"
    spec = importlib.util.spec_from_file_location("frozen_tp1_benchmark", path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    extension = build(args.config)
    if "--validate" in remaining:
        reference = build("w8u1")
        audited = set()

        def audited_route(*values):
            extension.conditional_router_page_lse(*values)
            output = values[9]
            pointer = output.data_ptr()
            if pointer not in audited:
                expected = torch.empty_like(output)
                reference.conditional_router_page_lse(*values[:9], expected, *values[10:])
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                audited.add(pointer)
                print(json.dumps(dict(router_audit="exact", layers=len(audited))), flush=True)

        routed = SimpleNamespace(**{k: v for k, v in vars(extension).items() if not k.startswith("__")})
        routed.conditional_router_page_lse = audited_route
    else:
        routed = extension
    original = harness._load_extension
    harness._load_extension = lambda **kw: routed if kw["page_size"] == 32 else original(**kw)
    sys.argv = [str(path), *remaining]
    harness.main()
    if "--validate" in remaining:
        assert len(audited) == 32


def paired(args):
    assert args.config != "w8u1"
    output = OUTPUT / "offload" if args.basis_storage == "offload" else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    suffix = "offload_full_reuse" if args.basis_storage == "offload" else "local"
    trials = []
    for length in (65536, 131072):
        for name, repeat in (("w8u1", 0), (args.config, 0), (args.config, 1), ("w8u1", 1)):
            tag = f"{name}_{length}_r{repeat}"
            folder = output / "decode"
            folder.mkdir(exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--phase", "decode",
                       "--config", name, "--mode", "optimized", "--storage", args.basis_storage,
                       "--routing", "full", "--length", str(length), "--warmup-steps", "16",
                       "--measure-steps", "128", "--validate", "--repeat", str(repeat),
                       "--tag", name, "--output-root", str(folder)]
            if args.basis_storage == "offload":
                command.append("--key-reuse")
            (folder / f"{tag}.command.json").write_text(json.dumps(command, indent=2) + "\n")
            print(json.dumps({"command": command}), flush=True)
            with (folder / f"{tag}.log").open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            result_path = folder / f"{name}_optimized_{suffix}_t{length}_r{repeat}/benchmark.json"
            trial = dict(config=name, length=length, repeat=repeat, returncode=completed.returncode,
                         storage=args.basis_storage, result=str(result_path))
            trials.append(trial)
            (output / "paired.json").write_text(json.dumps(trials, indent=2) + "\n")
            assert completed.returncode == 0
            result = json.loads(result_path.read_text())
            print(json.dumps(dict(completed=tag, median_ms=result["decode_cuda_median_ms"],
                                  finite=result["all_logits_finite"])), flush=True)
    rows = []
    for length in (65536, 131072):
        baseline = None
        for name in ("w8u1", args.config):
            results = [json.loads(Path(t["result"]).read_text()) for t in trials
                       if t["length"] == length and t["config"] == name]
            if baseline is None:
                baseline = results[0]["argmax_tokens"]
            row = dict(length=length, config=name, storage=args.basis_storage,
                       median_ms=statistics.median([x for r in results for x in r["decode_cuda_ms"]]),
                       repeat_medians=[r["decode_cuda_median_ms"] for r in results],
                       finite=all(r["all_logits_finite"] for r in results),
                       validated_layers=[r["validated_layers"] for r in results],
                       peak_gpu_gib=max(r["peak_allocated_gib"] for r in results),
                       last_step_hit_fractions=[r["last_step_key_hit_fraction"] for r in results],
                       argmax_exact=all(r["argmax_tokens"] == baseline for r in results))
            rows.append(row)
    (output / "decode_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"summary": rows}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("micro", "decode", "paired"), default="micro")
    parser.add_argument("--config", choices=CONFIGS, default="w8u1")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--basis-storage", choices=("local", "offload"), default="local")
    args, remaining = parser.parse_known_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "micro":
        assert not remaining
        micro(args)
    elif args.phase == "decode":
        decode(args, remaining)
    else:
        assert not remaining
        paired(args)


if __name__ == "__main__":
    main()
