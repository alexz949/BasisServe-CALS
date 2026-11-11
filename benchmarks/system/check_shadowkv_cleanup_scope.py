"""Which cleanup calls an upstream-ShadowKV benchmark arm really makes (not timed; one arm per process).

Counts gc.collect, torch.cuda.empty_cache, torch.cuda.synchronize and torch.svd by where they are called from:
inside the cache's get_svd, inside its H2D, inside the benchmark's reset, or elsewhere. One request at 8192
tokens: reset, prefill, H2D, two decode steps, reset. The arm is built by the benchmark's own
build_shadowkv_upstream, so the wrapper under test is the one the timed runs use.

Upstream get_svd (ShadowKVCache_CPU) calls, once per layer and before its torch.svd: synchronize, gc.collect,
empty_cache, synchronize. An arm with skip_in_get_svd must skip exactly the named calls there and nothing else.
Last, an exception inside get_svd (upstream fails on an invalid argument, in a worker thread) must leave
gc.collect and torch.cuda.empty_cache bound to what they were.

  python benchmarks/system/check_shadowkv_cleanup_scope.py --arm shadowkv_upstream_nohk_v128 --model SNAPSHOT \
      --inputs results/system_benchmarks/tp1_offline/inputs/prompts.safetensors --output DIR
"""

import argparse
import gc
import json
from pathlib import Path
import sys
import threading

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/system"))
LENGTH, LAYERS = 8192, 32
H2D_CALLS = dict(collect=2, empty_cache=2, synchronize=3)  # upstream ShadowKVCache_CPU.H2D


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.batch, args.length, args.output_tokens, args.mode = 1, LENGTH, 8, "request"
    import bench_v2q8_tp as bench
    from benchmarks.system import audit_tp1_request as frozen
    frozen.MODEL = args.model
    torch.cuda.set_device(0)

    counts, phase = {}, ["build"]

    def counted(owner, name):
        original = getattr(owner, name)

        def call(*positional, **keywords):
            counts[f"{phase[0]}:{name}"] = counts.get(f"{phase[0]}:{name}", 0) + 1
            return original(*positional, **keywords)

        setattr(owner, name, call)
        return call

    bound = {name: counted(owner, name) for owner, name in (
        (gc, "collect"), (torch.cuda, "empty_cache"), (torch.cuda, "synchronize"), (torch, "svd"))}

    def within(label, function):
        def call(*positional, **keywords):
            previous, phase[0] = phase[0], label
            output = function(*positional, **keywords)
            phase[0] = previous
            return output
        return call

    runtime = bench.build_shadowkv_upstream(args, 1, bench.Timers())
    phase[0] = "reset"
    runtime.reset()
    cache = runtime.llama.kv_cache
    # Outside the benchmark's own wrapper, so a call it turns into a no-op never reaches the counters.
    cache.get_svd = within("get_svd", cache.get_svd)
    cache.H2D = within("h2d", cache.H2D)
    prompts = load_file(str(args.inputs))["cohort_0"][:LENGTH][None].long().cuda()
    phase[0] = "prefill_outside_get_svd"
    token, _ = runtime.prefill_batch(prompts)
    phase[0] = "placement_outside_h2d"
    runtime.place()
    phase[0] = "decode"
    for step in range(2):
        token = runtime.forward(token, LENGTH + step).argmax(-1)[:, None]
    phase[0] = "reset"
    runtime.reset()
    phase[0] = "after"

    # An exception inside get_svd must not leave the cleanup functions replaced.
    seen = []
    threading.excepthook = lambda failure: seen.append(failure.exc_type.__name__)
    worker = threading.Thread(target=runtime.llama.kv_cache.get_svd, args=(None, 0))
    worker.start()
    worker.join()
    restored = gc.collect is bound["collect"] and torch.cuda.empty_cache is bound["empty_cache"]

    skip = bench.ARMS[args.arm].get("skip_in_get_svd", ())
    inside = {name: counts.get(f"get_svd:{name}", 0) for name in ("collect", "empty_cache", "synchronize", "svd")}
    expected = dict(collect=0 if "gc_collect" in skip else LAYERS, empty_cache=0 if "empty_cache" in skip else LAYERS,
                    synchronize=2 * LAYERS, svd=LAYERS)
    h2d = {name: counts.get(f"h2d:{name}", 0) for name in ("collect", "empty_cache", "synchronize")}
    result = dict(
        arm=args.arm, skip_in_get_svd=list(skip), context_tokens=LENGTH, layers=LAYERS,
        calls_inside_get_svd=inside, expected_inside_get_svd=expected,
        calls_inside_h2d=h2d, expected_inside_h2d=H2D_CALLS,
        gc_collect_calls_in_reset=counts.get("reset:collect", 0), all_counts=counts,
        exception_inside_get_svd=seen, bindings_restored_after_exception=restored,
        passed=bool(inside == expected and h2d == H2D_CALLS
                    and counts.get("reset:collect", 0) > 0 and seen == ["AttributeError"] and restored),
        scope="Call counts only, at 8192 tokens; not timed. Python's automatic garbage collector is untouched.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
