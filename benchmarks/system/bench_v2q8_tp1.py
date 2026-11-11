"""V2Q8 TP1 benchmark: one arm, one prompt cohort, one mode, fresh process.

Arms (Llama-3.1-8B-Instruct, BF16, one GPU, B1):
  paper Table 2 protocol (shared compressed V96 checkpoint):
    fullk_v96, lrqk_v96, shadowkv_v96   quality code path with preallocated storage (v2q8_tp1_v96)
    basis_v96                           B16R16 Page4, 2048 routed + 64 recent, no sink, exact K on mapped host
    basis_v96_reuse                     same, Base read from the rotated Value cache (no Base copy)
    basis_v96_resident                  same as basis_v96 with exact K on GPU
  native systems (dense Values, separately labelled):
    dense                               original model, GPU-resident K/V (archived TP1 dense runtime)
    basis_v128                          B16R16 Page32, 32 sink + 1952 routed + 64 recent, exact K on GPU
    lrqk_native, shadowkv_native        unmodified upstream LRQK / ShadowKV (archived TP1 factories)

Modes, each after a same-shape warmup request and a fresh request state:
  request  prefill + first token, placement, 127 decode calls; synchronized phase boundaries only
  steady   prefill, then 16 conditioning + 128 measured decode calls with CUDA events and synchronized wall
  profile  separately instrumented: attention-module time per decode call and prompt-specific construction
Model loading, JIT builds and upfront cache allocation happen before the request and are excluded.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/system"))

V96_ARMS = ("fullk_v96", "lrqk_v96", "shadowkv_v96")
BASIS_ARMS = {  # value width, page size, routed tokens (sink included), sink tokens, K on host, Value reuse
    "basis_v96": (96, 4, 2048, 0, True, False),
    "basis_v96_reuse": (96, 4, 2048, 0, True, True),
    "basis_v96_resident": (96, 4, 2048, 0, False, False),
    "basis_v128": (128, 32, 1984, 32, False, False),
}
NATIVE_ARMS = ("dense", "lrqk_native", "shadowkv_native")
ARMS = V96_ARMS + tuple(BASIS_ARMS) + NATIVE_ARMS


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def progress(args, phase):
    data = dict(arm=args.arm, mode=args.mode, phase=phase, context=args.length, cohort=args.cohort,
                time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (args.output / "progress.json").write_text(json.dumps(data) + "\n")
    print(json.dumps(data), flush=True)


def memory_state():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                device_used=total - free, process_rss=psutil.Process().memory_info().rss)


def build_runtime(args):
    from benchmarks.system import audit_tp1_request as frozen

    frozen_args = SimpleNamespace(method=args.arm.split("_")[0], length=args.length,
                                  output_tokens=args.output_tokens, profile_components=False, validate=False)
    if args.arm in V96_ARMS:
        from benchmarks.system import v2q8_tp1_v96
        return v2q8_tp1_v96.build(args.arm, model_path=frozen.MODEL, value_checkpoint=args.value_checkpoint,
                                  capacity=args.length + args.output_tokens)
    if args.arm == "dense":
        runtime = frozen.local_model(frozen_args, None)
        cells = dict(zip(runtime.prefill.__code__.co_freevars, runtime.prefill.__closure__))
        runtime.model = cells["model"].cell_contents
        runtime.configuration.update(value="original V128 and Wo")
        return runtime
    if args.arm == "lrqk_native":
        runtime = frozen.lrqk_model(frozen_args, None)
        runtime.configuration.update(value="original V128 and Wo", implementation="upstream tenghuilee/LRQK@caf16293")
        return runtime
    if args.arm == "shadowkv_native":
        runtime = frozen.shadow_model(frozen_args, None)
        runtime.configuration.update(value="original V128 and Wo",
                                     implementation="upstream ByteDance-Seed/ShadowKV@e51904cd ShadowKVCache_CPU")
        return runtime

    from table3_page_router import build as build_router
    from transformers import AutoModelForCausalLM
    from benchmarks.system import v2q8_basis

    value_dim, page_size, routed, sink, offload, reuse = BASIS_ARMS[args.arm]
    sys.path.insert(0, str(frozen.SNAPSHOT))
    harness = frozen.module_from_file(
        "frozen_request_harness", frozen.SNAPSHOT / "benchmarks/system/bench_tp1_sparse_full_v6.py")
    router, build = build_router(frozen.SNAPSHOT, args.build_root / f"p{page_size}_v{value_dim}_t{routed}_s{sink}",
                                 page_size, value_dim=value_dim, routed_tokens=routed, sink_tokens=sink)
    model = AutoModelForCausalLM.from_pretrained(frozen.MODEL, dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation="sdpa").to("cuda").eval()
    bank = args.bank_v96 if value_dim == 96 else args.bank_v128
    layers = v2q8_basis.install(
        model, harness=harness, router=router, bank=bank,
        value_checkpoint=args.value_checkpoint if value_dim == 96 else None,
        batch=1, capacity=args.length + args.output_tokens, page_size=page_size, routed_tokens=routed,
        sink_tokens=sink, offload=offload, value_reuse=reuse, build_root=args.build_root)

    def forward(tokens):
        start = layers[0].length
        positions = torch.arange(start, start + tokens.shape[1], device=tokens.device)[None]
        output = model.model(input_ids=tokens, position_ids=positions, use_cache=False)
        return model.lm_head(output.last_hidden_state[:, -1:])

    def reset():
        for layer in layers:
            layer.reset()

    return SimpleNamespace(prefill=forward, decode=forward, place=lambda: None, reset=reset,
        state=lambda: layers, model=model, configuration=dict(
            value=f"compressed V{value_dim} (paper Table 2 shared checkpoint)" if value_dim == 96 else "original V128 and Wo",
            storage="mapped pinned host K with GPU K slots, GPU V" if offload else "GPU-local K/V",
            router="B16R16 full-scan page router", bank=str(bank), page_size=page_size,
            routed_tokens_including_sink=routed, sink_tokens=sink, recent_tokens=64,
            physical_support=routed + 64,
            exact_keys="mapped pinned host memory with persistent GPU Key slots" if offload else "GPU-resident",
            base_code="first 16 coordinates of the rotated Value cache" if reuse else "separate Base cache",
            router_build_anchors=build))


def fresh(runtime):
    torch.cuda.synchronize()
    runtime.reset()
    gc.collect()
    torch.manual_seed(20260924)
    torch.cuda.synchronize()


def request(runtime, prompt, count, ready_marker):
    """One fixed-length greedy request. Memory is read after each timed interval closes."""
    from benchmarks.system.bench_tp1_offline_request import generate

    torch.cuda.reset_peak_memory_stats()
    measured = generate(runtime, prompt, count, ready_marker=ready_marker)
    measured["memory_after_request"] = memory_state()
    return measured


def steady(runtime, prompt, warmup, steps):
    memory = {}
    torch.cuda.reset_peak_memory_stats()
    begin = sync_time()
    logits = runtime.prefill(prompt)
    token = logits[:, -1].argmax(dim=-1, keepdim=True)
    runtime.place()
    prefill_seconds = sync_time() - begin
    memory["after_prefill_and_placement"] = memory_state()
    torch.cuda.reset_peak_memory_stats()
    cuda_ms, wall_ms, generated, finite = [], [], [token], True
    for index in range(warmup + steps):
        if index == warmup:
            memory["decode_ready"] = memory_state()
            torch.cuda.reset_peak_memory_stats()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = sync_time()
        start_event.record()
        logits = runtime.decode(token)
        end_event.record()
        stop = sync_time()
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(token)
        finite = finite and bool(torch.isfinite(logits).all())
        if index >= warmup:
            cuda_ms.append(start_event.elapsed_time(end_event))
            wall_ms.append(1000 * (stop - start))
    memory["decode_end"] = memory_state()
    ordered = sorted(wall_ms)
    return dict(cuda_ms=cuda_ms, wall_ms=wall_ms, cuda_median_ms=statistics.median(cuda_ms),
        cuda_mean_ms=statistics.fmean(cuda_ms), wall_median_ms=statistics.median(wall_ms),
        wall_mean_ms=statistics.fmean(wall_ms), wall_p95_ms=ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        wall_stdev_ms=statistics.stdev(wall_ms), decode_tokens_per_second=1000 / statistics.fmean(wall_ms),
        all_logits_finite=finite, generated_tokens=torch.cat(generated, dim=1).tolist(),
        prefill_and_placement_seconds=prefill_seconds, memory=memory)


def attention_modules(runtime):
    return [layer.self_attn for layer in runtime.model.model.layers] if hasattr(runtime, "model") else []


def profile(runtime, prompt, warmup, steps, arm):
    """Instrumented, separate from primary timing: attention-module CUDA time per decode call, the
    method's own component events (BasisKV), and synchronized prompt-specific construction time."""
    construction = []
    patched = []
    if arm in ("lrqk_v96", "shadowkv_v96"):
        from basisserve.checkpoint import c1_lrqk_qwen3, c1_shadowkv_qwen3
        owner, name = (c1_lrqk_qwen3, "LRQKState") if arm == "lrqk_v96" else (c1_shadowkv_qwen3, "C1ShadowKVState")
        original = getattr(owner, name)

        def timed(*positional, **keywords):
            begin = sync_time()
            state = original(*positional, **keywords)
            construction.append(sync_time() - begin)
            return state

        setattr(owner, name, timed)
        patched.append((owner, name, original))
    components = {}
    if arm == "shadowkv_native":
        fresh(runtime)  # the official cache is recreated here; time its build steps on that object
        cache = runtime.state()
        for name, label in (("get_svd", "svd_and_factor_storage"), ("prefill_kv_cache", "landmarks_outliers_and_cache"),
                            ("H2D", "post_prefill_placement")):
            original = getattr(cache, name)

            def timed(*positional, _original=original, _label=label, **keywords):
                begin = sync_time()
                output = _original(*positional, **keywords)
                components[_label] = components.get(_label, 0.0) + sync_time() - begin
                return output

            setattr(cache, name, timed)
        logits = runtime.prefill(prompt)
        runtime.place()
        runtime.decode(logits[:, -1].argmax(dim=-1, keepdim=True))
        return dict(construction_components_seconds=components, construction_seconds=sum(components.values()),
                    attention_module_ms=[], attention_module_mean_ms=None, components=[],
                    scope="Separate synchronized build profile of the official cache (get_svd includes upstream's "
                          "own gc.collect/empty_cache housekeeping); not subtractable from uninstrumented timings.")
    modules = attention_modules(runtime)
    events = {}

    def before(module, positional, keywords):
        if getattr(module, "_v2q8_profile", False):
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            pair[0].record()
            events.setdefault(id(module), []).append(pair)

    def after(module, positional, keywords, output):
        if getattr(module, "_v2q8_profile", False):
            events[id(module)][-1][1].record()

    handles = [m.register_forward_pre_hook(before, with_kwargs=True) for m in modules] + [
        m.register_forward_hook(after, with_kwargs=True) for m in modules]
    basis = arm in BASIS_ARMS
    if basis:
        names = ["attention_block", "cache_append", "router_scan", "page_selection", "sparse_attention"] + (
            ["key_refresh"] if BASIS_ARMS[arm][4] else [])
        for layer in runtime.state():
            layer.profile_events = {n: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                                    for n in names}
    fresh(runtime)
    logits = runtime.prefill(prompt)
    token = logits[:, -1].argmax(dim=-1, keepdim=True)
    runtime.place()
    torch.cuda.synchronize()
    attention_ms, components, generated = [], [], [token]
    for index in range(warmup + steps):
        record = index >= warmup
        for module in modules:
            module._v2q8_profile = record
        if basis:
            for layer in runtime.state():
                layer.profile_components = record
        logits = runtime.decode(token)
        torch.cuda.synchronize()
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(token)
        if record:
            attention_ms.append(sum(pairs[-1][0].elapsed_time(pairs[-1][1]) for pairs in events.values()))
            if basis:
                times = [layer.component_times() for layer in runtime.state()]
                components.append({key: sum(row[key] for row in times) for key in times[0]})
    for handle in handles:
        handle.remove()
    for module in modules:
        module._v2q8_profile = False
    if basis:
        for layer in runtime.state():
            layer.profile_components = False
    for owner, name, original in patched:
        setattr(owner, name, original)
    return dict(attention_module_ms=attention_ms,
        attention_module_mean_ms=statistics.fmean(attention_ms) if attention_ms else None,
        components=components, construction_seconds_per_layer=construction,
        construction_seconds=sum(construction) if construction else None,
        generated_tokens=torch.cat(generated, dim=1).tolist(),
        scope="Instrumented separately; hooks on every attention module, synchronized construction timers. "
              "Not subtractable from the uninstrumented request or steady timings.")


def inventory(runtime):
    """Bytes by tensor, once per backing storage, reachable from the model, the attention modules and the cache."""
    seen, rows = set(), []

    def plain(tensor):
        """The tensor that owns real storage: a DTensor's local shard; wrapper subclasses without storage
        (FakeTensor, FunctionalTensor from torch.compile caches) are skipped."""
        from torch.distributed.tensor import DTensor
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        return tensor if type(tensor) in (torch.Tensor, torch.nn.Parameter) else None

    def add(path, tensor):
        tensor = plain(tensor)
        if tensor is None:
            return
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        if pointer == 0 or pointer in seen:
            return
        seen.add(pointer)
        rows.append(dict(path=path, device="gpu" if tensor.is_cuda else "host", shape=list(tensor.shape),
                         dtype=str(tensor.dtype), pinned=None if tensor.is_cuda else bool(tensor.is_pinned()),
                         backing_bytes=storage.nbytes()))

    def walk(obj, path, depth):
        if torch.is_tensor(obj):
            add(path, obj)
        elif depth <= 0 or isinstance(obj, (str, bytes, int, float, bool, type(None))):
            return
        elif isinstance(obj, dict):
            for key, value in obj.items():
                walk(value, f"{path}[{key}]", depth - 1)
        elif isinstance(obj, (list, tuple)):
            for index, value in enumerate(obj):
                walk(value, f"{path}[{index}]", depth - 1)
        elif hasattr(obj, "__dict__") and not isinstance(obj, torch.nn.Module):
            for key, value in vars(obj).items():
                walk(value, f"{path}.{key}", depth - 1)

    model = getattr(runtime, "model", None)
    if model is not None:
        for name, parameter in model.named_parameters():
            add(f"parameter:{name}", parameter)
        for name, buffer in model.named_buffers():
            add(f"buffer:{name}", buffer)
        for index, module in enumerate(attention_modules(runtime)):
            walk({k: v for k, v in vars(module).items() if not k.startswith("_")}, f"attention[{index}]", 4)
    walk(runtime.state(), "state", 6)
    gpu = sum(r["backing_bytes"] for r in rows if r["device"] == "gpu")
    host = sum(r["backing_bytes"] for r in rows if r["device"] == "host")
    unowned = 0
    for candidate in gc.get_objects():
        candidate = plain(candidate) if torch.is_tensor(candidate) else None
        if candidate is not None and candidate.is_cuda and candidate.untyped_storage().data_ptr() not in seen:
            seen.add(candidate.untyped_storage().data_ptr())
            unowned += candidate.untyped_storage().nbytes()
    return dict(rows=sorted(rows, key=lambda r: -r["backing_bytes"])[:400], gpu_backing_bytes=gpu,
                host_backing_bytes=host, unowned_gpu_bytes=unowned, torch_allocated_bytes=torch.cuda.memory_allocated())


def environment(prompt):
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader"],
        text=True).strip()
    return dict(command=sys.argv, python=platform.python_version(), torch=torch.__version__,
        cuda_runtime=torch.version.cuda, nccl=".".join(map(str, torch.cuda.nccl.version())),
        conda_environment=Path(sys.prefix).name, host=platform.node(), gpus=gpu,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), cpu_affinity=sorted(os.sched_getaffinity(0)),
        torch_threads=torch.get_num_threads(), git_head=subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        prompt_tokens=prompt.shape[1])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--mode", choices=("request", "steady", "profile"), required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--cohort", type=int, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--bank-v96", type=Path, required=True)
    parser.add_argument("--bank-v128", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    warmup, steps, profile_steps = (2, 4, 4) if args.smoke else (16, 128, 32)
    args.output_tokens = (8 if args.smoke else 128) if args.mode == "request" else warmup + steps + 1
    assert args.length + args.output_tokens <= 131072
    args.output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    from v2q8_affinity import bind_to_gpu_node
    # One GPU per process; the physical index picks the half of the shared NUMA node.
    placement = bind_to_gpu_node(0, int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260924)
    progress(args, "model_load_and_build")
    started = time.perf_counter()
    runtime = build_runtime(args)
    setup_seconds = time.perf_counter() - started
    prompt = load_file(str(args.inputs))[f"cohort_{args.cohort}"][:args.length][None].long().cuda()
    assert prompt.shape[1] == args.length
    result = dict(status="measured", arm=args.arm, mode=args.mode, context_tokens=args.length, cohort=args.cohort,
                  output_tokens=args.output_tokens, configuration=runtime.configuration,
                  environment=environment(prompt), cpu_placement=placement, excluded_setup_seconds=setup_seconds,
                  memory_model_loaded=memory_state(),
                  # Upstream LRQK sets float32 matmul precision to 'high' at import; record what is actually in force.
                  numerics_after_build=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                                            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
    progress(args, "same_shape_warmup_request")
    fresh(runtime)
    warm = request(runtime, prompt, args.output_tokens, ready_marker=False)
    result["warmup_tokens_finite"] = warm["final_logits_finite"]
    del warm
    fresh(runtime)
    progress(args, f"measured_{args.mode}")
    if args.mode == "request":
        measured = request(runtime, prompt, args.output_tokens, ready_marker=args.arm == "lrqk_native")
        valid = measured["final_logits_finite"] and measured["actual_decode_calls"] == args.output_tokens - 1
    elif args.mode == "steady":
        measured = steady(runtime, prompt, warmup, steps)
        valid = measured["all_logits_finite"]
    else:
        measured = profile(runtime, prompt, warmup, profile_steps, args.arm)
        valid = bool(measured["attention_module_ms"]) or bool(measured.get("construction_components_seconds")) \
            or not attention_modules(runtime)
    result["measurement"] = measured
    if args.mode == "steady":
        result["inventory"] = inventory(runtime)
    result["status"] = "complete" if valid else "validation_failed"
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    progress(args, result["status"])


if __name__ == "__main__":
    main()
