"""Table 3 steady decode with matched Key placement, phase memory and a state inventory.

Runs one arm for one prompt cohort on the archived Table 3 runtime:

  dense_resident          archived dense baseline, exact K/V on GPU
  basis_resident          archived B16R16/Page32 full-scan runtime, exact K/V on GPU
  basis_resident_nobase   same, Base read from the first 16 Value coordinates
  basis_k_offload_nobase  same, exact Keys in mapped pinned host memory with GPU Key slots
"""

import argparse
import gc
import hashlib
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
from safetensors.torch import load_file, save_file

from table3_page_router import build as build_page_router

ARMS = ("dense_resident", "basis_resident", "basis_resident_nobase", "basis_k_offload_nobase")
CATEGORIES = {
    "key_cache": "exact historical Keys",
    "host_key": "exact historical Keys",
    "value_cache": "Values",
    "base_cache": "Value-derived routing state",
    "residual_cache": "Key residual routing state",
    "slot_key_cache": "selected-Key slots",
    "slot_resident": "slot maps",
    "slot_lookup": "slot maps",
    "selected_slots": "slot maps",
    "slot_missing": "slot maps",
    "slot_counts": "slot maps",
    "router_output": "page scores",
    "fine_router_output": "page scores",
    "selected_pages": "selected indices",
    "support_ids": "selected indices",
    "candidate_ids": "selected indices",
    "page_offsets": "index tables",
    "token_range": "index tables",
    "shared_rope": "RoPE tables",
    "base_left": "router factors",
    "base_right": "router factors",
    "base_bias": "router factors",
    "residual_encoder": "router factors",
    "residual_query": "router factors",
}


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def progress(args, phase):
    data = dict(arm=args.arm, mode=args.mode, phase=phase, context=args.length, cohort=args.cohort,
                page_size=args.page_size)
    (args.output / "progress.json").write_text(json.dumps(data) + "\n")
    print(json.dumps(data), flush=True)


def memory_state():
    """PyTorch and device memory now, plus PyTorch peaks since the last reset."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(), device_used=total - free,
                process_rss=psutil.Process().memory_info().rss)


def read_base_from_value(layer):
    """Rotate Value so its first 16 coordinates are the Base code, then drop the Base copy."""
    basis, upper = torch.linalg.qr(layer.base_left.double(), mode="complete")
    value_weight = layer.v_proj.weight.data.view(8, 128, -1)
    value_weight.copy_(basis.transpose(1, 2) @ value_weight.double())
    output_weight = layer.o_proj.weight.data.view(-1, 32, 128)
    output_weight.copy_(torch.einsum("ohd,hde->ohe", output_weight.double(),
                                     basis.repeat_interleave(4, dim=0)))
    layer.base_right.copy_(upper[:, :16] @ layer.base_right.double())
    layer.base_left.copy_(torch.eye(128, 16, device="cuda", dtype=torch.float64).expand(8, -1, -1))
    layer.base_cache = layer.value_cache[..., :16]


def build_runtime(args, frozen):
    if args.arm in ("dense_resident", "basis_resident"):
        runtime = frozen.local_model(SimpleNamespace(
            method=args.arm.split("_")[0], length=args.length, output_tokens=args.output_tokens,
            profile_components=False, validate=False), None)
        # The archived factory keeps the model only inside its forward closure.
        cells = dict(zip(runtime.prefill.__code__.co_freevars, runtime.prefill.__closure__))
        runtime.model = cells["model"].cell_contents
        return runtime
    config = frozen.module_from_file(
        "router_configuration", frozen.ROOT / "benchmarks/system/bench_tp1_router_config.py")
    sys.path.insert(0, str(frozen.SNAPSHOT))
    harness = frozen.module_from_file(
        "frozen_request_harness", frozen.SNAPSHOT / "benchmarks/system/bench_tp1_sparse_full_v6.py")
    from transformers import AutoModelForCausalLM

    offload = args.arm == "basis_k_offload_nobase"
    build = None
    if args.page_size:
        # The archived harness reads its page geometry from these module constants.
        harness.PAGE_SIZE = args.page_size
        harness.ROUTED_PAGES = 1984 // args.page_size
        router, build = build_page_router(
            frozen.SNAPSHOT, frozen.ROOT / "results/system_benchmarks/table3_page_router"
            / f"p{args.page_size}", args.page_size, value_dim=128, routed_tokens=1984, sink_tokens=32)
    else:
        router = config.build("w16u1")
    attention = harness._load_extension(value_dim=128, queries_per_kv=4, page_size=1,
                                        base_rank=16, residual_rank=16)
    model = AutoModelForCausalLM.from_pretrained(
        frozen.MODEL, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa").to("cuda").eval()
    layers = harness._install_sparse_attention(
        model, capacity=args.length + args.output_tokens,
        storage="offload" if offload else "local", mode="optimized", router_extension=router,
        fine_router_extension=None, attention_extension=attention, routing="full",
        key_reuse=offload, slot_extension=harness._load_slot_extension() if offload else None,
        postprocess_extension=None, profile_components=False, validate=False)
    for layer in layers:
        read_base_from_value(layer)

    def forward(tokens):
        start = layers[0].length
        positions = torch.arange(start, start + tokens.shape[1], device=tokens.device)[None]
        output = model.model(input_ids=tokens, position_ids=positions, use_cache=False)
        return model.lm_head(output.last_hidden_state[:, -1:])

    def reset():
        for layer in layers:
            layer.length = 0
            if offload:
                layer.slot_resident.fill_(-1)
                layer.slot_lookup.fill_(-1)

    return SimpleNamespace(prefill=forward, decode=forward, place=lambda: None, reset=reset,
        state=lambda: layers, model=model, configuration=dict(
            storage="mapped pinned host K with GPU K slots, GPU V" if offload else "GPU-local K/V",
            prompt_specific_fit=False, routing="full_scan", validation_enabled=False,
            base_code="first 16 coordinates of the rotated Value cache",
            page_size=harness.PAGE_SIZE, routed_pages=harness.ROUTED_PAGES,
            pinned_tokens=32, recent_tokens=harness.RECENT_TOKENS,
            router="page-size-general register router" if build else "archived w16u1 register router",
            router_build=build))


def steady(runtime, prompt, warmup, steps, release_cache):
    """Archived steady protocol; memory is read only at phase boundaries, outside step timing."""
    layers = runtime.state()
    slots = getattr(layers[0], "slot_counts", None) is not None
    memory = {}
    torch.cuda.reset_peak_memory_stats()
    begin_prefill = sync_time()
    logits = runtime.prefill(prompt)
    token = logits[:, -1].argmax(dim=-1, keepdim=True)
    runtime.place()
    prefill_seconds = sync_time() - begin_prefill
    memory["prefill"] = memory_state()
    if release_cache:
        torch.cuda.empty_cache()
        memory["prefill_cache_released"] = memory_state()
    torch.cuda.reset_peak_memory_stats()
    cuda_ms, wall_ms, components, generated, slot_steps = [], [], [], [], []
    finite = True
    for index in range(warmup + steps):
        if index == warmup:
            memory["conditioning"] = memory_state()
            torch.cuda.reset_peak_memory_stats()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = sync_time()
        begin.record()
        logits = runtime.decode(token)
        end.record()
        stop = sync_time()
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(token)
        finite = finite and bool(torch.isfinite(logits).all())
        if index >= warmup:
            cuda_ms.append(begin.elapsed_time(end))
            wall_ms.append(1000 * (stop - start))
            if layers[0].profile_components:
                times = [layer.component_times() for layer in layers]
                components.append({key: sum(row[key] for row in times) for key in times[0]})
                if slots:
                    hits, valid = torch.stack([layer.slot_counts for layer in layers]).sum(
                        dim=(0, 1, 2)).tolist()
                    slot_steps.append(dict(selected=valid, hits=hits, fetched=valid - hits,
                                           fetched_bytes=(valid - hits) * 256))
    memory["decode"] = memory_state()
    return dict(cuda_ms=cuda_ms, wall_ms=wall_ms, cuda_median_ms=statistics.median(cuda_ms),
        cuda_mean_ms=statistics.fmean(cuda_ms), wall_median_ms=statistics.median(wall_ms),
        wall_mean_ms=statistics.fmean(wall_ms), all_logits_finite=finite,
        generated_tokens=torch.cat(generated, dim=1).tolist(), components=components,
        slot_steps=slot_steps, prefill_and_placement_seconds=prefill_seconds, memory=memory)


def tensors_of(owner):
    """Named tensors held directly by an attention module, including tuples and dicts."""
    found = []
    held = {name: value for name, value in vars(owner).items() if not name.startswith("_")}
    for name, value in {**held, **owner._buffers}.items():
        values = value.values() if isinstance(value, dict) else value if isinstance(
            value, (tuple, list)) else [value]
        found.extend((name, item) for item in values if torch.is_tensor(item))
    return found


def inventory(model, layers):
    """Bytes by state, counted once per backing storage, plus CUDA tensors nothing here owns."""
    seen, rows = {}, {}

    def add(category, name, tensor, active_fraction):
        pointer = tensor.untyped_storage().data_ptr()
        backing = tensor.untyped_storage().nbytes()
        device = "gpu" if tensor.is_cuda else "host"
        row = rows.setdefault((device, category, name), dict(
            device=device, category=category, name=name, shape=list(tensor.shape),
            dtype=str(tensor.dtype), pinned=bool(tensor.is_pinned()) if not tensor.is_cuda else None,
            tensors=0, logical_bytes=0, backing_bytes=0, active_bytes=0, aliases=None,
            lifetime="persistent, allocated before the request"))
        row["tensors"] += 1
        row["logical_bytes"] += tensor.numel() * tensor.element_size()
        if pointer in seen:
            row["aliases"] = seen[pointer]
        else:
            seen[pointer] = name
            row["backing_bytes"] += backing
            row["active_bytes"] += int(backing * active_fraction)

    for parameter in model.parameters():
        add("model parameters", "parameters", parameter, 1.0)
    capacity = layers[0].capacity
    for layer in layers:
        for name, tensor in tensors_of(layer):
            add(CATEGORIES.get(name, "attention and router workspaces"), name, tensor,
                layer.length / capacity if capacity in tensor.shape else 1.0)
    unowned = {}
    for candidate in gc.get_objects():
        if torch.is_tensor(candidate) and candidate.is_cuda:
            pointer = candidate.untyped_storage().data_ptr()
            if pointer not in seen:
                unowned[pointer] = (candidate.untyped_storage().nbytes(), list(candidate.shape))
    largest = sorted(unowned.values(), reverse=True)[:8]
    gpu = sum(row["backing_bytes"] for row in rows.values() if row["device"] == "gpu")
    return dict(rows=list(rows.values()), capacity_tokens=capacity, active_tokens=layers[0].length,
        gpu_backing_bytes=gpu,
        host_backing_bytes=sum(r["backing_bytes"] for r in rows.values() if r["device"] == "host"),
        unowned_gpu_bytes=sum(size for size, _ in unowned.values()),
        largest_unowned_gpu=[dict(bytes=size, shape=shape) for size, shape in largest],
        torch_allocated_bytes=torch.cuda.memory_allocated())


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def selected_keys(layer):
    """Exact Keys the sparse attention read this step, in support order, invalid rows zeroed."""
    ids = layer.support_ids
    source, index = (layer.slot_key_cache, layer.selected_slots) if layer.key_reuse else (
        layer.key_cache, ids)
    keys = torch.gather(source, 2, index.clamp_min(0)[..., None].expand(-1, -1, -1, 128))
    return keys.masked_fill(ids[..., None] < 0, 0)


def check(runtime, prompt, forced):
    """Teacher-forced decode recording what placement must not change."""
    layers = runtime.state()
    runtime.prefill(prompt)
    logits, attention, support, keys = [], [], [], []
    for token in forced:
        logits.append(runtime.decode(token.view(1, 1)).float().cpu())
        attention.append(torch.stack([layer.attention_output.float() for layer in layers]).cpu())
        support.append([digest(layer.support_ids) for layer in layers])
        keys.append([digest(selected_keys(layer)) for layer in layers])
    length = layers[0].length
    caches = dict(
        key=[digest((layer.host_key if layer.key_cache is None else layer.key_cache)[:, :, :length])
             for layer in layers],
        value=[digest(layer.value_cache[:, :, :length]) for layer in layers],
        residual=[digest(layer.residual_cache[:, :, :length]) for layer in layers])
    tensors = dict(logits=torch.cat(logits, dim=1)[0], attention=torch.stack(attention))
    return tensors, dict(cache_tokens=length, support_sha256=support, selected_key_sha256=keys,
                         cache_sha256=caches, all_logits_finite=bool(torch.isfinite(tensors["logits"]).all()))


def environment(args, prompt):
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader"],
        text=True).strip()
    return dict(command=sys.argv, python=platform.python_version(), torch=torch.__version__,
        cuda_runtime=torch.version.cuda, conda_environment=Path(sys.prefix).name,
        host=platform.node(), gpu=gpu, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads(),
        slurm_job=os.environ.get("SLURM_JOB_ID"), runtime=str(args.runtime),
        prompt_sha256=digest(prompt.to(torch.int32)), prompt_tokens=prompt.shape[1])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--mode", choices=("steady", "check"), required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--cohort", type=int, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--page-size", type=int, choices=(4, 8, 16, 32),
                        help="Use the page-size-general router; omit for the archived Page32 router")
    args = parser.parse_args()
    args.runtime = args.runtime.resolve()
    sys.path.insert(0, str(args.runtime))
    from benchmarks.system import audit_tp1_request as frozen
    from benchmarks.system.bench_tp1_offline_request import fresh, generate

    warmup, steps, profile_steps, check_steps = (2, 4, 4, 4) if args.smoke else (16, 128, 32, 16)
    args.output_tokens = warmup + steps + 1 if args.mode == "steady" else check_steps + 1
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260924)
    progress(args, "model_load")
    runtime = build_runtime(args, frozen)
    layers = runtime.state()
    inputs = load_file(str(args.inputs))
    prompt = inputs[f"cohort_{args.cohort}"][:args.length][None].long().cuda()
    result = dict(status="measured", arm=args.arm, mode=args.mode, context_tokens=args.length,
        cohort=args.cohort, page_size=args.page_size or 32, conditioning_steps=warmup,
        measured_steps=steps,
        profile_steps=profile_steps, configuration=runtime.configuration,
        environment=environment(args, prompt))

    if args.mode == "check":
        progress(args, "teacher_forced_check")
        fresh(runtime)
        forced = inputs[f"cohort_{(args.cohort + 1) % 3}"][:check_steps].long().cuda()
        tensors, record = check(runtime, prompt, forced)
        save_file(tensors, str(args.output / "check.safetensors"))
        result.update(record, forced_tokens=forced.tolist(), status="complete")
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        progress(args, "complete")
        return

    result["memory_model_loaded"] = memory_state()
    progress(args, "same_shape_runtime_warmup")
    fresh(runtime)
    warm = generate(runtime, prompt, args.output_tokens)
    progress(args, "fresh_request_state")
    fresh(runtime)
    del warm
    progress(args, "measured_steady")
    measured = steady(runtime, prompt, warmup, steps, release_cache=False)
    result["measurement"] = measured
    result["inventory"] = inventory(runtime.model, layers)
    (args.output / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")

    progress(args, "separate_attention_profile")
    names = ["attention_block", "cache_append"] + (["dense_attention"] if args.arm == "dense_resident" else [
        "router_scan", "page_selection", "sparse_attention"]) + (
        ["key_refresh"] if args.arm == "basis_k_offload_nobase" else [])
    for layer in layers:
        layer.profile_events = {name: (torch.cuda.Event(enable_timing=True),
                                       torch.cuda.Event(enable_timing=True)) for name in names}
        layer.profile_components = True
    fresh(runtime)
    profile = steady(runtime, prompt, warmup, profile_steps, release_cache=True)
    profile["prefix_tokens_match"] = profile["generated_tokens"] == [
        measured["generated_tokens"][0][:warmup + profile_steps]]
    result["attention_profile"] = profile
    valid = measured["all_logits_finite"] and profile["all_logits_finite"] and profile["prefix_tokens_match"]
    result["status"] = "complete" if valid else "validation_failed"
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    progress(args, result["status"])


if __name__ == "__main__":
    main()
