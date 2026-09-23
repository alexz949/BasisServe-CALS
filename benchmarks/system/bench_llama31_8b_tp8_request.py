"""Fresh-process TP8 ShadowKV request and steady-decode trial."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import gc
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

import flash_attn
import pynvml
import torch
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from basisserve.core.llama31_8b_tp8_combined import (
    BASE_RANK, RESIDUAL_RANK, VALUE_RANK, install_tp8_combined,
    load_decode_postprocess_extension,
)
from basisserve.core.shadowkv_tp8_attention import cache_state_bytes, install_tp8_shadowkv
from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.mapped_host_paged_attention import _load_extension
from benchmarks.system.bench_llama31_8b_tp8_combined import (
    DEFAULT_FACTORS, DEFAULT_MODEL, DEFAULT_ROUTER, _bind_cpu, _choose,
    _load_prompts, _percentile,
)
from benchmarks.system.numa_memory import bind_host_allocations
from benchmarks.system.register_router import compile_register
from benchmarks.system.tp8_request_timing import (
    RequestPhaseRecorder, clock_id, summarize_request_timeline,
)
from evaluation import official_shadowkv_cpu


def git_text(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                          capture_output=True).stdout.strip()


def persistent_state(arm, layers, cache):
    if arm == "shadowkv":
        assert cache.v_cache_cpu.is_pinned()
        return cache_state_bytes(cache)
    names = ("key_cache", "value_cache") if arm == "dense" else (
        "value_cache", "residual_cache", "slot_key_cache", "slot_resident", "slot_lookup"
    )
    state = {name: sum(getattr(layer, name).numel() * getattr(layer, name).element_size()
                       for layer in layers) for name in names}
    if arm == "basis_joint":
        state["host_exact_key_bytes"] = sum(
            layer.host_key.numel() * layer.host_key.element_size() for layer in layers
        )
    return state


def nvml_process_bytes(handle):
    rows = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    own = [row for row in rows if row.pid == os.getpid()]
    assert len(own) == 1
    return int(own[0].usedGpuMemory)


def memory_snapshot(handle, recorder=None):
    if recorder is not None:
        recorder.capture_peak()
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": (
                recorder.peak_allocated_bytes if recorder is not None
                else torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": (
                recorder.peak_reserved_bytes if recorder is not None
                else torch.cuda.max_memory_reserved()),
            "nvml_process_bytes": nvml_process_bytes(handle)}


def extensions_for_arm(arm, rank):
    if arm == "shadowkv":
        return {}
    if arm == "dense":
        _load_extension(value_dim=128, queries_per_kv=4, page_size=32)
        return {}
    router = _load_extension(value_dim=VALUE_RANK, queries_per_kv=4,
                             page_size=32, base_rank=BASE_RANK,
                             residual_rank=RESIDUAL_RANK)
    build_root = Path("/tmp/basisserve_tp8_full_router")
    if rank == 0:
        register = compile_register(build_root, BASE_RANK, 8)
    dist.barrier()
    if rank != 0:
        register = compile_register(build_root, BASE_RANK, 8)
    return {"router_extension": router, "full_router_extension": register,
            "postprocess_extension": load_decode_postprocess_extension()}


def make_model(args, rank, length, decode_steps, cache_class, extensions, phase=None):
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        distributed_config=DistributedConfig(tp_size=8), local_files_only=True).eval()
    communicator = None
    cache = None
    if args.arm == "shadowkv":
        layers, cache = install_tp8_shadowkv(
            model, cache_class, batch=args.batch, length=length,
            decode_steps=decode_steps, phase=phase)
    else:
        if args.arm == "basis_joint":
            communicator = FeatureRaggedCommunicator.from_distributed(
                device=torch.device("cuda", rank))
        layers = install_tp8_combined(
            model, arm=args.arm, batch=args.batch, capacity=length + decode_steps,
            factor_root=args.factor_root if communicator is not None else None,
            router_root=args.router_root if communicator is not None else None,
            rank=rank, communicator=communicator, **extensions)
    return model, layers, cache, communicator


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dense", "basis_joint", "shadowkv"), required=True)
    parser.add_argument("--mode", choices=("request", "steady"), required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--factor-root", type=Path, default=DEFAULT_FACTORS)
    parser.add_argument("--router-root", type=Path, default=DEFAULT_ROUTER)
    parser.add_argument("--upstream", type=Path,
                        default=Path("/workspace/BasisServe-CALS/external/ShadowKV"))
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--cohort", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert 4096 <= args.length <= 130048 and args.batch in (1, 4, 8)
    assert args.length + 144 <= 131072
    assert args.model.is_dir() and args.tokens.is_file()
    if args.arm == "basis_joint":
        assert args.factor_root.is_dir() and args.router_root.is_dir()
    if args.arm == "shadowkv":
        assert args.upstream.is_dir()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 8 and rank == local_rank
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    numa_node = _bind_cpu(local_rank)
    bind_host_allocations(numa_node)
    dist.init_process_group("nccl", timeout=timedelta(minutes=90), device_id=device)
    assert dist.get_world_size() == 8 and torch.cuda.get_device_capability() == (8, 9)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"rank{rank}.log"

    def emit(status, **fields):
        row = {"status": status, "rank": rank, "time_utc": datetime.now(timezone.utc).isoformat(),
               **fields}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    emit("starting", arm=args.arm, mode=args.mode, prompt_tokens=args.length,
         batch=args.batch, cohort=args.cohort, numa_node=numa_node)
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
    extensions = extensions_for_arm(args.arm, rank)
    cache_class = None
    upstream_commit = None
    if args.arm == "shadowkv":
        official_shadowkv_cpu.ROOT = args.upstream.resolve()
        cache_class = official_shadowkv_cpu.load_cache_class()
        upstream_commit = subprocess.run(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
            check=True, text=True, capture_output=True).stdout.strip()
    prompts, cohort = _load_prompts(args.tokens, args.prompt_manifest,
                                    batch=args.batch, length=args.length)
    prompts = prompts.to(device)

    # Warm the actual projection, prefill, retrieval, and decode backends at B.
    # A fresh model/cache is then loaded for the recorded request.
    emit("warmup_start", tokens=4096, decode_steps=1)
    model, layers, cache, communicator = make_model(
        args, rank, 4096, 1, cache_class, extensions)
    warm = model(input_ids=prompts[:, :4096], use_cache=False, logits_to_keep=1)
    warm_token = _choose(warm.logits, model.config.vocab_size)
    del warm
    if cache is not None:
        cache.H2D()
    warm = model(input_ids=warm_token,
                 position_ids=torch.full((args.batch, 1), 4096, device=device, dtype=torch.long),
                 use_cache=False, logits_to_keep=1)
    assert bool(torch.isfinite(warm.logits).all())
    torch.cuda.synchronize()
    dist.barrier()
    del warm, warm_token, model, layers, cache
    if communicator is not None:
        communicator.close()
    del communicator
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    emit("warmup_complete")

    decode_steps = 128 if args.mode == "request" else 144
    recorder = RequestPhaseRecorder() if args.arm == "shadowkv" and args.mode == "request" else None
    model, layers, cache, communicator = make_model(
        args, rank, args.length, decode_steps, cache_class, extensions,
        phase=recorder.phase if recorder is not None else None)
    preallocated = memory_snapshot(nvml_handle)
    positions = [torch.full((args.batch, 1), args.length + step,
                            device=device, dtype=torch.long)
                 for step in range(decode_steps)]
    emit("prefill_start")
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    start_ns = time.perf_counter_ns()
    output = model(input_ids=prompts, use_cache=False, logits_to_keep=1)
    token = _choose(output.logits, model.config.vocab_size)
    finite = [torch.isfinite(output.logits).all()]
    generated = [token.clone()]
    del output
    torch.cuda.synchronize()
    first_ns = time.perf_counter_ns()
    prefill_memory = memory_snapshot(nvml_handle, recorder)
    emit("prefill_complete", elapsed_ms=(first_ns - start_ns) / 1e6)
    if cache is not None:
        if recorder is None:
            cache.H2D()
        else:
            with recorder.phase("other_prepare", 32, -1):
                cache.H2D()
    torch.cuda.synchronize()
    ready_ns = time.perf_counter_ns()
    ready_memory = memory_snapshot(nvml_handle, recorder)
    state = persistent_state(args.arm, layers, cache)
    emit("representation_ready", elapsed_ms=(ready_ns - start_ns) / 1e6)

    request_timeline = None
    steady = None
    if args.mode == "request":
        torch.cuda.reset_peak_memory_stats()
        emit("decode_start", output_tokens=128)
        for step in range(127):
            output = model(input_ids=token, position_ids=positions[step],
                           use_cache=False, logits_to_keep=1)
            token = _choose(output.logits, model.config.vocab_size)
            finite.append(torch.isfinite(output.logits).all())
            generated.append(token.clone())
            del output
        torch.cuda.synchronize()
        final_ns = time.perf_counter_ns()
        decode_memory = memory_snapshot(nvml_handle)
        if recorder is not None:
            recorder.capture_peak()
        request_timeline = {
            "rank": rank, "clock_id": clock_id(), "arm": args.arm,
            "output_tokens": 128, "request_start_ns": start_ns,
            "first_output_ns": first_ns, "representation_ready_ns": ready_ns,
            "final_output_ns": final_ns,
            "phases": recorder.phases if recorder is not None else [],
        }
    else:
        emit("conditioning_start", steps=16)
        for step in range(16):
            output = model(input_ids=token, position_ids=positions[step],
                           use_cache=False, logits_to_keep=1)
            token = _choose(output.logits, model.config.vocab_size)
            finite.append(torch.isfinite(output.logits).all())
            generated.append(token.clone())
            del output
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(128)]
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        emit("steady_decode_start", conditioning_steps=16, measured_steps=128)
        decode_wall_start = time.perf_counter()
        for index, (begin, end) in enumerate(events):
            begin.record()
            output = model(input_ids=token, position_ids=positions[16 + index],
                           use_cache=False, logits_to_keep=1)
            token = _choose(output.logits, model.config.vocab_size)
            finite.append(torch.isfinite(output.logits).all())
            generated.append(token.clone())
            end.record()
            del output
        torch.cuda.synchronize()
        decode_wall_seconds = time.perf_counter() - decode_wall_start
        decode_memory = memory_snapshot(nvml_handle)
        local_steps = torch.tensor([begin.elapsed_time(end) for begin, end in events],
                                   device=device, dtype=torch.float64)
        replica_wall = torch.tensor(decode_wall_seconds, device=device, dtype=torch.float64)
        dist.all_reduce(local_steps, op=dist.ReduceOp.MAX)
        dist.all_reduce(replica_wall, op=dist.ReduceOp.MAX)
        steps = local_steps.cpu().tolist()
        steady = {
            "mean_ms_per_step": statistics.fmean(steps),
            "p50_ms": statistics.median(steps),
            "p95_ms": _percentile(steps, 0.95),
            "aggregate_tokens_per_second": args.batch * 128 / float(replica_wall),
            "replica_wall_seconds": float(replica_wall),
            "step_ms_max_rank": steps,
            "conditioning_steps": 16, "measured_steps": 128,
        }

    assert bool(torch.stack(finite).all())
    assert len(generated) == (128 if args.mode == "request" else 145)
    assert all(layer.length == args.length + len(generated) - 1 for layer in layers)
    if args.arm == "shadowkv":
        assert all(len(layer.factor_records) == 1 for layer in layers)
        assert sum(row["svd_calls"] for layer in layers for row in layer.factor_records) == args.batch * 4
    generated_ids = torch.cat(generated, dim=-1).cpu().tolist()
    emit("complete", mode=args.mode, generated_tokens=len(generated))

    record = {
        "schema": "basisserve.llama31_8b.tp8_request.v1", "status": "complete",
        "arm": args.arm, "mode": args.mode, "tp": 8, "dp": 1, "pp": 1,
        "rank": rank, "local_rank": local_rank, "cohort": args.cohort,
        "prompt_tokens": args.length, "batch": args.batch,
        "numa_node": numa_node, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "dtype": "bfloat16", "tf32": False, "execution": "eager",
        "routing_mode": "full_scan_b16r16_persistent_slots" if args.arm == "basis_joint" else None,
        "shadow_rank": 160 if args.arm == "shadowkv" else None,
        "shadow_chunk_size": 8 if args.arm == "shadowkv" else None,
        "shadow_sparse_budget": 2048 if args.arm == "shadowkv" else None,
        "request_timeline": request_timeline, "steady_decode": steady,
        "memory": {"preallocated": preallocated, "prefill_end": prefill_memory,
                   "representation_ready": ready_memory, "decode_end": decode_memory,
                   "peak_request_allocated_bytes": (
                       max(ready_memory["peak_allocated_bytes"],
                           decode_memory["peak_allocated_bytes"])
                       if args.mode == "request" else None),
                   "owner_gather_svd_peak_allocated_bytes": (
                       recorder.owner_temporary_peak_allocated_bytes
                       if recorder is not None else None),
                   "pinned_host_value_bytes": (
                       cache.v_cache_cpu.numel() * cache.v_cache_cpu.element_size()
                       if cache is not None else 0),
                   "pinned_host_exact_key_bytes": (
                       state["host_exact_key_bytes"] if args.arm == "basis_joint" else 0)},
        "state": state,
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "generated_token_ids": generated_ids,
        "model": str(args.model.resolve()),
        "factor_root": str(args.factor_root.resolve()) if args.arm == "basis_joint" else None,
        "router_root": str(args.router_root.resolve()) if args.arm == "basis_joint" else None,
        "upstream": str(args.upstream.resolve()) if args.arm == "shadowkv" else None,
        "upstream_commit": upstream_commit,
        "tokens": str(args.tokens.resolve()), "prompt_cohort": cohort,
        "metadata": {
            "git_commit": git_text("rev-parse", "HEAD"),
            "git_status": git_text("status", "--porcelain"),
            "command_line": sys.argv,
            "pytorch": torch.__version__, "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "transformers": transformers.__version__,
            "flash_attn": flash_attn.__version__,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "warmup": "separate 4096-token request at same batch, then fresh model/cache",
            "numa_binding": "requested; OS may reject strict host page binding",
        },
    }
    (args.output_dir / f"rank{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
    gathered = [None] * 8
    dist.all_gather_object(gathered, record)
    if rank == 0:
        assert len(gathered) == 8 and {item["rank"] for item in gathered} == set(range(8))
        assert all(item["generated_token_ids"] == gathered[0]["generated_token_ids"]
                   for item in gathered)
        replica = {"status": "complete", "arm": args.arm, "mode": args.mode,
                   "prompt_tokens": args.length, "batch": args.batch, "cohort": args.cohort,
                   "successful_ranks": 8, "rank_results": [f"rank{index}.json" for index in range(8)],
                   "memory_max_rank": {
                       milestone: max(item["memory"][milestone]["allocated_bytes"]
                                      for item in gathered)
                       for milestone in ("prefill_end", "representation_ready", "decode_end")},
                   "peak_request_allocated_max_rank_bytes": max(
                       item["memory"]["peak_request_allocated_bytes"] or 0 for item in gathered),
                   "owner_gather_svd_peak_max_rank_bytes": max(
                       item["memory"]["owner_gather_svd_peak_allocated_bytes"] or 0
                       for item in gathered),
                   "pinned_host_value_sum_bytes": sum(
                       item["memory"]["pinned_host_value_bytes"] for item in gathered),
                   "pinned_host_exact_key_sum_bytes": sum(
                       item["memory"]["pinned_host_exact_key_bytes"] for item in gathered)}
        if args.mode == "request":
            replica["request"] = summarize_request_timeline(
                [item["request_timeline"] for item in gathered])
            replica["request"]["output_tokens_per_second"] = (
                args.batch * 128000 / replica["request"]["total_request_ms"])
        else:
            replica["steady_decode"] = steady
        (args.output_dir / "replica.json").write_text(json.dumps(replica, indent=2) + "\n")
    dist.barrier()
    if communicator is not None:
        communicator.close()
    pynvml.nvmlShutdown()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
