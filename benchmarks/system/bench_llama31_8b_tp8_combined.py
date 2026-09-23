"""Fixed-batch TP8 benchmark for Dense, Joint-ALS V96, and Basis joint routing."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from basisserve.core.llama31_8b_tp8_combined import (
    BASE_RANK,
    DecodeBreakdownRecorder,
    HEAD_DIM,
    PAGE_SIZE,
    RECENT_TOKENS,
    RESIDUAL_RANK,
    ROUTED_PAGES,
    SUPPORT_TOKENS,
    TP_SIZE,
    VALUE_RANK,
    install_tp8_combined,
    load_decode_postprocess_extension,
)
from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.mapped_host_paged_attention import _load_extension
from benchmarks.system.common import command
from benchmarks.system.numa_memory import bind_host_allocations
from benchmarks.system.register_router import compile_register


DEFAULT_MODEL = Path(
    "/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
DEFAULT_FACTORS = Path("/workspace/runs/l31-cal128/tp8-combined-v96")
DEFAULT_ROUTER = Path("/workspace/runs/l31-cal128/router/ours_b16r16")
DEFAULT_TOKENS = Path("/workspace/runs/l31-cal128/calibration/windows.safetensors")


def _trace_hooks(model):
    handles = []

    def attach(module, name):
        scopes = []

        def before(module, inputs):
            scope = torch.profiler.record_function(name)
            scope.__enter__()
            scopes.append(scope)

        def after(module, inputs, output):
            scopes.pop().__exit__(None, None, None)

        handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))

    for index, layer in enumerate(model.model.layers):
        attach(layer.self_attn, f"layer_{index}.attention")
        attach(layer.mlp, f"layer_{index}.mlp")
        attach(layer.mlp.gate_proj, f"layer_{index}.mlp.gate")
        attach(layer.mlp.up_proj, f"layer_{index}.mlp.up")
        attach(layer.mlp.down_proj, f"layer_{index}.mlp.down_reduce")
    return handles


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _local_numa_node(local_rank: int) -> int:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    rows = {int(line.split(",")[0]): line.split(",")[1].strip().lower() for line in query}
    bus = rows[local_rank]
    if bus.startswith("00000000:"):
        bus = "0000:" + bus.split(":", 1)[1]
    node_path = Path("/sys/bus/pci/devices") / bus / "numa_node"
    node = int(node_path.read_text().strip())
    assert node >= 0
    return node


def _bind_cpu(local_rank: int) -> int:
    node = _local_numa_node(local_rank)
    text = (Path("/sys/devices/system/node") / f"node{node}" / "cpulist").read_text().strip()
    cpus = []
    for item in text.split(","):
        if "-" in item:
            left, right = map(int, item.split("-"))
            cpus.extend(range(left, right + 1))
        else:
            cpus.append(int(item))
    local_peers = TP_SIZE // 2
    peer = local_rank % local_peers
    assigned = cpus[peer::local_peers]
    os.sched_setaffinity(0, assigned)
    return node


def _choose(logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    values, ids = logits.max(dim=-1)
    world = dist.get_world_size()
    if logits.shape[-1] != vocab_size:
        assert logits.shape[-1] * world == vocab_size
        ids = ids + dist.get_rank() * logits.shape[-1]
        all_values = [torch.empty_like(values) for _ in range(world)]
        all_ids = [torch.empty_like(ids) for _ in range(world)]
        dist.all_gather(all_values, values)
        dist.all_gather(all_ids, ids)
        winner = torch.stack(all_values).argmax(0, keepdim=True)
        ids = torch.stack(all_ids).gather(0, winner).squeeze(0)
    return ids


def _load_prompts(
    path: Path, manifest_path: Path | None, *, batch: int, length: int
) -> tuple[torch.Tensor, dict]:
    windows = load_file(str(path))["input_ids"]
    assert windows.ndim == 2 and windows.shape[0] >= batch and windows.shape[1] >= length
    prompts = windows[:batch, :length].long().contiguous()
    assert all(torch.unique(prompts[index]).numel() > 1 for index in range(batch))
    if manifest_path is None:
        prompt_metadata = {
            "tokens_file": str(path),
            "tokens_file_bytes": path.stat().st_size,
            "sample_ids": [f"row_{index}" for index in range(batch)],
            "cohort_hash": None,
        }
    else:
        prompt_metadata = json.loads(manifest_path.read_text())
        assert prompt_metadata["status"] == "complete"
        assert prompt_metadata["prompt_tokens"] == length
        assert prompt_metadata["maximum_batch"] >= batch
        prompt_metadata = {
            **prompt_metadata,
            "sample_ids": prompt_metadata["sample_ids"][:batch],
            "tokens_file_bytes": path.stat().st_size,
            "integrity_validation": "shape_and_manifest_fields_only",
        }
    return prompts, prompt_metadata


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dense", "als_full", "basis_joint"), required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--factor-root", type=Path, default=DEFAULT_FACTORS)
    parser.add_argument("--router-root", type=Path, default=DEFAULT_ROUTER)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--conditioning-steps", type=int, default=16)
    parser.add_argument("--measure-steps", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--tag", default="full_scan")
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("results/system_benchmarks/llama31_8b_tp8_full_scan"),
    )
    args = parser.parse_args()
    assert args.length >= 4096 and args.length <= 130048
    assert args.batch > 0 and args.conditioning_steps >= 0 and args.measure_steps > 0
    assert not (args.trace and args.profile_components)
    assert args.length + args.conditioning_steps + args.measure_steps <= 131072
    assert args.model.is_dir() and args.tokens.is_file()
    if args.arm != "dense":
        assert args.factor_root.is_dir() and args.router_root.is_dir()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == TP_SIZE
    numa_node = _bind_cpu(local_rank)
    bind_host_allocations(numa_node)
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    assert dist.get_world_size() == TP_SIZE and torch.cuda.get_device_capability() == (8, 9)

    folder = args.output_root / (
        f"{args.tag}_{args.arm}_p{args.length}_b{args.batch}_r{args.repeat}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / f"rank{rank}.log"

    def emit(payload: dict) -> None:
        payload = {"rank": rank, **payload}
        line = json.dumps(payload, sort_keys=True)
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    emit(
        {
            "status": "starting",
            "arm": args.arm,
            "prompt_tokens": args.length,
            "batch": args.batch,
            "conditioning_steps": args.conditioning_steps,
            "measure_steps": args.measure_steps,
            "numa_node": numa_node,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        }
    )

    router_extension = None
    full_router_extension = None
    postprocess_extension = None
    if args.arm == "basis_joint":
        router_extension = _load_extension(
            value_dim=VALUE_RANK,
            queries_per_kv=4,
            page_size=32,
            base_rank=BASE_RANK,
            residual_rank=RESIDUAL_RANK,
        )
        router_build_root = Path("/tmp/basisserve_tp8_full_router")
        if rank == 0:
            full_router_extension = compile_register(router_build_root, BASE_RANK, 8)
        dist.barrier()
        if rank != 0:
            full_router_extension = compile_register(router_build_root, BASE_RANK, 8)
        postprocess_extension = load_decode_postprocess_extension()
    else:
        _load_extension(
            value_dim=128 if args.arm == "dense" else VALUE_RANK,
            queries_per_kv=4,
            page_size=32,
        )
    communicator = (
        None
        if args.arm == "dense"
        else FeatureRaggedCommunicator.from_distributed(device=torch.device("cuda", local_rank))
    )
    emit({"status": "extensions_ready"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        distributed_config=DistributedConfig(tp_size=TP_SIZE),
        local_files_only=True,
    ).eval()
    capacity = args.length + args.conditioning_steps + args.measure_steps
    breakdown = DecodeBreakdownRecorder() if args.profile_components else None
    layers = install_tp8_combined(
        model,
        arm=args.arm,
        batch=args.batch,
        capacity=capacity,
        factor_root=args.factor_root if args.arm != "dense" else None,
        router_root=args.router_root if args.arm != "dense" else None,
        rank=rank,
        communicator=communicator,
        router_extension=router_extension,
        full_router_extension=full_router_extension,
        postprocess_extension=postprocess_extension,
        breakdown=breakdown,
    )
    prompts, prompt_metadata = _load_prompts(
        args.tokens, args.prompt_manifest, batch=args.batch, length=args.length
    )
    prompts = prompts.cuda()
    emit(
        {
            "status": "model_ready",
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "host_exact_key_bytes": (
                32 * args.batch * capacity * HEAD_DIM * 2
                if args.arm == "basis_joint" else 0
            ),
        }
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    dist.barrier()
    prefill_wall_start = time.perf_counter()
    prefill_begin = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    prefill_begin.record()
    output = model(input_ids=prompts, use_cache=False, logits_to_keep=1)
    token = _choose(output.logits, model.config.vocab_size)
    prefill_end.record()
    prefill_end.synchronize()
    prefill_wall_seconds = time.perf_counter() - prefill_wall_start
    prefill_cuda_ms = prefill_begin.elapsed_time(prefill_end)
    prefill_peak_allocated = torch.cuda.max_memory_allocated()
    prefill_peak_reserved = torch.cuda.max_memory_reserved()
    assert bool(torch.isfinite(output.logits).all())
    del output, prompts
    emit(
        {
            "status": "prefill_complete",
            "prefill_cuda_ms": prefill_cuda_ms,
            "prefill_wall_seconds": prefill_wall_seconds,
            "peak_allocated_bytes": prefill_peak_allocated,
        }
    )

    positions = [
        torch.full(
            (args.batch, 1), args.length + step,
            device="cuda", dtype=torch.long,
        )
        for step in range(args.conditioning_steps + args.measure_steps)
    ]
    generated = [token.clone()]
    finite = []
    for step in range(args.conditioning_steps):
        output = model(
            input_ids=token,
            position_ids=positions[step],
            use_cache=False,
            logits_to_keep=1,
        )
        token = _choose(output.logits, model.config.vocab_size)
        finite.append(torch.isfinite(output.logits).all())
        generated.append(token.clone())
        del output

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(args.measure_steps)
    ]
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    if breakdown is not None:
        breakdown.enabled = True
    trace = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) if args.trace else None
    trace_handles = _trace_hooks(model) if args.trace else []
    if trace is not None:
        trace.start()
        dist.barrier()
    wall_start = time.perf_counter()
    for measured_step, (begin, end) in enumerate(events):
        step = args.conditioning_steps + measured_step
        begin.record()
        with torch.profiler.record_function(f"decode_step_{measured_step}") if args.trace else nullcontext():
            output = model(
                input_ids=token,
                position_ids=positions[step],
                use_cache=False,
                logits_to_keep=1,
            )
            token = _choose(output.logits, model.config.vocab_size)
            finite.append(torch.isfinite(output.logits).all())
            generated.append(token.clone())
        end.record()
        del output
    torch.cuda.synchronize()
    if breakdown is not None:
        breakdown.enabled = False
    wall_seconds = time.perf_counter() - wall_start
    if trace is not None:
        trace.stop()
        trace.export_chrome_trace(str(folder / f"rank{rank}.trace.json"))
    for handle in trace_handles:
        handle.remove()
    decode_peak_allocated = torch.cuda.max_memory_allocated()
    decode_peak_reserved = torch.cuda.max_memory_reserved()
    assert bool(torch.stack(finite).all())

    local_times = torch.tensor(
        [begin.elapsed_time(end) for begin, end in events],
        device="cuda", dtype=torch.float64,
    )
    dist.all_reduce(local_times, op=dist.ReduceOp.MAX)
    replica_wall = torch.tensor(wall_seconds, device="cuda", dtype=torch.float64)
    dist.all_reduce(replica_wall, op=dist.ReduceOp.MAX)
    times = local_times.cpu().tolist()
    generated_ids = torch.cat(generated, dim=-1).cpu().tolist()
    slot_counts = None
    if args.arm == "basis_joint":
        counts = torch.stack([layer.slot_counts for layer in layers]).cpu()
        slot_counts = {
            "last_step_hits": int(counts[..., 0].sum()),
            "last_step_valid": int(counts[..., 1].sum()),
            "last_step_hit_rate": float(
                counts[..., 0].sum().float() / counts[..., 1].sum().clamp_min(1)
            ),
        }

    result = {
        "schema": "basisserve.llama31_8b.tp8_combined_decode.v1",
        "status": "complete",
        "arm": args.arm,
        "tp": TP_SIZE,
        "dp": 1,
        "pp": 1,
        "rank": rank,
        "local_rank": local_rank,
        "numa_node": numa_node,
        "prompt_tokens": args.length,
        "batch": args.batch,
        "conditioning_steps": args.conditioning_steps,
        "measured_steps": args.measure_steps,
        "trace_enabled": args.trace,
        "repeat": args.repeat,
        "dtype": "bfloat16",
        "value_rank": None if args.arm == "dense" else VALUE_RANK,
        "base_rank": BASE_RANK if args.arm == "basis_joint" else None,
        "residual_rank": RESIDUAL_RANK if args.arm == "basis_joint" else None,
        "page_size": PAGE_SIZE if args.arm == "basis_joint" else None,
        "routed_pages": ROUTED_PAGES if args.arm == "basis_joint" else None,
        "recent_tokens": RECENT_TOKENS if args.arm == "basis_joint" else None,
        "physical_support": SUPPORT_TOKENS if args.arm == "basis_joint" else None,
        "key_placement": "pinned_host_with_gpu_slots" if args.arm == "basis_joint" else "gpu",
        "routing_mode": "full_scan_b16r16_persistent_slots" if args.arm == "basis_joint" else "full",
        "prefill": "full_context_flash_attention_with_arm_value_representation",
        "prefill_cuda_ms": prefill_cuda_ms,
        "prefill_wall_seconds": prefill_wall_seconds,
        "decode_step_ms": times,
        "decode_step_mean_ms": statistics.fmean(times),
        "decode_step_p50_ms": statistics.median(times),
        "decode_step_p95_ms": _percentile(times, 0.95),
        "decode_wall_seconds": float(replica_wall),
        "decode_tokens_per_second": (
            args.batch * args.measure_steps / float(replica_wall)
        ),
        "prefill_peak_allocated_bytes": prefill_peak_allocated,
        "prefill_peak_reserved_bytes": prefill_peak_reserved,
        "decode_resident_allocated_bytes": torch.cuda.memory_allocated(),
        "decode_resident_reserved_bytes": torch.cuda.memory_reserved(),
        "decode_peak_allocated_bytes": decode_peak_allocated,
        "decode_peak_reserved_bytes": decode_peak_reserved,
        "host_persistent_exact_key_bytes": (
            32 * args.batch * capacity * HEAD_DIM * 2
            if args.arm == "basis_joint" else 0
        ),
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "slot_counts": slot_counts,
        "component_profile": (
            breakdown.summary(args.measure_steps) if breakdown is not None else None
        ),
        "generated_token_ids": generated_ids,
        "model": str(args.model.resolve()),
        "factor_root": str(args.factor_root.resolve()) if args.arm != "dense" else None,
        "router_root": str(args.router_root.resolve()) if args.arm != "dense" else None,
        "tokens": str(args.tokens.resolve()),
        "prompt_manifest": (
            str(args.prompt_manifest.resolve()) if args.prompt_manifest is not None else None
        ),
        "prompt_cohort": prompt_metadata,
        "metadata": {
            "git_commit": command(["git", "rev-parse", "HEAD"])["stdout"].strip(),
            "git_status": command(["git", "status", "--porcelain"])["stdout"],
            "command_line": sys.argv,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "torch_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "source_hash_validation": "not_performed",
        },
    }
    result_path = folder / f"rank{rank}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    emit(
        {
            "status": "complete",
            "decode_step_mean_ms": result["decode_step_mean_ms"],
            "decode_tokens_per_second": result["decode_tokens_per_second"],
            "result": str(result_path),
        }
    )
    dist.barrier()
    if communicator is not None:
        communicator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
