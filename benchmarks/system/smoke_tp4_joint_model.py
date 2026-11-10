"""Check TP4/TP8 Llama Instruct prefill and decode with frozen V96 factors."""

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

from safetensors import safe_open
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from basisserve.core.llama_tp4_uniform_als import install_llama_tp4_uniform_als
from benchmarks.system.bench_tp1_sparse_full_v6 import (
    _load_postprocess_extension, _load_slot_extension,
)
from benchmarks.system.tp4_joint_sparse import install_tp4_joint_sparse
from benchmarks.system.two_stage_router import compile_fine


def choose(logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    values, ids = logits[:, -1].max(-1)
    if logits.shape[-1] != vocab_size:
        world = dist.get_world_size()
        assert logits.shape[-1] * world == vocab_size
        ids += dist.get_rank() * logits.shape[-1]
        all_values = [torch.empty_like(values) for _ in range(world)]
        all_ids = [torch.empty_like(ids) for _ in range(world)]
        dist.all_gather(all_values, values)
        dist.all_gather(all_ids, ids)
        winner = torch.stack(all_values).argmax(0, keepdim=True)
        ids = torch.stack(all_ids).gather(0, winner).squeeze(0)
    return ids[:, None]


def load_prompts(windows: Path, prompt_tokens: int, batch: int, cohort: int,
                 benchmark: bool) -> tuple[torch.Tensor, list[dict[str, int]]]:
    if benchmark:
        max_batch = {4096: 128, 16384: 16, 65536: 16, 130048: 16}[prompt_tokens]
        assert 1 <= batch <= max_batch and 0 <= cohort < 3
        chunks = 1 if prompt_tokens == 130048 else 131072 // prompt_tokens
        first_window = 32 if prompt_tokens in (4096, 16384) else 0
        indices = range(cohort * max_batch, cohort * max_batch + batch)
        locations = [
            {"window": first_window + index // chunks,
             "offset": (index % chunks) * prompt_tokens}
            for index in indices
        ]
    else:
        assert batch == 1 and cohort == 0
        locations = [{"window": 32, "offset": 0}]
    assert all(item["window"] < 48 for item in locations)
    with safe_open(str(windows), framework="pt", device="cpu") as stream:
        source = stream.get_slice("input_ids")
        prompt = torch.cat([
            source[item["window"]:item["window"] + 1,
                   item["offset"]:item["offset"] + prompt_tokens].long()
            for item in locations
        ])
    assert prompt.shape == (batch, prompt_tokens)
    return prompt, locations


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("dense", "als_full", "basis_joint"), default="dense")
    parser.add_argument("--factors", type=Path)
    parser.add_argument("--router", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cohort", type=int, default=0)
    parser.add_argument("--conditioning-steps", type=int, default=0)
    parser.add_argument("--measured-steps", type=int, default=0)
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--collective-only", action="store_true")
    args = parser.parse_args()

    assert Path(sys.prefix).name == "basis"
    assert args.model.name == "0e9e39f249a16976918f6564b8830bc894c89659"
    assert 1 <= args.prompt_tokens <= 130048
    assert args.decode_steps > 0
    benchmark = args.measured_steps > 0
    profiling = args.profile_steps > 0
    assert not profiling or (
        not benchmark and args.profile_steps == 32 and args.conditioning_steps == 16
        and args.arm == "basis_joint"
    )
    timed = benchmark or profiling
    assert not benchmark or (
        args.prompt_tokens in (4096, 16384, 65536, 130048)
        and args.conditioning_steps == 16 and args.measured_steps == 128
    )
    total_steps = (
        args.conditioning_steps + args.measured_steps + args.profile_steps
        if timed else args.decode_steps
    )
    assert args.prompt_tokens + total_steps <= 131072
    assert args.arm == "dense" or (args.factors is not None and args.factors.is_dir())
    assert args.arm != "basis_joint" or (
        args.router is not None and args.router.is_dir() and args.prompt_tokens >= 4096
    )
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_capability(rank) == (8, 9)
    args.output.mkdir(parents=True, exist_ok=True)

    full_router = fine_router = slot_extension = postprocess_extension = None
    if args.arm == "basis_joint" and not args.collective_only:
        build_root = Path(os.environ.get("TMPDIR", "/tmp")) / "basisserve_tp8_joint_v96"
        full_router, fine_router = compile_fine(build_root, value_dim=96)
        slot_extension = _load_slot_extension()
        postprocess_extension = _load_postprocess_extension()
        print(f"rank={rank} routing extensions ready", flush=True)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", rank), timeout=timedelta(minutes=5)
    )
    assert dist.get_world_size() in (4, 8)

    if args.collective_only:
        total = torch.tensor(rank + 1, device="cuda", dtype=torch.int64)
        dist.all_reduce(total)
        torch.cuda.synchronize()
        assert total.item() == dist.get_world_size() * (dist.get_world_size() + 1) // 2
        print(f"rank={rank} NCCL all-reduce complete: {total.item()}", flush=True)
        dist.destroy_process_group()
        return

    prompt, prompt_locations = load_prompts(
        args.windows, args.prompt_tokens, args.batch, args.cohort, timed
    )
    prompt_hash = hashlib.sha256(prompt.numpy().tobytes()).hexdigest()
    prompt = prompt.cuda(rank)
    print(f"rank={rank} loading model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        tp_plan="auto", local_files_only=True,
    ).eval()
    installed_sparse = []
    if args.arm == "als_full":
        attention = model.model.layers[0].self_attn
        print(
            f"rank={rank} projection shapes "
            f"q={tuple(attention.q_proj.weight.shape)} "
            f"k={tuple(attention.k_proj.weight.shape)} "
            f"v={tuple(attention.v_proj.weight.shape)} "
            f"o={tuple(attention.o_proj.weight.shape)} "
            f"v_type={type(attention.v_proj.weight).__name__}",
            flush=True,
        )
        print(f"rank={rank} installing V96", flush=True)
        install_llama_tp4_uniform_als(model, args.factors, rank)
    elif args.arm == "basis_joint":
        print(f"rank={rank} installing V96/B16R16", flush=True)
        installed_sparse = install_tp4_joint_sparse(
            model, args.factors, args.router, rank,
            batch=args.batch, capacity=args.prompt_tokens + total_steps,
            full_router=full_router, fine_router=fine_router,
            slot_extension=slot_extension, postprocess_extension=postprocess_extension,
        )
    profiler = None
    if profiling:
        from benchmarks.system.tp_joint_profile import install_labels

        install_labels(installed_sparse)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=False, with_stack=False,
        )
    print(f"rank={rank} model loaded", flush=True)
    dist.barrier()
    print(f"rank={rank} model barrier complete", flush=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    print(f"rank={rank} prefill", flush=True)
    if args.arm == "basis_joint":
        positions = torch.arange(args.prompt_tokens, device="cuda", dtype=torch.long)
        output = model(
            input_ids=prompt, use_cache=False,
            position_ids=positions[None].expand(args.batch, -1),
            cache_position=positions, logits_to_keep=1,
        )
    else:
        output = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
    assert bool(torch.isfinite(output.logits).all())
    token = choose(output.logits, model.config.vocab_size)
    assert bool(((token >= 0) & (token < model.config.vocab_size)).all())
    if rank == 0:
        print(f"rank=0 first tokens={token.flatten().tolist()} vocab={model.config.vocab_size}", flush=True)
    cache = output.past_key_values
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    prefill_peak = torch.cuda.max_memory_allocated(rank)
    prefill_resident = torch.cuda.memory_allocated(rank)
    generated = [token]

    print(f"rank={rank} decode", flush=True)
    started = time.perf_counter()
    starts = []
    ends = []
    measured_started = None
    decode_ready = None
    for step in range(total_steps):
        if timed and step == args.conditioning_steps:
            torch.cuda.synchronize()
            dist.barrier()
            decode_ready = torch.cuda.memory_allocated(rank)
            torch.cuda.reset_peak_memory_stats(rank)
            if profiling:
                profiler.start()
            measured_started = time.perf_counter()
        if benchmark and step >= args.conditioning_steps:
            begin = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            begin.record()
            starts.append(begin)
        if args.arm == "basis_joint":
            position = torch.tensor([args.prompt_tokens + step], device="cuda", dtype=torch.long)
            output = model(
                input_ids=token, use_cache=False,
                position_ids=position[None].expand(args.batch, -1),
                cache_position=torch.zeros_like(position), logits_to_keep=1,
            )
        else:
            output = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
        if not timed:
            assert bool(torch.isfinite(output.logits).all())
        token = choose(output.logits, model.config.vocab_size)
        if benchmark and step >= args.conditioning_steps:
            finish.record()
            ends.append(finish)
        generated.append(token)
        cache = output.past_key_values
        if profiling and step >= args.conditioning_steps:
            profiler.step()
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - started
    measured_wall = time.perf_counter() - measured_started if benchmark else None
    if profiling:
        profiler.stop()
        profiler.export_chrome_trace(str(args.output / f"trace_rank{rank}.json"))
    assert bool(torch.isfinite(output.logits).all())
    assert bool(((token >= 0) & (token < model.config.vocab_size)).all())
    tokens = torch.cat(generated, dim=1)
    peers = [torch.empty_like(tokens) for _ in range(dist.get_world_size())]
    dist.all_gather(peers, tokens)
    assert all(torch.equal(tokens, peer) for peer in peers)

    result = {
        "status": "complete",
        "scope": (
            "Instrumented TP routing profile; not primary latency"
            if profiling else "Experiment B fixed-batch decode"
            if benchmark else "TP model smoke; timing is diagnostic"
        ),
        "tp_size": dist.get_world_size(),
        "arm": args.arm,
        "rank": rank, "host": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(rank), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "model_revision": args.model.name, "prompt_sha256": prompt_hash,
        "prompt_locations": prompt_locations,
        "prompt_tokens": args.prompt_tokens, "batch": args.batch,
        "cohort": args.cohort, "decode_steps": total_steps,
        "generated_token_ids": tokens.tolist(),
        "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(rank),
        "gpu_prefill_peak_allocated_bytes": prefill_peak,
        "gpu_prefill_resident_allocated_bytes": prefill_resident,
        "nccl_p2p_disabled": os.environ.get("NCCL_P2P_DISABLE") == "1",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "factor_manifest_sha256": (
            hashlib.sha256((args.factors / "manifest.json").read_bytes()).hexdigest()
            if args.factors else None
        ),
    }
    if profiling:
        result["profile_steps"] = args.profile_steps
        result["conditioning_steps"] = args.conditioning_steps
        result["trace_file"] = f"trace_rank{rank}.json"
        averages = profiler.key_averages(group_by_input_shape=True)
        (args.output / f"operators_rank{rank}.txt").write_text(
            averages.table(sort_by="self_device_time_total", row_limit=100)
        )
    if benchmark:
        wall = torch.tensor(measured_wall, device="cuda", dtype=torch.float64)
        dist.all_reduce(wall, op=dist.ReduceOp.MAX)
        step_ms = [begin.elapsed_time(finish) for begin, finish in zip(starts, ends)]
        assert len(step_ms) == args.measured_steps
        replica_steps = torch.tensor(step_ms, device="cuda", dtype=torch.float64)
        dist.all_reduce(replica_steps, op=dist.ReduceOp.MAX)
        replica_step_ms = replica_steps.cpu().tolist()
        result.update({
            "conditioning_steps": args.conditioning_steps,
            "measured_steps": args.measured_steps,
            "decode_ready_allocated_bytes": decode_ready,
            "gpu_peak_decode_allocated_bytes": torch.cuda.max_memory_allocated(rank),
            "gpu_final_allocated_bytes": torch.cuda.memory_allocated(rank),
            "replica_wall_seconds": wall.item(),
            "aggregate_tokens_per_second": args.batch * args.measured_steps / wall.item(),
            "step_cuda_ms": step_ms,
            "step_mean_cuda_ms": statistics.mean(step_ms),
            "step_p50_cuda_ms": statistics.median(step_ms),
            "step_p95_cuda_ms": sorted(step_ms)[int(0.95 * (len(step_ms) - 1))],
            "replica_step_cuda_ms": replica_step_ms,
            "replica_step_mean_cuda_ms": statistics.mean(replica_step_ms),
            "replica_step_p50_cuda_ms": statistics.median(replica_step_ms),
            "replica_step_p95_cuda_ms": sorted(replica_step_ms)[
                int(0.95 * (len(replica_step_ms) - 1))
            ],
        })
    if installed_sparse:
        hit = sum(int(layer.slot_counts[..., 0].sum()) for layer in installed_sparse)
        selected = sum(int(layer.slot_counts[..., 1].sum()) for layer in installed_sparse)
        result["slot_hit_rate_last_step"] = hit / selected
        result["host_key_capacity_bytes"] = sum(
            layer.host_key.numel() * layer.host_key.element_size()
            for layer in installed_sparse
        )
    (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
