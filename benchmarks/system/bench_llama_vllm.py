"""Three-arm Llama TP benchmark using native vLLM scheduling and model execution."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

import numpy as np
from safetensors import safe_open
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind

from basisserve.vllm import register


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def append(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def percentile(values, q):
    return float(np.percentile(values, q))


def prompts(path, length, batch):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        view = handle.get_slice("input_ids")
        rows, width = view.get_shape()
        segments = width // length
        assert rows * segments >= 3 * batch
        cohorts, manifests = [], []
        # Cohorts 0..2 use distinct non-overlapping windows/segments.
        # At 130048, the warmup uses a distinct shifted input from the same
        # source documents; this source-text overlap is recorded explicitly.
        for cohort in range(4):
            sequences, records = [], []
            for index in range(batch):
                absolute = cohort * batch + index
                shifted = absolute >= rows * segments
                if shifted:
                    absolute %= rows * segments
                row, segment = divmod(absolute, segments)
                offset = width - length if shifted else segment * length
                token_ids = view[row, offset:offset + length].long()
                sequences.append(token_ids.tolist())
                records.append(dict(window=row, offset=offset, tokens=length,
                                    shifted_warmup=shifted,
                                    sha256=hashlib.sha256(token_ids.numpy().tobytes()).hexdigest()))
            cohorts.append(sequences)
            manifests.append(records)
    assert len({r["sha256"] for group in manifests for r in group}) == 4 * batch
    return cohorts, manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, choices=(4, 8), required=True)
    parser.add_argument("--arm", choices=("dense", "als_full", "basis_joint", "palu_g4", "palu_g4_sharded"), required=True,
                        help="palu_g4: PaLU G-LRD4 with the group latent replicated per rank; --factors is its checkpoint")
    parser.add_argument("--prompt", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--experiment", choices=("request", "decode", "profile"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--graph", choices=("full", "eager"), default="full",
                        help="CUDA Graph policy; the sharded PaLU arm always runs eager")
    args = parser.parse_args()
    if args.arm == "palu_g4_sharded":
        args.graph = "eager"
    assert Path(sys.prefix).name == "basis"
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert os.environ["NCCL_P2P_DISABLE"] == "1"
    assert os.environ["NVIDIA_TF32_OVERRIDE"] == "0"
    assert torch.cuda.device_count() == args.tp
    assert args.batch > 0 and args.prompt in (4096, 16384, 65536, 130048)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    register()
    generation = 8 if args.smoke else 128
    conditioning = 2 if args.smoke else 16
    steps = 4 if args.smoke else 32 if args.experiment == "profile" else 128
    repetitions = 1 if args.smoke or args.experiment == "profile" else 3
    config = json.loads((args.model / "config.json").read_text())
    assert args.prompt + max(generation, 1 + conditioning + steps + 1) <= config["max_position_embeddings"]
    max_length = min(args.prompt + 1024, config["max_position_embeddings"])
    cohorts, manifests = prompts(args.windows, args.prompt, args.batch)
    factor_bytes = (args.factors / "manifest.json").read_bytes()
    factor_manifest = json.loads(factor_bytes)
    assert hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest() == factor_manifest["model"]["config_sha256"]
    overrides = {}
    if args.arm.startswith("palu_g4"):
        overrides.update(architectures=["BasisServeLlamaPaluG4ShardedForCausalLM" if args.arm.endswith("sharded")
                                        else "BasisServeLlamaPaluG4ForCausalLM"],
                         basisserve_palu_factor_dir=str(args.factors),
                         basisserve_palu_manifest_sha256=hashlib.sha256(factor_bytes).hexdigest())
    elif args.arm != "dense":
        overrides.update(architectures=["BasisServeLlamaJointALSForCausalLM"],
                         basisserve_joint_factor_dir=str(args.factors),
                         basisserve_joint_manifest_sha256=hashlib.sha256(factor_bytes).hexdigest())
    if args.arm == "basis_joint":
        overrides.update(basisserve_joint_router_dir=str(args.factors / "router_b16r16"),
                         basisserve_host_budget_bytes_per_rank=40 * 1024**3)
    capture_sizes = sorted({args.batch, *[2**i for i in range(args.batch.bit_length()) if 2**i <= args.batch]})
    configuration = dict(
        model=str(args.model), tensor_parallel_size=args.tp,
        distributed_executor_backend="mp", dtype="bfloat16", seed=0,
        max_model_len=max_length, max_num_seqs=args.batch, max_num_batched_tokens=8192,
        enable_chunked_prefill=True, enable_prefix_caching=False, async_scheduling=False,
        gpu_memory_utilization=.8, block_size=32, disable_log_stats=False,
        disable_custom_all_reduce=False,
        enforce_eager=args.graph == "eager",
        compilation_config=dict(cudagraph_mode="NONE" if args.graph == "eager" else "FULL_AND_PIECEWISE",
                                cudagraph_capture_sizes=capture_sizes),
        scheduler_cls="benchmarks.system.vllm_cohort_scheduler.CohortScheduler",
        worker_extension_cls="benchmarks.system.vllm_bench_worker.LlamaBenchWorker",
        hf_overrides=overrides,
    )
    if args.experiment == "profile":
        configuration["profiler_config"] = dict(
            profiler="torch", torch_profiler_dir=str((args.output / "profiles").resolve()),
            torch_profiler_with_stack=False, torch_profiler_record_shapes=True,
        )
    source_files = sorted([*Path("basisserve/vllm").glob("llama_joint*.py"),
                           *Path("basisserve/vllm").glob("*palu*.py"),
                           *Path("basisserve/kernels").glob("paged_joint*.py"),
                           Path(__file__), Path("benchmarks/system/vllm_cohort_scheduler.py"),
                           Path("benchmarks/system/vllm_bench_worker.py")])
    metadata = dict(
        status="initializing", experiment=args.experiment, arm=args.arm, tp=args.tp, graph=args.graph,
        prompt_tokens=args.prompt, batch=args.batch, smoke=args.smoke,
        generation_tokens=generation, conditioning_steps=conditioning, measured_decode_steps=steps,
        repetitions=repetitions, configuration=configuration, prompt_cohorts=manifests,
        conda_environment=sys.prefix, command=shlex.join([sys.executable, *sys.argv]),
        host=os.uname().nodename, visible_gpus=os.environ["CUDA_VISIBLE_DEVICES"],
        software={name: importlib.metadata.version(name) for name in ("vllm", "torch", "triton", "transformers")},
        factor_manifest_sha256=hashlib.sha256(factor_bytes).hexdigest(),
        router_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((args.factors / "router_b16r16").glob("*.safetensors"))],
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        timing_scope="in-process vLLM scheduler/engine, multiprocessing TP workers; no tokenizer, detokenizer or network",
        request_boundary="common monotonic release before first scheduler step, native first/last token timestamps",
        decode_boundary="all requests fully prefilled before conditioning; one native scheduler step emits B tokens",
        nccl_p2p_disabled=True, host_key_pool_limit_per_rank=40 * 1024**3,
        model_loaded_memory=None,
    )
    write(args.output / "config.json", metadata)
    print(f"BENCH load {args.arm} TP{args.tp} P{args.prompt} B{args.batch} {args.experiment}", flush=True)
    llm = LLM(**configuration)
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    assert type(scheduler).__name__ == "CohortScheduler"
    configuration["actual_compilation_config"] = str(llm.llm_engine.vllm_config.compilation_config)
    ready = llm.collective_rpc("bench_install")
    assert args.graph == "eager" or all(row["cudagraph_captures"] > 0 for row in ready)
    append(args.output / "memory_by_rank.jsonl", dict(phase="graph_ready", ranks=ready))

    def failed_cohort(reason, **details):
        metadata.update(status="failed", failure_reason=reason, failure_details=details)
        write(args.output / "config.json", metadata)
        append(args.output / "request_trials.jsonl", dict(
            status="failed", failure_reason=reason, experiment=args.experiment,
            arm=args.arm, tp=args.tp, prompt_tokens=args.prompt, batch=args.batch, **details))
        print("BENCH failed " + json.dumps(dict(reason=reason, **details)), flush=True)

    if args.experiment != "request":
        # These arms use one full-attention block pool with a 32-token block.
        # Include the final measured token and the pool's reserved null block.
        needed = args.batch * ((args.prompt + conditioning + steps + 31) // 32)
        available = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        if needed > available:
            failed_cohort("gpu_cache_capacity", required_blocks=needed, free_blocks=available,
                          failure_stage="before_prefill", block_size=32)
            return

    def add_cohort(index, count):
        assert not engine.has_unfinished_requests()
        sampling = SamplingParams(temperature=0, max_tokens=count, ignore_eos=True,
                                  detokenize=False, output_kind=RequestOutputKind.DELTA)
        records = {}
        for i, tokens in enumerate(cohorts[index]):
            request_id = f"c{index}-r{i}"
            accepted_before = time.monotonic()
            engine.add_request(request_id, {"prompt_token_ids": tokens}, sampling)
            records[request_id] = dict(request_id=request_id, prompt=manifests[index][i],
                                       accepted_before=accepted_before, accepted_after=time.monotonic(),
                                       token_ids=[], observed_token_times=[], metrics=None)
        return records

    def step(records):
        begin = time.monotonic()
        outputs = engine.step()
        end = time.monotonic()
        emitted = 0
        for output in outputs:
            record = records[output.request_id]
            assert len(output.outputs) == 1
            ids = list(output.outputs[0].token_ids)
            record["token_ids"].extend(ids)
            record["observed_token_times"].extend([end] * len(ids))
            emitted += len(ids)
            if output.metrics is not None:
                record["metrics"] = asdict(output.metrics)
        return (end - begin) * 1000, emitted

    print("BENCH separate full warmup cohort", flush=True)
    warmup = add_cohort(3, generation)
    while engine.has_unfinished_requests():
        step(warmup)
    assert all(len(row["token_ids"]) == generation for row in warmup.values())
    llm.collective_rpc("bench_snapshot")
    trials = []
    for trial in range(repetitions):
        print(f"BENCH trial={trial} start", flush=True)
        scheduler.trial_steps.clear()
        scheduler.record_steps = True
        scheduler.hold_ready_decodes = args.experiment != "request"
        requested = generation if args.experiment == "request" else 1 + conditioning + steps + 1
        records = add_cohort(trial, requested)
        before = llm.collective_rpc("bench_begin")
        append(args.output / "memory_by_rank.jsonl", dict(trial=trial, phase="request_start", ranks=before))
        release = time.monotonic()
        wall_steps = []
        if args.experiment == "request":
            while engine.has_unfinished_requests():
                duration, _ = step(records)
                wall_steps.append(duration)
            finish = time.monotonic()
            ranks = llm.collective_rpc("bench_end")
            metrics = [row["metrics"] for row in records.values()]
            assert all(m is not None and not m["is_corrupted"] for m in metrics)
            assert all(len(row["token_ids"]) == generation for row in records.values())
            ttft = [(m["first_token_ts"] - release) * 1000 for m in metrics]
            tpot = [(m["last_token_ts"] - m["first_token_ts"]) * 1000 / (generation - 1) for m in metrics]
            e2e = max(m["last_token_ts"] for m in metrics) - release
            summary = dict(cohort_e2e_s=e2e, engine_loop_s=finish - release,
                           cohort_output_tokens_s=args.batch * generation / e2e,
                           ttft_mean_ms=statistics.mean(ttft), ttft_p50_ms=statistics.median(ttft),
                           tpot_mean_ms=statistics.mean(tpot), tpot_p50_ms=statistics.median(tpot),
                           ttft_p95_ms=percentile(ttft, 95) if args.batch >= 20 else None,
                           tpot_p95_ms=percentile(tpot, 95) if args.batch >= 20 else None)
            append(args.output / "memory_by_rank.jsonl", dict(trial=trial, phase="request_end", ranks=ranks))
        else:
            while not all(len(row["token_ids"]) == 1 for row in records.values()):
                step(records)
                assert all(len(row["token_ids"]) <= 1 for row in records.values())
                latest = scheduler.trial_steps[-1]
                if latest["preempted_ids"] or not latest["requests"]:
                    failed_cohort("preempted" if latest["preempted_ids"] else "gpu_cache_capacity",
                                  trial=trial, failure_stage="prefill", scheduler_step=latest)
                    return
            prefill_ranks = llm.collective_rpc("bench_end")
            scheduler.release_decodes()
            assert len(scheduler.running) == args.batch
            assert all(r.num_computed_tokens == args.prompt for r in scheduler.running)
            append(args.output / "memory_by_rank.jsonl", dict(trial=trial, phase="decode_ready", ranks=prefill_ranks))
            for _ in range(conditioning):
                _, emitted = step(records)
                assert emitted == args.batch
            lengths_before = {r.request_id: r.num_computed_tokens for r in scheduler.running}
            assert set(lengths_before.values()) == {args.prompt + conditioning}
            scheduler.trial_steps.clear()
            # Support collection is confined to the separately instrumented profile.
            llm.collective_rpc("bench_begin", kwargs=dict(support_steps=steps if args.experiment == "profile" else 0))
            if args.experiment == "profile":
                llm.start_profile()
            release = time.monotonic()
            for _ in range(steps):
                duration, emitted = step(records)
                wall_steps.append(duration)
                assert emitted == args.batch
            finish = time.monotonic()
            if args.experiment == "profile":
                llm.stop_profile()
            ranks = llm.collective_rpc("bench_end")
            lengths_after = {r.request_id: r.num_computed_tokens for r in scheduler.running}
            assert set(lengths_after.values()) == {args.prompt + conditioning + steps}
            assert len(scheduler.trial_steps) == steps
            assert all(len(row["requests"]) == args.batch and all(
                req["scheduled_tokens"] == 1 and not req["is_prefill"] for req in row["requests"]
            ) for row in scheduler.trial_steps)
            assert all(len(row["step_cuda_ms"]) == steps for row in ranks)
            summary = dict(decode_step_mean_ms=statistics.mean(wall_steps),
                           decode_step_p50_ms=statistics.median(wall_steps),
                           decode_step_p95_ms=percentile(wall_steps, 95),
                           decode_tokens_s=args.batch * steps / (finish - release),
                           decode_wall_s=finish - release, cache_lengths_before=lengths_before,
                           cache_lengths_after=lengths_after, actual_active_decode_batch=args.batch)
            append(args.output / "memory_by_rank.jsonl", dict(trial=trial, phase="decode_end", ranks=ranks))
            engine.abort_request(list(records))
        preemptions = sum(len(row["preempted_ids"]) for row in scheduler.trial_steps)
        summary.update(trial=trial, experiment=args.experiment, arm=args.arm, tp=args.tp,
                       prompt_tokens=args.prompt, batch=args.batch, release_t0=release,
                       status="complete" if preemptions == 0 else "preempted", preemptions=preemptions,
                       gpu_peak_maxrank_bytes=max(row["peak_allocated_bytes"] for row in ranks),
                       host_pool_total_bytes=sum(row["host_key_pool_bytes"] for row in ranks),
                       prompt_cohort_sha256=hashlib.sha256(json.dumps(manifests[trial], sort_keys=True).encode()).hexdigest())
        for record in records.values():
            record.update(trial=trial, release_t0=release)
            append(args.output / "requests.jsonl", record)
        append(args.output / "scheduler_steps.jsonl", dict(trial=trial, steps=scheduler.trial_steps))
        for row in ranks:
            if row["support_counts"] is not None:
                append(args.output / "support_stats.jsonl", dict(trial=trial, rank=row["rank"], counts=row["support_counts"],
                                                              scope="separate instrumented profile", count_order=["hits", "misses"]))
        np.savez(args.output / f"steps_{trial}.npz", wall_ms=np.asarray(wall_steps),
                 **{f"rank{row['rank']}_cuda_ms": np.asarray(row["step_cuda_ms"]) for row in ranks})
        append(args.output / "request_trials.jsonl", summary)
        trials.append(summary)
        scheduler.record_steps = False
        print("BENCH " + json.dumps(summary), flush=True)
    metadata.update(status="complete", trials=trials)
    write(args.output / "config.json", metadata)
    write(args.output / "summary.json", dict(trials=trials, aggregation="median across three distinct prompt cohorts",
          medians={key: statistics.median([r[key] for r in trials]) for key in trials[0]
                   if key.endswith(("_ms", "_s")) and all(isinstance(r[key], (int, float)) for r in trials)}))
    print("BENCH complete", flush=True)


if __name__ == "__main__":
    main()
