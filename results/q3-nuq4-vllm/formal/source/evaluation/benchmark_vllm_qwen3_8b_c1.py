#!/usr/bin/env python3
"""Fixed-cohort TP8 dense/C1 benchmark, with separate rank-zero profiles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basisserve.core.qwen3_8b_vllm_c1 import file_sha256, load_manifest
from basisserve.vllm import QWEN3_8B_C1_MODEL_ARCHITECTURE, QWEN3_32B_TP8_C1_MODEL_ARCHITECTURE, register


def distribution(values):
    ordered = sorted(values)
    return dict(mean=statistics.fmean(ordered), median=statistics.median(ordered),
                p95=ordered[min(len(ordered)-1, int(len(ordered)*0.95))],
                min=ordered[0], max=ordered[-1])


def run_batch(llm, prompts, sampling, decode_tokens):
    preemptions_before = sum(metric.value for metric in llm.get_metrics()
                             if metric.name == "vllm:num_preemptions")
    # Admission is outside timing. All requests are queued before scheduling.
    llm.sleep(level=0, mode="keep")
    request_ids = llm.enqueue(prompts, sampling, use_tqdm=False)
    assert len(request_ids) == len(prompts)
    start = time.perf_counter()
    llm.wake_up(tags=["scheduling"])
    outputs = llm.wait_for_completion(use_tqdm=False)
    seconds = time.perf_counter() - start
    preemptions_after = sum(metric.value for metric in llm.get_metrics()
                            if metric.name == "vllm:num_preemptions")
    assert len(outputs) == len(prompts)
    assert all(len(output.outputs[0].token_ids) == decode_tokens for output in outputs)
    requests = []
    for output in outputs:
        metric = output.metrics
        assert metric is not None and not metric.is_corrupted
        assert metric.last_token_ts >= metric.first_token_ts >= metric.scheduled_ts >= metric.queued_ts
        requests.append(dict(
            request_id=output.request_id,
            queued_ts=metric.queued_ts, scheduled_ts=metric.scheduled_ts,
            first_token_ts=metric.first_token_ts, last_token_ts=metric.last_token_ts,
            ttft_ms=1000*(metric.first_token_ts-metric.queued_ts),
            tpot_ms=1000*(metric.last_token_ts-metric.first_token_ts)/(decode_tokens-1),
            output_token_ids=list(output.outputs[0].token_ids),
        ))
    return dict(wall_seconds=seconds, preemptions=preemptions_after-preemptions_before,
                output_tokens_per_second=len(outputs)*decode_tokens/seconds,
                ttft_ms=distribution([row["ttft_ms"] for row in requests]),
                tpot_ms=distribution([row["tpot_ms"] for row in requests]), requests=requests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dense", "c1"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--factor-validation", choices=("sha256", "structure"), default="sha256")
    parser.add_argument("--output-dir", type=Path, default=Path("results/vllm_tp8"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1,2,4,8,16,32,64,128,256])
    parser.add_argument("--prefill-tokens", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--profile-batches", type=int, nargs="*", default=[1,32,256])
    args = parser.parse_args()
    assert args.decode_tokens > 1 and args.repeats > 0 and args.warmups > 0
    assert set(args.profile_batches) <= set(args.batch_sizes)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.arm}.json"
    assert not result_path.exists(), f"Keep existing results: {result_path}"
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    register()
    root = args.factor_dir.resolve()
    sha = file_sha256(root / "results.json") if args.factor_validation == "sha256" else None
    manifest = load_manifest(root, sha, validation=args.factor_validation)
    layers = manifest["fit_config"]["num_hidden_layers"]
    architecture = QWEN3_8B_C1_MODEL_ARCHITECTURE if layers == 36 else QWEN3_32B_TP8_C1_MODEL_ARCHITECTURE
    if args.factor_validation == "sha256":
        assert file_sha256(args.model / "config.json") == manifest["fit_config"]["model_config_sha256"]
    model_config = json.loads((args.model / "config.json").read_text())
    fit = manifest["fit_config"]
    for model_key, fit_key in (("hidden_size", "hidden_size"),
                               ("num_attention_heads", "num_query_heads"),
                               ("num_key_value_heads", "num_physical_kv_heads"),
                               ("num_hidden_layers", "num_hidden_layers"),
                               ("head_dim", "head_dim")):
        assert model_config[model_key] == fit[fit_key]
    overrides = {} if args.arm == "dense" else {
        "architectures": [architecture],
        "basisserve_c1_factor_dir": str(root), "basisserve_c1_result_sha256": sha,
        "basisserve_c1_validation": args.factor_validation,
    }
    configuration = dict(
        model=str(args.model.resolve()), tensor_parallel_size=8, dtype="bfloat16",
        max_model_len=args.prefill_tokens+args.decode_tokens,
        max_num_seqs=max(args.batch_sizes), max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        async_scheduling=False, gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False, disable_log_stats=False, seed=0,
        compilation_config={"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes": sorted(set(args.batch_sizes)),
                            "cudagraph_num_of_warmups": 1},
        hf_overrides=overrides,
        worker_extension_cls="basisserve.vllm.worker_extension.BasisServeWorkerExtension",
    )
    payload = dict(
        status="initializing", arm=args.arm, created_at=datetime.now(timezone.utc).isoformat(),
        command=shlex.join([sys.executable, *sys.argv]), conda_environment="basis",
        torch=torch.__version__, torch_cuda=torch.version.cuda, vllm=vllm.__version__,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        factor_sha256=sha, factor_validation=args.factor_validation, configuration=configuration,
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        metric_notes={"throughput": "All output tokens divided by cohort wall time, includes prefill.",
                      "ttft": "Per request: first_token_ts minus queued_ts; includes paused admission time.",
                      "tpot": "Per request: (last_token_ts-first_token_ts)/(decode_tokens-1); includes scheduling interleaving.",
                      "profiles": "Separate unmeasured full-cohort runs, rank zero only; summed kernel durations are not wall time."},
        batches=[], profiles=[],
    )

    def save():
        result_path.write_text(json.dumps(payload, indent=2)+"\n")

    save()
    llm = LLM(**configuration)
    assert any(metric.name == "vllm:num_preemptions" for metric in llm.get_metrics())
    tokenizer = llm.get_tokenizer()
    seed = tokenizer.encode("The capital of France is Paris. Tensor parallelism distributes a model across GPUs. ", add_special_tokens=False)
    tail = (seed*((args.prefill_tokens+len(seed)-1)//len(seed)))[:args.prefill_tokens]
    sampling = SamplingParams(temperature=0, max_tokens=args.decode_tokens, ignore_eos=True, detokenize=False)
    payload["status"] = "running"
    save()
    for batch in args.batch_sizes:
        prompts = []
        for index in range(batch):
            prefix = tokenizer.encode(f"Request {index}: ", add_special_tokens=False)
            prompts.append({"prompt_token_ids": (prefix+tail)[:args.prefill_tokens]})
        print(f"BENCH arm={args.arm} batch={batch} phase=warmup", flush=True)
        for _ in range(args.warmups):
            run_batch(llm, prompts, sampling, args.decode_tokens)
        row = dict(batch=batch, runs=[])
        payload["batches"].append(row)
        for repeat in range(args.repeats):
            measured = run_batch(llm, prompts, sampling, args.decode_tokens)
            row["runs"].append(measured)
            save()
            print(json.dumps(dict(event="BENCH", arm=args.arm, batch=batch, repeat=repeat,
                                  wall_seconds=measured["wall_seconds"],
                                  preemptions=measured["preemptions"],
                                  output_tokens_per_second=measured["output_tokens_per_second"],
                                  ttft_ms=measured["ttft_ms"], tpot_ms=measured["tpot_ms"])), flush=True)
        row["workers"] = llm.collective_rpc("basisserve_c1_statistics" if args.arm == "c1" else "basisserve_cuda_statistics")
        assert len(row["workers"]) == 8
        if args.arm == "c1":
            assert all(worker["loaded_value_layers"] == layers and worker["capture_calls"] > 0
                       and worker["value_head_size"] == fit["cache_rank_per_head"]
                       for worker in row["workers"])
        save()
    # Keep CUPTI and trace export entirely after the timed sweep.
    for batch in args.profile_batches:
        prompts = [{"prompt_token_ids": (tokenizer.encode(f"Request {index}: ", add_special_tokens=False)+tail)[:args.prefill_tokens]}
                   for index in range(batch)]
        prefix = output_dir / "profiles" / f"{args.arm}_b{batch}"
        print(f"PROFILE arm={args.arm} batch={batch} prefix={prefix}", flush=True)
        llm.collective_rpc("basisserve_profile_start")
        run_batch(llm, prompts, sampling, args.decode_tokens)
        llm.collective_rpc("basisserve_profile_stop", args=(str(prefix),))
        payload["profiles"].append(str(prefix))
        save()
    payload["status"] = "complete"
    save()
    print(f"COMPLETE {result_path}", flush=True)


if __name__ == "__main__":
    main()
