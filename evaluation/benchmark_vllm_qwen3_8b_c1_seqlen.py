#!/usr/bin/env python3
"""Fixed-batch TP8 dense/C1 sweep over prompt lengths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basisserve.core.qwen3_8b_vllm_c1 import file_sha256, load_manifest
from basisserve.vllm import QWEN3_8B_C1_MODEL_ARCHITECTURE, register
from evaluation.benchmark_vllm_qwen3_8b_c1 import run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dense", "c1"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/vllm_8b_tp8_seqlen"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefill-lengths", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192, 16384, 32640])
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--profile-prefill-lengths", type=int, nargs="*",
                        default=[512, 4096, 16384, 32640])
    args = parser.parse_args()
    assert args.batch_size > 0 and args.decode_tokens > 1
    assert args.repeats > 0 and args.warmups > 0
    assert args.prefill_lengths == sorted(set(args.prefill_lengths))
    assert set(args.profile_prefill_lengths) <= set(args.prefill_lengths)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.arm}.json"
    assert not result_path.exists(), f"Keep existing results: {result_path}"

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    register()
    root = args.factor_dir.resolve()
    sha = file_sha256(root / "results.json")
    manifest = load_manifest(root, sha)
    fit = manifest["fit_config"]
    assert (fit["hidden_size"], fit["num_query_heads"], fit["num_hidden_layers"]) == (4096, 32, 36)
    assert file_sha256(args.model / "config.json") == fit["model_config_sha256"]
    max_model_len = max(args.prefill_lengths) + args.decode_tokens
    assert max_model_len <= 32768

    overrides = {} if args.arm == "dense" else {
        "architectures": [QWEN3_8B_C1_MODEL_ARCHITECTURE],
        "basisserve_c1_factor_dir": str(root),
        "basisserve_c1_result_sha256": sha,
    }
    graph_sizes = [size for size in (1, 2, 4, 8, 16, 32, 64, 128, 256)
                   if size <= args.batch_size]
    configuration = dict(
        model=str(args.model.resolve()), tensor_parallel_size=8, dtype="bfloat16",
        max_model_len=max_model_len, max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        async_scheduling=False, gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False, disable_log_stats=False, seed=0,
        compilation_config={"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes": graph_sizes,
                            "cudagraph_num_of_warmups": 1},
        hf_overrides=overrides,
        worker_extension_cls="basisserve.vllm.worker_extension.BasisServeWorkerExtension",
    )
    payload = dict(
        status="initializing", arm=args.arm,
        created_at=datetime.now(timezone.utc).isoformat(),
        command=shlex.join([sys.executable, *sys.argv]), conda_environment="basis",
        torch=torch.__version__, torch_cuda=torch.version.cuda, vllm=vllm.__version__,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        factor_sha256=sha, configuration=configuration,
        arguments={key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        metric_notes={
            "throughput": "All output tokens divided by cohort wall time, includes prefill.",
            "ttft": "Per request: first_token_ts minus queued_ts; includes paused admission time.",
            "tpot": "Per request: (last_token_ts-first_token_ts)/(decode_tokens-1); includes scheduling interleaving.",
            "profiles": "Separate unmeasured full-cohort runs, rank zero only; summed kernel durations are not wall time.",
        },
        lengths=[], profiles=[],
    )

    def save():
        result_path.write_text(json.dumps(payload, indent=2) + "\n")

    save()
    llm = LLM(**configuration)
    assert any(metric.name == "vllm:num_preemptions" for metric in llm.get_metrics())
    tokenizer = llm.get_tokenizer()
    seed = tokenizer.encode(
        "The capital of France is Paris. Tensor parallelism distributes a model across GPUs. ",
        add_special_tokens=False,
    )
    sampling = SamplingParams(temperature=0, max_tokens=args.decode_tokens,
                              ignore_eos=True, detokenize=False)
    payload["status"] = "running"
    save()

    def prompts_for(prefill_tokens):
        tail = (seed * ((prefill_tokens + len(seed) - 1) // len(seed)))[:prefill_tokens]
        prompts = []
        for index in range(args.batch_size):
            prefix = tokenizer.encode(f"Request {index}: ", add_special_tokens=False)
            prompts.append({"prompt_token_ids": (prefix + tail)[:prefill_tokens]})
        assert all(len(prompt["prompt_token_ids"]) == prefill_tokens for prompt in prompts)
        return prompts

    for prefill_tokens in args.prefill_lengths:
        prompts = prompts_for(prefill_tokens)
        print(f"BENCH arm={args.arm} batch={args.batch_size} prefill={prefill_tokens} phase=warmup", flush=True)
        for _ in range(args.warmups):
            run_batch(llm, prompts, sampling, args.decode_tokens)
        row = dict(prefill_tokens=prefill_tokens, batch=args.batch_size, runs=[])
        payload["lengths"].append(row)
        for repeat in range(args.repeats):
            measured = run_batch(llm, prompts, sampling, args.decode_tokens)
            row["runs"].append(measured)
            save()
            print(json.dumps(dict(event="BENCH", arm=args.arm,
                                  batch=args.batch_size, prefill_tokens=prefill_tokens,
                                  repeat=repeat, wall_seconds=measured["wall_seconds"],
                                  preemptions=measured["preemptions"],
                                  output_tokens_per_second=measured["output_tokens_per_second"],
                                  ttft_ms=measured["ttft_ms"],
                                  tpot_ms=measured["tpot_ms"])), flush=True)
        row["workers"] = llm.collective_rpc(
            "basisserve_c1_statistics" if args.arm == "c1"
            else "basisserve_cuda_statistics")
        assert len(row["workers"]) == 8
        if args.arm == "c1":
            assert all(worker["loaded_value_layers"] == 36 and worker["capture_calls"] > 0
                       for worker in row["workers"])
        save()

    for prefill_tokens in args.profile_prefill_lengths:
        prompts = prompts_for(prefill_tokens)
        prefix = output_dir / "profiles" / f"{args.arm}_s{prefill_tokens}"
        print(f"PROFILE arm={args.arm} batch={args.batch_size} prefill={prefill_tokens} prefix={prefix}", flush=True)
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
