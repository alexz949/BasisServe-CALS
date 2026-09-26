"""Qwen3-8B-Base TP8 packed NUQ4 + A8/W8 full-request benchmark."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basisserve.core.qwen3_nuq4_artifacts import QwenNUQ4Artifacts
from basisserve.vllm import register
from evaluation.benchmark_vllm_qwen3_8b_c1 import run_batch


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--arm", choices=("dense", "nuq4"), default="nuq4")
    parser.add_argument("--rank", type=int, choices=(64, 96), default=64)
    parser.add_argument("--phase", choices=("smoke", "preflight", "formal"), default="smoke")
    parser.add_argument("--prior", type=Path, default=ROOT / "results/q3-kv4-fp8/formal")
    parser.add_argument("--output", type=Path, default=ROOT / "results/q3-nuq4-vllm")
    parser.add_argument("--prefill-tokens", type=int, default=4096)
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--profile-batches", type=int, nargs="*", default=[])
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    assert args.prefill_tokens in (128, 4096)
    quantized = args.arm == "nuq4"
    artifacts = QwenNUQ4Artifacts(args.prior, args.rank) if quantized else None
    model = (artifacts.model if quantized else
             Path(json.loads((args.prior / "manifest.json").read_text())["protocol"]["model"]))
    directory = args.output / args.phase / (f"r{args.rank}" if quantized else "dense")
    directory.mkdir(parents=True, exist_ok=True)
    kernel = "splitk" if quantized else "flash"
    suffix = f"{'eager' if args.eager else 'graph'}_{kernel}_{args.prefill_tokens}"
    result = directory / f"{suffix}.json"
    assert not result.exists(), "Preserve completed results"
    journal = directory / f"{suffix}.jsonl"

    def record(event):
        row = dict(utc=datetime.now(timezone.utc).isoformat(), **event)
        with journal.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    register()
    from basisserve.vllm.nuq4_attention import SERVING_EXCEPTIONS_PER_TOKEN
    batches = args.batch_sizes or ([1, 2] if args.phase == "smoke" else [1, 256] if args.phase == "preflight"
                                  else [1, 2, 4, 8, 16, 32, 64, 128, 256])
    assert batches == sorted(set(batches)) and 1 <= min(batches) <= max(batches) <= 256
    assert set(args.profile_batches) <= set(batches)
    decode = 4 if args.phase == "smoke" else 128
    repeats = 3 if args.phase == "formal" else 1
    configuration = dict(model=str(model), tensor_parallel_size=8, dtype="bfloat16",
        max_model_len=args.prefill_tokens+decode, max_num_seqs=256 if args.phase != "smoke" else max(batches),
        max_num_batched_tokens=8192, block_size=16,
        enable_chunked_prefill=True, enable_prefix_caching=False, async_scheduling=False,
        gpu_memory_utilization=0.2 if args.phase == "smoke" else 0.8,
        enforce_eager=args.eager, disable_log_stats=False, seed=0,
        compilation_config={"mode": "NONE", "cudagraph_mode": "NONE" if args.eager else "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256] if args.phase != "smoke" else batches,
            "cudagraph_num_of_warmups": 1},
        hf_overrides=({"architectures": ["BasisServeQwen3NUQ4ForCausalLM"],
            "basisserve_nuq4_prior": str(args.prior.resolve()), "basisserve_nuq4_rank": args.rank} if quantized else {}),
        worker_extension_cls="basisserve.vllm.worker_extension.BasisServeWorkerExtension")
    if not quantized:
        configuration["attention_config"] = dict(backend="FLASH_ATTN")
    payload = dict(environment="basis", command=shlex.join(sys.argv), phase=args.phase,
        arm=args.arm, nominal_rank=args.rank if quantized else None,
        torch=torch.__version__, vllm=vllm.__version__,
        configuration=configuration, value_ranks=artifacts.schedule if quantized else None, batches=[], profiles=[],
        exceptions_per_token=SERVING_EXCEPTIONS_PER_TOKEN if quantized else 0,
        decode_kernel="nuq4_split_k" if quantized else "FLASH_ATTN",
        metric_notes=dict(throughput="All output tokens / cohort wall seconds, including prefill",
            ttft="First token minus queued timestamp, including paused admission time",
            tpot="Last minus first token timestamp / (output length - 1)",
            output_length=decode, decode_forwards=decode-1,
            smoke="20% memory budget, four outputs, one repeat; not a speed conclusion",
            preflight="80% memory budget, 128 outputs, one repeat; not formal results",
            profiles="Separate unmeasured full-cohort runs after timing, rank zero only; kernel sums are not wall time"),
        notes=("Actual packed NUQ4; global V stats collective included; no SHA256; full attention; BF16 encoder; A8/W8 decoder."
               if quantized else "Original Qwen3-8B-Base, full BF16 K/V on GPU, FlashAttention; no SHA256."))
    record(dict(event="initializing", **payload))
    llm = LLM(**configuration)
    tokenizer = llm.get_tokenizer()
    seed = tokenizer.encode("The capital of France is Paris. Tensor parallelism distributes a model across GPUs. ", add_special_tokens=False)
    tail = (seed * ((args.prefill_tokens+len(seed)-1)//len(seed)))[:args.prefill_tokens]
    sampling = SamplingParams(temperature=0, max_tokens=decode, ignore_eos=True, detokenize=False)
    for batch in batches:
        prompts = [{"prompt_token_ids": (tokenizer.encode(f"Request {i}: ", add_special_tokens=False)+tail)[:args.prefill_tokens]}
                   for i in range(batch)]
        record(dict(event="warmup", batch=batch))
        run_batch(llm, prompts, sampling, decode)
        rpc = "basisserve_c1_statistics" if quantized else "basisserve_cuda_statistics"
        workers = llm.collective_rpc(rpc)
        assert len(workers) == 8
        record(dict(event="warmup_workers", batch=batch, workers=workers))
        if quantized:
            assert all(w["loaded_value_layers"] == 36 and not w["overflow_layers"] for w in workers)
        row = dict(batch=batch, runs=[])
        for repeat in range(repeats):
            measured = run_batch(llm, prompts, sampling, decode)
            workers = llm.collective_rpc(rpc)
            record(dict(event="trial", batch=batch, repeat=repeat, measured=measured, workers=workers))
            if quantized:
                assert all(not w["overflow_layers"] for w in workers), "NUQ4 exception capacity failure, not a valid result"
            if quantized and not args.eager:
                assert all(w["capture_calls"] > 0 for w in workers)
            row["runs"].append(measured)
        row["workers"] = workers
        payload["batches"].append(row)
    for batch in args.profile_batches:
        prompts = [{"prompt_token_ids": (tokenizer.encode(f"Request {i}: ", add_special_tokens=False)+tail)[:args.prefill_tokens]}
                   for i in range(batch)]
        prefix = directory / "profiles" / f"b{batch}"
        record(dict(event="profile_start", batch=batch))
        llm.collective_rpc("basisserve_profile_start")
        run_batch(llm, prompts, sampling, decode)
        llm.collective_rpc("basisserve_profile_stop", args=(str(prefix),))
        workers = llm.collective_rpc(rpc)
        if quantized:
            assert all(not w["overflow_layers"] for w in workers)
        payload["profiles"].append(str(prefix))
        record(dict(event="profile_complete", batch=batch, prefix=str(prefix), workers=workers))
    payload["status"] = "complete"
    result.write_text(json.dumps(payload, indent=2) + "\n")
    record(dict(event="complete", result=str(result)))


if __name__ == "__main__":
    main()
