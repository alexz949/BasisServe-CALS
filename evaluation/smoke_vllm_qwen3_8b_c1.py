#!/usr/bin/env python3
"""Smoke real TP8 compact-C1 requests, chunked prefill and decode CUDA Graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basisserve.core.qwen3_8b_vllm_c1 import file_sha256, load_manifest
from basisserve.vllm import QWEN3_8B_C1_MODEL_ARCHITECTURE, QWEN3_32B_TP8_C1_MODEL_ARCHITECTURE, register


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--execution-mode", choices=("eager", "cuda_graph"), default="cuda_graph")
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    args = parser.parse_args()
    root = args.factor_dir.resolve()
    sha = file_sha256(root / "results.json")
    manifest = load_manifest(root, sha)
    layers = manifest["fit_config"]["num_hidden_layers"]
    architecture = QWEN3_8B_C1_MODEL_ARCHITECTURE if layers == 36 else QWEN3_32B_TP8_C1_MODEL_ARCHITECTURE
    assert file_sha256(args.model / "config.json") == manifest["fit_config"]["model_config_sha256"]
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    register()
    graph = args.execution_mode == "cuda_graph"
    llm = LLM(
        model=str(args.model.resolve()), tensor_parallel_size=8, dtype="bfloat16",
        max_model_len=args.prefill_tokens + args.decode_tokens,
        max_num_seqs=8, max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        async_scheduling=False, gpu_memory_utilization=0.5,
        enforce_eager=not graph, disable_log_stats=False,
        compilation_config={"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY" if graph else "NONE",
                            "cudagraph_capture_sizes": [1, 2, 4, 8] if graph else [],
                            "cudagraph_num_of_warmups": 2},
        hf_overrides={"architectures": [architecture],
                      "basisserve_c1_factor_dir": str(root), "basisserve_c1_result_sha256": sha},
        worker_extension_cls="basisserve.vllm.worker_extension.BasisServeWorkerExtension",
    )
    tokenizer = llm.get_tokenizer()
    seed = tokenizer.encode("The capital of France is Paris. Tensor parallelism distributes a model across GPUs. ", add_special_tokens=False)
    sampling = SamplingParams(temperature=0, max_tokens=args.decode_tokens, ignore_eos=True, detokenize=False)
    runs = []
    for batch in (1, 3):
        prompts = [{"prompt_token_ids": (seed * ((args.prefill_tokens + len(seed) - 1)//len(seed)))[:args.prefill_tokens - i * 7]}
                   for i in range(batch)]
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        tokens = [list(output.outputs[0].token_ids) for output in outputs]
        assert len(outputs) == batch and all(len(row) == args.decode_tokens for row in tokens)
        runs.append(dict(batch=batch, output_tokens=tokens, seconds=time.perf_counter()-start))
    statistics = llm.collective_rpc("basisserve_c1_statistics")
    assert len(statistics) == 8
    assert all(item["loaded_value_layers"] == layers for item in statistics)
    assert all(item["capture_calls"] > 0 for item in statistics) if graph else all(item["capture_calls"] == 0 for item in statistics)
    if args.reference_json:
        reference = json.loads(args.reference_json.read_text())
        assert [row["output_tokens"] for row in runs] == [row["output_tokens"] for row in reference["runs"]]
    payload = dict(status="passed", command=shlex.join([sys.executable, *sys.argv]),
                   torch=torch.__version__, vllm=vllm.__version__, factor_sha256=sha,
                   execution_mode=args.execution_mode, prefill_tokens=args.prefill_tokens,
                   decode_tokens=args.decode_tokens, max_num_batched_tokens=args.max_num_batched_tokens,
                   runs=runs, workers=statistics)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
