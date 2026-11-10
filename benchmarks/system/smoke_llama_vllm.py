"""Real vLLM Llama TP smoke; these timings are not benchmark results."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys

print(f"SMOKE pid={os.getpid()} importing torch/vLLM", flush=True)
from safetensors import safe_open
import torch
import vllm
from vllm import LLM, SamplingParams
from basisserve.vllm import register


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--tp", type=int, choices=(4, 8), required=True)
    parser.add_argument("--graph", choices=("eager", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--arm", choices=("dense", "als_full", "basis_joint"), default="dense")
    parser.add_argument("--factors", type=Path)
    args = parser.parse_args()
    print(f"SMOKE pid={os.getpid()} checking CUDA and native dependencies", flush=True)
    assert Path(sys.prefix).name == "basis"
    assert os.environ.get("NCCL_P2P_DISABLE") == "1"
    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"
    assert torch.cuda.device_count() == args.tp
    assert all(torch.cuda.get_device_capability(i) == (8, 9) for i in range(args.tp))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import torchvision
    import torchaudio
    import triton

    register()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with safe_open(str(args.windows), framework="pt", device="cpu") as handle:
        assert "input_ids" in handle.keys()
        inputs = handle.get_slice("input_ids")[32:35, :4096]
    overrides = {}
    if args.arm != "dense":
        assert args.factors is not None
        manifest_bytes = (args.factors / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        assert hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest() == manifest["model"]["config_sha256"]
        overrides = dict(
            architectures=["BasisServeLlamaJointALSForCausalLM"],
            basisserve_joint_factor_dir=str(args.factors),
            basisserve_joint_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        )
        if args.arm == "basis_joint":
            overrides.update(basisserve_joint_router_dir=str(args.factors / "router_b16r16"),
                             basisserve_host_budget_bytes_per_rank=32 * 1024**3)
    configuration = dict(
        model=str(args.model), tensor_parallel_size=args.tp,
        distributed_executor_backend="mp", dtype="bfloat16",
        max_model_len=4112, max_num_seqs=1, max_num_batched_tokens=8192,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        async_scheduling=False, gpu_memory_utilization=0.8,
        enforce_eager=args.graph == "eager", disable_log_stats=False,
        disable_custom_all_reduce=True, seed=0,
        compilation_config={
            "mode": "NONE",
            "cudagraph_mode": "NONE" if args.graph == "eager" else "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1], "cudagraph_num_of_warmups": 1,
        },
        worker_extension_cls="benchmarks.system.vllm_smoke_worker.LlamaSmokeWorker",
        hf_overrides=overrides, block_size=32,
    )
    result = dict(
        status="initializing", scope="vLLM TP and CUDA Graph correctness smoke",
        command=shlex.join([sys.executable, *sys.argv]),
        conda_environment=sys.prefix, torch=torch.__version__,
        cuda_runtime=torch.version.cuda, vllm=vllm.__version__,
        torchvision=torchvision.__version__, torchaudio=torchaudio.__version__,
        triton=triton.__version__, arm=args.arm,
        transformers=importlib.metadata.version("transformers"),
        configuration=configuration, host=os.uname().nodename,
        visible_gpus=os.environ.get("CUDA_VISIBLE_DEVICES"),
        nccl_p2p_disabled=True, tf32_override=os.environ["NVIDIA_TF32_OVERRIDE"],
        prompt_sha256=hashlib.sha256(inputs[:2].numpy().tobytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"SMOKE initializing vLLM arm={args.arm} graph={args.graph}", flush=True)
    llm = LLM(**configuration)
    sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True, detokenize=False)
    print("SMOKE separate warmup request", flush=True)
    llm.generate([{"prompt_token_ids": inputs[2].tolist()}], sampling, use_tqdm=False)
    generated = []
    for prompt in inputs[:2]:
        print(f"SMOKE test request={len(generated)}", flush=True)
        outputs = llm.generate([{"prompt_token_ids": prompt.tolist()}], sampling, use_tqdm=False)
        assert len(outputs) == 1
        tokens = list(outputs[0].outputs[0].token_ids)
        assert len(tokens) == 16 and all(0 <= token < 128256 for token in tokens)
        generated.append(tokens)
    workers = llm.collective_rpc("llama_smoke_statistics")
    assert len(workers) == args.tp
    if args.arm != "dense":
        assert all(worker["joint"]["loaded_value_layers"] == 32 for worker in workers)
    if args.graph == "full":
        assert all(worker["cudagraph_captures"] > 0 for worker in workers)
    if args.reference:
        reference = json.loads(args.reference.read_text())
        assert reference["status"] == "complete"
        assert reference["arm"] == args.arm
        assert reference["configuration"]["tensor_parallel_size"] == args.tp
        assert reference["prompt_sha256"] == result["prompt_sha256"]
        assert reference["generated_token_ids"] == generated
    result.update(status="complete", generated_token_ids=generated, workers=workers,
                  reference=str(args.reference) if args.reference else None)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
