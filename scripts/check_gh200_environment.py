"""Short Slurm GPU check before installing or fitting model checkpoints."""

import importlib.metadata
import json
import os
import platform
import sys
import time

import torch


def main():
    torch.set_num_threads(2)
    report = {
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_build": torch.version.cuda,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "datasets", "lm-eval", "kernels")
        },
        "cuda_available": torch.cuda.is_available(),
    }
    print(json.dumps(report), flush=True)
    if not report["cuda_available"]:
        return 1
    torch.manual_seed(0)
    results = []
    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        prop = torch.cuda.get_device_properties(device)
        start = time.monotonic()
        a = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        b = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        product = a @ b
        reference = a.float() @ b.float()
        relative_error = float((product.float() - reference).norm() / reference.norm())
        finite = bool(torch.isfinite(product).all())
        torch.cuda.synchronize(device)
        results.append({
            "index": index,
            "name": prop.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_gib": prop.total_memory / 2**30,
            "bf16_matmul_finite": finite,
            "relative_error_to_fp32": relative_error,
            "seconds": time.monotonic() - start,
            "passed": finite and relative_error < 0.02,
        })
    print(json.dumps({"gpu_checks": results}), flush=True)
    # Import the evaluation adapter and target model after testing CUDA.
    from lm_eval.models.huggingface import HFLM
    from transformers import Qwen3_5ForCausalLM

    print(json.dumps({"imports": [HFLM.__name__, Qwen3_5ForCausalLM.__name__]}), flush=True)
    return 0 if all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    sys.exit(main())
