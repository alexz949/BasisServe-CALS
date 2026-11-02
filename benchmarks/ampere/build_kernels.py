"""Prepare the TP8 Ampere extensions in a single-GPU Slurm allocation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch

from basisserve.kernels.feature_ragged_allgather import prepare_feature_ragged_extension
from basisserve.kernels.mapped_host_paged_attention import prepare_mapped_host_paged_attention_extension
from benchmarks.ampere.kernel_smoke import extension
from benchmarks.ampere.two_stage import compile_fine


def configure_local_caches(rank):
    os.environ["TORCH_EXTENSIONS_DIR"] = f"/tmp/basisserve_ampere_extensions_{os.getuid()}"
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/basisserve_ampere_triton_{os.getuid()}_{rank}"
    os.environ["CUDA_CACHE_PATH"] = f"/tmp/basisserve_ampere_cuda_{os.getuid()}_{rank}"


def load_kernels(build_dir):
    builders = (
        ("fine", lambda: compile_fine(build_dir)),
        ("postprocess", lambda: extension("postprocess", "basisserve/kernels/csrc/fused_decode_postprocess.cu")),
        ("slots", lambda: extension("slots", "basisserve/kernels/csrc/persistent_key_slots.cu")),
        ("mapped_host", prepare_mapped_host_paged_attention_extension),
        ("feature_allgather", prepare_feature_ragged_extension),
    )
    kernels = {}
    for name, build in builders:
        started = time.perf_counter()
        print(json.dumps({"phase": "extension_start", "extension": name,
                          "rank": int(os.environ.get("LOCAL_RANK", "0")),
                          "time_utc": datetime.now(timezone.utc).isoformat()}), flush=True)
        result = build()
        if name in ("fine", "postprocess", "slots"):
            kernels[name] = result
        print(json.dumps({"phase": "extension_ready", "extension": name,
                          "rank": int(os.environ.get("LOCAL_RANK", "0")),
                          "seconds": time.perf_counter() - started}), flush=True)
    return kernels


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_local_caches(0)
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability() in ((8, 0), (8, 6))
    load_kernels(args.build_dir)
    print(json.dumps({"phase": "all_extensions_ready",
                      "gpu": torch.cuda.get_device_name(),
                      "cache": os.environ["TORCH_EXTENSIONS_DIR"]}), flush=True)


if __name__ == "__main__":
    main()
