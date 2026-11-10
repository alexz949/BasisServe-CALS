"""Compile the TP8 V96/B16R16 decode kernels on one L40S GPU."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

from basisserve.kernels.feature_ragged_allgather import _load_extension as load_allgather
from basisserve.kernels.mapped_host_paged_attention import prepare_mapped_host_paged_attention_extension
from basisserve.kernels.slot_indexed_attention import slot_indexed_attention
from benchmarks.system.bench_tp1_sparse_full_v6 import _load_postprocess_extension, _load_slot_extension
from benchmarks.system.two_stage_router import compile_fine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    assert Path(sys.prefix).name == "basis"
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    assert args.build_only or torch.cuda.get_device_capability(0) == (8, 9)
    assert os.environ.get("CUDA_HOME")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    os.environ.setdefault("MAX_JOBS", "2")

    args.output.mkdir(parents=True, exist_ok=True)
    record_path = args.output / "compile.jsonl"
    record_path.write_text("")
    hardware = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_home": os.environ["CUDA_HOME"],
        "arch": os.environ["TORCH_CUDA_ARCH_LIST"],
        "host": os.uname().nodename,
        "build_only": args.build_only,
    }

    def record(name: str, operation) -> None:
        started = time.perf_counter()
        print(f"starting {name}", flush=True)
        operation()
        entry = {"component": name, "status": "ready", "seconds": time.perf_counter() - started}
        with record_path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    (args.output / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    build_root = Path(os.environ.get("TMPDIR", "/tmp")) / "basisserve_tp8_joint_v96"
    record(
        "mapped_host_v96_page32",
        lambda: prepare_mapped_host_paged_attention_extension(
            value_dim=96, queries_per_kv=4, page_size=32, base_rank=16, residual_rank=16
        ),
    )
    record("two_stage_v96", lambda: compile_fine(build_root, value_dim=96))
    record("persistent_k_slots", _load_slot_extension)
    record("fused_postprocess", _load_postprocess_extension)
    record("feature_allgather", load_allgather)

    def smoke_slot_attention() -> None:
        query = torch.randn(1, 4, 1, 128, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(1, 1, 64, 128, device="cuda", dtype=torch.bfloat16)
        value = torch.randn(1, 1, 64, 96, device="cuda", dtype=torch.bfloat16)
        selected = torch.arange(64, device="cuda").view(1, 1, 64)
        slots = selected.clone()
        partial = torch.empty(1, 4, 2, 96, device="cuda", dtype=torch.float32)
        lse = torch.empty(4, 2, device="cuda", dtype=torch.float32)
        output = torch.empty(1, 4, 1, 96, device="cuda", dtype=torch.float32)
        result = slot_indexed_attention(
            query, key, value, selected, slots, (partial, lse, output), scale=128**-0.5
        )
        torch.cuda.synchronize()
        assert bool(torch.isfinite(result).all())
        scores = torch.matmul(query.float(), key.expand(1, 4, 64, 128).float().transpose(-1, -2))
        probabilities = torch.softmax(scores * 128**-0.5, dim=-1)
        expected = torch.matmul(probabilities, value.expand(1, 4, 64, 96).float())
        torch.testing.assert_close(result, expected, rtol=3e-3, atol=3e-3)

    if not args.build_only:
        record("slot_attention_v96_smoke", smoke_slot_attention)
    print("TP8 V96/B16R16 compile setup complete", flush=True)


if __name__ == "__main__":
    main()
