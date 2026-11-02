"""Check native V96 prefill correctness and latency on an Ampere GPU."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from basisserve.kernels.compressed_v_decode_attention import compressed_v_prefill_attention


ROOT = Path(__file__).resolve().parents[2]


def time_cuda(function, repeats=5):
    for _ in range(2):
        function()
    torch.cuda.synchronize()
    durations = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        stop.record()
        stop.synchronize()
        durations.append(start.elapsed_time(stop))
    return statistics.median(durations)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", choices=("triton", "cudnn"), default="triton")
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() in ((8, 0), (8, 6))
    torch.manual_seed(20260922)
    cases = []
    for length in (512, 4096):
        query = torch.randn(1, 4, length, 128, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(1, 1, length, 128, device="cuda", dtype=torch.bfloat16)
        value = torch.randn(1, 1, length, 96, device="cuda", dtype=torch.bfloat16)

        def native():
            if args.candidate == "triton":
                return compressed_v_prefill_attention(query, key, value, scale=128**-0.5)
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return F.scaled_dot_product_attention(
                    query, key, value, is_causal=True, enable_gqa=True,
                    scale=128**-0.5,
                )

        def padded():
            wide = F.pad(value, (0, 32))
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    query, key, wide, is_causal=True, enable_gqa=True,
                    scale=128**-0.5,
                )[..., :96]

        observed = native()
        expected = padded()
        torch.testing.assert_close(observed.float(), expected.float(), rtol=0.03, atol=0.03)
        cases.append({
            "length": length,
            "candidate_v96_ms": time_cuda(native),
            "padded_flash_ms": time_cuda(padded),
            "relative_l2_error": float((observed.float()-expected.float()).norm()
                                       / expected.float().norm()),
            "maximum_absolute_error": float((observed.float()-expected.float()).abs().max()),
        })
    source = ROOT / "basisserve/kernels/compressed_v_decode_attention.py"
    result = {
        "status": "complete",
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "candidate": args.candidate,
        "cases": cases,
        "source_sha256": {
            "basisserve/kernels/compressed_v_decode_attention.py": hashlib.sha256(source.read_bytes()).hexdigest(),
            "benchmarks/ampere/prefill_smoke.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
