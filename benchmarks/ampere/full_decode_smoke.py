"""Check fixed-capacity V96 decode masking against a dense FP32 reference."""

import argparse
import json
from pathlib import Path

import torch

from basisserve.kernels.compressed_v_decode_attention import compressed_v_decode_attention_triton
from benchmarks.ampere.build_kernels import configure_local_caches


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_local_caches(0)
    torch.manual_seed(20260922)
    torch.set_num_threads(2)
    checks = []
    for capacity in (4241, 32913):
        query = torch.randn(1, 4, 1, 128, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(1, 1, capacity, 128, device="cuda", dtype=torch.bfloat16)
        value = torch.randn(1, 1, capacity, 96, device="cuda", dtype=torch.bfloat16)
        valid = torch.empty((), device="cuda", dtype=torch.int32)
        for length in (capacity - 145, capacity - 1):
            valid.fill_(length)
            # Poison unused storage: masks must exclude the entire unused tail.
            key[:, :, length:] = torch.nan
            value[:, :, length:] = torch.nan
            actual = compressed_v_decode_attention_triton(query, key, value, valid_sequence_length=valid)
            scores = query.float() @ key[:, :, :length].float().transpose(-1, -2) * (128**-0.5)
            expected = scores.softmax(-1) @ value[:, :, :length].float()
            torch.testing.assert_close(actual.float(), expected, atol=0.002, rtol=0.02)
            checks.append({"capacity": capacity, "length": length,
                           "max_abs_error": float((actual.float() - expected).abs().max())})
            key[:, :, length:] = torch.randn_like(key[:, :, length:])
            value[:, :, length:] = torch.randn_like(value[:, :, length:])
    result = {"status": "complete", "checks": checks,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
