#!/usr/bin/env python3
"""Measure numerical PSD margins of stored C1 covariance matrices."""

from __future__ import annotations

import argparse

from safetensors.torch import load_file
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    tensors = load_file(args.layer_file, device="cpu")
    for key in ("fit_covariance", "heldout_covariance"):
        covariance = tensors[key].to(device=args.device, dtype=torch.float64)
        covariance = 0.5 * (covariance + covariance.T)
        trace_scale = float(torch.diagonal(covariance).mean())
        minimum_eigenvalue = float(torch.linalg.eigvalsh(covariance)[0])
        required_relative_shift = max(0.0, -minimum_eigenvalue) / trace_scale
        print(
            f"[PSD] key={key} min_eigenvalue={minimum_eigenvalue:.12e} "
            f"trace_scale={trace_scale:.12e} "
            f"required_relative_shift={required_relative_shift:.12e} "
            f"damped_1e-5_min={minimum_eigenvalue + 1e-5 * trace_scale:.12e}",
            flush=True,
        )
        del covariance
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
