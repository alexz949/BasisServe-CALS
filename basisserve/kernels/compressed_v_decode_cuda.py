"""JIT loader for exact-rank grouped-GQA CUDA decode attention."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import torch


_EXTENSION_BASENAME = "basisserve_compressed_v_decode_cuda_v8"


@lru_cache(maxsize=1)
def _load_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("compressed-V CUDA decode requires CUDA")
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not configured:
        raise FileNotFoundError(
            "CUDA_HOME is unset; load nvidia/cuda12/cuda/12.4.1 before running"
        )
    cuda_home = Path(configured).expanduser().resolve()
    if not (cuda_home / "bin" / "nvcc").is_file():
        raise FileNotFoundError(f"CUDA compiler is unavailable under {cuda_home}")

    source_root = Path(__file__).resolve().parent / "csrc"
    sources = (
        source_root / "compressed_v_decode_attention.cpp",
        source_root / "compressed_v_decode_attention.cu",
    )
    if any(not source.is_file() for source in sources):
        raise FileNotFoundError("compressed-V CUDA extension sources are incomplete")

    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.9;9.0+PTX")
    major, minor = torch.cuda.get_device_capability()
    extension_name = f"{_EXTENSION_BASENAME}_sm{major}{minor}"
    from torch.utils import cpp_extension

    cpp_extension.CUDA_HOME = str(cuda_home)
    return cpp_extension.load(
        name=extension_name,
        sources=[str(source) for source in sources],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--threads", "4"],
        with_cuda=True,
        build_directory=os.environ.get("BASISSERVE_EXT_BUILD_DIR"),
        verbose=os.environ.get("BASISSERVE_VERBOSE_BUILD", "0") == "1",
    )


def launch_compressed_v_decode_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    workspace: torch.Tensor,
    output: torch.Tensor,
    *,
    architecture: int,
    scale: float,
    splits: int,
    feature_major_output: bool,
) -> torch.Tensor:
    """Launch the compiled exact-rank CUDA kernel."""

    return _load_extension().decode(
        query,
        key,
        value,
        workspace,
        output,
        int(architecture),
        float(scale),
        int(splits),
        bool(feature_major_output),
    )


__all__ = ["launch_compressed_v_decode_cuda"]
