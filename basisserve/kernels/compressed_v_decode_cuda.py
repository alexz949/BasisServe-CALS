"""JIT loader for exact-rank grouped-GQA CUDA decode attention."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import torch


_EXTENSION_BASENAME = "basisserve_compressed_v_decode_cuda_v15"


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
        source_root / "c1_r32_routing.cu",
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


def launch_c1_paged_sparse_decode_cuda(
    query: torch.Tensor,
    packed_key_pages: torch.Tensor,
    value: torch.Tensor,
    selected_page_ids: torch.Tensor,
    workspace: torch.Tensor,
    output: torch.Tensor,
    *,
    scale: float,
    splits: int,
) -> torch.Tensor:
    """Launch shared-GQA sparse attention over packed exact-Key pages."""

    return _load_extension().paged_sparse_decode(
        query,
        packed_key_pages,
        value,
        selected_page_ids,
        workspace,
        output,
        float(scale),
        int(splits),
    )


def launch_c1_pack_exact_key_pages_cuda(
    exact_key: torch.Tensor,
    selected_page_ids: torch.Tensor,
    packed_key_pages: torch.Tensor,
) -> torch.Tensor:
    """Pack selected resident exact-Key pages into the offload staging layout."""

    return _load_extension().pack_exact_key_pages(
        exact_key,
        selected_page_ids,
        packed_key_pages,
    )


def launch_c1_r32_page_lse_cuda(
    query: torch.Tensor,
    routing_sidecar: torch.Tensor,
    query_projector: torch.Tensor,
    query_code: torch.Tensor,
    page_log_mass: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Project one GQA Query and stream R32 token codes into page LSEs."""

    return _load_extension().r32_page_lse(
        query,
        routing_sidecar,
        query_projector,
        query_code,
        page_log_mass,
        float(scale),
    )


def launch_c1_r32_topk_gqa_union_cuda(
    page_log_mass: torch.Tensor,
    selected_page_ids: torch.Tensor,
    selected_page_counts: torch.Tensor,
    *,
    top_pages_per_query: int,
) -> torch.Tensor:
    """Select per-Query Top-pages and emit sorted compact GQA unions."""

    return _load_extension().r32_topk_gqa_union(
        page_log_mass,
        selected_page_ids,
        selected_page_counts,
        int(top_pages_per_query),
    )


def launch_c1_dense_gqa_v96_decode_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_sequence_length: torch.Tensor,
    workspace: torch.Tensor,
    output: torch.Tensor,
    *,
    scale: float,
    splits: int,
) -> torch.Tensor:
    """Launch dense shared-GQA attention for exact K128 and C1-V96."""

    return _load_extension().dense_gqa_v96_decode(
        query,
        key,
        value,
        valid_sequence_length,
        workspace,
        output,
        float(scale),
        int(splits),
    )


__all__ = [
    "launch_c1_dense_gqa_v96_decode_cuda",
    "launch_c1_pack_exact_key_pages_cuda",
    "launch_c1_paged_sparse_decode_cuda",
    "launch_c1_r32_page_lse_cuda",
    "launch_c1_r32_topk_gqa_union_cuda",
    "launch_compressed_v_decode_cuda",
]
