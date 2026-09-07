"""GPU-driven Page32 exact-K attention over CUDA-mapped host memory."""

from __future__ import annotations

import fcntl
from functools import lru_cache
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor


_EXTENSION_BASENAME = "basisserve_mapped_host_paged_attention_v4"
_PAGE_SIZE = 32
_QK_DIM = 128
_VALUE_DIM = 80
_QUERIES_PER_KV = 4
_BASE_RANK = 16
_RESIDUAL_RANK = 8
_DEFAULT_SPLITS = 32
_MAX_SPLITS = 128


@lru_cache(maxsize=1)
def _load_extension():
    assert torch.cuda.is_available()
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    assert configured is not None
    cuda_home = Path(configured).expanduser().resolve()
    assert (cuda_home / "bin" / "nvcc").is_file()
    source_root = Path(__file__).resolve().parent / "csrc"
    sources = (
        source_root / "mapped_host_paged_attention.cpp",
        source_root / "mapped_host_paged_attention.cu",
        source_root / "conditional_router_page32.cu",
    )
    assert all(source.is_file() for source in sources)
    major, minor = torch.cuda.get_device_capability()
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    extension_name = f"{_EXTENSION_BASENAME}_sm{major}{minor}"
    from torch.utils import cpp_extension

    cpp_extension.CUDA_HOME = str(cuda_home)
    lock_path = Path(tempfile.gettempdir()) / (
        f"{extension_name}.{os.getuid()}.lock"
    )
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        extension = cpp_extension.load(
            name=extension_name,
            sources=[str(source) for source in sources],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "--threads",
                "4",
            ],
            with_cuda=True,
            build_directory=os.environ.get("BASISSERVE_EXT_BUILD_DIR"),
            verbose=os.environ.get("BASISSERVE_VERBOSE_BUILD", "0") == "1",
        )
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return extension


def prepare_mapped_host_paged_attention_extension() -> None:
    """Compile and load the mapped-host CUDA extension."""

    _load_extension()


def conditional_router_page32_lse(
    query: Tensor,
    base_code: Tensor,
    residual_code: Tensor,
    *,
    base_right: Tensor,
    base_bias: Tensor,
    residual_query: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    scale: float,
    query_code: Tensor | None = None,
    output: Tensor | None = None,
) -> Tensor:
    """Run the cached-Base16/R8/Page32 conditional router."""

    batch, kv_heads, tokens, base_rank = map(int, base_code.shape)
    query_heads = int(query.shape[1])
    pages = math.ceil(tokens / _PAGE_SIZE)
    assert query.is_cuda and base_code.is_cuda and residual_code.is_cuda
    assert base_right.is_cuda and base_bias.is_cuda
    assert residual_query.is_cuda and rope_cos.is_cuda and rope_sin.is_cuda
    tensors = (
        query,
        base_code,
        residual_code,
        base_right,
        base_bias,
        residual_query,
        rope_cos,
        rope_sin,
    )
    assert all(tensor.dtype == torch.bfloat16 for tensor in tensors)
    assert torch.cuda.get_device_capability(query.device)[0] >= 8
    assert tuple(query.shape) == (batch, query_heads, 1, _QK_DIM)
    assert query_heads == _QUERIES_PER_KV * kv_heads
    assert base_rank == _BASE_RANK and tokens > 0
    assert tuple(residual_code.shape) == (
        batch,
        kv_heads,
        tokens,
        _RESIDUAL_RANK,
    )
    assert tuple(base_right.shape) == (kv_heads, _BASE_RANK, _QK_DIM)
    assert tuple(base_bias.shape) == (kv_heads, _QK_DIM)
    assert tuple(residual_query.shape) == (
        query_heads,
        _QK_DIM,
        _RESIDUAL_RANK,
    )
    assert tuple(rope_cos.shape) == (tokens, _QK_DIM // 2)
    assert tuple(rope_sin.shape) == tuple(rope_cos.shape)
    assert all(tensor.device == query.device for tensor in tensors)
    assert all(tensor.stride(-1) == 1 for tensor in tensors)
    assert base_right.is_contiguous()
    assert base_bias.is_contiguous() and residual_query.is_contiguous()
    assert rope_cos.is_contiguous() and rope_sin.is_contiguous()
    if query_code is None:
        query_code = torch.empty(
            batch,
            kv_heads,
            _QUERIES_PER_KV,
            _RESIDUAL_RANK,
            dtype=torch.bfloat16,
            device=query.device,
        )
    if output is None:
        output = torch.empty(
            batch,
            kv_heads,
            _QUERIES_PER_KV,
            pages,
            dtype=torch.float32,
            device=query.device,
        )
    assert tuple(query_code.shape) == (
        batch,
        kv_heads,
        _QUERIES_PER_KV,
        _RESIDUAL_RANK,
    )
    assert query_code.dtype == torch.bfloat16 and query_code.device == query.device
    assert query_code.is_contiguous()
    assert tuple(output.shape) == (batch, kv_heads, _QUERIES_PER_KV, pages)
    assert output.dtype == torch.float32 and output.device == query.device
    selected_scale = float(scale)
    assert math.isfinite(selected_scale) and selected_scale > 0.0
    return _load_extension().conditional_router_page32_lse(
        query,
        base_code,
        residual_code,
        base_right,
        base_bias,
        residual_query,
        rope_cos,
        rope_sin,
        query_code,
        output,
        selected_scale,
    )


def conditional_router_append_decode(
    key: Tensor,
    value: Tensor,
    *,
    base_left: Tensor,
    base_right: Tensor,
    base_bias: Tensor,
    residual_encoder: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    value_cache: Tensor,
    base_cache: Tensor,
    residual_cache: Tensor,
    rope_cos_cache: Tensor,
    rope_sin_cache: Tensor,
    start: int,
    write_rope: bool,
) -> None:
    """Append one C1 routing token with one fixed-geometry CUDA kernel."""

    batch, kv_heads, tokens, head_dim = map(int, key.shape)
    assert tokens == 1 and head_dim == _QK_DIM
    assert tuple(value.shape) == (batch, kv_heads, 1, _VALUE_DIM)
    assert tuple(base_left.shape) == (kv_heads, _VALUE_DIM, _BASE_RANK)
    assert tuple(base_right.shape) == (kv_heads, _BASE_RANK, _QK_DIM)
    assert tuple(base_bias.shape) == (kv_heads, _QK_DIM)
    assert tuple(residual_encoder.shape) == (
        kv_heads,
        _QK_DIM,
        _RESIDUAL_RANK,
    )
    assert tuple(rope_cos.shape) == (1, _QK_DIM // 2)
    assert tuple(rope_sin.shape) == tuple(rope_cos.shape)
    capacity = int(value_cache.shape[2])
    assert tuple(value_cache.shape) == (batch, kv_heads, capacity, _VALUE_DIM)
    assert tuple(base_cache.shape) == (
        batch,
        kv_heads,
        capacity,
        _BASE_RANK,
    )
    assert tuple(residual_cache.shape) == (
        batch,
        kv_heads,
        capacity,
        _RESIDUAL_RANK,
    )
    assert tuple(rope_cos_cache.shape) == (capacity, _QK_DIM // 2)
    assert tuple(rope_sin_cache.shape) == tuple(rope_cos_cache.shape)
    tensors = (
        key,
        value,
        base_left,
        base_right,
        base_bias,
        residual_encoder,
        rope_cos,
        rope_sin,
        value_cache,
        base_cache,
        residual_cache,
        rope_cos_cache,
        rope_sin_cache,
    )
    assert all(tensor.is_cuda for tensor in tensors)
    assert all(tensor.device == key.device for tensor in tensors)
    assert all(tensor.dtype == torch.bfloat16 for tensor in tensors)
    assert all(tensor.stride(-1) == 1 for tensor in tensors)
    assert base_left.is_contiguous() and base_right.is_contiguous()
    assert base_bias.is_contiguous() and residual_encoder.is_contiguous()
    assert 0 <= int(start) < capacity
    assert torch.cuda.get_device_capability(key.device)[0] >= 8
    _load_extension().conditional_router_append_decode(
        key,
        value,
        base_left,
        base_right,
        base_bias,
        residual_encoder,
        rope_cos,
        rope_sin,
        value_cache,
        base_cache,
        residual_cache,
        rope_cos_cache,
        rope_sin_cache,
        int(start),
        bool(write_rope),
    )


def select_fixed_group_max_pages_cuda(
    page_log_mass: Tensor,
    *,
    pages_per_kv_head: int,
    pinned_prefix_pages: int,
    force_current_page: bool = True,
    output: Tensor | None = None,
) -> Tensor:
    """Normalize, aggregate, and select one physical page list per KV head."""

    batch, kv_heads, queries_per_kv, pages = map(int, page_log_mass.shape)
    selected_count = min(int(pages_per_kv_head), pages)
    prefix_count = min(int(pinned_prefix_pages), pages)
    fixed_count = prefix_count + int(
        bool(force_current_page) and pages - 1 >= prefix_count
    )
    assert page_log_mass.is_cuda and page_log_mass.dtype == torch.float32
    assert queries_per_kv == _QUERIES_PER_KV
    assert page_log_mass.stride(-1) == 1
    assert 0 < pages <= 4096
    assert 0 < selected_count <= 128
    assert 0 <= prefix_count and fixed_count <= selected_count
    if output is None:
        output = torch.empty(
            batch,
            kv_heads,
            selected_count,
            dtype=torch.long,
            device=page_log_mass.device,
        )
    assert tuple(output.shape) == (batch, kv_heads, selected_count)
    assert output.dtype == torch.long and output.device == page_log_mass.device
    assert output.is_contiguous()
    return _load_extension().select_fixed_group_max_pages(
        page_log_mass,
        output,
        int(pages_per_kv_head),
        int(pinned_prefix_pages),
        bool(force_current_page),
    )


def mapped_host_bf16_empty(
    *,
    batch: int,
    kv_heads: int,
    capacity: int,
    head_dim: int = _QK_DIM,
) -> Tensor:
    """Allocate CUDA-mapped BF16 host storage owned by a CPU tensor."""

    assert batch > 0 and kv_heads > 0 and capacity > 0
    assert head_dim == _QK_DIM
    return _load_extension().mapped_host_bf16_empty(
        int(batch),
        int(kv_heads),
        int(capacity),
        int(head_dim),
    )


def append_mapped_host_key(host_key: Tensor, key: Tensor, *, start: int) -> Tensor:
    """Append one contiguous CUDA exact-Key block directly into mapped host RAM."""

    assert host_key.device.type == "cpu" and host_key.dtype == torch.bfloat16
    assert key.is_cuda and key.dtype == torch.bfloat16 and key.ndim == 4
    assert key.stride(-1) == 1
    return _load_extension().append(host_key, key, int(start))


def mapped_host_device_pointer(host_key: Tensor) -> int:
    """Return the stable device alias of a CUDA-mapped host allocation."""

    assert host_key.device.type == "cpu" and host_key.dtype == torch.bfloat16
    return int(_load_extension().device_pointer(host_key))


def mapped_host_page32_v80_attention(
    host_key: Tensor,
    query: Tensor,
    value: Tensor,
    selected_page_ids: Tensor,
    *,
    sequence_length: int,
    scale: float | None = None,
    splits: int = _DEFAULT_SPLITS,
    host_key_device_pointer: int | None = None,
    workspace: Tensor | None = None,
    output: Tensor | None = None,
) -> Tensor:
    """Compute exact selected-page QK/softmax/V80 without a GPU K staging tensor."""

    batch, query_heads, query_tokens, qk_dim = map(int, query.shape)
    value_batch, kv_heads, value_tokens, value_dim = map(int, value.shape)
    page_batch, page_heads, page_slots = map(int, selected_page_ids.shape)
    assert host_key.device.type == "cpu" and host_key.dtype == torch.bfloat16
    assert query.is_cuda and value.is_cuda and selected_page_ids.is_cuda
    assert query.dtype == value.dtype == torch.bfloat16
    assert selected_page_ids.dtype == torch.int64
    assert query_tokens == 1 and qk_dim == _QK_DIM
    assert value_dim == _VALUE_DIM and value_batch == batch
    assert query_heads == _QUERIES_PER_KV * kv_heads
    assert (page_batch, page_heads) == (batch, kv_heads)
    assert 0 < sequence_length <= value_tokens <= int(host_key.shape[2])
    selected_splits = min(int(splits), page_slots)
    assert 0 < selected_splits <= _MAX_SPLITS
    if workspace is None:
        workspace = torch.empty(
            batch * query_heads,
            _MAX_SPLITS,
            _VALUE_DIM + 2,
            dtype=torch.float32,
            device=query.device,
        )
    if output is None:
        output = torch.empty(
            batch,
            query_heads,
            1,
            _VALUE_DIM,
            dtype=torch.bfloat16,
            device=query.device,
        )
    assert workspace.shape[0] == batch * query_heads
    assert workspace.shape[1] >= selected_splits
    assert workspace.shape[2] == _VALUE_DIM + 2
    assert workspace.dtype == torch.float32 and workspace.device == query.device
    assert tuple(output.shape) == (batch, query_heads, 1, _VALUE_DIM)
    assert output.dtype == torch.bfloat16 and output.device == query.device
    selected_scale = qk_dim**-0.5 if scale is None else float(scale)
    assert math.isfinite(selected_scale) and selected_scale > 0.0
    selected_pointer = (
        mapped_host_device_pointer(host_key)
        if host_key_device_pointer is None
        else int(host_key_device_pointer)
    )
    assert selected_pointer != 0
    return _load_extension().attention(
        selected_pointer,
        int(host_key.shape[2]),
        query,
        value,
        selected_page_ids.contiguous(),
        workspace,
        output,
        int(sequence_length),
        selected_scale,
        selected_splits,
    )


def gpu_page32_v80_attention(
    key: Tensor,
    query: Tensor,
    value: Tensor,
    selected_page_ids: Tensor,
    *,
    sequence_length: int,
    scale: float | None = None,
    splits: int = _DEFAULT_SPLITS,
    workspace: Tensor | None = None,
    output: Tensor | None = None,
) -> Tensor:
    """Run the mapped-host kernel against a GPU-resident exact-K oracle."""

    batch, query_heads, query_tokens, qk_dim = map(int, query.shape)
    value_batch, kv_heads, value_tokens, value_dim = map(int, value.shape)
    page_batch, page_heads, page_slots = map(int, selected_page_ids.shape)
    assert key.is_cuda and key.dtype == torch.bfloat16 and key.ndim == 4
    assert query.is_cuda and value.is_cuda and selected_page_ids.is_cuda
    assert query.dtype == value.dtype == torch.bfloat16
    assert selected_page_ids.dtype == torch.int64
    assert query_tokens == 1 and qk_dim == _QK_DIM
    assert value_dim == _VALUE_DIM and value_batch == batch
    assert query_heads == _QUERIES_PER_KV * kv_heads
    assert (page_batch, page_heads) == (batch, kv_heads)
    assert tuple(key.shape[:2]) == (batch, kv_heads)
    assert int(key.shape[-1]) == _QK_DIM
    assert 0 < sequence_length <= value_tokens <= int(key.shape[2])
    selected_splits = min(int(splits), page_slots)
    assert 0 < selected_splits <= _MAX_SPLITS
    if workspace is None:
        workspace = torch.empty(
            batch * query_heads,
            _MAX_SPLITS,
            _VALUE_DIM + 2,
            dtype=torch.float32,
            device=query.device,
        )
    if output is None:
        output = torch.empty(
            batch,
            query_heads,
            1,
            _VALUE_DIM,
            dtype=torch.bfloat16,
            device=query.device,
        )
    assert workspace.shape[0] == batch * query_heads
    assert workspace.shape[1] >= selected_splits
    assert workspace.shape[2] == _VALUE_DIM + 2
    assert workspace.dtype == torch.float32 and workspace.device == query.device
    assert tuple(output.shape) == (batch, query_heads, 1, _VALUE_DIM)
    assert output.dtype == torch.bfloat16 and output.device == query.device
    selected_scale = qk_dim**-0.5 if scale is None else float(scale)
    assert math.isfinite(selected_scale) and selected_scale > 0.0
    return _load_extension().attention(
        int(key.data_ptr()),
        int(key.shape[2]),
        query,
        value,
        selected_page_ids.contiguous(),
        workspace,
        output,
        int(sequence_length),
        selected_scale,
        selected_splits,
    )


__all__ = [
    "append_mapped_host_key",
    "conditional_router_append_decode",
    "conditional_router_page32_lse",
    "gpu_page32_v80_attention",
    "mapped_host_bf16_empty",
    "mapped_host_device_pointer",
    "mapped_host_page32_v80_attention",
    "prepare_mapped_host_paged_attention_extension",
    "select_fixed_group_max_pages_cuda",
]
