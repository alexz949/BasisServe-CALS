"""GPU-driven paged K128 attention over CUDA-mapped host memory."""

from __future__ import annotations

import fcntl
from functools import lru_cache
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor


_EXTENSION_BASENAME = "basisserve_mapped_host_paged_attention_v6"
_QK_DIM = 128
_DEFAULT_SPLITS = 32
_MAX_SPLITS = 128
_MAPPED_POINTER_ATTRIBUTE = "_basisserve_mapped_device_pointer"


@lru_cache(maxsize=None)
def _load_extension(value_dim=80, queries_per_kv=4, page_size=32, base_rank=16, residual_rank=8):
    assert 0 < value_dim <= 256 and 0 < queries_per_kv <= 16
    assert page_size in (1, 16, 32, 64) and 0 <= base_rank <= 128 and 0 < residual_rank <= 128
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
    extension_name = f"{_EXTENSION_BASENAME}_sm{major}{minor}_v{value_dim}_g{queries_per_kv}_p{page_size}_b{base_rank}_r{residual_rank}"
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
                f"-DBASIS_VALUE_DIM={value_dim}",
                f"-DBASIS_GQA={queries_per_kv}",
                f"-DBASIS_PAGE_SIZE={page_size}",
                f"-DBASIS_BASE_RANK={base_rank}",
                f"-DBASIS_RESIDUAL_RANK={residual_rank}",
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "--threads",
                "4",
            ],
            with_cuda=True,

            verbose=os.environ.get("BASISSERVE_VERBOSE_BUILD", "0") == "1",
        )
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return extension


def prepare_mapped_host_paged_attention_extension(
    *, value_dim: int = 80, queries_per_kv: int = 4, page_size: int = 32,
    base_rank: int = 16, residual_rank: int = 8,
) -> None:
    """Compile and load the mapped-host CUDA extension."""

    _load_extension(value_dim, queries_per_kv, page_size, base_rank, residual_rank)


def conditional_router_page_lse(
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
    page_size: int = 32,
    query_code: Tensor | None = None,
    query_code_prepared: bool = False,
    output: Tensor | None = None,
) -> Tensor:
    """Run K128 routing; infer ranks/GQA, specialize Page16/32/64 at compile time."""

    assert page_size in (16, 32, 64)
    batch, kv_heads, tokens, base_rank = map(int, base_code.shape)
    query_heads = int(query.shape[1])
    residual_rank = int(residual_code.shape[-1])
    assert query_heads % kv_heads == 0
    queries_per_kv = query_heads // kv_heads
    pages = math.ceil(tokens / page_size)
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
    assert query_heads == queries_per_kv * kv_heads
    assert 0 <= base_rank <= 128 and tokens > 0
    assert tuple(residual_code.shape) == (
        batch,
        kv_heads,
        tokens,
        residual_rank,
    )
    assert tuple(base_right.shape) == (kv_heads, base_rank, _QK_DIM)
    assert tuple(base_bias.shape) == (kv_heads, _QK_DIM)
    assert tuple(residual_query.shape) == (
        query_heads,
        _QK_DIM,
        residual_rank,
    )
    assert tuple(rope_cos.shape) == (tokens, _QK_DIM // 2)
    assert tuple(rope_sin.shape) == tuple(rope_cos.shape)
    assert all(tensor.device == query.device for tensor in tensors)
    assert all(tensor.stride(-1) == 1 for tensor in tensors)
    assert base_right.is_contiguous()
    assert base_bias.is_contiguous() and residual_query.is_contiguous()
    assert rope_cos.is_contiguous() and rope_sin.is_contiguous()
    assert not query_code_prepared or query_code is not None
    if query_code is None:
        query_code = torch.empty(
            batch,
            kv_heads,
            queries_per_kv,
            residual_rank,
            dtype=torch.bfloat16,
            device=query.device,
        )
    if output is None:
        output = torch.empty(
            batch,
            kv_heads,
            queries_per_kv,
            pages,
            dtype=torch.float32,
            device=query.device,
        )
    assert tuple(query_code.shape) == (
        batch,
        kv_heads,
        queries_per_kv,
        residual_rank,
    )
    assert query_code.dtype == torch.bfloat16 and query_code.device == query.device
    assert query_code.is_contiguous()
    assert tuple(output.shape) == (batch, kv_heads, queries_per_kv, pages)
    assert output.dtype == torch.float32 and output.device == query.device
    selected_scale = float(scale)
    assert math.isfinite(selected_scale) and selected_scale > 0.0
    return _load_extension(queries_per_kv=queries_per_kv, page_size=page_size, base_rank=base_rank, residual_rank=residual_rank).conditional_router_page_lse(
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
        bool(query_code_prepared),
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
    mapped_host_key: Tensor | None = None,
    mapped_host_key_device_pointer: int | None = None,
) -> None:
    """Append one C1 routing token with one shape-specialized CUDA kernel."""

    batch, kv_heads, tokens, head_dim = map(int, key.shape)
    value_dim = int(value.shape[-1])
    base_rank = int(base_left.shape[-1])
    residual_rank = int(residual_encoder.shape[-1])
    assert tokens == 1 and head_dim == _QK_DIM
    assert tuple(value.shape) == (batch, kv_heads, 1, value_dim)
    assert tuple(base_left.shape) == (kv_heads, value_dim, base_rank)
    assert tuple(base_right.shape) == (kv_heads, base_rank, _QK_DIM)
    assert tuple(base_bias.shape) == (kv_heads, _QK_DIM)
    assert tuple(residual_encoder.shape) == (
        kv_heads,
        _QK_DIM,
        residual_rank,
    )
    assert tuple(rope_cos.shape) == (1, _QK_DIM // 2)
    assert tuple(rope_sin.shape) == tuple(rope_cos.shape)
    capacity = int(value_cache.shape[2])
    assert tuple(value_cache.shape) == (batch, kv_heads, capacity, value_dim)
    assert tuple(base_cache.shape) == (
        batch,
        kv_heads,
        capacity,
        base_rank,
    )
    assert tuple(residual_cache.shape) == (
        batch,
        kv_heads,
        capacity,
        residual_rank,
    )
    expected_rope_rows = capacity if write_rope else 1
    assert tuple(rope_cos_cache.shape) == (expected_rope_rows, _QK_DIM // 2)
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
    if mapped_host_key is None:
        assert mapped_host_key_device_pointer is None
        mapped_pointer = 0
        mapped_capacity = 0
    else:
        assert mapped_host_key.device.type == "cpu"
        assert mapped_host_key.dtype == torch.bfloat16
        assert mapped_host_key.is_contiguous()
        assert tuple(mapped_host_key.shape) == (
            batch,
            kv_heads,
            capacity,
            _QK_DIM,
        )
        mapped_pointer = (
            mapped_host_device_pointer(mapped_host_key)
            if mapped_host_key_device_pointer is None
            else int(mapped_host_key_device_pointer)
        )
        mapped_capacity = capacity
    _load_extension(value_dim=value_dim, base_rank=base_rank, residual_rank=residual_rank).conditional_router_append_decode(
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
        mapped_pointer,
        mapped_capacity,
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
    assert 0 < queries_per_kv <= 16
    assert page_log_mass.stride(-1) == 1
    assert 0 < pages <= 8192
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
    return _load_extension(queries_per_kv=queries_per_kv).select_fixed_group_max_pages(
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
    extension = _load_extension()
    host_key = extension.mapped_host_bf16_empty(
        int(batch),
        int(kv_heads),
        int(capacity),
        int(head_dim),
    )
    setattr(
        host_key,
        _MAPPED_POINTER_ATTRIBUTE,
        int(extension.device_pointer(host_key)),
    )
    return host_key


def append_mapped_host_key(host_key: Tensor, key: Tensor, *, start: int) -> Tensor:
    """Append one contiguous CUDA exact-Key block directly into mapped host RAM."""

    assert host_key.device.type == "cpu" and host_key.dtype == torch.bfloat16
    assert key.is_cuda and key.dtype == torch.bfloat16 and key.ndim == 4
    assert key.stride(-1) == 1
    return _load_extension().append(host_key, key, int(start))


def mapped_host_device_pointer(host_key: Tensor) -> int:
    """Return the stable device alias of a CUDA-mapped host allocation."""

    assert host_key.device.type == "cpu" and host_key.dtype == torch.bfloat16
    pointer = getattr(host_key, _MAPPED_POINTER_ATTRIBUTE, None)
    if pointer is None:
        pointer = int(_load_extension().device_pointer(host_key))
        setattr(host_key, _MAPPED_POINTER_ATTRIBUTE, pointer)
    return int(pointer)


def mapped_host_paged_attention(
    host_key: Tensor,
    query: Tensor,
    value: Tensor,
    selected_page_ids: Tensor,
    *,
    sequence_length: int,
    page_size: int = 32,
    scale: float | None = None,
    splits: int = _DEFAULT_SPLITS,
    host_key_device_pointer: int | None = None,
    workspace: Tensor | None = None,
    output: Tensor | None = None,
    value_prefix: Tensor | None = None,
) -> Tensor:
    """Read mapped-host K128; infer V1..256/GQA1..16, use Page16/32/64.

    Negative page IDs are padding. Empty support returns zeros. Keys must be
    contiguous; query/value feature strides must be one.
    """

    batch, query_heads, query_tokens, qk_dim = map(int, query.shape)
    value_batch, kv_heads, value_tokens, value_dim = map(int, value.shape)
    prefix_width = 0 if value_prefix is None else int(value_prefix.shape[-1])
    if value_prefix is not None:
        assert value_prefix.shape[:3] == value.shape[:3]
        assert value_prefix.device == value.device and value_prefix.dtype == value.dtype
        assert value_prefix.stride(-1) == 1
        value_dim += prefix_width
    page_batch, page_heads, page_slots = map(int, selected_page_ids.shape)
    assert host_key.device.type == "cpu" and host_key.dtype == torch.bfloat16
    assert query.is_cuda and value.is_cuda and selected_page_ids.is_cuda
    assert query.dtype == value.dtype == torch.bfloat16
    assert selected_page_ids.dtype == torch.int64
    assert query_tokens == 1 and qk_dim == _QK_DIM
    assert 0 < value_dim <= 256 and value_batch == batch
    assert query_heads % kv_heads == 0
    queries_per_kv = query_heads // kv_heads
    assert value.stride(-1) == query.stride(-1) == 1
    assert value.device == selected_page_ids.device == query.device
    assert (page_batch, page_heads) == (batch, kv_heads)
    assert 0 < sequence_length <= value_tokens <= int(host_key.shape[2])
    selected_splits = min(int(splits), page_slots)
    assert 0 < selected_splits <= _MAX_SPLITS
    if workspace is None:
        workspace = torch.empty(
            batch * query_heads,
            _MAX_SPLITS,
            value_dim + 2,
            dtype=torch.float32,
            device=query.device,
        )
    if output is None:
        output = torch.empty(
            batch,
            query_heads,
            1,
            value_dim,
            dtype=torch.bfloat16,
            device=query.device,
        )
    assert workspace.shape[0] == batch * query_heads
    assert workspace.shape[1] >= selected_splits
    assert workspace.shape[2] == value_dim + 2
    assert workspace.dtype == torch.float32 and workspace.device == query.device
    assert tuple(output.shape) == (batch, query_heads, 1, value_dim)
    assert output.dtype == torch.bfloat16 and output.device == query.device
    selected_scale = qk_dim**-0.5 if scale is None else float(scale)
    assert math.isfinite(selected_scale) and selected_scale > 0.0
    selected_pointer = (
        mapped_host_device_pointer(host_key)
        if host_key_device_pointer is None
        else int(host_key_device_pointer)
    )
    assert host_key.is_contiguous()
    assert tuple(host_key.shape[:2]) == (batch, kv_heads) and host_key.shape[-1] == 128
    assert selected_pointer != 0
    return _load_extension(value_dim=value_dim, queries_per_kv=queries_per_kv, page_size=page_size).attention(
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
        value if value_prefix is None else value_prefix,
        prefix_width,
    )


def gpu_paged_attention(
    key: Tensor,
    query: Tensor,
    value: Tensor,
    selected_page_ids: Tensor,
    *,
    sequence_length: int,
    page_size: int = 32,
    scale: float | None = None,
    splits: int = _DEFAULT_SPLITS,
    workspace: Tensor | None = None,
    output: Tensor | None = None,
    value_prefix: Tensor | None = None,
) -> Tensor:
    """Run the mapped-host kernel against a GPU-resident exact-K oracle."""

    batch, query_heads, query_tokens, qk_dim = map(int, query.shape)
    value_batch, kv_heads, value_tokens, value_dim = map(int, value.shape)
    prefix_width = 0 if value_prefix is None else int(value_prefix.shape[-1])
    if value_prefix is not None:
        assert value_prefix.shape[:3] == value.shape[:3]
        assert value_prefix.device == value.device and value_prefix.dtype == value.dtype
        assert value_prefix.stride(-1) == 1
        value_dim += prefix_width
    page_batch, page_heads, page_slots = map(int, selected_page_ids.shape)
    assert key.is_contiguous() and key.device == query.device
    assert key.is_cuda and key.dtype == torch.bfloat16 and key.ndim == 4
    assert query.is_cuda and value.is_cuda and selected_page_ids.is_cuda
    assert query.dtype == value.dtype == torch.bfloat16
    assert selected_page_ids.dtype == torch.int64
    assert query_tokens == 1 and qk_dim == _QK_DIM
    assert 0 < value_dim <= 256 and value_batch == batch
    assert query_heads % kv_heads == 0
    queries_per_kv = query_heads // kv_heads
    assert value.stride(-1) == query.stride(-1) == 1
    assert value.device == selected_page_ids.device == query.device
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
            value_dim + 2,
            dtype=torch.float32,
            device=query.device,
        )
    if output is None:
        output = torch.empty(
            batch,
            query_heads,
            1,
            value_dim,
            dtype=torch.bfloat16,
            device=query.device,
        )
    assert workspace.shape[0] == batch * query_heads
    assert workspace.shape[1] >= selected_splits
    assert workspace.shape[2] == value_dim + 2
    assert workspace.dtype == torch.float32 and workspace.device == query.device
    assert tuple(output.shape) == (batch, query_heads, 1, value_dim)
    assert output.dtype == torch.bfloat16 and output.device == query.device
    selected_scale = qk_dim**-0.5 if scale is None else float(scale)
    assert math.isfinite(selected_scale) and selected_scale > 0.0
    return _load_extension(value_dim=value_dim, queries_per_kv=queries_per_kv, page_size=page_size).attention(
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
        value if value_prefix is None else value_prefix,
        prefix_width,
    )


__all__ = [
    "append_mapped_host_key",
    "conditional_router_append_decode",
    "conditional_router_page_lse",
    "gpu_paged_attention",
    "mapped_host_bf16_empty",
    "mapped_host_device_pointer",
    "mapped_host_paged_attention",
    "prepare_mapped_host_paged_attention_extension",
    "select_fixed_group_max_pages_cuda",
]


def conditional_router_query_code(query: Tensor, residual_query: Tensor, output: Tensor) -> Tensor:
    """Native BF16 residual-query projection, separated from the context scan."""
    residual_rank = int(residual_query.shape[-1])
    return _load_extension(residual_rank=residual_rank).conditional_router_query_code(query, residual_query, output)
