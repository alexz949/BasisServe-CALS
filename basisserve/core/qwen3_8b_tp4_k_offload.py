"""Mapped-host exact-Key offload for the Qwen3-8B TP4 C1 runtime.

Exact post-RoPE Keys are stored in CUDA-mapped host memory. The compact C1
Value cache, Base16+R8 routing state, and selected page IDs remain on the GPU.
One CUDA kernel directly reads selected Page32 Keys over PCIe and performs
exact QK, online softmax, and V80 accumulation without CPU packing or a GPU K
staging tensor. The existing C1 full-batch latent AllGather and decoder remain
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Sequence

from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch import Tensor, nn

from basisserve.core.qwen3_8b_tp4_decode import (
    HEAD_DIM,
    KV_HEADS_PER_PROCESS,
    QUERY_HEADS_PER_PROCESS,
    TP_SIZE,
    Qwen3TP4C1DecodeAttention,
    Qwen3TP4C1FactorLayer,
    _Qwen3TP4StaticDecodeAttention,
    file_sha256,
    load_qwen3_tp4_c1_factor_layer,
)
from basisserve.kernels.compressed_v_decode_attention import (
    compressed_v_prefill_attention,
)
from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.mapped_host_paged_attention import (
    append_mapped_host_key,
    conditional_router_append_decode,
    conditional_router_page32_lse,
    gpu_page32_v80_attention,
    mapped_host_bf16_empty,
    mapped_host_device_pointer,
    mapped_host_page32_v80_attention,
    select_fixed_group_max_pages_cuda,
)


@dataclass(frozen=True)
class Qwen3TP4ConditionalRouterLayer:
    """The process-local Base-rank predictive map and residual-rank router."""

    layer_index: int
    base_rank: int
    residual_rank: int
    base_left: Tensor
    base_right: Tensor
    base_bias: Tensor
    residual_encoder: Tensor
    residual_query: Tensor
    path: Path
    sha256: str


def _conditional_factor_path(root: str | Path, layer_index: int) -> Path:
    selected = sorted(
        Path(root).expanduser().resolve().rglob(
            f"layer_{int(layer_index):03d}.safetensors"
        )
    )
    assert len(selected) == 1
    return selected[0]


def load_qwen3_tp4_conditional_router_layer(
    root: str | Path,
    layer_index: int,
    *,
    base_rank: int = 16,
    residual_rank: int = 8,
) -> Qwen3TP4ConditionalRouterLayer:
    """Load and slice one all-head conditional router for the current TP rank."""

    path = _conditional_factor_path(root, layer_index)
    payload = load_file(str(path), device="cpu")
    names = {
        "base_left": f"base_left_b{base_rank}",
        "base_right": f"base_right_b{base_rank}",
        "base_bias": f"base_bias_b{base_rank}",
        "residual_encoder": f"residual_encoder_b{base_rank}_r{residual_rank}",
        "residual_query": f"residual_query_b{base_rank}_r{residual_rank}",
    }
    assert all(name in payload for name in names.values())
    process_rank = dist.get_rank()
    kv_start = process_rank * KV_HEADS_PER_PROCESS
    kv_stop = kv_start + KV_HEADS_PER_PROCESS
    query_start = process_rank * QUERY_HEADS_PER_PROCESS
    query_stop = query_start + QUERY_HEADS_PER_PROCESS
    return Qwen3TP4ConditionalRouterLayer(
        layer_index=int(layer_index),
        base_rank=int(base_rank),
        residual_rank=int(residual_rank),
        base_left=payload[names["base_left"]][kv_start:kv_stop].contiguous(),
        base_right=payload[names["base_right"]][kv_start:kv_stop].contiguous(),
        base_bias=payload[names["base_bias"]][kv_start:kv_stop].contiguous(),
        residual_encoder=payload[names["residual_encoder"]][
            kv_start:kv_stop
        ].contiguous(),
        residual_query=payload[names["residual_query"]][
            query_start:query_stop
        ].contiguous(),
        path=path,
        sha256=file_sha256(path),
    )


def _apply_rope(values: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = int(values.shape[-1]) // 2
    first = values[..., :half]
    second = values[..., half:]
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin),
        dim=-1,
    )


def conditional_router_page_log_mass(
    query: Tensor,
    value: Tensor,
    residual_code: Tensor,
    *,
    base_left: Tensor,
    base_right: Tensor,
    base_bias: Tensor,
    residual_query: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    page_size: int,
    page_chunk: int,
    scale: float,
) -> Tensor:
    """Return Base+Residual page LSE as ``[B, Hkv, Hq/Hkv, pages]``."""

    batch, kv_heads, tokens, value_rank = map(int, value.shape)
    query_heads = int(query.shape[1])
    heads_per_kv = query_heads // kv_heads
    residual_rank = int(residual_code.shape[-1])
    assert tuple(query.shape) == (batch, query_heads, 1, HEAD_DIM)
    assert tuple(residual_code.shape[:3]) == (batch, kv_heads, tokens)
    assert tuple(base_left.shape) == (kv_heads, value_rank, base_right.shape[1])
    assert tuple(base_right.shape[::2]) == (kv_heads, HEAD_DIM)
    assert tuple(base_bias.shape) == (kv_heads, HEAD_DIM)
    assert tuple(residual_query.shape) == (
        query_heads,
        HEAD_DIM,
        residual_rank,
    )
    assert tuple(rope_cos.shape) == (tokens, HEAD_DIM // 2)
    assert tuple(rope_sin.shape) == tuple(rope_cos.shape)
    grouped_query = query[:, :, 0].reshape(
        batch,
        kv_heads,
        heads_per_kv,
        HEAD_DIM,
    )
    grouped_residual_query = residual_query.reshape(
        kv_heads,
        heads_per_kv,
        HEAD_DIM,
        residual_rank,
    )
    query_residual_code = torch.einsum(
        "bghd,ghdr->bghr",
        grouped_query,
        grouped_residual_query,
    )
    pages = math.ceil(tokens / page_size)
    chunk_tokens = int(page_chunk) * int(page_size)
    chunks: list[Tensor] = []
    for start in range(0, tokens, chunk_tokens):
        stop = min(start + chunk_tokens, tokens)
        base_code = torch.einsum(
            "bgtv,gvr->bgtr",
            value[:, :, start:stop],
            base_left,
        )
        base_pre = torch.einsum(
            "bgtr,grd->bgtd",
            base_code,
            base_right,
        )
        base_pre.add_(base_bias[None, :, None, :])
        cos = rope_cos[start:stop][None, None]
        sin = rope_sin[start:stop][None, None]
        base_post = _apply_rope(base_pre, cos, sin)
        token_scores = torch.einsum(
            "bghd,bgtd->bght",
            grouped_query,
            base_post,
        )
        token_scores.add_(
            torch.einsum(
                "bghr,bgtr->bght",
                query_residual_code,
                residual_code[:, :, start:stop],
            )
        )
        token_scores.mul_(float(scale))
        padding = (-int(token_scores.shape[-1])) % int(page_size)
        if padding:
            token_scores = torch.nn.functional.pad(
                token_scores,
                (0, padding),
                value=-torch.inf,
            )
        chunks.append(
            torch.logsumexp(
                token_scores.float().reshape(
                    batch,
                    kv_heads,
                    heads_per_kv,
                    -1,
                    page_size,
                ),
                dim=-1,
            )
        )
    result = torch.cat(chunks, dim=-1)
    assert tuple(result.shape) == (batch, kv_heads, heads_per_kv, pages)
    return result


def quest_page_scores(
    query: Tensor,
    page_minimum: Tensor,
    page_maximum: Tensor,
    *,
    scale: float,
) -> Tensor:
    """Return QUEST coordinate-bound scores for every GQA Query head."""

    batch, query_heads, query_tokens, head_dim = map(int, query.shape)
    page_batch, kv_heads, pages, page_dim = map(int, page_minimum.shape)
    heads_per_kv = query_heads // kv_heads
    assert query_tokens == 1 and head_dim == HEAD_DIM
    assert page_maximum.shape == page_minimum.shape
    assert (page_batch, page_dim) == (batch, HEAD_DIM)
    grouped_query = query[:, :, 0].reshape(
        batch,
        kv_heads,
        heads_per_kv,
        HEAD_DIM,
    ).float()
    minimum = page_minimum[:, :, None].float()
    maximum = page_maximum[:, :, None].float()
    result = torch.maximum(
        grouped_query[:, :, :, None] * minimum,
        grouped_query[:, :, :, None] * maximum,
    ).sum(dim=-1)
    result.mul_(float(scale))
    assert tuple(result.shape) == (batch, kv_heads, heads_per_kv, pages)
    return result


def shadowkv_page_scores(
    query: Tensor,
    page_mean: Tensor,
    *,
    scale: float,
) -> Tensor:
    """Return one post-RoPE mean-landmark score per Page32 and Query head."""

    batch, query_heads, query_tokens, head_dim = map(int, query.shape)
    page_batch, kv_heads, pages, page_dim = map(int, page_mean.shape)
    heads_per_kv = query_heads // kv_heads
    assert query_tokens == 1 and head_dim == HEAD_DIM
    assert (page_batch, page_dim) == (batch, HEAD_DIM)
    grouped_query = query[:, :, 0].reshape(
        batch,
        kv_heads,
        heads_per_kv,
        HEAD_DIM,
    )
    result = torch.einsum(
        "bghd,bgpd->bghp",
        grouped_query.float(),
        page_mean.float(),
    )
    result.mul_(float(scale))
    assert tuple(result.shape) == (batch, kv_heads, heads_per_kv, pages)
    return result


def select_fixed_group_max_pages(
    page_log_mass: Tensor,
    *,
    pages_per_kv_head: int,
    pinned_prefix_pages: int,
    force_current_page: bool = True,
) -> Tensor:
    """Select one fixed physical page budget after max-over-query-head routing."""

    batch, kv_heads, _, page_count = map(int, page_log_mass.shape)
    selected_count = min(int(pages_per_kv_head), page_count)
    assert selected_count > 0
    prefix_count = min(int(pinned_prefix_pages), page_count)
    fixed = torch.zeros(page_count, dtype=torch.bool, device=page_log_mass.device)
    fixed[:prefix_count] = True
    if force_current_page:
        fixed[-1] = True
    fixed_ids = torch.nonzero(fixed, as_tuple=False).flatten()
    assert int(fixed_ids.numel()) <= selected_count
    if selected_count == page_count:
        return torch.arange(
            page_count,
            dtype=torch.long,
            device=page_log_mass.device,
        ).view(1, 1, page_count).expand(batch, kv_heads, page_count)
    remaining = selected_count - int(fixed_ids.numel())
    routable_log_mass = page_log_mass.float().masked_fill(
        fixed.view(1, 1, 1, -1),
        -torch.inf,
    )
    group_scores = torch.softmax(routable_log_mass, dim=-1).amax(dim=2)
    routed = torch.topk(
        group_scores,
        k=remaining,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    fixed_batch = fixed_ids.view(1, 1, -1).expand(batch, kv_heads, -1)
    return torch.sort(torch.cat((fixed_batch, routed), dim=-1), dim=-1).values


class PinnedCPUPageMajorKeyCache:
    """Preallocated page-major exact-Key cache with one packed H2D staging area."""

    def __init__(
        self,
        *,
        batch_size: int,
        kv_heads: int,
        capacity: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch_size = int(batch_size)
        self.kv_heads = int(kv_heads)
        self.capacity = int(capacity)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.page_capacity = math.ceil(self.capacity / self.page_size)
        self.token_capacity = self.page_capacity * self.page_size
        self.pages = torch.zeros(
            self.batch_size,
            self.kv_heads,
            self.page_capacity,
            self.page_size,
            self.head_dim,
            dtype=dtype,
            device="cpu",
            pin_memory=torch.cuda.is_available(),
        )
        self.length = 0
        self._append_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        self._host_staging: Tensor | None = None
        self._staging_slots = 0
        self.d2h_seconds = 0.0
        self.exact_key_bytes_d2h = 0
        self.last_host_pack_seconds = 0.0
        self.last_h2d_seconds = 0.0
        self.last_requested_bytes = 0

    @property
    def resident_bytes(self) -> int:
        return self.pages.numel() * self.pages.element_size()

    def wait_for_appends(self) -> None:
        if self._append_events is not None:
            start, stop = self._append_events
            stop.synchronize()
            self.d2h_seconds += start.elapsed_time(stop) / 1000.0
            self._append_events = None

    def append(self, key: Tensor) -> None:
        self.wait_for_appends()
        tokens = int(key.shape[2])
        assert tuple(key.shape[:2]) == (self.batch_size, self.kv_heads)
        assert int(key.shape[-1]) == self.head_dim
        start = self.length
        stop = start + tokens
        assert stop <= self.capacity
        destination = self.pages.view(
            self.batch_size,
            self.kv_heads,
            self.token_capacity,
            self.head_dim,
        )[:, :, start:stop]
        start_event = None
        stop_event = None
        if key.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(key.device))
        destination.copy_(key.detach(), non_blocking=key.is_cuda)
        if stop_event is not None:
            stop_event.record(torch.cuda.current_stream(key.device))
            self._append_events = (start_event, stop_event)
        self.exact_key_bytes_d2h += key.numel() * key.element_size()
        self.length = stop

    def reset(self) -> None:
        self.wait_for_appends()
        self.length = 0
        self.d2h_seconds = 0.0
        self.exact_key_bytes_d2h = 0

    def fetch(self, selected_page_ids: Tensor, *, device: torch.device) -> Tensor:
        """Pack selected pages on CPU and issue one blocking contiguous H2D copy."""

        self.wait_for_appends()
        ids_started = time.perf_counter()
        ids = selected_page_ids.detach().to(device="cpu", dtype=torch.long)
        batch, heads, slots = map(int, ids.shape)
        assert (batch, heads) == (self.batch_size, self.kv_heads)
        assert bool(((ids >= 0) & (ids < math.ceil(self.length / self.page_size))).all())
        flat_ids = (
            (
                torch.arange(batch).view(batch, 1, 1) * heads
                + torch.arange(heads).view(1, heads, 1)
            )
            * self.page_capacity
            + ids
        ).reshape(-1)
        requests = batch * heads * slots
        if self._host_staging is None or self._staging_slots < slots:
            self._host_staging = torch.empty(
                batch,
                heads,
                slots,
                self.page_size,
                self.head_dim,
                dtype=self.pages.dtype,
                device="cpu",
                pin_memory=self.pages.is_pinned(),
            )
            self._staging_slots = slots
        staging = self._host_staging[:, :, :slots]
        bank = self.pages.reshape(
            batch * heads * self.page_capacity,
            self.page_size,
            self.head_dim,
        )
        torch.index_select(
            bank,
            0,
            flat_ids,
            out=staging.reshape(requests, self.page_size, self.head_dim),
        )
        self.last_host_pack_seconds = time.perf_counter() - ids_started
        self.last_requested_bytes = staging.numel() * staging.element_size()
        if device.type == "cpu":
            self.last_h2d_seconds = 0.0
            return staging.clone()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        destination = torch.empty_like(staging, device=device)
        start.record(torch.cuda.current_stream(device))
        destination.copy_(staging, non_blocking=True)
        stop.record(torch.cuda.current_stream(device))
        stop.synchronize()
        self.last_h2d_seconds = start.elapsed_time(stop) / 1000.0
        return destination


class MappedHostExactKeyCache:
    """Exact BF16 K storage directly addressable by CUDA kernels over PCIe."""

    def __init__(
        self,
        *,
        batch_size: int,
        kv_heads: int,
        capacity: int,
        head_dim: int,
    ) -> None:
        self.batch_size = int(batch_size)
        self.kv_heads = int(kv_heads)
        self.capacity = int(capacity)
        self.head_dim = int(head_dim)
        self.key = mapped_host_bf16_empty(
            batch=self.batch_size,
            kv_heads=self.kv_heads,
            capacity=self.capacity,
            head_dim=self.head_dim,
        )
        self.device_pointer = mapped_host_device_pointer(self.key)
        self.length = 0
        self.exact_key_bytes_d2h = 0

    @property
    def resident_bytes(self) -> int:
        return self.key.numel() * self.key.element_size()

    def append(self, key: Tensor) -> None:
        tokens = int(key.shape[2])
        assert tuple(key.shape[:2]) == (self.batch_size, self.kv_heads)
        assert int(key.shape[-1]) == self.head_dim
        assert key.dtype == torch.bfloat16 and key.is_cuda
        stop = self.length + tokens
        assert stop <= self.capacity
        append_mapped_host_key(self.key, key, start=self.length)
        self.exact_key_bytes_d2h += key.numel() * key.element_size()
        self.length = stop

    def reset(self) -> None:
        self.length = 0
        self.exact_key_bytes_d2h = 0


class GPUExactKeyCache:
    """GPU-resident exact K used to isolate mapped-host numerical behavior."""

    def __init__(
        self,
        *,
        batch_size: int,
        kv_heads: int,
        capacity: int,
        head_dim: int,
        device: torch.device,
    ) -> None:
        self.key = torch.empty(
            int(batch_size),
            int(kv_heads),
            int(capacity),
            int(head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self.length = 0
        self.exact_key_bytes_d2h = 0

    @property
    def resident_bytes(self) -> int:
        return self.key.numel() * self.key.element_size()

    def append(self, key: Tensor) -> None:
        tokens = int(key.shape[2])
        stop = self.length + tokens
        assert tuple(key.shape[:2]) == tuple(self.key.shape[:2])
        assert int(key.shape[-1]) == int(self.key.shape[-1])
        assert stop <= int(self.key.shape[2])
        self.key[:, :, self.length : stop].copy_(key)
        self.length = stop

    def reset(self) -> None:
        self.length = 0


class _SharedRopeCache:
    """One position table shared by every C1 routing layer on a TP rank."""

    def __init__(self) -> None:
        self.cos: Tensor | None = None
        self.sin: Tensor | None = None

    def configure(
        self,
        *,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        shape = (int(capacity), HEAD_DIM // 2)
        if (
            self.cos is None
            or tuple(self.cos.shape) != shape
            or self.cos.device != device
            or self.cos.dtype != dtype
        ):
            self.cos = torch.empty(shape, device=device, dtype=dtype)
            self.sin = torch.empty_like(self.cos)
        assert self.sin is not None
        return self.cos, self.sin

    def clear(self) -> None:
        self.cos = None
        self.sin = None


class Qwen3TP4C1KOffloadDecodeAttention(Qwen3TP4C1DecodeAttention):
    """TP4 C1 attention with selectable GPU routing and exact-K placement."""

    def __init__(
        self,
        base_attention: nn.Module,
        c1_factors: Qwen3TP4C1FactorLayer,
        router: Qwen3TP4ConditionalRouterLayer | None,
        communicator: FeatureRaggedCommunicator,
        shared_rope_cache: _SharedRopeCache | None,
        *,
        owns_shared_rope_cache: bool,
        routing_mode: str,
        exact_key_storage: str,
        page_size: int,
        exact_token_budget: int,
        pinned_prefix_pages: int,
        allgather_backend: str,
    ) -> None:
        super().__init__(
            base_attention,
            c1_factors,
            communicator,
            "triton",
            "bfloat16",
            None,
            allgather_backend,
            "auto",
            0,
        )
        assert c1_factors.source_rank == 80
        assert routing_mode in ("c1_base16_r8", "quest", "shadowkv")
        assert exact_key_storage in ("mapped_host", "gpu")
        assert (router is not None) == (routing_mode == "c1_base16_r8")
        assert (shared_rope_cache is not None) == (
            routing_mode == "c1_base16_r8"
        )
        assert not owns_shared_rope_cache or shared_rope_cache is not None
        if router is not None:
            assert router.layer_index == self.layer_idx
            assert router.base_rank == 16 and router.residual_rank == 8
        assert exact_token_budget % page_size == 0
        assert page_size == 32
        self.routing_mode = routing_mode
        self.exact_key_storage = exact_key_storage
        self.page_size = int(page_size)
        self.pages_per_kv_head = int(exact_token_budget) // self.page_size
        self.pinned_prefix_pages = int(pinned_prefix_pages)
        self.shared_rope_cache = shared_rope_cache
        self.owns_shared_rope_cache = bool(owns_shared_rope_cache)
        device = base_attention.q_proj.weight.device
        dtype = base_attention.q_proj.weight.dtype
        self.register_buffer(
            "offload_base_left",
            None if router is None else router.base_left.to(device=device, dtype=dtype),
        )
        self.register_buffer(
            "offload_base_right",
            None if router is None else router.base_right.to(device=device, dtype=dtype),
        )
        self.register_buffer(
            "offload_base_bias",
            None if router is None else router.base_bias.to(device=device, dtype=dtype),
        )
        self.register_buffer(
            "offload_residual_encoder",
            None
            if router is None
            else router.residual_encoder.to(device=device, dtype=dtype),
        )
        self.register_buffer(
            "offload_residual_query",
            None
            if router is None
            else router.residual_query.to(device=device, dtype=dtype),
        )
        self.router_factor_path = None if router is None else str(router.path)
        self.router_factor_sha256 = None if router is None else router.sha256
        self.residual_rank = 0 if router is None else int(router.residual_rank)
        self.register_buffer("base_cache", None, persistent=False)
        self.register_buffer("residual_cache", None, persistent=False)
        self.register_buffer("rope_cos_cache", None, persistent=False)
        self.register_buffer("rope_sin_cache", None, persistent=False)
        self.register_buffer("quest_page_minimum", None, persistent=False)
        self.register_buffer("quest_page_maximum", None, persistent=False)
        self.register_buffer("shadowkv_page_mean", None, persistent=False)
        self.register_buffer("mapped_attention_workspace", None, persistent=False)
        self.register_buffer("mapped_attention_output", None, persistent=False)
        self.register_buffer("router_query_code", None, persistent=False)
        self.register_buffer("router_page_log_mass", None, persistent=False)
        self.register_buffer("selected_page_ids", None, persistent=False)
        self.exact_key_cache: MappedHostExactKeyCache | GPUExactKeyCache | None = None
        self.reset_offload_statistics()

    def reset_offload_statistics(self) -> None:
        self._offload_totals = {
            "decode_calls": 0.0,
            "selected_pages": 0.0,
            "exact_key_bytes_direct_read": 0.0,
        }

    def offload_statistics(self) -> dict[str, float]:
        totals = dict(self._offload_totals)
        if self.exact_key_cache is not None:
            totals["exact_key_bytes_d2h"] = float(
                self.exact_key_cache.exact_key_bytes_d2h
            )
        return totals

    @property
    def gpu_cache_bytes(self) -> int:
        tensors = (
            self.value_cache,
            self.base_cache,
            self.residual_cache,
            self.quest_page_minimum,
            self.quest_page_maximum,
            self.shadowkv_page_mean,
            self.router_query_code,
            self.router_page_log_mass,
            self.selected_page_ids,
        )
        result = sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
            if tensor is not None
        )
        if self.owns_shared_rope_cache:
            for tensor in (self.rope_cos_cache, self.rope_sin_cache):
                if tensor is not None:
                    result += tensor.numel() * tensor.element_size()
        if isinstance(self.exact_key_cache, GPUExactKeyCache):
            result += self.exact_key_cache.resident_bytes
        return result

    @property
    def cpu_cache_bytes(self) -> int:
        if isinstance(self.exact_key_cache, MappedHostExactKeyCache):
            return self.exact_key_cache.resident_bytes
        return 0

    @property
    def cache_bytes(self) -> int:
        return self.gpu_cache_bytes + self.cpu_cache_bytes

    def configure_cache(
        self,
        *,
        batch_size: int,
        capacity: int,
        max_forward_tokens: int = 1,
    ) -> None:
        del max_forward_tokens
        batch = int(batch_size)
        length = int(capacity)
        device = self.q_proj.weight.device
        dtype = self.q_proj.weight.dtype
        self.key_cache = None
        self.value_cache = torch.empty(
            batch,
            KV_HEADS_PER_PROCESS,
            length,
            self.value_head_dim,
            device=device,
            dtype=dtype,
        )
        page_capacity = math.ceil(length / self.page_size)
        if self.routing_mode == "c1_base16_r8":
            assert self.shared_rope_cache is not None
            self.base_cache = torch.empty(
                batch,
                KV_HEADS_PER_PROCESS,
                length,
                16,
                device=device,
                dtype=dtype,
            )
            self.residual_cache = torch.empty(
                batch,
                KV_HEADS_PER_PROCESS,
                length,
                self.residual_rank,
                device=device,
                dtype=dtype,
            )
            self.rope_cos_cache, self.rope_sin_cache = (
                self.shared_rope_cache.configure(
                    capacity=length,
                    device=device,
                    dtype=dtype,
                )
            )
            self.router_query_code = torch.empty(
                batch,
                KV_HEADS_PER_PROCESS,
                QUERY_HEADS_PER_PROCESS // KV_HEADS_PER_PROCESS,
                self.residual_rank,
                device=device,
                dtype=dtype,
            )
            self.router_page_log_mass = torch.empty(
                batch,
                KV_HEADS_PER_PROCESS,
                QUERY_HEADS_PER_PROCESS // KV_HEADS_PER_PROCESS,
                page_capacity,
                device=device,
                dtype=torch.float32,
            )
        elif self.routing_mode == "quest":
            page_shape = (
                batch,
                KV_HEADS_PER_PROCESS,
                page_capacity,
                HEAD_DIM,
            )
            self.quest_page_minimum = torch.empty(
                page_shape,
                device=device,
                dtype=dtype,
            )
            self.quest_page_maximum = torch.empty_like(self.quest_page_minimum)
        else:
            self.shadowkv_page_mean = torch.empty(
                batch,
                KV_HEADS_PER_PROCESS,
                page_capacity,
                HEAD_DIM,
                device=device,
                dtype=dtype,
            )
        assert dtype == torch.bfloat16
        assert self.page_size == 32
        if self.exact_key_storage == "mapped_host":
            self.exact_key_cache = MappedHostExactKeyCache(
                batch_size=batch,
                kv_heads=KV_HEADS_PER_PROCESS,
                capacity=length,
                head_dim=HEAD_DIM,
            )
        else:
            self.exact_key_cache = GPUExactKeyCache(
                batch_size=batch,
                kv_heads=KV_HEADS_PER_PROCESS,
                capacity=length,
                head_dim=HEAD_DIM,
                device=device,
            )
        self.mapped_attention_workspace = torch.empty(
            batch * QUERY_HEADS_PER_PROCESS,
            32,
            self.value_head_dim + 2,
            dtype=torch.float32,
            device=device,
        )
        self.mapped_attention_output = torch.empty(
            batch,
            QUERY_HEADS_PER_PROCESS,
            1,
            self.value_head_dim,
            dtype=dtype,
            device=device,
        )
        maximum_selected_pages = min(self.pages_per_kv_head, page_capacity)
        self.selected_page_ids = torch.empty(
            batch * KV_HEADS_PER_PROCESS * maximum_selected_pages,
            dtype=torch.long,
            device=device,
        )
        self._cache_length = 0
        self.reset_offload_statistics()

    def clear_cache(self) -> None:
        self.key_cache = None
        self.value_cache = None
        self.base_cache = None
        self.residual_cache = None
        self.rope_cos_cache = None
        self.rope_sin_cache = None
        self.quest_page_minimum = None
        self.quest_page_maximum = None
        self.shadowkv_page_mean = None
        self.router_query_code = None
        self.router_page_log_mass = None
        self.selected_page_ids = None
        self.mapped_attention_workspace = None
        self.mapped_attention_output = None
        self.exact_key_cache = None
        if self.owns_shared_rope_cache:
            assert self.shared_rope_cache is not None
            self.shared_rope_cache.clear()
        self._cache_length = 0
        self._decode_workspace = None
        self._uniform_allgather.clear()

    def reset_cache(self) -> None:
        assert self.exact_key_cache is not None
        self.exact_key_cache.reset()
        self._cache_length = 0
        self.reset_offload_statistics()

    def set_cache_length(self, length: int) -> None:
        assert int(length) == 0
        self.reset_cache()

    def _append_cache(
        self,
        key: Tensor,
        value: Tensor,
        *,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        assert self.value_cache is not None
        assert self.exact_key_cache is not None
        assert position_embeddings is not None
        tokens = int(key.shape[2])
        assert tokens == 1 or self._cache_length == 0
        start = self._cache_length
        stop = start + tokens
        assert stop <= int(self.value_cache.shape[2])
        cos, sin = position_embeddings
        selected_cos = cos[0] if cos.ndim == 3 else cos
        selected_sin = sin[0] if sin.ndim == 3 else sin
        selected_cos = selected_cos[:, : HEAD_DIM // 2]
        selected_sin = selected_sin[:, : HEAD_DIM // 2]
        if self.routing_mode == "c1_base16_r8":
            assert self.base_cache is not None
            assert self.residual_cache is not None
            assert self.rope_cos_cache is not None
            assert self.rope_sin_cache is not None
            assert self.offload_base_left is not None
            assert self.offload_base_right is not None
            assert self.offload_base_bias is not None
            assert self.offload_residual_encoder is not None
            if tokens == 1:
                conditional_router_append_decode(
                    key,
                    value,
                    base_left=self.offload_base_left,
                    base_right=self.offload_base_right,
                    base_bias=self.offload_base_bias,
                    residual_encoder=self.offload_residual_encoder,
                    rope_cos=selected_cos,
                    rope_sin=selected_sin,
                    value_cache=self.value_cache,
                    base_cache=self.base_cache,
                    residual_cache=self.residual_cache,
                    rope_cos_cache=self.rope_cos_cache,
                    rope_sin_cache=self.rope_sin_cache,
                    start=start,
                    write_rope=self.owns_shared_rope_cache,
                )
            else:
                self.value_cache[:, :, start:stop].copy_(value)
                if self.owns_shared_rope_cache:
                    self.rope_cos_cache[start:stop].copy_(selected_cos)
                    self.rope_sin_cache[start:stop].copy_(selected_sin)
                base_code = torch.einsum(
                    "bgtv,gvr->bgtr",
                    value,
                    self.offload_base_left,
                )
                self.base_cache[:, :, start:stop].copy_(base_code)
                base_pre = torch.einsum(
                    "bgtr,grd->bgtd",
                    base_code,
                    self.offload_base_right,
                )
                base_pre.add_(self.offload_base_bias[None, :, None, :])
                base_post = _apply_rope(
                    base_pre,
                    selected_cos[None, None],
                    selected_sin[None, None],
                )
                residual = key - base_post
                residual_code = torch.einsum(
                    "bgtd,gdr->bgtr",
                    residual,
                    self.offload_residual_encoder,
                )
                self.residual_cache[:, :, start:stop].copy_(residual_code)
        elif self.routing_mode == "quest":
            self.value_cache[:, :, start:stop].copy_(value)
            assert self.quest_page_minimum is not None
            assert self.quest_page_maximum is not None
            if start == 0:
                full_pages = tokens // self.page_size
                if full_pages:
                    complete = key[:, :, : full_pages * self.page_size].float()
                    complete = complete.reshape(
                        int(key.shape[0]),
                        KV_HEADS_PER_PROCESS,
                        full_pages,
                        self.page_size,
                        HEAD_DIM,
                    )
                    self.quest_page_minimum[:, :, :full_pages].copy_(
                        complete.amin(dim=3)
                    )
                    self.quest_page_maximum[:, :, :full_pages].copy_(
                        complete.amax(dim=3)
                    )
                if tokens % self.page_size:
                    tail = key[:, :, full_pages * self.page_size :].float()
                    self.quest_page_minimum[:, :, full_pages].copy_(
                        tail.amin(dim=2)
                    )
                    self.quest_page_maximum[:, :, full_pages].copy_(
                        tail.amax(dim=2)
                    )
            else:
                assert tokens == 1
                page = start // self.page_size
                offset = start % self.page_size
                current = key[:, :, 0]
                if offset == 0:
                    self.quest_page_minimum[:, :, page].copy_(current)
                    self.quest_page_maximum[:, :, page].copy_(current)
                else:
                    torch.minimum(
                        self.quest_page_minimum[:, :, page],
                        current,
                        out=self.quest_page_minimum[:, :, page],
                    )
                    torch.maximum(
                        self.quest_page_maximum[:, :, page],
                        current,
                        out=self.quest_page_maximum[:, :, page],
                    )
        else:
            self.value_cache[:, :, start:stop].copy_(value)
            assert self.shadowkv_page_mean is not None
            if start == 0:
                full_pages = tokens // self.page_size
                if full_pages:
                    complete = key[:, :, : full_pages * self.page_size].float()
                    complete = complete.reshape(
                        int(key.shape[0]),
                        KV_HEADS_PER_PROCESS,
                        full_pages,
                        self.page_size,
                        HEAD_DIM,
                    )
                    self.shadowkv_page_mean[:, :, :full_pages].copy_(
                        complete.mean(dim=3)
                    )
                if tokens % self.page_size:
                    tail = key[:, :, full_pages * self.page_size :].float()
                    self.shadowkv_page_mean[:, :, full_pages].copy_(
                        tail.mean(dim=2)
                    )
            else:
                assert tokens == 1
                page = start // self.page_size
                offset = start % self.page_size
                current = key[:, :, 0].float()
                if offset == 0:
                    self.shadowkv_page_mean[:, :, page].copy_(current)
                else:
                    updated = (
                        self.shadowkv_page_mean[:, :, page].float() * offset
                        + current
                    ) / (offset + 1)
                    self.shadowkv_page_mean[:, :, page].copy_(updated)
        self.exact_key_cache.append(key)
        self._cache_length = stop
        return key, self.value_cache[:, :, :stop]

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        is_prefill: bool,
    ) -> Tensor:
        if is_prefill:
            return compressed_v_prefill_attention(
                query,
                key,
                value,
                scale=self.scaling,
            )
        assert self.exact_key_cache is not None
        assert self.mapped_attention_workspace is not None
        assert self.mapped_attention_output is not None
        page_count = math.ceil(self._cache_length / self.page_size)
        if self.routing_mode == "c1_base16_r8":
            assert self.base_cache is not None
            assert self.residual_cache is not None
            assert self.rope_cos_cache is not None
            assert self.rope_sin_cache is not None
            assert self.offload_base_right is not None
            assert self.offload_base_bias is not None
            assert self.offload_residual_query is not None
            assert self.router_query_code is not None
            assert self.router_page_log_mass is not None
            page_scores = conditional_router_page32_lse(
                query,
                self.base_cache[:, :, : self._cache_length],
                self.residual_cache[:, :, : self._cache_length],
                base_right=self.offload_base_right,
                base_bias=self.offload_base_bias,
                residual_query=self.offload_residual_query,
                rope_cos=self.rope_cos_cache[: self._cache_length],
                rope_sin=self.rope_sin_cache[: self._cache_length],
                scale=self.scaling,
                query_code=self.router_query_code,
                output=self.router_page_log_mass[:, :, :, :page_count],
            )
        elif self.routing_mode == "quest":
            assert self.quest_page_minimum is not None
            assert self.quest_page_maximum is not None
            page_scores = quest_page_scores(
                query,
                self.quest_page_minimum[:, :, :page_count],
                self.quest_page_maximum[:, :, :page_count],
                scale=self.scaling,
            )
        else:
            assert self.shadowkv_page_mean is not None
            page_scores = shadowkv_page_scores(
                query,
                self.shadowkv_page_mean[:, :, :page_count],
                scale=self.scaling,
            )
        assert self.selected_page_ids is not None
        selected_count = min(self.pages_per_kv_head, page_count)
        selected_output = self.selected_page_ids[
            : int(page_scores.shape[0])
            * KV_HEADS_PER_PROCESS
            * selected_count
        ].view(int(page_scores.shape[0]), KV_HEADS_PER_PROCESS, selected_count)
        selected_page_ids = select_fixed_group_max_pages_cuda(
            page_scores,
            pages_per_kv_head=self.pages_per_kv_head,
            pinned_prefix_pages=self.pinned_prefix_pages,
            force_current_page=True,
            output=selected_output,
        )
        self._offload_totals["selected_pages"] += float(selected_page_ids.numel())
        self._offload_totals["exact_key_bytes_direct_read"] += float(
            selected_page_ids.numel()
            * self.page_size
            * HEAD_DIM
            * self.exact_key_cache.key.element_size()
        )
        if isinstance(self.exact_key_cache, MappedHostExactKeyCache):
            output = mapped_host_page32_v80_attention(
                self.exact_key_cache.key,
                query,
                value,
                selected_page_ids,
                sequence_length=self._cache_length,
                scale=self.scaling,
                splits=32,
                host_key_device_pointer=self.exact_key_cache.device_pointer,
                workspace=self.mapped_attention_workspace,
                output=self.mapped_attention_output,
            )
        else:
            output = gpu_page32_v80_attention(
                self.exact_key_cache.key,
                query,
                value,
                selected_page_ids,
                sequence_length=self._cache_length,
                scale=self.scaling,
                splits=32,
                workspace=self.mapped_attention_workspace,
                output=self.mapped_attention_output,
            )
        self._offload_totals["decode_calls"] += 1.0
        return output


def install_qwen3_tp4_k_offload_attention(
    model: nn.Module,
    *,
    c1_factor_dir: str | Path,
    routing_factor_dir: str | Path | None,
    routing_mode: str = "c1_base16_r8",
    exact_key_storage: str = "mapped_host",
    base_rank: int = 16,
    residual_rank: int = 8,
    page_size: int = 32,
    exact_token_budget: int = 4096,
    pinned_prefix_pages: int = 1,
    allgather_backend: str = "uniform_nccl",
) -> tuple[Qwen3TP4C1KOffloadDecodeAttention, ...]:
    """Install one TP4 sparse router and a shared exact-K attention path."""

    assert dist.is_initialized() and dist.get_world_size() == TP_SIZE
    assert routing_mode in ("c1_base16_r8", "quest", "shadowkv")
    assert exact_key_storage in ("mapped_host", "gpu")
    assert (routing_factor_dir is not None) == (routing_mode == "c1_base16_r8")
    communicator = FeatureRaggedCommunicator.from_distributed(
        device=model.model.embed_tokens.weight.device,
    )
    shared_rope_cache = (
        _SharedRopeCache() if routing_mode == "c1_base16_r8" else None
    )
    installed: list[Qwen3TP4C1KOffloadDecodeAttention] = []
    for layer_index, layer in enumerate(model.model.layers):
        c1_factors = load_qwen3_tp4_c1_factor_layer(c1_factor_dir, layer_index)
        router = (
            load_qwen3_tp4_conditional_router_layer(
                routing_factor_dir,
                layer_index,
                base_rank=base_rank,
                residual_rank=residual_rank,
            )
            if routing_factor_dir is not None
            else None
        )
        replacement = Qwen3TP4C1KOffloadDecodeAttention(
            layer.self_attn,
            c1_factors,
            router,
            communicator,
            shared_rope_cache,
            owns_shared_rope_cache=(
                shared_rope_cache is not None and layer_index == 0
            ),
            routing_mode=routing_mode,
            exact_key_storage=exact_key_storage,
            page_size=page_size,
            exact_token_budget=exact_token_budget,
            pinned_prefix_pages=pinned_prefix_pages,
            allgather_backend=allgather_backend,
        )
        replacement.eval()
        layer.self_attn = replacement
        installed.append(replacement)
    return tuple(installed)


def tp4_k_offload_cache_bytes(
    modules: Sequence[_Qwen3TP4StaticDecodeAttention],
) -> dict[str, int]:
    offloaded = tuple(
        module
        for module in modules
        if isinstance(module, Qwen3TP4C1KOffloadDecodeAttention)
    )
    return {
        "gpu_bytes_per_rank": sum(module.gpu_cache_bytes for module in offloaded),
        "cpu_pinned_bytes_per_rank": sum(
            module.cpu_cache_bytes for module in offloaded
        ),
    }


__all__ = [
    "GPUExactKeyCache",
    "PinnedCPUPageMajorKeyCache",
    "MappedHostExactKeyCache",
    "Qwen3TP4C1KOffloadDecodeAttention",
    "Qwen3TP4ConditionalRouterLayer",
    "conditional_router_page_log_mass",
    "install_qwen3_tp4_k_offload_attention",
    "load_qwen3_tp4_conditional_router_layer",
    "quest_page_scores",
    "select_fixed_group_max_pages",
    "shadowkv_page_scores",
    "tp4_k_offload_cache_bytes",
]
