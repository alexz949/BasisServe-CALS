"""Pinned-CPU exact-Key offload with GPU-resident KQ routing and C1 Values.

The runtime in this module is the serving counterpart of the Reverse ShadowKV
quality oracle.  Exact post-RoPE Keys live in pinned host memory.  A compact
post-RoPE K sidecar and the C1 Value payload stay on the accelerator.  Each
decode query scans the sidecar, fetches only selected exact-Key pages, and runs
one sparse exact attention operation over those pages.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as F


class ExactKeyPageSource(Protocol):
    """Structural interface shared with the Reverse ShadowKV page store."""

    def get_pages(
        self,
        *,
        layer_idx: int,
        batch_indices: Tensor,
        kv_head_indices: Tensor,
        page_ids: Tensor,
        page_size: int,
        device: torch.device,
    ) -> Tensor:
        """Return exact pages in request order as ``[requests, page, D]``."""


class PinnedCPUExactKeyPageStore:
    """Page-granular exact post-RoPE Key storage in pinned CPU memory.

    The selected coordinates are compactly copied from the accelerator to the
    host, gathered into a reusable pinned staging buffer, and copied back with
    ``non_blocking=True``.  Host staging is protected by a CUDA event so it is
    never overwritten while a prior DMA operation is still reading it.
    """

    def __init__(self, exact_key: Tensor, *, layer_idx: int = 0) -> None:
        if exact_key.ndim != 4 or not exact_key.is_floating_point():
            raise ValueError(
                "exact Key must be floating [batch, KV heads, sequence, head dim]"
            )
        if exact_key.device.type == "cuda":
            host_key = torch.empty(
                exact_key.shape,
                dtype=exact_key.dtype,
                device="cpu",
                pin_memory=True,
            )
            host_key.copy_(exact_key.detach(), non_blocking=True)
            torch.cuda.current_stream(exact_key.device).synchronize()
        elif exact_key.device.type == "cpu":
            source = exact_key.detach().contiguous()
            if torch.cuda.is_available() and not source.is_pinned():
                host_key = torch.empty(
                    source.shape,
                    dtype=source.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                host_key.copy_(source)
            else:
                host_key = source
        else:
            raise ValueError("exact Key storage must originate on CPU or CUDA")
        self.exact_key = host_key
        self.layer_idx = int(layer_idx)
        self.last_request_count = 0
        self.last_requested_bytes = 0
        self.last_host_gather_seconds = 0.0
        self._page_banks: dict[int, Tensor] = {}
        self._host_staging: dict[int, Tensor] = {}
        self._host_staging_events: dict[int, torch.cuda.Event] = {}

    @property
    def resident_bytes(self) -> int:
        return self.exact_key.numel() * self.exact_key.element_size()

    def _page_bank(self, page_size: int) -> Tensor:
        if page_size <= 0:
            raise ValueError("page size must be positive")
        cached = self._page_banks.get(page_size)
        if cached is not None:
            return cached
        batch, heads, sequence, head_dim = map(int, self.exact_key.shape)
        page_count = math.ceil(sequence / page_size)
        padded_sequence = page_count * page_size
        if padded_sequence == sequence:
            padded = self.exact_key
        else:
            padded = torch.zeros(
                batch,
                heads,
                padded_sequence,
                head_dim,
                dtype=self.exact_key.dtype,
                device="cpu",
                pin_memory=self.exact_key.is_pinned(),
            )
            padded[:, :, :sequence].copy_(self.exact_key)
        bank = padded.reshape(batch * heads * page_count, page_size, head_dim)
        self._page_banks[page_size] = bank
        return bank

    def _staging(self, *, page_size: int, request_count: int) -> Tensor:
        staging = self._host_staging.get(page_size)
        if staging is not None and int(staging.shape[0]) >= request_count:
            event = self._host_staging_events.get(page_size)
            if event is not None:
                event.synchronize()
            return staging
        capacity = 1 << max(request_count - 1, 0).bit_length()
        staging = torch.empty(
            capacity,
            page_size,
            int(self.exact_key.shape[-1]),
            dtype=self.exact_key.dtype,
            device="cpu",
            pin_memory=self.exact_key.is_pinned(),
        )
        self._host_staging[page_size] = staging
        self._host_staging_events.pop(page_size, None)
        return staging

    def get_pages(
        self,
        *,
        layer_idx: int,
        batch_indices: Tensor,
        kv_head_indices: Tensor,
        page_ids: Tensor,
        page_size: int,
        device: torch.device,
    ) -> Tensor:
        if int(layer_idx) != self.layer_idx:
            raise KeyError(f"page store has layer {self.layer_idx}, requested {layer_idx}")
        if not (
            batch_indices.ndim == kv_head_indices.ndim == page_ids.ndim == 1
            and len(batch_indices) == len(kv_head_indices) == len(page_ids)
        ):
            raise ValueError("page request indices must be equal-length vectors")
        request_count = int(len(page_ids))
        self.last_request_count = request_count
        self.last_requested_bytes = (
            request_count
            * page_size
            * int(self.exact_key.shape[-1])
            * self.exact_key.element_size()
        )
        if request_count == 0:
            return torch.empty(
                0,
                page_size,
                int(self.exact_key.shape[-1]),
                dtype=self.exact_key.dtype,
                device=device,
            )

        coordinates = torch.stack(
            (batch_indices, kv_head_indices, page_ids), dim=-1
        ).detach().to(device="cpu", dtype=torch.long)
        batch, heads, sequence, _ = map(int, self.exact_key.shape)
        page_count = math.ceil(sequence / page_size)
        if bool(
            ((coordinates[:, 0] < 0) | (coordinates[:, 0] >= batch)).any()
            or ((coordinates[:, 1] < 0) | (coordinates[:, 1] >= heads)).any()
            or ((coordinates[:, 2] < 0) | (coordinates[:, 2] >= page_count)).any()
        ):
            raise IndexError("exact-Key page request is outside the cache geometry")
        flat_indices = (
            (coordinates[:, 0] * heads + coordinates[:, 1]) * page_count
            + coordinates[:, 2]
        )
        bank = self._page_bank(page_size)
        gather_start = time.perf_counter()
        staging = self._staging(
            page_size=page_size, request_count=request_count
        )
        torch.index_select(
            bank, 0, flat_indices, out=staging[:request_count]
        )
        self.last_host_gather_seconds = time.perf_counter() - gather_start

        if device.type == "cpu":
            return staging[:request_count].clone()
        if device.type != "cuda":
            raise ValueError("exact-Key pages can only be fetched to CPU or CUDA")
        if not self.exact_key.is_pinned():
            raise RuntimeError("asynchronous CUDA fetch requires pinned exact-Key storage")
        destination = torch.empty(
            request_count,
            page_size,
            int(self.exact_key.shape[-1]),
            dtype=self.exact_key.dtype,
            device=device,
        )
        destination.copy_(staging[:request_count], non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        self._host_staging_events[page_size] = event
        return destination


@dataclass(frozen=True)
class KQPageSelection:
    """Physical GQA pages selected by a low-rank KQ scan."""

    page_mask: Tensor
    page_log_mass: Tensor
    refined_query_mask: Tensor | None
    tail_mass_ratio: Tensor | None


@dataclass(frozen=True)
class PackedExactKeyPages:
    """Selected exact Keys packed to a common per-group page width."""

    exact_key: Tensor
    page_ids: Tensor
    valid_pages: Tensor
    requested_pages: int
    requested_bytes: int


@dataclass(frozen=True)
class OffloadedC1Attention:
    """Output and materialized routing/offload decisions for one decode step."""

    output: Tensor
    selection: KQPageSelection
    pages: PackedExactKeyPages


def _validate_routing_geometry(
    query: Tensor,
    routing_sidecar: Tensor,
    query_projector: Tensor,
) -> tuple[int, int, int, int, int, int]:
    if query.ndim != 4 or int(query.shape[2]) != 1:
        raise ValueError("query must be [batch, Query heads, 1, head dim]")
    if routing_sidecar.ndim != 4 or query_projector.ndim != 3:
        raise ValueError("routing sidecar/projector must be rank four/rank three")
    batch, query_heads, _, head_dim = map(int, query.shape)
    sidecar_batch, kv_heads, sequence, rank = map(int, routing_sidecar.shape)
    if sidecar_batch != batch or query_heads % kv_heads:
        raise ValueError("query and routing sidecar have incompatible GQA geometry")
    if tuple(query_projector.shape[1:]) != (head_dim, rank):
        raise ValueError("routing Query projector has incompatible geometry")
    if int(query_projector.shape[0]) not in (kv_heads, query_heads):
        raise ValueError("routing Query projector must be per-KV or per-Query head")
    if not (query.is_floating_point() and routing_sidecar.is_floating_point()):
        raise TypeError("query and routing sidecar must be floating point")
    if not (
        query.device == routing_sidecar.device == query_projector.device
    ):
        raise ValueError("query, routing sidecar, and projector must share a device")
    return batch, query_heads, kv_heads, sequence, head_dim, rank


def kq_svd_gqa_page_selection(
    query: Tensor,
    routing_sidecar: Tensor,
    query_projector: Tensor,
    *,
    page_size: int,
    pages_per_query_head: int,
    adaptive_max_pages_per_query_head: int | None = None,
    adaptive_tail_mass_ratio_threshold: float | None = None,
) -> KQPageSelection:
    """Scan a resident R-dimensional sidecar and union pages within GQA groups."""

    batch, query_heads, kv_heads, sequence, head_dim, _ = (
        _validate_routing_geometry(query, routing_sidecar, query_projector)
    )
    if page_size <= 0 or pages_per_query_head <= 0:
        raise ValueError("page size and page budget must be positive")
    if (adaptive_max_pages_per_query_head is None) != (
        adaptive_tail_mass_ratio_threshold is None
    ):
        raise ValueError("adaptive maximum pages and threshold must be set together")
    if adaptive_max_pages_per_query_head is not None:
        if adaptive_max_pages_per_query_head <= pages_per_query_head:
            raise ValueError("adaptive maximum pages must exceed the base budget")
        assert adaptive_tail_mass_ratio_threshold is not None
        if not 0.0 < adaptive_tail_mass_ratio_threshold <= 1.0:
            raise ValueError("adaptive tail mass threshold must lie in (0, 1]")

    heads_per_group = query_heads // kv_heads
    grouped_query = query[:, :, 0].reshape(
        batch, kv_heads, heads_per_group, head_dim
    )
    if int(query_projector.shape[0]) == kv_heads:
        query_code = torch.einsum(
            "bghd,gdr->bghr", grouped_query, query_projector
        )
    else:
        grouped_projector = query_projector.reshape(
            kv_heads, heads_per_group, head_dim, -1
        )
        query_code = torch.einsum(
            "bghd,ghdr->bghr", grouped_query, grouped_projector
        )
    proxy_scores = torch.einsum(
        "bghr,bgtr->bght", query_code, routing_sidecar
    ) / math.sqrt(head_dim)
    page_count = math.ceil(sequence / page_size)
    padding = page_count * page_size - sequence
    padded_scores = F.pad(proxy_scores.float(), (0, padding), value=-torch.inf)
    page_log_mass = torch.logsumexp(
        padded_scores.reshape(
            batch, query_heads, page_count, page_size
        ),
        dim=-1,
    )

    maximum = min(
        adaptive_max_pages_per_query_head or pages_per_query_head,
        page_count,
    )
    top_values, top_indices = torch.topk(
        page_log_mass, k=maximum, dim=-1, largest=True, sorted=True
    )
    positions = torch.arange(maximum, device=query.device).view(1, 1, maximum)
    base_count = min(pages_per_query_head, maximum)
    refined: Tensor | None = None
    tail_mass_ratio: Tensor | None = None
    if adaptive_max_pages_per_query_head is None:
        selected_top = (positions < base_count).expand(
            batch, query_heads, maximum
        )
    else:
        base_log_mass = torch.logsumexp(top_values[..., :base_count], dim=-1)
        tail_log_mass = torch.logsumexp(top_values[..., base_count:], dim=-1)
        tail_mass_ratio = torch.exp(tail_log_mass - base_log_mass)
        assert adaptive_tail_mass_ratio_threshold is not None
        refined = tail_mass_ratio >= adaptive_tail_mass_ratio_threshold
        counts = torch.where(
            refined,
            torch.full_like(refined, maximum, dtype=torch.long),
            torch.full_like(refined, base_count, dtype=torch.long),
        )
        selected_top = positions < counts[..., None]
    query_page_mask = torch.zeros(
        batch,
        query_heads,
        page_count,
        dtype=torch.bool,
        device=query.device,
    )
    query_page_mask.scatter_(2, top_indices, selected_top)
    page_mask = query_page_mask.reshape(
        batch, kv_heads, heads_per_group, page_count
    ).any(dim=2)
    return KQPageSelection(
        page_mask=page_mask,
        page_log_mass=page_log_mass,
        refined_query_mask=refined,
        tail_mass_ratio=tail_mass_ratio,
    )


def fetch_and_pack_exact_key_pages(
    page_store: ExactKeyPageSource,
    page_mask: Tensor,
    *,
    layer_idx: int,
    page_size: int,
    head_dim: int,
    device: torch.device,
) -> PackedExactKeyPages:
    """Fetch selected CPU pages and pack them for one batched sparse attention."""

    if page_mask.ndim != 3 or page_mask.dtype != torch.bool:
        raise ValueError("page mask must be boolean [batch, KV heads, pages]")
    if page_mask.device != device:
        raise ValueError("page mask and attention must share a device")
    batch, kv_heads, _ = map(int, page_mask.shape)
    counts = page_mask.sum(dim=-1)
    if bool((counts == 0).any()):
        raise ValueError("every physical KV head must select at least one page")
    max_pages = int(counts.max().item())
    slots_by_page = page_mask.cumsum(dim=-1) - 1
    requests = torch.nonzero(page_mask, as_tuple=False)
    request_slots = slots_by_page[
        requests[:, 0], requests[:, 1], requests[:, 2]
    ]
    page_ids = torch.full(
        (batch, kv_heads, max_pages),
        -1,
        dtype=torch.long,
        device=device,
    )
    page_ids[
        requests[:, 0], requests[:, 1], request_slots
    ] = requests[:, 2]
    fetched = page_store.get_pages(
        layer_idx=layer_idx,
        batch_indices=requests[:, 0],
        kv_head_indices=requests[:, 1],
        page_ids=requests[:, 2],
        page_size=page_size,
        device=device,
    )
    expected = (len(requests), page_size, head_dim)
    if tuple(fetched.shape) != expected:
        raise ValueError(
            f"exact-Key store returned {tuple(fetched.shape)}, expected {expected}"
        )
    packed = torch.zeros(
        batch,
        kv_heads,
        max_pages,
        page_size,
        head_dim,
        dtype=fetched.dtype,
        device=device,
    )
    packed[
        requests[:, 0], requests[:, 1], request_slots
    ] = fetched
    return PackedExactKeyPages(
        exact_key=packed,
        page_ids=page_ids,
        valid_pages=page_ids >= 0,
        requested_pages=int(len(requests)),
        requested_bytes=(
            int(len(requests)) * page_size * head_dim * fetched.element_size()
        ),
    )


def sparse_exact_k_c1_attention(
    query: Tensor,
    c1_value: Tensor,
    pages: PackedExactKeyPages,
    *,
    page_size: int,
) -> Tensor:
    """Run one exact sparse SDPA call over packed K pages and resident C1-V."""

    if query.ndim != 4 or int(query.shape[2]) != 1 or c1_value.ndim != 4:
        raise ValueError("query/C1 Value must be rank-four decode tensors")
    batch, query_heads, _, head_dim = map(int, query.shape)
    value_batch, kv_heads, sequence, value_rank = map(int, c1_value.shape)
    if value_batch != batch or query_heads % kv_heads:
        raise ValueError("query and C1 Value have incompatible GQA geometry")
    if int(pages.exact_key.shape[-1]) != head_dim:
        raise ValueError("packed exact-Key width differs from query")
    if tuple(pages.page_ids.shape[:2]) != (batch, kv_heads):
        raise ValueError("packed exact-Key ownership differs from C1 Value")
    if not (query.device == c1_value.device == pages.exact_key.device):
        raise ValueError("query, exact K, and C1 Value must share a device")
    if not (query.dtype == c1_value.dtype == pages.exact_key.dtype):
        raise TypeError("query, exact K, and C1 Value must share a dtype")

    page_count = math.ceil(sequence / page_size)
    padding = page_count * page_size - sequence
    padded_value = F.pad(c1_value, (0, 0, 0, padding))
    paged_value = padded_value.reshape(
        batch, kv_heads, page_count, page_size, value_rank
    )
    safe_page_ids = pages.page_ids.clamp_min(0)
    gather_ids = safe_page_ids[..., None, None].expand(
        batch,
        kv_heads,
        int(safe_page_ids.shape[-1]),
        page_size,
        value_rank,
    )
    selected_value = paged_value.gather(2, gather_ids)
    token_positions = (
        safe_page_ids[..., None] * page_size
        + torch.arange(page_size, device=query.device)
    )
    valid_tokens = pages.valid_pages[..., None] & (token_positions < sequence)
    selected_tokens = int(safe_page_ids.shape[-1]) * page_size
    packed_key = pages.exact_key.reshape(
        batch, kv_heads, selected_tokens, head_dim
    )
    packed_value = selected_value.reshape(
        batch, kv_heads, selected_tokens, value_rank
    )
    heads_per_group = query_heads // kv_heads
    attention_mask = valid_tokens.repeat_interleave(
        heads_per_group, dim=1
    ).reshape(batch, query_heads, 1, selected_tokens)
    return F.scaled_dot_product_attention(
        query,
        packed_key,
        packed_value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=heads_per_group > 1,
    )


def dense_exact_k_c1_attention(
    query: Tensor,
    exact_key: Tensor,
    c1_value: Tensor,
) -> Tensor:
    """Full-scan exact-K/C1-V decode baseline using the same SDPA backend."""

    if query.ndim != 4 or exact_key.ndim != 4 or c1_value.ndim != 4:
        raise ValueError("dense Q/K/C1-V tensors must be rank four")
    query_heads = int(query.shape[1])
    kv_heads = int(exact_key.shape[1])
    if query_heads % kv_heads or tuple(exact_key.shape[:3]) != tuple(c1_value.shape[:3]):
        raise ValueError("dense Q/K/C1-V tensors have incompatible GQA geometry")
    if not (query.device == exact_key.device == c1_value.device):
        raise ValueError("dense Q/K/C1-V tensors must share a device")
    if not (query.dtype == exact_key.dtype == c1_value.dtype):
        raise TypeError("dense Q/K/C1-V tensors must share a dtype")
    return F.scaled_dot_product_attention(
        query,
        exact_key,
        c1_value,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=query_heads > kv_heads,
    )


def offloaded_kq_svd_c1_attention(
    query: Tensor,
    routing_sidecar: Tensor,
    routing_query_projector: Tensor,
    c1_value: Tensor,
    page_store: ExactKeyPageSource,
    *,
    layer_idx: int,
    page_size: int,
    pages_per_query_head: int,
    adaptive_max_pages_per_query_head: int | None = None,
    adaptive_tail_mass_ratio_threshold: float | None = None,
) -> OffloadedC1Attention:
    """Execute routing, pinned-host exact-K fetch, and sparse exact C1 attention."""

    selection = kq_svd_gqa_page_selection(
        query,
        routing_sidecar,
        routing_query_projector,
        page_size=page_size,
        pages_per_query_head=pages_per_query_head,
        adaptive_max_pages_per_query_head=adaptive_max_pages_per_query_head,
        adaptive_tail_mass_ratio_threshold=adaptive_tail_mass_ratio_threshold,
    )
    pages = fetch_and_pack_exact_key_pages(
        page_store,
        selection.page_mask,
        layer_idx=layer_idx,
        page_size=page_size,
        head_dim=int(query.shape[-1]),
        device=query.device,
    )
    output = sparse_exact_k_c1_attention(
        query, c1_value, pages, page_size=page_size
    )
    return OffloadedC1Attention(
        output=output,
        selection=selection,
        pages=pages,
    )


class PreparedExactKeyPageFetch:
    """Reusable staging for the existing page store; no hot-path allocations.

    GPU page IDs are copied to a preallocated pinned index buffer. The CPU
    gathers requested pages into pinned staging, then one asynchronous DMA
    copies that staging to the GPU. The D2H dependency is intentionally kept.
    """

    def __init__(self, store: PinnedCPUExactKeyPageStore, *, pages_per_head: int,
                 page_size: int, device: torch.device):
        assert pages_per_head > 0 and page_size > 0 and device.type == 'cuda'
        assert store.exact_key.is_pinned()
        self.bank = store._page_bank(page_size)
        batch, heads, length, dim = store.exact_key.shape
        self.batch, self.heads = batch, heads
        self.page_size, self.pages_per_head = page_size, pages_per_head
        count = batch * heads * pages_per_head
        self.gpu_indices = torch.empty(batch, heads, pages_per_head, device=device, dtype=torch.int64)
        self.host_indices = torch.empty(count, dtype=torch.int64, pin_memory=True)
        self.offsets = (torch.arange(batch * heads, device=device).view(batch, heads, 1)
                        * math.ceil(length / page_size))
        self.staging = torch.empty(count, page_size, dim, dtype=store.exact_key.dtype, pin_memory=True)
        self.destination = torch.empty_like(self.staging, device=device)
        self.indices_ready = torch.cuda.Event()
        self.requested_bytes = self.staging.numel() * self.staging.element_size()
        self.device = device

    def __call__(self, page_ids: Tensor) -> Tensor:
        assert page_ids.shape == self.gpu_indices.shape
        torch.add(page_ids, self.offsets, out=self.gpu_indices)
        self.host_indices.copy_(self.gpu_indices.view(-1), non_blocking=True)
        self.indices_ready.record(torch.cuda.current_stream(self.device))
        self.indices_ready.synchronize()
        torch.index_select(self.bank, 0, self.host_indices, out=self.staging)
        self.destination.copy_(self.staging, non_blocking=True)
        return self.destination.view(self.batch, self.heads,
                                     self.pages_per_head * self.page_size, -1)


class PreparedQueryKeyFetch:
    """Deduplicated token DMA, with an inverse map for independent query supports.

    The union controls only the bytes fetched. Attention must use inverse_ids
    for K and the original per-query token IDs for V. Buffers are reused on
    one CUDA stream; the D2H event also protects the prior staging H2D read.
    """
    def __init__(self,store: PinnedCPUExactKeyPageStore,*,groups:int,
                 tokens_per_query:int,device:torch.device):
        assert store.exact_key.is_pinned() and groups>0 and tokens_per_query>0
        b,h,t,d=store.exact_key.shape
        self.shape=(b,h,groups,tokens_per_query);self.length=t
        self.bank=store._page_bank(1).view(b*h*t,d)
        self.count=b*h*groups*tokens_per_query
        self.group_requests=groups*tokens_per_query
        self.sentinel=b*h*t;self.device=device
        self.encoded=torch.empty(self.count,device=device,dtype=torch.int64)
        self.sorted_ids=torch.empty_like(self.encoded);self.permutation=torch.empty_like(self.encoded)
        self.starts=torch.empty(self.count,device=device,dtype=torch.bool)
        self.positions=torch.empty_like(self.encoded)
        self.unique_ids=torch.empty_like(self.encoded)
        self.inverse_ids=torch.empty(self.shape,device=device,dtype=torch.int64)
        self.gpu_count=torch.empty((),device=device,dtype=torch.int64)
        self.host_count=torch.empty((),dtype=torch.int64,pin_memory=True)
        self.host_indices=torch.empty(self.count,dtype=torch.int64,pin_memory=True)
        self.staging=torch.empty(self.count,d,dtype=store.exact_key.dtype,pin_memory=True)
        self.destination=torch.empty_like(self.staging,device=device)
        self.ready=torch.cuda.Event();self.actual_count=0

    def compact(self,ids:Tensor):
        from basisserve.kernels.selected_key_compaction import encode_requests,pack_requests
        assert ids.shape==self.shape and ids.is_contiguous() and ids.dtype==torch.int64
        grid=((self.count+255)//256,)
        encode_requests[grid](ids,self.encoded,LENGTH=self.length,GROUP_REQUESTS=self.group_requests,COUNT=self.count,BLOCK=256)
        torch.sort(self.encoded,out=(self.sorted_ids,self.permutation))
        self.starts[:1].fill_(True)
        torch.ne(self.sorted_ids[1:],self.sorted_ids[:-1],out=self.starts[1:])
        torch.cumsum(self.starts,0,out=self.positions)
        pack_requests[grid](self.sorted_ids,self.permutation,self.positions,self.unique_ids,
            self.inverse_ids,self.gpu_count,SENTINEL=self.sentinel,COUNT=self.count,BLOCK=256)

    def __call__(self,ids:Tensor):
        self.compact(ids)
        self.host_count.copy_(self.gpu_count,non_blocking=True)
        self.ready.record(torch.cuda.current_stream(self.device));self.ready.synchronize()
        self.actual_count=int(self.host_count)
        n=self.actual_count
        self.host_indices[:n].copy_(self.unique_ids[:n],non_blocking=True)
        self.ready.record(torch.cuda.current_stream(self.device));self.ready.synchronize()
        torch.index_select(self.bank,0,self.host_indices[:n],out=self.staging[:n])
        self.destination[:n].copy_(self.staging[:n],non_blocking=True)
        return self.destination

    def traffic(self):
        return dict(unique_key_tokens=self.actual_count,
            logical_k_bytes=self.actual_count*self.destination.shape[-1]*self.destination.element_size(),
            h2d_dma_payload_bytes=self.actual_count*self.destination.shape[-1]*self.destination.element_size(),
            d2h_index_payload_bytes=(self.actual_count+1)*8,
            measured_pcie_bus_bytes=None)
