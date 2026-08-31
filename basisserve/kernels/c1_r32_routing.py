"""CUDA routing kernels for the Qwen3 GQA4 R32 page selector."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


_HEAD_DIM = 128
_ROUTING_RANK = 32
_PAGE_SIZE = 64
_QUERIES_PER_KV = 4
_MAX_TOP_PAGES = 32
_MAX_PAGES = 2048


def reference_c1_r32_page_lse(
    query: Tensor,
    routing_sidecar: Tensor,
    query_projector: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Materialize token proxy scores and reduce them to page log-masses."""

    if query.ndim != 4 or routing_sidecar.ndim != 4:
        raise ValueError("R32 routing Query and sidecar must be rank four")
    if query_projector.ndim != 3:
        raise ValueError("R32 routing Query projector must be rank three")
    batch, query_heads, query_tokens, head_dim = map(int, query.shape)
    sidecar_batch, kv_heads, tokens, rank = map(int, routing_sidecar.shape)
    if (
        batch != sidecar_batch
        or query_tokens != 1
        or query_heads != _QUERIES_PER_KV * kv_heads
        or tuple(query_projector.shape[1:]) != (head_dim, rank)
        or int(query_projector.shape[0]) not in (kv_heads, query_heads)
    ):
        raise ValueError("R32 routing geometry is incompatible")
    selected_scale = head_dim**-0.5 if scale is None else float(scale)
    grouped_query = query[:, :, 0].reshape(
        batch,
        kv_heads,
        _QUERIES_PER_KV,
        head_dim,
    )
    projector = query_projector.to(device=query.device, dtype=query.dtype)
    if int(projector.shape[0]) == kv_heads:
        query_code = torch.einsum("bhgd,hdr->bhgr", grouped_query, projector)
    else:
        query_code = torch.einsum(
            "bhgd,hgdr->bhgr",
            grouped_query,
            projector.reshape(
                kv_heads,
                _QUERIES_PER_KV,
                head_dim,
                rank,
            ),
        )
    token_scores = torch.einsum(
        "bhgr,bhtr->bhgt",
        query_code,
        routing_sidecar.to(device=query.device, dtype=query.dtype),
    ) * selected_scale
    pages = math.ceil(tokens / _PAGE_SIZE)
    padded = F.pad(
        token_scores.reshape(batch, query_heads, tokens),
        (0, pages * _PAGE_SIZE - tokens),
        value=-torch.inf,
    )
    return torch.logsumexp(
        padded.reshape(batch, query_heads, pages, _PAGE_SIZE),
        dim=-1,
    )


def reference_c1_r32_topk_gqa_union(
    page_log_mass: Tensor,
    *,
    pages_per_query_head: int,
) -> tuple[Tensor, Tensor]:
    """Reference sorted compact GQA union with fixed-capacity row padding."""

    if page_log_mass.ndim != 3 or not page_log_mass.is_floating_point():
        raise ValueError("page log-masses must be floating [B,Hq,P]")
    batch, query_heads, pages = map(int, page_log_mass.shape)
    if query_heads <= 0 or query_heads % _QUERIES_PER_KV:
        raise ValueError("page log-masses require GQA ratio four")
    if pages_per_query_head <= 0:
        raise ValueError("pages per Query head must be positive")
    kv_heads = query_heads // _QUERIES_PER_KV
    selected_per_query = min(int(pages_per_query_head), pages)
    top_pages = page_log_mass.reshape(
        batch,
        kv_heads,
        _QUERIES_PER_KV,
        pages,
    ).topk(selected_per_query, dim=-1).indices
    page_mask = torch.zeros(
        batch,
        kv_heads,
        pages,
        dtype=torch.bool,
        device=page_log_mass.device,
    )
    page_mask.scatter_(2, top_pages.reshape(batch, kv_heads, -1), True)
    counts = page_mask.sum(dim=-1, dtype=torch.int32)
    output_slots = min(pages, pages_per_query_head * _QUERIES_PER_KV)
    selected_page_ids = torch.full(
        (batch, kv_heads, output_slots),
        -1,
        dtype=torch.int64,
        device=page_log_mass.device,
    )
    page_indices = torch.arange(
        pages,
        dtype=torch.int64,
        device=page_log_mass.device,
    )
    for batch_index in range(batch):
        for kv_head in range(kv_heads):
            chosen = page_indices[page_mask[batch_index, kv_head]]
            selected_page_ids[
                batch_index,
                kv_head,
                : int(chosen.numel()),
            ] = chosen
    return selected_page_ids, counts


def c1_r32_page_lse_cuda(
    query: Tensor,
    routing_sidecar: Tensor,
    query_projector: Tensor,
    *,
    scale: float | None = None,
    query_code: Tensor | None = None,
    output: Tensor | None = None,
) -> Tensor:
    """Project Q once and stream R32 token codes into page64 log-masses."""

    if query.ndim != 4 or routing_sidecar.ndim != 4:
        raise ValueError("R32 routing Query and sidecar must be rank four")
    if query_projector.ndim != 3:
        raise ValueError("R32 routing Query projector must be rank three")
    batch, query_heads, query_tokens, head_dim = map(int, query.shape)
    sidecar_batch, kv_heads, tokens, rank = map(int, routing_sidecar.shape)
    pages = math.ceil(tokens / _PAGE_SIZE)
    if (
        sidecar_batch != batch
        or query_tokens != 1
        or head_dim != _HEAD_DIM
        or rank != _ROUTING_RANK
        or query_heads != _QUERIES_PER_KV * kv_heads
        or tuple(query_projector.shape[1:]) != (_HEAD_DIM, _ROUTING_RANK)
        or int(query_projector.shape[0]) not in (kv_heads, query_heads)
    ):
        raise ValueError(
            "fused R32 page routing requires QK128, R32, and GQA ratio four"
        )
    if tokens <= 0 or pages > _MAX_PAGES:
        raise ValueError("fused R32 page routing supports 1 through 128K tokens")
    tensors = (query, routing_sidecar, query_projector)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("fused R32 page routing requires CUDA tensors")
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("fused R32 page-routing tensors must share one device")
    if query.dtype not in (torch.float16, torch.bfloat16) or any(
        tensor.dtype != query.dtype for tensor in tensors
    ):
        raise TypeError("fused R32 page routing requires one FP16/BF16 dtype")
    if any(tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError("R32 routing feature dimensions must be contiguous")
    selected_scale = _HEAD_DIM**-0.5 if scale is None else float(scale)
    if not math.isfinite(selected_scale) or selected_scale <= 0.0:
        raise ValueError("R32 routing scale must be finite and positive")

    expected_query_code = (batch, query_heads, _ROUTING_RANK)
    if query_code is None:
        query_code = torch.empty(
            expected_query_code,
            dtype=query.dtype,
            device=query.device,
        )
    elif (
        tuple(query_code.shape) != expected_query_code
        or query_code.dtype != query.dtype
        or query_code.device != query.device
        or not query_code.is_contiguous()
    ):
        raise ValueError("invalid fused R32 Query-code workspace")
    expected_output = (batch, query_heads, pages)
    if output is None:
        output = torch.empty(
            expected_output,
            dtype=query.dtype,
            device=query.device,
        )
    elif (
        tuple(output.shape) != expected_output
        or output.dtype != query.dtype
        or output.device != query.device
        or not output.is_contiguous()
    ):
        raise ValueError("invalid fused R32 page-score output")

    from basisserve.kernels.compressed_v_decode_cuda import (
        launch_c1_r32_page_lse_cuda,
    )

    return launch_c1_r32_page_lse_cuda(
        query,
        routing_sidecar,
        query_projector,
        query_code,
        output,
        scale=selected_scale,
    )


def c1_r32_topk_gqa_union_cuda(
    page_log_mass: Tensor,
    *,
    pages_per_query_head: int,
    selected_page_ids: Tensor | None = None,
    selected_page_counts: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Fuse per-Query Top-k with a sorted, left-compacted GQA page union."""

    if page_log_mass.ndim != 3 or not page_log_mass.is_floating_point():
        raise ValueError("page log-masses must be floating [B,Hq,P]")
    batch, query_heads, pages = map(int, page_log_mass.shape)
    if query_heads <= 0 or query_heads % _QUERIES_PER_KV:
        raise ValueError("fused R32 Top-k requires GQA ratio four")
    if not 0 < pages_per_query_head <= _MAX_TOP_PAGES:
        raise ValueError("fused R32 Top-k supports 1 through 32 pages per Q head")
    if not 0 < pages <= _MAX_PAGES:
        raise ValueError("fused R32 Top-k supports 1 through 2048 pages")
    if not page_log_mass.is_cuda:
        raise ValueError("fused R32 Top-k requires CUDA page scores")
    if page_log_mass.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("fused R32 Top-k requires FP16 or BF16 page scores")
    if not page_log_mass.is_contiguous():
        raise ValueError("fused R32 Top-k requires contiguous page scores")
    kv_heads = query_heads // _QUERIES_PER_KV
    output_slots = min(pages, pages_per_query_head * _QUERIES_PER_KV)
    expected_ids = (batch, kv_heads, output_slots)
    if selected_page_ids is None:
        selected_page_ids = torch.empty(
            expected_ids,
            dtype=torch.int64,
            device=page_log_mass.device,
        )
    elif (
        tuple(selected_page_ids.shape) != expected_ids
        or selected_page_ids.dtype != torch.int64
        or selected_page_ids.device != page_log_mass.device
        or not selected_page_ids.is_contiguous()
    ):
        raise ValueError("invalid fused R32 compact page-ID output")
    expected_counts = (batch, kv_heads)
    if selected_page_counts is None:
        selected_page_counts = torch.empty(
            expected_counts,
            dtype=torch.int32,
            device=page_log_mass.device,
        )
    elif (
        tuple(selected_page_counts.shape) != expected_counts
        or selected_page_counts.dtype != torch.int32
        or selected_page_counts.device != page_log_mass.device
        or not selected_page_counts.is_contiguous()
    ):
        raise ValueError("invalid fused R32 selected-page count output")

    from basisserve.kernels.compressed_v_decode_cuda import (
        launch_c1_r32_topk_gqa_union_cuda,
    )

    launch_c1_r32_topk_gqa_union_cuda(
        page_log_mass,
        selected_page_ids,
        selected_page_counts,
        top_pages_per_query=pages_per_query_head,
    )
    return selected_page_ids, selected_page_counts


__all__ = [
    "c1_r32_page_lse_cuda",
    "c1_r32_topk_gqa_union_cuda",
    "reference_c1_r32_page_lse",
    "reference_c1_r32_topk_gqa_union",
]
