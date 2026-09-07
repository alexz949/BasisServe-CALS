"""Loki PCA Top-K attention with a resident compressed C1 Value cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from basisserve.core.c1_k_routing_sidecar import build_routing_sidecar
from basisserve.kernels.indexed_sparse_decode_attention import (
    gqa_indexed_sparse_decode_attention_triton,
    gqa_proxy_scores_triton,
)


@dataclass(frozen=True)
class C1LokiAttentionResult:
    """Block attention output and logical routing statistics."""

    output: Tensor
    statistics: dict[str, float | Tensor]


def _head_to_kv(query_heads: int, kv_heads: int, device: torch.device) -> Tensor:
    assert kv_heads > 0 and query_heads % kv_heads == 0
    return torch.arange(kv_heads, device=device).repeat_interleave(
        query_heads // kv_heads
    )


def _query_projector(
    projector: Tensor,
    *,
    query_heads: int,
    kv_heads: int,
    head_to_kv: Tensor,
) -> Tensor:
    assert int(projector.shape[0]) in (query_heads, kv_heads)
    return (
        projector
        if int(projector.shape[0]) == query_heads
        else projector.index_select(0, head_to_kv)
    )


def _valid_attention_support(
    attention_mask: Tensor | None,
    *,
    batch: int,
    query_start: int,
    query_stop: int,
    query_length: int,
    sequence_length: int,
    device: torch.device,
) -> Tensor:
    query_positions = (
        sequence_length
        - query_length
        + torch.arange(query_start, query_stop, device=device)
    )
    key_positions = torch.arange(sequence_length, device=device)
    valid = (key_positions[None, :] <= query_positions[:, None])[None, None].expand(
        batch, 1, -1, -1
    )
    if attention_mask is None:
        return valid

    candidate = attention_mask.to(device=device)
    if candidate.ndim == 2:
        candidate = candidate[:, None, None, :]
    else:
        assert candidate.ndim == 4
        if int(candidate.shape[-2]) == query_length:
            candidate = candidate[..., query_start:query_stop, :]
        else:
            assert int(candidate.shape[-2]) in (1, query_stop - query_start)
    allowed = (
        candidate
        if candidate.dtype == torch.bool
        else torch.isfinite(candidate) & (candidate > -1.0e20)
    )
    return valid & allowed


def _physical_union_count(
    indices: Tensor,
    selected_valid: Tensor,
    *,
    kv_heads: int,
    sequence_length: int,
) -> Tensor:
    batch, query_heads, queries, selected = map(int, indices.shape)
    heads_per_group = query_heads // kv_heads
    grouped_indices = indices.reshape(
        batch, kv_heads, heads_per_group, queries, selected
    )
    grouped_valid = selected_valid.reshape(
        batch, kv_heads, heads_per_group, queries, selected
    )
    physical = torch.zeros(
        batch,
        kv_heads,
        queries,
        sequence_length,
        dtype=torch.bool,
        device=indices.device,
    )
    for head in range(heads_per_group):
        contribution = torch.zeros_like(physical)
        contribution.scatter_(
            -1,
            grouped_indices[:, :, head],
            grouped_valid[:, :, head],
        )
        physical |= contribution
    return physical.sum()


@torch.inference_mode()
def c1_loki_pca_topk_attention(
    query: Tensor,
    exact_post_key: Tensor,
    c1_value: Tensor,
    key_projector: Tensor,
    query_projector: Tensor,
    *,
    top_k: int,
    scale: float,
    query_block_size: int,
    attention_mask: Tensor | None = None,
    routing_sidecar: Tensor | None = None,
    collect_statistics: bool = True,
) -> C1LokiAttentionResult:
    """Apply repository-faithful Loki selection and exact attention.

    Loki projects post-RoPE Q/K with a Key-PCA basis, independently selects
    token Top-K support for every Query head, and evaluates original exact-QK
    on that support.  Query-axis tiling changes only peak memory.  Values stay
    in C1 coordinates and are decoded by the caller's compressed ``o_proj``.
    """

    batch, query_heads, query_length, head_dim = map(int, query.shape)
    key_batch, kv_heads, sequence_length, key_dim = map(int, exact_post_key.shape)
    value_batch, value_heads, value_length, value_rank = map(int, c1_value.shape)
    routing_rank = int(key_projector.shape[-1])
    assert batch == key_batch == value_batch
    assert kv_heads == value_heads and sequence_length == value_length
    assert head_dim == key_dim
    assert tuple(key_projector.shape) == (kv_heads, head_dim, routing_rank)
    assert tuple(query_projector.shape[1:]) == (head_dim, routing_rank)
    assert top_k > 0 and query_block_size > 0

    head_to_kv = _head_to_kv(query_heads, kv_heads, query.device)
    sidecar = (
        build_routing_sidecar(exact_post_key, key_projector)
        if routing_sidecar is None
        else routing_sidecar
    )
    expanded_query_projector = _query_projector(
        query_projector,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_to_kv=head_to_kv,
    ).to(device=query.device, dtype=query.dtype)
    use_decode_kernels = (
        query_length == 1
        and attention_mask is None
        and query.is_cuda
        and exact_post_key.is_cuda
        and c1_value.is_cuda
        and sidecar.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and exact_post_key.dtype == query.dtype
        and c1_value.dtype == query.dtype
        and sidecar.dtype == query.dtype
        and head_dim <= 256
        and value_rank <= 128
        and routing_rank <= 256
    )
    expanded_key = (
        None if use_decode_kernels else exact_post_key.index_select(1, head_to_kv)
    )
    expanded_value = (
        None if use_decode_kernels else c1_value.index_select(1, head_to_kv)
    )
    expanded_sidecar = (
        None if use_decode_kernels else sidecar.index_select(1, head_to_kv)
    )
    selected_count = min(top_k, sequence_length)
    heads_per_group = query_heads // kv_heads
    output_blocks = []
    logical_selected_tokens = 0
    physical_selected_tokens = 0
    query_valid_tokens = 0
    physical_valid_tokens = 0

    for query_start in range(0, query_length, query_block_size):
        query_stop = min(query_start + query_block_size, query_length)
        query_block = query[:, :, query_start:query_stop]
        valid = _valid_attention_support(
            attention_mask,
            batch=batch,
            query_start=query_start,
            query_stop=query_stop,
            query_length=query_length,
            sequence_length=sequence_length,
            device=query.device,
        )
        expanded_valid = valid.expand(batch, query_heads, -1, -1)
        query_code = torch.einsum(
            "bhqd,hdr->bhqr", query_block, expanded_query_projector
        )
        if use_decode_kernels:
            approximate_scores = gqa_proxy_scores_triton(
                query_code[:, :, 0].contiguous(),
                sidecar,
                scale=scale,
            )
        else:
            assert expanded_sidecar is not None
            approximate_scores = torch.matmul(
                query_code,
                expanded_sidecar.transpose(-1, -2),
            ).mul_(scale)
        approximate_scores.masked_fill_(~expanded_valid, -torch.inf)
        selected_indices = torch.topk(
            approximate_scores,
            selected_count,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices

        selected_valid = expanded_valid.gather(-1, selected_indices)
        if use_decode_kernels:
            output_blocks.append(
                gqa_indexed_sparse_decode_attention_triton(
                    query_block,
                    exact_post_key,
                    c1_value,
                    selected_indices,
                    scale=scale,
                )
            )
        else:
            assert expanded_key is not None and expanded_value is not None
            block_queries = query_stop - query_start
            selected_key = torch.gather(
                expanded_key[:, :, None].expand(
                    batch,
                    query_heads,
                    block_queries,
                    sequence_length,
                    head_dim,
                ),
                3,
                selected_indices[..., None].expand(
                    batch,
                    query_heads,
                    block_queries,
                    selected_count,
                    head_dim,
                ),
            )
            selected_scores = torch.einsum(
                "bhqd,bhqkd->bhqk", query_block, selected_key
            ).mul_(scale)
            del selected_key
            selected_scores.masked_fill_(~selected_valid, -torch.inf)
            selected_probability = (
                torch.softmax(selected_scores.float(), dim=-1)
                .to(dtype=query.dtype)
                .masked_fill_(~selected_valid, 0.0)
            )
            selected_value = torch.gather(
                expanded_value[:, :, None].expand(
                    batch,
                    query_heads,
                    block_queries,
                    sequence_length,
                    value_rank,
                ),
                3,
                selected_indices[..., None].expand(
                    batch,
                    query_heads,
                    block_queries,
                    selected_count,
                    value_rank,
                ),
            )
            output_blocks.append(
                torch.einsum(
                    "bhqk,bhqkv->bhqv",
                    selected_probability,
                    selected_value,
                )
            )

        if collect_statistics:
            union_count = _physical_union_count(
                selected_indices,
                selected_valid,
                kv_heads=kv_heads,
                sequence_length=sequence_length,
            )
            if use_decode_kernels:
                logical_selected_tokens += batch * query_heads * selected_count
                physical_selected_tokens += union_count
                query_valid_tokens += batch * query_heads * sequence_length
                physical_valid_tokens += batch * kv_heads * sequence_length
            else:
                logical_selected_tokens += int(selected_valid.sum().item())
                physical_selected_tokens += int(union_count.item())
                query_valid_tokens += int(expanded_valid.sum().item())
                physical_valid_tokens += int(
                    expanded_valid.reshape(
                        batch,
                        kv_heads,
                        heads_per_group,
                        query_stop - query_start,
                        sequence_length,
                    )
                    .any(dim=2)
                    .sum()
                    .item()
                )

    sidecar_bytes = sidecar.numel() * sidecar.element_size()
    return C1LokiAttentionResult(
        output=torch.cat(output_blocks, dim=2).to(dtype=c1_value.dtype),
        statistics={
            "queries": float(batch * query_length),
            "physical_valid_tokens": float(physical_valid_tokens),
            "query_valid_tokens": float(query_valid_tokens),
            "selected_tokens": physical_selected_tokens,
            "query_selected_tokens": float(logical_selected_tokens),
            "selected_pages": physical_selected_tokens,
            "logical_selected_pages": float(logical_selected_tokens),
            "oracle_page_store_key_bytes_read": (
                physical_selected_tokens
                * head_dim
                * exact_post_key.element_size()
            ),
            "resident_selector_metadata_bytes": float(sidecar_bytes),
            "selection_qk_flops": float(
                2
                * batch
                * routing_rank
                * (
                    kv_heads * sequence_length * head_dim
                    + query_heads * query_length * head_dim
                    + query_heads * query_length * sequence_length
                )
            ),
            "sparse_exact_qk_flops": float(2 * logical_selected_tokens * head_dim),
            "sparse_c1_value_flops": float(2 * logical_selected_tokens * value_rank),
            "adaptive_eligible_query_heads": 0.0,
            "adaptive_refined_query_heads": 0.0,
            "adaptive_tail_mass_ratio_sum": 0.0,
        },
    )


__all__ = ["C1LokiAttentionResult", "c1_loki_pca_topk_attention"]
