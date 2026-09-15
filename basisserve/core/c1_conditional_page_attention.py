"""Full-query Page32 attention for a C1 conditional Key router."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from basisserve.kernels.indexed_sparse_decode_attention import (
    gqa_page32_log_mass_triton,
)
from basisserve.kernels.mapped_host_paged_attention import (
    gpu_paged_attention,
    select_fixed_group_max_pages_cuda,
)


@dataclass(frozen=True)
class C1ConditionalPageAttentionResult:
    """Causal attention output and logical physical-page statistics."""

    output: Tensor
    statistics: dict[str, float | Tensor]


def _head_to_kv(query_heads: int, kv_heads: int, device: torch.device) -> Tensor:
    assert kv_heads > 0 and query_heads % kv_heads == 0
    return torch.arange(kv_heads, device=device).repeat_interleave(
        query_heads // kv_heads
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


def _selected_pages(
    proxy_scores: Tensor,
    valid: Tensor,
    *,
    kv_heads: int,
    page_size: int,
    page_budget: int,
    pinned_prefix_pages: int,
) -> tuple[Tensor, Tensor]:
    """Select one fixed physical page set per query and GQA group."""

    batch, query_heads, queries, sequence_length = map(int, proxy_scores.shape)
    heads_per_group = query_heads // kv_heads
    page_count = math.ceil(sequence_length / page_size)
    padded_tokens = page_count * page_size
    padding = padded_tokens - sequence_length
    padded_scores = proxy_scores.float()
    padded_valid = valid
    if padding:
        padded_scores = torch.nn.functional.pad(
            padded_scores,
            (0, padding),
            value=-torch.inf,
        )
        padded_valid = torch.nn.functional.pad(valid, (0, padding), value=False)

    page_valid = padded_valid.reshape(
        batch, query_heads, queries, page_count, page_size
    ).any(dim=-1)
    page_indices = torch.arange(page_count, device=proxy_scores.device)
    routed_page_valid = page_valid & (
        page_indices[None, None, None, :] >= pinned_prefix_pages
    )
    page_log_mass = torch.logsumexp(
        padded_scores.reshape(
            batch, query_heads, queries, page_count, page_size
        ),
        dim=-1,
    ).masked_fill(~routed_page_valid, -torch.inf)
    has_routed_page = routed_page_valid.any(dim=-1, keepdim=True)
    safe_page_log_mass = torch.where(
        has_routed_page,
        page_log_mass,
        torch.zeros_like(page_log_mass),
    )
    normalized_page_mass = torch.softmax(safe_page_log_mass, dim=-1).masked_fill(
        ~routed_page_valid,
        0.0,
    )
    group_scores = normalized_page_mass.reshape(
        batch,
        kv_heads,
        heads_per_group,
        queries,
        page_count,
    ).amax(dim=2)
    group_page_valid = page_valid.reshape(
        batch,
        kv_heads,
        heads_per_group,
        queries,
        page_count,
    ).any(dim=2)
    group_routed_valid = group_page_valid & (
        page_indices[None, None, None, :] >= pinned_prefix_pages
    )
    group_scores.masked_fill_(~group_routed_valid, -torch.inf)

    pinned_count = min(pinned_prefix_pages, page_budget, page_count)
    routed_count = min(page_budget - pinned_count, page_count - pinned_count)
    selected_indices = []
    selected_valid = []
    if pinned_count:
        prefix = torch.arange(
            pinned_count,
            dtype=torch.long,
            device=proxy_scores.device,
        ).view(1, 1, 1, pinned_count)
        prefix = prefix.expand(batch, kv_heads, queries, pinned_count)
        selected_indices.append(prefix)
        selected_valid.append(group_page_valid.gather(-1, prefix))
    if routed_count:
        routed = torch.topk(
            group_scores,
            routed_count,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        selected_indices.append(routed)
        selected_valid.append(group_routed_valid.gather(-1, routed))

    assert selected_indices
    return torch.cat(selected_indices, dim=-1), torch.cat(selected_valid, dim=-1)


@torch.inference_mode()
def c1_conditional_page_topk_attention(
    query: Tensor,
    exact_post_key: Tensor,
    c1_value: Tensor,
    routing_sidecar: Tensor,
    routing_query_projector: Tensor,
    *,
    page_size: int,
    exact_token_budget: int,
    pinned_prefix_pages: int,
    scale: float,
    query_block_size: int,
    attention_mask: Tensor | None = None,
    collect_statistics: bool = True,
) -> C1ConditionalPageAttentionResult:
    """Apply full-causal Base+Residual routing and selected exact-QK attention.

    Proxy token scores are converted to Page log-masses, normalized independently
    for every Query head, and reduced with a max across the Query heads sharing one
    physical GQA Key head.  The selected Page set is therefore physical and fixed
    per KV head.  Exact 128-dimensional QK and the C1 Value payload are evaluated
    only on tokens in those selected pages.
    """

    batch, query_heads, query_length, head_dim = map(int, query.shape)
    key_batch, kv_heads, sequence_length, key_dim = map(int, exact_post_key.shape)
    value_batch, value_heads, value_length, value_rank = map(int, c1_value.shape)
    sidecar_batch, sidecar_heads, sidecar_length, routing_rank = map(
        int, routing_sidecar.shape
    )
    assert batch == key_batch == value_batch == sidecar_batch
    assert kv_heads == value_heads == sidecar_heads
    assert sequence_length == value_length == sidecar_length
    assert head_dim == key_dim and query_heads % kv_heads == 0
    assert tuple(routing_query_projector.shape) == (
        query_heads,
        head_dim,
        routing_rank,
    )
    assert page_size > 0 and exact_token_budget > 0 and query_block_size > 0
    page_budget = math.ceil(exact_token_budget / page_size)
    assert 0 <= pinned_prefix_pages <= page_budget

    head_to_kv = _head_to_kv(query_heads, kv_heads, query.device)
    query_projector = routing_query_projector.to(
        device=query.device,
        dtype=query.dtype,
    )
    heads_per_group = query_heads // kv_heads
    page_count = math.ceil(sequence_length / page_size)
    use_decode_kernels = (
        query_length == 1
        and attention_mask is None
        and page_size == 32
        and page_budget < page_count
        and page_budget <= 128
        and page_count <= 4096
        and query_heads % kv_heads == 0
        and query_heads // kv_heads <= 16
        and head_dim == 128
        and 0 < value_rank <= 256
        and routing_rank <= 256
        and query.is_cuda
        and exact_post_key.is_cuda
        and c1_value.is_cuda
        and routing_sidecar.is_cuda
        and query.dtype == torch.bfloat16
        and exact_post_key.dtype == query.dtype
        and c1_value.dtype == query.dtype
        and routing_sidecar.dtype == query.dtype
        and torch.cuda.get_device_capability(query.device)[0] >= 8
    )
    expanded_key = (
        None if use_decode_kernels else exact_post_key.index_select(1, head_to_kv)
    )
    expanded_value = (
        None if use_decode_kernels else c1_value.index_select(1, head_to_kv)
    )
    expanded_sidecar = (
        None if use_decode_kernels else routing_sidecar.index_select(1, head_to_kv)
    )
    token_offsets = torch.arange(page_size, device=query.device)
    output_blocks = []
    logical_selected_tokens = 0
    physical_selected_tokens = 0
    logical_selected_pages = 0
    physical_selected_pages = 0
    query_valid_tokens = 0
    physical_valid_tokens = 0

    for query_start in range(0, query_length, query_block_size):
        query_stop = min(query_start + query_block_size, query_length)
        block_queries = query_stop - query_start
        query_block = query[:, :, query_start:query_stop]
        if use_decode_kernels:
            query_code = torch.einsum(
                "bhqd,hdr->bhqr",
                query_block,
                query_projector,
            )[:, :, 0].contiguous()
            page_log_mass = gqa_page32_log_mass_triton(
                query_code,
                routing_sidecar,
                scale=scale,
            )
            selected_page_ids = select_fixed_group_max_pages_cuda(
                page_log_mass,
                pages_per_kv_head=page_budget,
                pinned_prefix_pages=pinned_prefix_pages,
                force_current_page=False,
            )
            output_blocks.append(
                gpu_paged_attention(
                    exact_post_key,
                    query_block,
                    c1_value,
                    selected_page_ids,
                    sequence_length=sequence_length,
                    scale=scale,
                    splits=32,
                )
            )
            if collect_statistics:
                selected_widths = (
                    sequence_length - selected_page_ids * page_size
                ).clamp(min=0, max=page_size)
                block_physical_tokens = selected_widths.sum()
                block_physical_pages = int(selected_page_ids.numel())
                physical_selected_tokens += block_physical_tokens
                logical_selected_tokens += (
                    block_physical_tokens * heads_per_group
                )
                physical_selected_pages += block_physical_pages
                logical_selected_pages += block_physical_pages * heads_per_group
                query_valid_tokens += batch * query_heads * sequence_length
                physical_valid_tokens += batch * kv_heads * sequence_length
            continue

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
            "bhqd,hdr->bhqr",
            query_block,
            query_projector,
        )
        assert expanded_sidecar is not None
        proxy_scores = torch.matmul(
            query_code,
            expanded_sidecar.transpose(-1, -2),
        ).mul_(scale)
        proxy_scores.masked_fill_(~expanded_valid, -torch.inf)
        selected_page_ids, selected_page_valid = _selected_pages(
            proxy_scores,
            expanded_valid,
            kv_heads=kv_heads,
            page_size=page_size,
            page_budget=page_budget,
            pinned_prefix_pages=pinned_prefix_pages,
        )

        raw_token_ids = (
            selected_page_ids[..., None] * page_size
            + token_offsets.view(1, 1, 1, 1, page_size)
        ).flatten(-2)
        selected_physical_valid = selected_page_valid[..., None].expand(
            *selected_page_valid.shape,
            page_size,
        ).flatten(-2)
        # A one-page expansion can retain zero strides; do not mutate that view.
        selected_physical_valid = selected_physical_valid & (raw_token_ids < sequence_length)
        physical_valid = expanded_valid.reshape(
            batch,
            kv_heads,
            heads_per_group,
            block_queries,
            sequence_length,
        ).any(dim=2)
        selected_token_ids = raw_token_ids.clamp_max(sequence_length - 1)
        selected_physical_valid &= physical_valid.gather(-1, selected_token_ids)

        query_token_ids = selected_token_ids.index_select(1, head_to_kv)
        selected_valid = selected_physical_valid.index_select(1, head_to_kv)
        assert expanded_key is not None and expanded_value is not None
        selected_key = torch.gather(
            expanded_key[:, :, None].expand(
                batch,
                query_heads,
                block_queries,
                sequence_length,
                head_dim,
            ),
            3,
            query_token_ids[..., None].expand(
                batch,
                query_heads,
                block_queries,
                int(query_token_ids.shape[-1]),
                head_dim,
            ),
        )
        selected_scores = torch.einsum(
            "bhqd,bhqkd->bhqk",
            query_block,
            selected_key,
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
            query_token_ids[..., None].expand(
                batch,
                query_heads,
                block_queries,
                int(query_token_ids.shape[-1]),
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
            logical_selected_tokens += int(selected_valid.sum().item())
            physical_selected_tokens += int(selected_physical_valid.sum().item())
            logical_selected_pages += int(
                selected_page_valid.sum().item() * heads_per_group
            )
            physical_selected_pages += int(selected_page_valid.sum().item())
            query_valid_tokens += int(expanded_valid.sum().item())
            physical_valid_tokens += int(physical_valid.sum().item())

    return C1ConditionalPageAttentionResult(
        output=torch.cat(output_blocks, dim=2).to(dtype=c1_value.dtype),
        statistics={
            "queries": float(batch * query_length),
            "physical_valid_tokens": float(physical_valid_tokens),
            "query_valid_tokens": float(query_valid_tokens),
            "selected_tokens": physical_selected_tokens,
            "query_selected_tokens": float(logical_selected_tokens),
            "selected_pages": float(physical_selected_pages),
            "logical_selected_pages": float(logical_selected_pages),
            "oracle_page_store_key_bytes_read": (
                physical_selected_tokens
                * head_dim
                * exact_post_key.element_size()
            ),
            "resident_selector_metadata_bytes": float(
                routing_sidecar.numel() * routing_sidecar.element_size()
            ),
            "selection_qk_flops": float(
                2
                * batch
                * query_heads
                * query_length
                * sequence_length
                * routing_rank
            ),
            "sparse_exact_qk_flops": float(
                2 * logical_selected_tokens * head_dim
            ),
            "sparse_c1_value_flops": float(
                2 * logical_selected_tokens * value_rank
            ),
            "adaptive_eligible_query_heads": 0.0,
            "adaptive_refined_query_heads": 0.0,
            "adaptive_tail_mass_ratio_sum": 0.0,
        },
    )


__all__ = [
    "C1ConditionalPageAttentionResult",
    "c1_conditional_page_topk_attention",
]
