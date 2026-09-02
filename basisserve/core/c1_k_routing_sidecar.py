"""Low-dimensional post-RoPE Key sidecars for exact-Key page offload.

The routing sidecar is deliberately separate from the resident C1 Value
payload.  It produces page identifiers only; attention over the selected
pages still uses exact post-RoPE Keys.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from transformers import DynamicCache

from basisserve.core.exact_qk_v_offload import gqa_group_max_page_mass_mask


ROUTING_PROXY_IMPLEMENTATION = "group_normalized_max_fixed_budget_v1"


@dataclass(frozen=True)
class RoutingSidecarSelection:
    """Proxy scores and the physical pages selected for exact-Key fetch."""

    proxy_scores: Tensor
    token_mask: Tensor
    page_mask: Tensor


class RoutingDynamicCache(DynamicCache):
    """Dynamic KV cache with an aligned, incrementally projected Key sidecar."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._routing_sidecars: list[Tensor | None] = [None] * len(self.layers)

    def _ensure_routing_layer(self, layer_idx: int) -> None:
        if layer_idx < 0:
            raise ValueError("routing cache layer index must be nonnegative")
        while len(self._routing_sidecars) <= layer_idx:
            self._routing_sidecars.append(None)

    def update_routing_sidecar(
        self,
        new_exact_post_key: Tensor,
        key_projector: Tensor,
        layer_idx: int,
    ) -> Tensor:
        """Project and append only the newly committed post-RoPE Keys."""

        update = build_routing_sidecar(new_exact_post_key, key_projector)
        return self.update_precomputed_routing_sidecar(update, layer_idx)

    def update_precomputed_routing_sidecar(
        self,
        update: Tensor,
        layer_idx: int,
    ) -> Tensor:
        """Append routing coordinates already computed by an attention module."""

        self._ensure_routing_layer(layer_idx)
        cached = self._routing_sidecars[layer_idx]
        if cached is None:
            sidecar = update
        else:
            if (
                tuple(cached.shape[:-2]) != tuple(update.shape[:-2])
                or int(cached.shape[-1]) != int(update.shape[-1])
                or cached.device != update.device
                or cached.dtype != update.dtype
            ):
                raise ValueError("routing sidecar update geometry changed")
            sidecar = torch.cat((cached, update), dim=-2)
        expected_length = int(self.get_seq_length(layer_idx))
        if int(sidecar.shape[-2]) != expected_length:
            raise RuntimeError(
                "routing sidecar and exact-Key cache lengths diverged: "
                f"{int(sidecar.shape[-2])} vs {expected_length}"
            )
        self._routing_sidecars[layer_idx] = sidecar
        return sidecar

    def routing_sidecar(self, layer_idx: int) -> Tensor | None:
        """Return the complete sidecar for one layer, if it has been cached."""

        if layer_idx < 0 or layer_idx >= len(self._routing_sidecars):
            return None
        return self._routing_sidecars[layer_idx]

    def crop(self, max_length: int) -> None:
        for layer_idx, sidecar in enumerate(self._routing_sidecars):
            if sidecar is None:
                continue
            current_length = int(sidecar.shape[-2])
            target_length = (
                current_length - abs(max_length)
                if max_length <= 0
                else max_length
            )
            target_length = max(target_length, 0)
            if current_length > target_length:
                self._routing_sidecars[layer_idx] = sidecar[
                    ..., :target_length, :
                ]
        super().crop(max_length)

    def reorder_cache(self, beam_idx: Tensor) -> None:
        super().reorder_cache(beam_idx)
        for layer_idx, sidecar in enumerate(self._routing_sidecars):
            if sidecar is not None:
                self._routing_sidecars[layer_idx] = sidecar.index_select(
                    0, beam_idx.to(sidecar.device)
                )

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        for layer_idx, sidecar in enumerate(self._routing_sidecars):
            if sidecar is not None:
                self._routing_sidecars[layer_idx] = sidecar.repeat_interleave(
                    repeats, dim=0
                )

    def batch_select_indices(self, indices: Tensor) -> None:
        super().batch_select_indices(indices)
        for layer_idx, sidecar in enumerate(self._routing_sidecars):
            if sidecar is not None:
                self._routing_sidecars[layer_idx] = sidecar[
                    indices.to(sidecar.device)
                ]

    def reset(self) -> None:
        super().reset()
        self._routing_sidecars = [None] * len(self.layers)


def _validate_key_projector(key_projector: Tensor) -> tuple[int, int, int]:
    if key_projector.ndim != 3 or not key_projector.is_floating_point():
        raise ValueError(
            "Key projector must be floating [KV heads, head dim, routing rank]"
        )
    kv_heads, head_dim, rank = map(int, key_projector.shape)
    if min(kv_heads, head_dim, rank) <= 0 or rank > head_dim:
        raise ValueError("routing projector has invalid head/rank geometry")
    return kv_heads, head_dim, rank


def truncate_routing_projectors(
    key_projector: Tensor,
    query_projector: Tensor,
    *,
    rank: int,
) -> tuple[Tensor, Tensor]:
    """Take a prefix of ordered PCA/KQ-SVD routing factors."""

    kv_heads, head_dim, available_rank = _validate_key_projector(key_projector)
    if query_projector.ndim != 3 or not query_projector.is_floating_point():
        raise ValueError(
            "Query projector must be floating [KV or Query heads, head dim, rank]"
        )
    if int(query_projector.shape[0]) % kv_heads:
        raise ValueError("Query projector heads must be divisible by KV heads")
    if tuple(query_projector.shape[1:]) != (head_dim, available_rank):
        raise ValueError("Key and Query routing factors have incompatible geometry")
    if not 0 < rank <= available_rank:
        raise ValueError(f"routing rank must be in [1, {available_rank}]")
    return (
        key_projector[..., :rank].contiguous(),
        query_projector[..., :rank].contiguous(),
    )


def build_routing_sidecar(
    exact_post_key: Tensor,
    key_projector: Tensor,
) -> Tensor:
    """Project exact post-RoPE Keys into a persistent routing-only code.

    ``exact_post_key`` may be either ``[KV heads, tokens, head dim]`` or
    ``[batch, KV heads, tokens, head dim]``.  The returned tensor keeps the
    same prefix and replaces ``head dim`` by ``routing rank``.
    """

    kv_heads, head_dim, _ = _validate_key_projector(key_projector)
    if exact_post_key.ndim not in (3, 4) or not exact_post_key.is_floating_point():
        raise ValueError("exact post-RoPE Key must be a floating rank-3/4 tensor")
    if tuple(exact_post_key.shape[-3::2]) != (kv_heads, head_dim):
        raise ValueError("exact Key and routing projector geometry differ")
    return torch.einsum(
        "...htd,hdr->...htr",
        exact_post_key,
        key_projector.to(
            device=exact_post_key.device,
            dtype=exact_post_key.dtype,
        ),
    )


def routing_proxy_scores(
    query: Tensor,
    routing_sidecar: Tensor,
    query_projector: Tensor,
    *,
    head_dim: int,
) -> Tensor:
    """Compute low-dimensional GQA routing scores for one decode query.

    Args:
        query: exact post-RoPE Query, ``[Query heads, head dim]``.
        routing_sidecar: projected Keys, ``[KV heads, tokens, rank]``.
        query_projector: factors for each KV head or Query head,
            ``[heads, head dim, rank]``.
        head_dim: original exact-QK head width used for score scaling.
    """

    if query.ndim != 2 or not query.is_floating_point():
        raise ValueError("Query must be floating [Query heads, head dim]")
    if routing_sidecar.ndim != 3 or not routing_sidecar.is_floating_point():
        raise ValueError("routing sidecar must be floating [KV heads, tokens, rank]")
    if query_projector.ndim != 3 or not query_projector.is_floating_point():
        raise ValueError("Query projector must be a floating rank-three tensor")
    query_heads, observed_head_dim = map(int, query.shape)
    kv_heads, tokens, rank = map(int, routing_sidecar.shape)
    if head_dim <= 0 or observed_head_dim != head_dim:
        raise ValueError("Query and declared head dimensions differ")
    if tokens <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("routing sidecar has invalid GQA geometry")
    if tuple(query_projector.shape[1:]) != (head_dim, rank):
        raise ValueError("Query projector width differs from routing sidecar")
    if int(query_projector.shape[0]) not in (kv_heads, query_heads):
        raise ValueError("Query projector must be per-KV-head or per-Query-head")

    heads_per_group = query_heads // kv_heads
    grouped_query = query.reshape(kv_heads, heads_per_group, head_dim)
    projector = query_projector.to(device=query.device, dtype=query.dtype)
    if int(projector.shape[0]) == kv_heads:
        query_code = torch.einsum("hgd,hdr->hgr", grouped_query, projector)
    else:
        query_code = torch.einsum(
            "hgd,hgdr->hgr",
            grouped_query,
            projector.reshape(kv_heads, heads_per_group, head_dim, rank),
        )
    grouped_scores = torch.matmul(
        query_code,
        routing_sidecar.to(
            device=query.device,
            dtype=query.dtype,
        ).transpose(-1, -2),
    )
    return grouped_scores.reshape(query_heads, tokens) / math.sqrt(
        head_dim
    )


def select_routing_pages(
    query: Tensor,
    routing_sidecar: Tensor,
    query_projector: Tensor,
    *,
    head_dim: int,
    page_size: int,
    nominal_token_budget: int,
) -> RoutingSidecarSelection:
    """Select a fixed physical page budget from group-normalized max scores."""

    if page_size <= 0 or nominal_token_budget <= 0:
        raise ValueError("page size and nominal token budget must be positive")
    scores = routing_proxy_scores(
        query,
        routing_sidecar,
        query_projector,
        head_dim=head_dim,
    )
    token_mask, page_mask = gqa_group_max_page_mass_mask(
        scores,
        num_kv_heads=int(routing_sidecar.shape[0]),
        page_size=page_size,
        pages_per_kv_head=math.ceil(nominal_token_budget / page_size),
    )
    return RoutingSidecarSelection(
        proxy_scores=scores,
        token_mask=token_mask,
        page_mask=page_mask,
    )


def routing_storage_ratio(
    *,
    value_rank: int,
    routing_rank: int,
    key_width: int,
    value_width: int,
) -> float:
    """Persistent GPU scalar ratio versus a dense same-precision KV cache."""

    if min(value_rank, routing_rank, key_width, value_width) <= 0:
        raise ValueError("cache widths and ranks must be positive")
    return (value_rank + routing_rank) / (key_width + value_width)


__all__ = [
    "ROUTING_PROXY_IMPLEMENTATION",
    "RoutingDynamicCache",
    "RoutingSidecarSelection",
    "build_routing_sidecar",
    "routing_proxy_scores",
    "routing_storage_ratio",
    "select_routing_pages",
    "truncate_routing_projectors",
]
