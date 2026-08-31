"""Correctness oracle for K-only Reverse ShadowKV with resident C1 Values.

Deployable selectors store compact post-RoPE Key metadata on the accelerator.
Selected physical GQA Key pages are read through :class:`ExactKeyPageStore`, then
exact sparse attention reads only the matching slices of the resident C1 Value
cache. Teacher selectors deliberately consult the full exact Key tensor and are
not serving policies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor

from basisserve.core.c1_k_refine import ExactKeyPageStore, GPUExactKeyPageStore
from basisserve.core.c1_k_routing_sidecar import (
    build_routing_sidecar,
    routing_proxy_scores,
)
from basisserve.core.exact_qk_v_offload import gqa_union_page_mass_mask
from basisserve.core.exact_qk_v_offload import (
    gqa_union_adaptive_page_mass_mask,
)


LandmarkSelector = Literal[
    "teacher_exact",
    "teacher_mass",
    "teacher_output",
    "teacher_influence",
    "mean_landmark",
    "centroid_radius",
    "quest_minmax",
    "quest_k",
    "quest_c1_latent",
    "quest_c1_output",
    "kq_svd",
]

QuestSupport = Literal["physical_shared", "per_query_head"]
QueryHeadAggregation = Literal["max_head", "logsumexp_head"]


_LANDMARK_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class ReverseShadowConfig:
    """Selection and cache geometry for one Reverse ShadowKV decode step."""

    page_size: int
    exact_token_budget: int
    recent_exact_window: int = 0
    landmarks_per_page: int = 1
    selector: LandmarkSelector = "mean_landmark"
    landmark_dtype: str = "bfloat16"
    quest_support: QuestSupport = "physical_shared"
    query_head_aggregation: QueryHeadAggregation = "max_head"
    adaptive_max_token_budget: int | None = None
    adaptive_tail_mass_ratio_threshold: float | None = None

    def validate(self, head_dim: int) -> None:
        if self.page_size <= 0:
            raise ValueError("page size must be positive")
        if self.exact_token_budget < 0:
            raise ValueError("exact-token budget must be nonnegative")
        if self.recent_exact_window < 0:
            raise ValueError("recent exact window must be nonnegative")
        if not 1 <= self.landmarks_per_page <= self.page_size:
            raise ValueError("landmarks per page must be in [1, page size]")
        if self.selector not in (
            "teacher_exact",
            "teacher_mass",
            "teacher_output",
            "teacher_influence",
            "mean_landmark",
            "centroid_radius",
            "quest_minmax",
            "quest_k",
            "quest_c1_latent",
            "quest_c1_output",
            "kq_svd",
        ):
            raise ValueError(f"unsupported selector: {self.selector}")
        if self.landmark_dtype not in _LANDMARK_DTYPES:
            raise ValueError(f"unsupported landmark dtype: {self.landmark_dtype}")
        if self.quest_support not in ("physical_shared", "per_query_head"):
            raise ValueError(f"unsupported QUEST support: {self.quest_support}")
        if self.query_head_aggregation not in ("max_head", "logsumexp_head"):
            raise ValueError(
                "query-head aggregation must be max_head or logsumexp_head"
            )
        adaptive_values = (
            self.adaptive_max_token_budget,
            self.adaptive_tail_mass_ratio_threshold,
        )
        if any(value is not None for value in adaptive_values) and not all(
            value is not None for value in adaptive_values
        ):
            raise ValueError(
                "adaptive maximum budget and tail mass threshold must be set together"
            )
        if self.adaptive_max_token_budget is not None:
            if self.selector != "kq_svd":
                raise ValueError("adaptive budget is only defined for KQ-SVD")
            if self.adaptive_max_token_budget <= self.exact_token_budget:
                raise ValueError("adaptive maximum budget must exceed base budget")
            assert self.adaptive_tail_mass_ratio_threshold is not None
            if not 0.0 < self.adaptive_tail_mass_ratio_threshold <= 1.0:
                raise ValueError("adaptive tail mass threshold must lie in (0, 1]")
        if self.quest_support == "per_query_head" and self.selector != "quest_minmax":
            raise ValueError("per-query-head support is only defined for QUEST")
        if head_dim <= 0:
            raise ValueError("head dimension must be positive")

    @property
    def page_budget(self) -> int:
        if self.exact_token_budget == 0:
            return 0
        return math.ceil(self.exact_token_budget / self.page_size)

    @property
    def adaptive_max_page_budget(self) -> int | None:
        if self.adaptive_max_token_budget is None:
            return None
        return math.ceil(self.adaptive_max_token_budget / self.page_size)

    @property
    def torch_landmark_dtype(self) -> torch.dtype:
        try:
            return _LANDMARK_DTYPES[self.landmark_dtype]
        except KeyError as error:
            raise ValueError(
                f"unsupported landmark dtype: {self.landmark_dtype}"
            ) from error


@dataclass(frozen=True)
class PostRoPEKLandmarks:
    """Mean landmarks and QUEST bounds for each physical post-RoPE Key page."""

    values: Tensor
    radii: Tensor
    valid: Tensor
    page_mins: Tensor
    page_maxes: Tensor
    page_bounds_valid: Tensor
    sequence_length: int
    page_size: int

    def validate(
        self,
        *,
        batch: int,
        kv_heads: int,
        head_dim: int,
        landmarks_per_page: int,
        device: torch.device,
    ) -> None:
        page_count = math.ceil(self.sequence_length / self.page_size)
        expected = (batch, kv_heads, page_count, landmarks_per_page, head_dim)
        if tuple(self.values.shape) != expected:
            raise ValueError(
                f"landmarks must have shape {expected}, got {tuple(self.values.shape)}"
            )
        if tuple(self.valid.shape) != expected[:-1]:
            raise ValueError("landmark validity geometry differs from landmark values")
        if tuple(self.radii.shape) != expected[:-1]:
            raise ValueError("landmark radius geometry differs from landmark values")
        page_expected = (batch, kv_heads, page_count, head_dim)
        if tuple(self.page_mins.shape) != page_expected:
            raise ValueError("QUEST page-min geometry differs from Key pages")
        if tuple(self.page_maxes.shape) != page_expected:
            raise ValueError("QUEST page-max geometry differs from Key pages")
        if tuple(self.page_bounds_valid.shape) != page_expected[:-1]:
            raise ValueError("QUEST page-bound validity geometry differs from Key pages")
        if self.valid.dtype != torch.bool:
            raise TypeError("landmark validity must be boolean")
        if self.page_bounds_valid.dtype != torch.bool:
            raise TypeError("QUEST page-bound validity must be boolean")
        if not all(
            tensor.is_floating_point()
            for tensor in (
                self.values,
                self.radii,
                self.page_mins,
                self.page_maxes,
            )
        ):
            raise TypeError("Key selector metadata must be floating point")
        if torch.any(self.radii < 0):
            raise ValueError("landmark radii must be nonnegative")
        if torch.any(
            (self.page_mins > self.page_maxes)
            & self.page_bounds_valid[..., None]
        ):
            raise ValueError("QUEST page minima must not exceed page maxima")
        metadata = (
            self.values,
            self.radii,
            self.valid,
            self.page_mins,
            self.page_maxes,
            self.page_bounds_valid,
        )
        if any(tensor.device != device for tensor in metadata):
            raise ValueError("query and landmarks must share a device")


@dataclass(frozen=True)
class C1PageMetadata:
    """C1-Value page norms used by output-aware physical-page selectors."""

    latent_max_norm: Tensor
    decoded_output_max_norm: Tensor
    valid_token_count: Tensor
    decoder_gram: Tensor
    sequence_length: int
    page_size: int

    def validate(
        self,
        *,
        batch: int,
        kv_heads: int,
        query_heads: int,
        value_rank: int,
        device: torch.device,
    ) -> None:
        page_count = math.ceil(self.sequence_length / self.page_size)
        if tuple(self.latent_max_norm.shape) != (batch, kv_heads, page_count):
            raise ValueError("C1 latent page-norm geometry is incompatible")
        if tuple(self.decoded_output_max_norm.shape) != (
            batch,
            query_heads,
            page_count,
        ):
            raise ValueError("C1 decoded-output page-norm geometry is incompatible")
        if tuple(self.valid_token_count.shape) != (batch, kv_heads, page_count):
            raise ValueError("C1 page token-count geometry is incompatible")
        if tuple(self.decoder_gram.shape) != (
            query_heads,
            value_rank,
            value_rank,
        ):
            raise ValueError("C1 decoder-Gram geometry is incompatible")
        if not all(
            tensor.is_floating_point()
            for tensor in (
                self.latent_max_norm,
                self.decoded_output_max_norm,
                self.decoder_gram,
            )
        ):
            raise TypeError("C1 page norms and decoder Gram must be floating point")
        if self.valid_token_count.dtype not in (torch.int32, torch.int64):
            raise TypeError("C1 page token counts must be integral")
        if torch.any(self.latent_max_norm < 0) or torch.any(
            self.decoded_output_max_norm < 0
        ):
            raise ValueError("C1 page norms must be nonnegative")
        if torch.any(self.valid_token_count < 0):
            raise ValueError("C1 page token counts must be nonnegative")
        metadata = (
            self.latent_max_norm,
            self.decoded_output_max_norm,
            self.valid_token_count,
            self.decoder_gram,
        )
        if any(tensor.device != device for tensor in metadata):
            raise ValueError("query and C1 page metadata must share a device")


@dataclass
class ReverseShadowResult:
    output: Tensor
    selected_page_ids: Tensor
    selected_page_mask: Tensor
    selected_token_count: int
    page_scores: Tensor
    running_lse: Tensor
    statistics: dict[str, float]


def _block_query_mask(
    attention_mask: Tensor | None,
    *,
    query_index: int,
    query_length: int,
    visible_sequence: int,
) -> Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 2:
        return attention_mask[..., :visible_sequence]
    if attention_mask.ndim == 4:
        mask_queries = int(attention_mask.shape[-2])
        if mask_queries not in (1, query_length):
            raise ValueError(
                "block attention mask query axis must be singleton or match query"
            )
        index = 0 if mask_queries == 1 else query_index
        return attention_mask[..., index : index + 1, :visible_sequence]
    raise ValueError("block Reverse ShadowKV mask must be rank 2 or rank 4")


def c1_k_reverse_shadow_block_attention(
    query: Tensor,
    exact_key: Tensor,
    c1_value: Tensor,
    config: ReverseShadowConfig,
    attention_mask: Tensor | None = None,
    *,
    routing_key_projector: Tensor | None = None,
    routing_query_projector: Tensor | None = None,
    routing_sidecar: Tensor | None = None,
    layer_idx: int = 0,
) -> tuple[Tensor, tuple[ReverseShadowResult, ...]]:
    """Apply query-wise causal Reverse ShadowKV to one target block.

    The cache contains a committed prefix followed by every Key/Value in the
    pending target block. Query ``j`` may see exactly ``prefix + j + 1`` cache
    positions. Building its metadata from that visible prefix prevents future
    tokens in the partially filled final page from influencing QUEST selection.
    """

    if query.ndim != 4 or exact_key.ndim != 4 or c1_value.ndim != 4:
        raise ValueError("block Q/K/V must have rank four")
    batch, _, query_length, _ = map(int, query.shape)
    if batch != int(exact_key.shape[0]) or batch != int(c1_value.shape[0]):
        raise ValueError("block Q/K/V batch axes differ")
    if tuple(exact_key.shape[:3]) != tuple(c1_value.shape[:3]):
        raise ValueError("block exact K and C1 V cache geometry differs")
    sequence = int(exact_key.shape[2])
    prefix_length = sequence - query_length
    if query_length <= 0 or prefix_length < 0:
        raise ValueError("block cache must contain the complete query block")
    if config.selector == "kq_svd":
        if routing_query_projector is None:
            raise ValueError("KQ-SVD routing requires a Query projector")
        if routing_sidecar is None:
            if routing_key_projector is None:
                raise ValueError(
                    "KQ-SVD routing requires a cached sidecar or Key projector"
                )
            full_routing_sidecar = build_routing_sidecar(
                exact_key, routing_key_projector
            )
        else:
            if (
                routing_sidecar.ndim != 4
                or tuple(routing_sidecar.shape[:3])
                != tuple(exact_key.shape[:3])
            ):
                raise ValueError("cached routing sidecar and exact K differ")
            full_routing_sidecar = routing_sidecar
    else:
        if (
            routing_key_projector is not None
            or routing_query_projector is not None
            or routing_sidecar is not None
        ):
            raise ValueError("routing projectors require selector='kq_svd'")
        full_routing_sidecar = None

    outputs = []
    results = []
    for query_index in range(query_length):
        visible_sequence = prefix_length + query_index + 1
        visible_key = exact_key[:, :, :visible_sequence]
        visible_value = c1_value[:, :, :visible_sequence]
        visible_mask = _block_query_mask(
            attention_mask,
            query_index=query_index,
            query_length=query_length,
            visible_sequence=visible_sequence,
        )
        landmarks = (
            build_quest_minmax_landmarks(
                visible_key,
                page_size=config.page_size,
                landmarks_per_page=config.landmarks_per_page,
                attention_mask=visible_mask,
                landmark_dtype=config.landmark_dtype,
            )
            if config.selector == "quest_minmax"
            else build_routing_page_geometry(
                visible_key,
                page_size=config.page_size,
                attention_mask=visible_mask,
                landmark_dtype=config.landmark_dtype,
            )
            if config.selector == "kq_svd"
            else build_post_rope_k_landmarks(
                visible_key,
                page_size=config.page_size,
                landmarks_per_page=config.landmarks_per_page,
                attention_mask=visible_mask,
                landmark_dtype=config.landmark_dtype,
            )
        )
        result = c1_k_reverse_shadow_attention(
            query[:, :, query_index : query_index + 1],
            landmarks,
            visible_value,
            config,
            visible_key,
            visible_mask,
            layer_idx=layer_idx,
            routing_sidecar=(
                None
                if full_routing_sidecar is None
                else full_routing_sidecar[:, :, :visible_sequence]
            ),
            routing_query_projector=routing_query_projector,
            vectorized_reference=True,
        )
        outputs.append(result.output)
        results.append(result)
    return torch.cat(outputs, dim=2), tuple(results)


def _physical_attention_mask(
    mask: Tensor | None,
    *,
    batch: int,
    sequence: int,
    device: torch.device,
) -> Tensor:
    """Normalize a cache-construction mask and reject head-dependent validity."""

    if mask is None:
        return torch.ones(batch, sequence, dtype=torch.bool, device=device)
    candidate = mask.to(device=device)
    if candidate.ndim == 4 and int(candidate.shape[-2]) == 1:
        candidate = candidate.squeeze(-2)
    if candidate.ndim == 2:
        candidate = candidate[:, None, :]
    if candidate.ndim != 3 or int(candidate.shape[0]) not in (1, batch):
        raise ValueError("landmark mask must broadcast to [batch, heads, sequence]")
    if int(candidate.shape[-1]) != sequence:
        raise ValueError("landmark mask sequence length differs from exact Key")
    candidate = candidate.expand(batch, candidate.shape[1], sequence)
    if candidate.dtype == torch.bool:
        valid = candidate
    else:
        bias = candidate.float()
        valid = torch.isfinite(bias) & (bias > -1.0e20)
    if int(valid.shape[1]) > 1 and not torch.equal(
        valid, valid[:, :1].expand_as(valid)
    ):
        raise ValueError("physical Key landmarks require head-independent validity")
    return valid[:, 0]


def build_quest_minmax_landmarks(
    exact_key: Tensor,
    *,
    page_size: int,
    landmarks_per_page: int = 1,
    attention_mask: Tensor | None = None,
    landmark_dtype: str = "bfloat16",
) -> PostRoPEKLandmarks:
    """Vectorized QUEST-only metadata builder for correctness replay."""

    if exact_key.ndim != 4 or not exact_key.is_floating_point():
        raise ValueError("exact Key must be a floating [batch, heads, tokens, dim]")
    batch, kv_heads, sequence, head_dim = map(int, exact_key.shape)
    config = ReverseShadowConfig(
        page_size=page_size,
        exact_token_budget=0,
        landmarks_per_page=landmarks_per_page,
        selector="quest_minmax",
        landmark_dtype=landmark_dtype,
    )
    config.validate(head_dim)
    if sequence <= 0:
        raise ValueError("exact Key sequence must be nonempty")
    valid_tokens = _physical_attention_mask(
        attention_mask,
        batch=batch,
        sequence=sequence,
        device=exact_key.device,
    )
    page_count = math.ceil(sequence / page_size)
    padded_sequence = page_count * page_size
    padding = padded_sequence - sequence
    padded_key = exact_key.float()
    padded_valid = valid_tokens
    if padding:
        padded_key = torch.cat(
            (
                padded_key,
                torch.zeros(
                    batch,
                    kv_heads,
                    padding,
                    head_dim,
                    dtype=padded_key.dtype,
                    device=padded_key.device,
                ),
            ),
            dim=2,
        )
        padded_valid = torch.cat(
            (
                padded_valid,
                torch.zeros(
                    batch,
                    padding,
                    dtype=torch.bool,
                    device=padded_valid.device,
                ),
            ),
            dim=1,
        )
    key_pages = padded_key.reshape(
        batch, kv_heads, page_count, page_size, head_dim
    )
    valid_pages = padded_valid.reshape(batch, page_count, page_size)
    page_has_tokens = valid_pages.any(dim=-1)
    expanded_valid = valid_pages[:, None, :, :, None]
    page_mins = key_pages.masked_fill(~expanded_valid, torch.inf).amin(dim=3)
    page_maxes = key_pages.masked_fill(~expanded_valid, -torch.inf).amax(dim=3)
    page_valid = page_has_tokens[:, None, :].expand(batch, kv_heads, page_count)
    page_mins = torch.where(
        page_valid[..., None], page_mins, torch.zeros_like(page_mins)
    )
    page_maxes = torch.where(
        page_valid[..., None], page_maxes, torch.zeros_like(page_maxes)
    )
    dummy_shape = (
        batch,
        kv_heads,
        page_count,
        landmarks_per_page,
        head_dim,
    )
    return PostRoPEKLandmarks(
        values=torch.zeros(
            dummy_shape,
            dtype=config.torch_landmark_dtype,
            device=exact_key.device,
        ),
        radii=torch.zeros(dummy_shape[:-1], dtype=torch.float32, device=exact_key.device),
        valid=page_valid[..., None].expand(dummy_shape[:-1]),
        page_mins=page_mins.to(config.torch_landmark_dtype),
        page_maxes=page_maxes.to(config.torch_landmark_dtype),
        page_bounds_valid=page_valid,
        sequence_length=sequence,
        page_size=page_size,
    )


def build_routing_page_geometry(
    exact_key: Tensor,
    *,
    page_size: int,
    attention_mask: Tensor | None = None,
    landmark_dtype: str = "bfloat16",
) -> PostRoPEKLandmarks:
    """Build only page validity/geometry for a routing-sidecar selector.

    KQ-SVD selection consumes the token-level low-rank sidecar, so materializing
    mean landmarks or QUEST extrema would add unrelated storage and work.
    """

    if exact_key.ndim != 4 or not exact_key.is_floating_point():
        raise ValueError("exact Key must be a floating [batch, heads, tokens, dim]")
    batch, kv_heads, sequence, head_dim = map(int, exact_key.shape)
    config = ReverseShadowConfig(
        page_size=page_size,
        exact_token_budget=0,
        selector="kq_svd",
        landmark_dtype=landmark_dtype,
    )
    config.validate(head_dim)
    if sequence <= 0:
        raise ValueError("exact Key sequence must be nonempty")
    valid_tokens = _physical_attention_mask(
        attention_mask,
        batch=batch,
        sequence=sequence,
        device=exact_key.device,
    )
    page_count = math.ceil(sequence / page_size)
    padding = page_count * page_size - sequence
    if padding:
        valid_tokens = torch.cat(
            (
                valid_tokens,
                torch.zeros(
                    batch,
                    padding,
                    dtype=torch.bool,
                    device=exact_key.device,
                ),
            ),
            dim=-1,
        )
    page_valid = valid_tokens.reshape(batch, page_count, page_size).any(dim=-1)
    page_valid = page_valid[:, None].expand(batch, kv_heads, page_count)
    landmark_shape = (batch, kv_heads, page_count, 1, head_dim)
    page_shape = (batch, kv_heads, page_count, head_dim)
    return PostRoPEKLandmarks(
        values=torch.zeros(
            landmark_shape,
            dtype=config.torch_landmark_dtype,
            device=exact_key.device,
        ),
        radii=torch.zeros(
            landmark_shape[:-1], dtype=torch.float32, device=exact_key.device
        ),
        valid=page_valid[..., None],
        page_mins=torch.zeros(
            page_shape,
            dtype=config.torch_landmark_dtype,
            device=exact_key.device,
        ),
        page_maxes=torch.zeros(
            page_shape,
            dtype=config.torch_landmark_dtype,
            device=exact_key.device,
        ),
        page_bounds_valid=page_valid,
        sequence_length=sequence,
        page_size=page_size,
    )


def build_post_rope_k_landmarks(
    exact_key: Tensor,
    *,
    page_size: int,
    landmarks_per_page: int = 1,
    attention_mask: Tensor | None = None,
    landmark_dtype: str = "bfloat16",
) -> PostRoPEKLandmarks:
    """Build mean/radius landmarks and QUEST Min/Max page metadata.

    With one landmark, each page contributes its mean.  With ``M > 1``, a page
    is split into ``M`` fixed contiguous regions before taking means.  Empty
    regions in a short final page are marked invalid and stored as zeros. QUEST
    bounds always summarize the entire physical page, independent of ``M``.
    """

    if exact_key.ndim != 4:
        raise ValueError("exact Key must be [batch, KV heads, sequence, head dim]")
    if not exact_key.is_floating_point():
        raise TypeError("exact Key must be floating point")
    batch, kv_heads, sequence, head_dim = map(int, exact_key.shape)
    config = ReverseShadowConfig(
        page_size=page_size,
        exact_token_budget=0,
        landmarks_per_page=landmarks_per_page,
        landmark_dtype=landmark_dtype,
    )
    config.validate(head_dim)
    if sequence <= 0:
        raise ValueError("exact Key sequence must be nonempty")
    valid_tokens = _physical_attention_mask(
        attention_mask,
        batch=batch,
        sequence=sequence,
        device=exact_key.device,
    )
    page_count = math.ceil(sequence / page_size)
    values = torch.zeros(
        batch,
        kv_heads,
        page_count,
        landmarks_per_page,
        head_dim,
        dtype=torch.float32,
        device=exact_key.device,
    )
    landmark_valid = torch.zeros(
        batch,
        kv_heads,
        page_count,
        landmarks_per_page,
        dtype=torch.bool,
        device=exact_key.device,
    )
    page_mins = torch.zeros(
        batch,
        kv_heads,
        page_count,
        head_dim,
        dtype=torch.float32,
        device=exact_key.device,
    )
    page_maxes = torch.zeros_like(page_mins)
    page_bounds_valid = torch.zeros(
        batch,
        kv_heads,
        page_count,
        dtype=torch.bool,
        device=exact_key.device,
    )
    for page in range(page_count):
        page_start = page * page_size
        page_stop = min(page_start + page_size, sequence)
        page_region_valid = valid_tokens[:, page_start:page_stop]
        page_has_tokens = page_region_valid.any(dim=-1)
        page_keys = exact_key[:, :, page_start:page_stop].float()
        minimum = page_keys.masked_fill(
            ~page_region_valid[:, None, :, None], torch.inf
        ).amin(dim=2)
        maximum = page_keys.masked_fill(
            ~page_region_valid[:, None, :, None], -torch.inf
        ).amax(dim=2)
        page_mins[:, :, page] = torch.where(
            page_has_tokens[:, None, None], minimum, torch.zeros_like(minimum)
        )
        page_maxes[:, :, page] = torch.where(
            page_has_tokens[:, None, None], maximum, torch.zeros_like(maximum)
        )
        page_bounds_valid[:, :, page] = page_has_tokens[:, None]
        for landmark in range(landmarks_per_page):
            start = page_start + (landmark * page_size) // landmarks_per_page
            stop = page_start + ((landmark + 1) * page_size) // landmarks_per_page
            stop = min(stop, sequence)
            if start >= stop:
                continue
            region_valid = valid_tokens[:, start:stop]
            counts = region_valid.sum(dim=-1)
            safe_counts = counts.clamp_min(1).to(torch.float32)
            weighted = exact_key[:, :, start:stop].float() * region_valid[
                :, None, :, None
            ]
            values[:, :, page, landmark] = weighted.sum(dim=2) / safe_counts[
                :, None, None
            ]
            landmark_valid[:, :, page, landmark] = (counts > 0)[:, None]
    stored_values = values.to(config.torch_landmark_dtype)
    # Compute radii around the values that will actually be stored.  Keeping the
    # scalar radius in FP32 avoids invalidating the Cauchy upper bound through a
    # downward BF16/FP16 radius rounding.
    radii = torch.zeros(
        batch,
        kv_heads,
        page_count,
        landmarks_per_page,
        dtype=torch.float32,
        device=exact_key.device,
    )
    for page in range(page_count):
        page_start = page * page_size
        for landmark in range(landmarks_per_page):
            start = page_start + (landmark * page_size) // landmarks_per_page
            stop = page_start + ((landmark + 1) * page_size) // landmarks_per_page
            stop = min(stop, sequence)
            if start >= stop:
                continue
            region_valid = valid_tokens[:, start:stop]
            residual = exact_key[:, :, start:stop].float() - stored_values[
                :, :, page, landmark, None
            ].float()
            residual_norm = torch.linalg.vector_norm(residual, dim=-1).masked_fill(
                ~region_valid[:, None], -torch.inf
            )
            maximum = residual_norm.amax(dim=-1)
            radii[:, :, page, landmark] = torch.where(
                landmark_valid[:, :, page, landmark],
                maximum,
                torch.zeros_like(maximum),
            )
    return PostRoPEKLandmarks(
        values=stored_values,
        radii=radii,
        valid=landmark_valid,
        page_mins=page_mins.to(config.torch_landmark_dtype),
        page_maxes=page_maxes.to(config.torch_landmark_dtype),
        page_bounds_valid=page_bounds_valid,
        sequence_length=sequence,
        page_size=page_size,
    )


def build_c1_decoder_gram(decoder: Tensor) -> Tensor:
    """Return per-query-head ``D_h D_h^T`` matrices in FP32."""

    if decoder.ndim == 4:
        decoder = decoder.reshape(-1, decoder.shape[-2], decoder.shape[-1])
    if decoder.ndim != 3 or not decoder.is_floating_point():
        raise ValueError("C1 decoder must be [query heads, value rank, output dim]")
    return torch.matmul(decoder.float(), decoder.float().transpose(-1, -2))


def build_c1_page_metadata(
    c1_value: Tensor,
    decoder: Tensor,
    *,
    page_size: int,
    attention_mask: Tensor | None = None,
    decoder_gram: Tensor | None = None,
) -> C1PageMetadata:
    """Build latent and decoded-output maximum norms for every C1 page.

    The decoded norm is evaluated as ``sqrt(C @ (D D^T) @ C^T)``.  Hidden-size
    decoded vectors are never materialized.
    """

    if c1_value.ndim != 4 or not c1_value.is_floating_point():
        raise ValueError(
            "C1 Value must be a floating [batch, KV heads, sequence, value rank]"
        )
    if page_size <= 0:
        raise ValueError("page size must be positive")
    batch, kv_heads, sequence, value_rank = map(int, c1_value.shape)
    if sequence <= 0 or value_rank <= 0:
        raise ValueError("C1 Value sequence and rank must be positive")
    normalized_decoder = decoder
    if normalized_decoder.ndim == 4:
        normalized_decoder = normalized_decoder.reshape(
            -1, normalized_decoder.shape[-2], normalized_decoder.shape[-1]
        )
    if normalized_decoder.ndim != 3 or int(normalized_decoder.shape[1]) != value_rank:
        raise ValueError("C1 decoder rank is incompatible with C1 Value")
    query_heads = int(normalized_decoder.shape[0])
    head_to_kv = _head_to_kv(query_heads, kv_heads, c1_value.device)
    if normalized_decoder.device != c1_value.device:
        raise ValueError("C1 Value and decoder must share a device")
    gram = (
        build_c1_decoder_gram(normalized_decoder)
        if decoder_gram is None
        else decoder_gram.float()
    )
    expected_gram = (query_heads, value_rank, value_rank)
    if tuple(gram.shape) != expected_gram or gram.device != c1_value.device:
        raise ValueError(
            f"decoder Gram must have shape {expected_gram} on the C1 Value device"
        )

    valid_tokens = _physical_attention_mask(
        attention_mask,
        batch=batch,
        sequence=sequence,
        device=c1_value.device,
    )
    page_count = math.ceil(sequence / page_size)
    latent_max = torch.zeros(
        batch, kv_heads, page_count, dtype=torch.float32, device=c1_value.device
    )
    output_max = torch.zeros(
        batch, query_heads, page_count, dtype=torch.float32, device=c1_value.device
    )
    valid_count = torch.zeros(
        batch, kv_heads, page_count, dtype=torch.int32, device=c1_value.device
    )
    values = c1_value.float()
    for page in range(page_count):
        start = page * page_size
        stop = min(start + page_size, sequence)
        page_valid = valid_tokens[:, start:stop]
        counts = page_valid.sum(dim=-1, dtype=torch.int32)
        valid_count[:, :, page] = counts[:, None]

        page_value = values[:, :, start:stop]
        latent_norm = torch.linalg.vector_norm(page_value, dim=-1).masked_fill(
            ~page_valid[:, None], -torch.inf
        )
        latent_page_max = latent_norm.amax(dim=-1)
        latent_max[:, :, page] = torch.where(
            counts[:, None] > 0,
            latent_page_max,
            torch.zeros_like(latent_page_max),
        )

        expanded_value = page_value.index_select(1, head_to_kv)
        decoded_norm_squared = torch.einsum(
            "bhsr,hrt,bhst->bhs",
            expanded_value,
            gram,
            expanded_value,
        ).clamp_min_(0)
        decoded_norm = decoded_norm_squared.sqrt_().masked_fill(
            ~page_valid[:, None], -torch.inf
        )
        decoded_page_max = decoded_norm.amax(dim=-1)
        output_max[:, :, page] = torch.where(
            counts[:, None] > 0,
            decoded_page_max,
            torch.zeros_like(decoded_page_max),
        )

    return C1PageMetadata(
        latent_max_norm=latent_max,
        decoded_output_max_norm=output_max,
        valid_token_count=valid_count,
        decoder_gram=gram,
        sequence_length=sequence,
        page_size=page_size,
    )


def _attention_mask(
    mask: Tensor | None,
    *,
    batch: int,
    query_heads: int,
    sequence: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if mask is None:
        valid = torch.ones(batch, query_heads, sequence, dtype=torch.bool, device=device)
        return valid, torch.zeros(
            batch, query_heads, sequence, dtype=torch.float32, device=device
        )
    candidate = mask.to(device=device)
    if candidate.ndim == 2:
        candidate = candidate[:, None, :]
    elif candidate.ndim == 4 and int(candidate.shape[-2]) == 1:
        candidate = candidate.squeeze(-2)
    if candidate.ndim != 3 or int(candidate.shape[0]) not in (1, batch):
        raise ValueError("attention mask must broadcast to [batch, query heads, sequence]")
    if int(candidate.shape[1]) not in (1, query_heads) or int(candidate.shape[2]) != sequence:
        raise ValueError("attention mask must broadcast to [batch, query heads, sequence]")
    candidate = candidate.expand(batch, query_heads, sequence)
    if candidate.dtype == torch.bool:
        valid = candidate
        bias = torch.zeros(candidate.shape, dtype=torch.float32, device=device)
    else:
        bias = candidate.float()
        valid = torch.isfinite(bias) & (bias > -1.0e20)
        bias = torch.where(valid, bias, torch.zeros_like(bias))
    return valid, bias


def _head_to_kv(query_heads: int, kv_heads: int, device: torch.device) -> Tensor:
    if kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must divide evenly across physical KV heads")
    return torch.arange(kv_heads, device=device).repeat_interleave(
        query_heads // kv_heads
    )


def _page_validity(
    valid: Tensor,
    *,
    kv_heads: int,
    page_size: int,
) -> tuple[Tensor, Tensor]:
    batch, query_heads, sequence = map(int, valid.shape)
    heads_per_group = query_heads // kv_heads
    physical_valid = valid.reshape(
        batch, kv_heads, heads_per_group, sequence
    ).any(dim=2)
    page_count = math.ceil(sequence / page_size)
    padding = page_count * page_size - sequence
    padded = physical_valid
    if padding:
        padded = torch.cat(
            (
                padded,
                torch.zeros(
                    batch,
                    kv_heads,
                    padding,
                    dtype=torch.bool,
                    device=valid.device,
                ),
            ),
            dim=2,
        )
    page_valid = padded.reshape(
        batch, kv_heads, page_count, page_size
    ).any(dim=-1)
    return physical_valid, page_valid


def _page_maximum_bias(valid: Tensor, bias: Tensor, page_size: int) -> Tensor:
    batch, query_heads, sequence = map(int, valid.shape)
    page_count = math.ceil(sequence / page_size)
    padding = page_count * page_size - sequence
    masked = bias.masked_fill(~valid, -torch.inf)
    if padding:
        masked = torch.cat(
            (
                masked,
                torch.full(
                    (batch, query_heads, padding),
                    -torch.inf,
                    dtype=masked.dtype,
                    device=masked.device,
                ),
            ),
            dim=2,
        )
    return masked.reshape(batch, query_heads, page_count, page_size).amax(dim=-1)


def _landmark_page_scores(
    query: Tensor,
    landmarks: PostRoPEKLandmarks,
    valid: Tensor,
    bias: Tensor,
    *,
    use_radius: bool,
) -> Tensor:
    batch, query_heads, _, head_dim = map(int, query.shape)
    kv_heads = int(landmarks.values.shape[1])
    heads_per_group = query_heads // kv_heads
    head_to_kv = _head_to_kv(query_heads, kv_heads, query.device)
    expanded = landmarks.values.index_select(1, head_to_kv).float()
    scores = torch.einsum("bhd,bhpmd->bhpm", query[:, :, 0].float(), expanded)
    if use_radius:
        expanded_radii = landmarks.radii.index_select(1, head_to_kv)
        query_norm = torch.linalg.vector_norm(query[:, :, 0].float(), dim=-1)
        scores.add_(query_norm[:, :, None, None] * expanded_radii.float())
    scores.mul_(head_dim**-0.5)
    expanded_landmark_valid = landmarks.valid.index_select(1, head_to_kv)
    page_count = int(scores.shape[2])
    scores.add_(_page_maximum_bias(valid, bias, landmarks.page_size).unsqueeze(-1))
    scores.masked_fill_(~expanded_landmark_valid, -torch.inf)
    return scores.reshape(
        batch, kv_heads, heads_per_group, page_count, -1
    ).amax(dim=(2, 4))


def _teacher_page_scores(
    query: Tensor,
    exact_key: Tensor,
    valid: Tensor,
    bias: Tensor,
    *,
    page_size: int,
) -> Tensor:
    batch, query_heads, _, head_dim = map(int, query.shape)
    kv_heads = int(exact_key.shape[1])
    heads_per_group = query_heads // kv_heads
    head_to_kv = _head_to_kv(query_heads, kv_heads, query.device)
    page_count = math.ceil(int(exact_key.shape[2]) / page_size)
    result = torch.full(
        (batch, kv_heads, page_count),
        -torch.inf,
        dtype=torch.float32,
        device=query.device,
    )
    for page in range(page_count):
        start = page * page_size
        stop = min(start + page_size, int(exact_key.shape[2]))
        expanded_key = exact_key[:, :, start:stop].index_select(1, head_to_kv)
        scores = torch.einsum(
            "bhd,bhsd->bhs", query[:, :, 0].float(), expanded_key.float()
        )
        scores.mul_(head_dim**-0.5).add_(bias[:, :, start:stop])
        scores.masked_fill_(~valid[:, :, start:stop], -torch.inf)
        result[:, :, page] = scores.amax(dim=-1).reshape(
            batch, kv_heads, heads_per_group
        ).amax(dim=2)
    return result


def _quest_query_head_page_scores(
    query: Tensor,
    landmarks: PostRoPEKLandmarks,
    valid: Tensor,
    bias: Tensor,
) -> Tensor:
    """Return one QUEST bounding-box upper score per query head and page."""

    batch, query_heads, _, head_dim = map(int, query.shape)
    kv_heads = int(landmarks.page_mins.shape[1])
    head_to_kv = _head_to_kv(query_heads, kv_heads, query.device)
    page_mins = landmarks.page_mins.index_select(1, head_to_kv).float()
    page_maxes = landmarks.page_maxes.index_select(1, head_to_kv).float()
    current_query = query[:, :, 0].float()[:, :, None]
    scores = torch.maximum(
        current_query * page_mins,
        current_query * page_maxes,
    ).sum(dim=-1)
    scores.mul_(head_dim**-0.5)
    scores.add_(_page_maximum_bias(valid, bias, landmarks.page_size))
    bounds_valid = landmarks.page_bounds_valid.index_select(1, head_to_kv)
    scores.masked_fill_(~bounds_valid, -torch.inf)
    return scores


def _aggregate_query_head_page_scores(
    scores: Tensor,
    *,
    kv_heads: int,
    aggregation: QueryHeadAggregation,
) -> Tensor:
    batch, query_heads, page_count = map(int, scores.shape)
    heads_per_group = query_heads // kv_heads
    grouped = scores.reshape(batch, kv_heads, heads_per_group, page_count)
    if aggregation == "max_head":
        return grouped.amax(dim=2)
    if aggregation == "logsumexp_head":
        return torch.logsumexp(grouped, dim=2)
    raise ValueError(f"unsupported query-head aggregation: {aggregation}")


def _quest_page_scores(
    query: Tensor,
    landmarks: PostRoPEKLandmarks,
    valid: Tensor,
    bias: Tensor,
    *,
    support: QuestSupport,
    aggregation: QueryHeadAggregation = "max_head",
) -> Tensor:
    """Return QUEST scores at the requested GQA ownership granularity."""

    scores = _quest_query_head_page_scores(query, landmarks, valid, bias)
    if support == "per_query_head":
        return scores
    return _aggregate_query_head_page_scores(
        scores,
        kv_heads=int(landmarks.page_mins.shape[1]),
        aggregation=aggregation,
    )


def _c1_aware_quest_page_scores(
    query: Tensor,
    landmarks: PostRoPEKLandmarks,
    c1_metadata: C1PageMetadata | None,
    valid: Tensor,
    bias: Tensor,
    *,
    selector: LandmarkSelector,
    aggregation: QueryHeadAggregation,
) -> Tensor:
    """Return physical-page QUEST, latent-aware, or decoded-output priorities."""

    query_scores = _quest_query_head_page_scores(query, landmarks, valid, bias)
    query_heads = int(query_scores.shape[1])
    kv_heads = int(landmarks.page_mins.shape[1])
    if selector == "quest_k":
        return _aggregate_query_head_page_scores(
            query_scores,
            kv_heads=kv_heads,
            aggregation=aggregation,
        )
    if c1_metadata is None:
        raise ValueError(f"{selector} requires C1 page metadata")
    head_to_kv = _head_to_kv(query_heads, kv_heads, query.device)
    counts = c1_metadata.valid_token_count.index_select(1, head_to_kv).float()
    log_count = counts.clamp_min(1).log()
    if selector == "quest_c1_latent":
        norms = c1_metadata.latent_max_norm.index_select(1, head_to_kv)
    elif selector == "quest_c1_output":
        norms = c1_metadata.decoded_output_max_norm
    else:
        raise ValueError(f"unsupported C1-aware QUEST selector: {selector}")
    weighted = query_scores + norms.float().clamp_min(1.0e-30).log() + log_count
    weighted.masked_fill_(counts == 0, -torch.inf)
    return _aggregate_query_head_page_scores(
        weighted,
        kv_heads=kv_heads,
        aggregation=aggregation,
    )


def _teacher_mass_page_scores(
    query: Tensor,
    exact_key: Tensor,
    valid: Tensor,
    bias: Tensor,
    *,
    page_size: int,
) -> Tensor:
    """Return exact page probability mass summed over each GQA head group."""

    batch, query_heads, _, head_dim = map(int, query.shape)
    kv_heads, sequence = map(int, exact_key.shape[1:3])
    heads_per_group = query_heads // kv_heads
    head_to_kv = _head_to_kv(query_heads, kv_heads, query.device)
    expanded_key = exact_key.index_select(1, head_to_kv).float()
    scores = torch.einsum(
        "bhd,bhsd->bhs", query[:, :, 0].float(), expanded_key
    )
    scores.mul_(head_dim**-0.5).add_(bias)
    scores.masked_fill_(~valid, -torch.inf)
    probability = torch.softmax(scores, dim=-1, dtype=torch.float32)
    page_count = math.ceil(sequence / page_size)
    result = torch.zeros(
        batch,
        kv_heads,
        page_count,
        dtype=torch.float32,
        device=query.device,
    )
    for page in range(page_count):
        start = page * page_size
        stop = min(start + page_size, sequence)
        result[:, :, page] = probability[:, :, start:stop].sum(dim=-1).reshape(
            batch, kv_heads, heads_per_group
        ).sum(dim=2)
    return result


def _teacher_decoded_page_scores(
    query: Tensor,
    exact_key: Tensor,
    c1_value: Tensor,
    decoder: Tensor,
    valid: Tensor,
    bias: Tensor,
    *,
    page_size: int,
    normalization_aware: bool,
) -> Tensor:
    """Return exact decoded page contribution or leave-one-page-out influence.

    This evaluation-only oracle materializes decoded hidden-size page vectors.
    For the normalization-aware mode, each head uses its exact output change
    after removing the page and renormalizing the remaining attention.  Query
    heads sharing one physical KV group are summed before taking the norm,
    matching the folded C1 layer-output geometry.
    """

    batch, query_heads, _, head_dim = map(int, query.shape)
    kv_heads, sequence = map(int, exact_key.shape[1:3])
    value_rank = int(c1_value.shape[-1])
    heads_per_group = query_heads // kv_heads
    head_to_kv = _head_to_kv(query_heads, kv_heads, query.device)
    normalized_decoder = decoder
    if normalized_decoder.ndim == 4:
        normalized_decoder = normalized_decoder.reshape(
            query_heads,
            normalized_decoder.shape[-2],
            normalized_decoder.shape[-1],
        )
    if (
        normalized_decoder.ndim != 3
        or tuple(normalized_decoder.shape[:2]) != (query_heads, value_rank)
        or normalized_decoder.device != query.device
    ):
        raise ValueError(
            "teacher decoded selector requires decoder "
            "[query heads, value rank, output dim] on the query device"
        )

    expanded_key = exact_key.index_select(1, head_to_kv).float()
    exact_scores = torch.einsum(
        "bhd,bhsd->bhs", query[:, :, 0].float(), expanded_key
    )
    exact_scores.mul_(head_dim**-0.5).add_(bias)
    exact_scores.masked_fill_(~valid, -torch.inf)
    probability = torch.softmax(exact_scores, dim=-1, dtype=torch.float32)
    grouped_probability = probability.reshape(
        batch, kv_heads, heads_per_group, sequence
    )
    full_latent = torch.einsum(
        "bghs,bgsv->bghv", grouped_probability, c1_value.float()
    )
    grouped_decoder = normalized_decoder.float().reshape(
        kv_heads,
        heads_per_group,
        value_rank,
        normalized_decoder.shape[-1],
    )
    page_count = math.ceil(sequence / page_size)
    result = torch.zeros(
        batch,
        kv_heads,
        page_count,
        dtype=torch.float32,
        device=query.device,
    )
    for page in range(page_count):
        start = page * page_size
        stop = min(start + page_size, sequence)
        page_probability = grouped_probability[..., start:stop]
        page_mass = page_probability.sum(dim=-1)
        page_latent = torch.einsum(
            "bghs,bgsv->bghv",
            page_probability,
            c1_value[:, :, start:stop].float(),
        )
        signal = page_latent
        if normalization_aware:
            remaining_mass = (1.0 - page_mass).clamp_min(1.0e-12)
            signal = (
                page_latent - page_mass[..., None] * full_latent
            ) / remaining_mass[..., None]
        decoded = torch.einsum(
            "bghv,ghvo->bgho", signal, grouped_decoder
        ).sum(dim=2)
        result[:, :, page] = torch.linalg.vector_norm(decoded, dim=-1)
    return result


def _normalize_forced_pages(
    forced_page_mask: Tensor | None,
    *,
    expected: tuple[int, int, int],
    device: torch.device,
) -> Tensor:
    if forced_page_mask is None:
        return torch.zeros(expected, dtype=torch.bool, device=device)
    candidate = forced_page_mask.to(device=device)
    if candidate.dtype != torch.bool or tuple(candidate.shape) != expected:
        raise ValueError(f"forced page mask must be boolean with shape {expected}")
    return candidate.clone()


def _query_page_validity(valid: Tensor, page_size: int) -> Tensor:
    batch, query_heads, sequence = map(int, valid.shape)
    page_count = math.ceil(sequence / page_size)
    padding = page_count * page_size - sequence
    padded = valid
    if padding:
        padded = torch.cat(
            (
                padded,
                torch.zeros(
                    batch,
                    query_heads,
                    padding,
                    dtype=torch.bool,
                    device=valid.device,
                ),
            ),
            dim=2,
        )
    return padded.reshape(batch, query_heads, page_count, page_size).any(dim=-1)


def _physical_page_union(
    selected: Tensor,
    *,
    query_heads: int,
    kv_heads: int,
) -> Tensor:
    if int(selected.shape[1]) == kv_heads:
        return selected
    if int(selected.shape[1]) != query_heads:
        raise ValueError("selected pages must belong to physical KV or Query heads")
    heads_per_group = query_heads // kv_heads
    return selected.reshape(
        int(selected.shape[0]), kv_heads, heads_per_group, int(selected.shape[2])
    ).any(dim=2)


def _select_pages(
    page_scores: Tensor,
    owner_valid: Tensor,
    page_valid: Tensor,
    config: ReverseShadowConfig,
    forced_page_mask: Tensor | None,
) -> Tensor:
    batch, owners, page_count = map(int, page_scores.shape)
    selected = _normalize_forced_pages(
        forced_page_mask,
        expected=(batch, owners, page_count),
        device=page_scores.device,
    )
    selected &= page_valid
    for batch_index in range(batch):
        for owner in range(owners):
            if config.recent_exact_window:
                valid_tokens = torch.nonzero(
                    owner_valid[batch_index, owner], as_tuple=False
                ).flatten()
                if len(valid_tokens):
                    last = int(valid_tokens[-1])
                    first = max(0, last - config.recent_exact_window + 1)
                    recent_pages = torch.unique(
                        valid_tokens[valid_tokens >= first] // config.page_size,
                        sorted=True,
                    )
                    selected[batch_index, owner, recent_pages] = True
            remaining = max(
                config.page_budget - int(selected[batch_index, owner].sum()), 0
            )
            if remaining == 0:
                continue
            candidates = torch.nonzero(
                page_valid[batch_index, owner]
                & ~selected[batch_index, owner],
                as_tuple=False,
            ).flatten()
            if len(candidates):
                order = torch.argsort(
                    page_scores[batch_index, owner, candidates],
                    descending=True,
                    stable=True,
                )
                selected[batch_index, owner, candidates[order[:remaining]]] = True
    return selected


def _selected_page_ids(selected: Tensor) -> Tensor:
    width = int(selected.sum(dim=-1).max().item()) if selected.numel() else 0
    result = torch.full(
        (*selected.shape[:2], width),
        -1,
        dtype=torch.long,
        device=selected.device,
    )
    for batch in range(int(selected.shape[0])):
        for group in range(int(selected.shape[1])):
            pages = torch.nonzero(selected[batch, group], as_tuple=False).flatten()
            result[batch, group, : len(pages)] = pages
    return result


def _page_requests(selected: Tensor) -> tuple[Tensor, Tensor]:
    coordinates = torch.nonzero(selected, as_tuple=False)
    if len(coordinates) == 0:
        return coordinates, torch.empty(
            0, dtype=torch.long, device=selected.device
        )
    return coordinates, coordinates[:, 2]


def _selected_token_mask(
    selected: Tensor,
    valid: Tensor,
    *,
    page_size: int,
    kv_heads: int,
) -> Tensor:
    batch, query_heads, sequence = map(int, valid.shape)
    if int(selected.shape[1]) == query_heads:
        expanded = selected
    elif int(selected.shape[1]) == kv_heads:
        head_to_kv = _head_to_kv(query_heads, kv_heads, valid.device)
        expanded = selected.index_select(1, head_to_kv)
    else:
        raise ValueError("selected pages must belong to physical KV or Query heads")
    page_ids = torch.arange(sequence, device=valid.device) // page_size
    expanded = expanded.gather(
        2, page_ids.view(1, 1, sequence).expand(batch, query_heads, sequence)
    )
    return expanded & valid


def _kq_svd_page_selection(
    query: Tensor,
    routing_sidecar: Tensor,
    routing_query_projector: Tensor,
    valid: Tensor,
    bias: Tensor,
    physical_valid: Tensor,
    page_valid: Tensor,
    config: ReverseShadowConfig,
    forced_page_mask: Tensor | None,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Rank proxy page mass per Query head and fetch each GQA-group union."""

    batch, query_heads, _, head_dim = map(int, query.shape)
    sidecar_batch, kv_heads, sequence, rank = map(int, routing_sidecar.shape)
    if sidecar_batch != batch or tuple(valid.shape) != (
        batch,
        query_heads,
        sequence,
    ):
        raise ValueError("routing sidecar and attention geometry differ")
    if routing_sidecar.device != query.device:
        raise ValueError("routing sidecar and query must share a device")
    if routing_query_projector.ndim != 3 or tuple(
        routing_query_projector.shape[1:]
    ) != (head_dim, rank):
        raise ValueError("routing Query projector and sidecar geometry differ")
    if int(routing_query_projector.shape[0]) not in (kv_heads, query_heads):
        raise ValueError("routing Query projector must be per-KV or per-Query head")
    if config.page_budget <= 0:
        raise ValueError("KQ-SVD routing requires a positive page budget")

    page_count = int(page_valid.shape[-1])
    heads_per_group = query_heads // kv_heads
    selected_batches = []
    score_batches = []
    adaptive_eligible_query_heads = 0
    adaptive_refined_query_heads = 0
    adaptive_tail_mass_ratio_sum = 0.0
    for batch_index in range(batch):
        token_scores = routing_proxy_scores(
            query[batch_index, :, 0],
            routing_sidecar[batch_index],
            routing_query_projector,
            head_dim=head_dim,
        ).float()
        token_scores.add_(bias[batch_index]).masked_fill_(
            ~valid[batch_index], -torch.inf
        )
        if config.adaptive_max_page_budget is None:
            _, selected_pages = gqa_union_page_mass_mask(
                token_scores,
                num_kv_heads=kv_heads,
                page_size=config.page_size,
                pages_per_query_head=config.page_budget,
            )
        else:
            assert config.adaptive_tail_mass_ratio_threshold is not None
            adaptive = gqa_union_adaptive_page_mass_mask(
                token_scores,
                num_kv_heads=kv_heads,
                page_size=config.page_size,
                base_pages_per_query_head=config.page_budget,
                max_pages_per_query_head=config.adaptive_max_page_budget,
                tail_mass_ratio_threshold=(
                    config.adaptive_tail_mass_ratio_threshold
                ),
            )
            selected_pages = adaptive.page_mask
            adaptive_eligible_query_heads += adaptive.eligible_query_heads
            adaptive_refined_query_heads += adaptive.refined_query_heads
            adaptive_tail_mass_ratio_sum += adaptive.tail_mass_ratio_sum
        selected_batches.append(selected_pages)
        padding = page_count * config.page_size - sequence
        padded_scores = (
            torch.cat(
                (
                    token_scores,
                    torch.full(
                        (query_heads, padding),
                        -torch.inf,
                        dtype=token_scores.dtype,
                        device=token_scores.device,
                    ),
                ),
                dim=-1,
            )
            if padding
            else token_scores
        )
        per_query_page_mass = torch.logsumexp(
            padded_scores.reshape(
                query_heads, page_count, config.page_size
            ),
            dim=-1,
        )
        score_batches.append(
            per_query_page_mass.reshape(
                kv_heads, heads_per_group, page_count
            ).amax(dim=1)
        )
    selected = torch.stack(selected_batches) & page_valid
    page_scores = torch.stack(score_batches).masked_fill(~page_valid, -torch.inf)

    if forced_page_mask is not None:
        forced = forced_page_mask.to(device=query.device, dtype=torch.bool)
        if tuple(forced.shape) == (batch, query_heads, page_count):
            forced = _physical_page_union(
                forced, query_heads=query_heads, kv_heads=kv_heads
            )
        forced = _normalize_forced_pages(
            forced,
            expected=(batch, kv_heads, page_count),
            device=query.device,
        )
        selected |= forced & page_valid
    if config.recent_exact_window:
        for batch_index in range(batch):
            for group in range(kv_heads):
                valid_tokens = torch.nonzero(
                    physical_valid[batch_index, group], as_tuple=False
                ).flatten()
                if len(valid_tokens):
                    last = int(valid_tokens[-1])
                    first = max(0, last - config.recent_exact_window + 1)
                    selected[batch_index, group, torch.unique(
                        valid_tokens[valid_tokens >= first] // config.page_size
                    )] = True
    return page_scores, selected, {
        "adaptive_eligible_query_heads": float(adaptive_eligible_query_heads),
        "adaptive_refined_query_heads": float(adaptive_refined_query_heads),
        "adaptive_tail_mass_ratio_sum": float(adaptive_tail_mass_ratio_sum),
    }


def _logical_statistics(
    *,
    config: ReverseShadowConfig,
    landmarks: PostRoPEKLandmarks,
    c1_value: Tensor,
    selected: Tensor,
    selected_token_mask: Tensor,
    valid: Tensor,
    physical_valid: Tensor,
    query_heads: int,
    head_dim: int,
    exact_element_size: int,
    c1_page_metadata: C1PageMetadata | None = None,
    routing_sidecar: Tensor | None = None,
) -> tuple[int, dict[str, float]]:
    batch = int(selected.shape[0])
    kv_heads = int(c1_value.shape[1])
    sequence = int(c1_value.shape[2])
    physical_union = _physical_page_union(
        selected, query_heads=query_heads, kv_heads=kv_heads
    )
    page_ids = torch.arange(sequence, device=selected.device) // config.page_size
    physical_selected = physical_union.gather(
        2,
        page_ids.view(1, 1, sequence).expand(batch, kv_heads, sequence),
    )
    physical_selected_tokens = int((physical_selected & physical_valid).sum())
    physical_valid_tokens = max(int(physical_valid.sum()), 1)
    logical_selected_tokens = int(selected_token_mask.sum())
    logical_valid_tokens = max(int(valid.sum()), 1)
    selected_pages = int(physical_union.sum())
    logical_selected_pages = int(selected.sum())
    heads_per_group = query_heads // kv_heads
    if config.selector in (
        "teacher_exact",
        "teacher_mass",
        "teacher_output",
        "teacher_influence",
    ):
        selection_flops = 2 * batch * query_heads * sequence * head_dim
    elif config.selector == "kq_svd":
        if routing_sidecar is None:
            raise ValueError("KQ-SVD statistics require the routing sidecar")
        routing_rank = int(routing_sidecar.shape[-1])
        selection_flops = (
            2
            * batch
            * query_heads
            * routing_rank
            * (head_dim + sequence)
        )
    elif config.selector in (
        "quest_minmax",
        "quest_k",
        "quest_c1_latent",
        "quest_c1_output",
    ):
        valid_pages = int(landmarks.page_bounds_valid.sum())
        selection_flops = 4 * valid_pages * heads_per_group * head_dim
    else:
        valid_landmarks = int(landmarks.valid.sum())
        selection_flops = 2 * valid_landmarks * heads_per_group * head_dim
    if config.selector == "kq_svd":
        assert routing_sidecar is not None
        selector_metadata_bytes = (
            routing_sidecar.numel() * routing_sidecar.element_size()
        )
    elif config.selector in (
        "quest_minmax",
        "quest_k",
        "quest_c1_latent",
        "quest_c1_output",
    ):
        selector_metadata_bytes = (
            landmarks.page_mins.numel() * landmarks.page_mins.element_size()
            + landmarks.page_maxes.numel() * landmarks.page_maxes.element_size()
            + landmarks.page_bounds_valid.numel()
            * landmarks.page_bounds_valid.element_size()
        )
    elif config.selector == "centroid_radius":
        selector_metadata_bytes = (
            landmarks.values.numel() * landmarks.values.element_size()
            + landmarks.radii.numel() * landmarks.radii.element_size()
            + landmarks.valid.numel() * landmarks.valid.element_size()
        )
    elif config.selector == "mean_landmark":
        selector_metadata_bytes = (
            landmarks.values.numel() * landmarks.values.element_size()
            + landmarks.valid.numel() * landmarks.valid.element_size()
        )
    else:
        selector_metadata_bytes = 0
    latent_norm_bytes = 0
    output_norm_bytes = 0
    token_count_bytes = 0
    decoder_gram_bytes = 0
    if c1_page_metadata is not None and config.selector in (
        "quest_c1_latent",
        "quest_c1_output",
    ):
        token_count_bytes = (
            c1_page_metadata.valid_token_count.numel()
            * c1_page_metadata.valid_token_count.element_size()
        )
        selector_metadata_bytes += token_count_bytes
        if config.selector == "quest_c1_latent":
            latent_norm_bytes = (
                c1_page_metadata.latent_max_norm.numel()
                * c1_page_metadata.latent_max_norm.element_size()
            )
            selector_metadata_bytes += latent_norm_bytes
        else:
            output_norm_bytes = (
                c1_page_metadata.decoded_output_max_norm.numel()
                * c1_page_metadata.decoded_output_max_norm.element_size()
            )
            decoder_gram_bytes = (
                c1_page_metadata.decoder_gram.numel()
                * c1_page_metadata.decoder_gram.element_size()
            )
            selector_metadata_bytes += output_norm_bytes + decoder_gram_bytes
    routing_sidecar_bytes = (
        selector_metadata_bytes if config.selector == "kq_svd" else 0
    )
    uses_landmarks = config.selector != "kq_svd"
    return physical_selected_tokens, {
        "resident_c1_value_bytes": float(c1_value.numel() * c1_value.element_size()),
        "resident_selector_metadata_bytes": float(selector_metadata_bytes),
        "resident_routing_sidecar_bytes": float(routing_sidecar_bytes),
        "resident_c1_latent_norm_bytes": float(latent_norm_bytes),
        "resident_c1_output_norm_bytes": float(output_norm_bytes),
        "resident_c1_page_count_bytes": float(token_count_bytes),
        "resident_c1_decoder_gram_bytes": float(decoder_gram_bytes),
        "resident_landmark_key_bytes": float(
            landmarks.values.numel() * landmarks.values.element_size()
            if uses_landmarks
            else 0
        ),
        "resident_landmark_radius_bytes": float(
            landmarks.radii.numel() * landmarks.radii.element_size()
            if uses_landmarks
            else 0
        ),
        "resident_landmark_validity_bytes": float(
            landmarks.valid.numel() * landmarks.valid.element_size()
            if uses_landmarks
            else 0
        ),
        "resident_quest_minmax_key_bytes": float(
            (
                landmarks.page_mins.numel() * landmarks.page_mins.element_size()
                + landmarks.page_maxes.numel()
                * landmarks.page_maxes.element_size()
            )
            if uses_landmarks
            else 0
        ),
        "resident_quest_validity_bytes": float(
            (
                landmarks.page_bounds_valid.numel()
                * landmarks.page_bounds_valid.element_size()
            )
            if uses_landmarks
            else 0
        ),
        "physical_valid_tokens": float(int(physical_valid.sum())),
        "query_valid_tokens": float(int(valid.sum())),
        "selected_pages": float(selected_pages),
        "logical_selected_pages": float(logical_selected_pages),
        "selected_tokens": float(physical_selected_tokens),
        "query_selected_tokens": float(logical_selected_tokens),
        "selected_token_fraction": physical_selected_tokens / physical_valid_tokens,
        "query_selected_token_fraction": logical_selected_tokens / logical_valid_tokens,
        "oracle_page_store_key_bytes_read": float(
            selected_pages * config.page_size * head_dim * exact_element_size
        ),
        "selected_c1_value_bytes_read": float(
            logical_selected_tokens
            * int(c1_value.shape[-1])
            * c1_value.element_size()
        ),
        "selection_qk_flops": float(selection_flops),
        "sparse_exact_qk_flops": float(
            2 * logical_selected_tokens * head_dim
        ),
        "sparse_c1_value_flops": float(
            2 * logical_selected_tokens * int(c1_value.shape[-1])
        ),
        "full_exact_qk_flops": float(
            2 * batch * query_heads * sequence * head_dim
        ),
    }


def c1_k_reverse_shadow_attention(
    query: Tensor,
    landmarks: PostRoPEKLandmarks,
    c1_value: Tensor,
    config: ReverseShadowConfig,
    exact_key: Tensor | None = None,
    attention_mask: Tensor | None = None,
    *,
    page_store: ExactKeyPageStore | None = None,
    c1_page_metadata: C1PageMetadata | None = None,
    decoder: Tensor | None = None,
    layer_idx: int = 0,
    forced_page_mask: Tensor | None = None,
    routing_sidecar: Tensor | None = None,
    routing_query_projector: Tensor | None = None,
    vectorized_reference: bool = False,
) -> ReverseShadowResult:
    """Select physical K pages, fetch exact K, and attend to resident C1-V.

    This query-length-one reference never materializes a dense score tensor or
    expands C1 Values across GQA query heads.  The current page-store call is
    synchronous; it is an algorithm oracle, not a measured CPU-offload kernel.
    """

    if query.ndim != 4 or int(query.shape[2]) != 1:
        raise ValueError("query must have shape [batch, query heads, 1, head dim]")
    if c1_value.ndim != 4:
        raise ValueError("C1 Value must be [batch, KV heads, sequence, value rank]")
    batch, query_heads, _, head_dim = map(int, query.shape)
    value_batch, kv_heads, sequence, value_rank = map(int, c1_value.shape)
    if value_batch != batch or sequence <= 0 or value_rank <= 0:
        raise ValueError("query and C1 Value cache geometry differs")
    _head_to_kv(query_heads, kv_heads, query.device)
    config.validate(head_dim)
    if landmarks.sequence_length != sequence or landmarks.page_size != config.page_size:
        raise ValueError("landmark and attention cache geometry differs")
    landmarks.validate(
        batch=batch,
        kv_heads=kv_heads,
        head_dim=head_dim,
        landmarks_per_page=config.landmarks_per_page,
        device=query.device,
    )
    if c1_value.device != query.device:
        raise ValueError("query, landmarks, and C1 Value must share a device")
    if c1_page_metadata is not None:
        if (
            c1_page_metadata.sequence_length != sequence
            or c1_page_metadata.page_size != config.page_size
        ):
            raise ValueError("C1 page metadata and attention cache geometry differs")
        c1_page_metadata.validate(
            batch=batch,
            kv_heads=kv_heads,
            query_heads=query_heads,
            value_rank=value_rank,
            device=query.device,
        )
    if config.selector in ("quest_c1_latent", "quest_c1_output") and (
        c1_page_metadata is None
    ):
        raise ValueError(f"{config.selector} requires C1 page metadata")
    expected_key_shape = (batch, kv_heads, sequence, head_dim)
    if exact_key is not None and tuple(exact_key.shape) != expected_key_shape:
        raise ValueError("exact Key geometry is incompatible")
    if exact_key is not None and exact_key.device != query.device:
        raise ValueError("teacher exact Key and query must share a device")
    teacher_key_selectors = (
        "teacher_exact",
        "teacher_mass",
        "teacher_output",
        "teacher_influence",
    )
    if config.selector in teacher_key_selectors and exact_key is None:
        raise ValueError(f"{config.selector} requires the full exact Key tensor")
    if config.selector in ("teacher_output", "teacher_influence") and decoder is None:
        raise ValueError(f"{config.selector} requires the C1 output decoder")
    if config.selector == "kq_svd":
        if routing_sidecar is None or routing_query_projector is None:
            raise ValueError("KQ-SVD routing requires a sidecar and Query projector")
    elif routing_sidecar is not None or routing_query_projector is not None:
        raise ValueError("routing sidecar inputs require selector='kq_svd'")
    if page_store is None:
        if exact_key is None:
            raise ValueError("exact Key or an exact-Key page store is required")
        page_store = GPUExactKeyPageStore(exact_key, layer_idx=layer_idx)

    valid, bias = _attention_mask(
        attention_mask,
        batch=batch,
        query_heads=query_heads,
        sequence=sequence,
        device=query.device,
    )
    physical_valid, page_valid = _page_validity(
        valid, kv_heads=kv_heads, page_size=config.page_size
    )
    selection_valid = physical_valid
    selection_page_valid = page_valid
    selection_forced_page_mask = forced_page_mask
    if config.quest_support == "per_query_head":
        selection_valid = valid
        selection_page_valid = _query_page_validity(valid, config.page_size)
        if forced_page_mask is not None:
            if tuple(forced_page_mask.shape) == tuple(page_valid.shape):
                head_to_kv = _head_to_kv(query_heads, kv_heads, query.device)
                selection_forced_page_mask = forced_page_mask.index_select(
                    1, head_to_kv
                )
            elif tuple(forced_page_mask.shape) != tuple(selection_page_valid.shape):
                raise ValueError(
                    "forced page mask must use physical-KV or Query-head geometry"
                )
    routing_selected = None
    routing_selection_statistics = {
        "adaptive_eligible_query_heads": 0.0,
        "adaptive_refined_query_heads": 0.0,
        "adaptive_tail_mass_ratio_sum": 0.0,
    }
    if config.selector == "teacher_exact":
        assert exact_key is not None
        page_scores = _teacher_page_scores(
            query,
            exact_key,
            valid,
            bias,
            page_size=config.page_size,
        )
    elif config.selector == "teacher_mass":
        assert exact_key is not None
        page_scores = _teacher_mass_page_scores(
            query,
            exact_key,
            valid,
            bias,
            page_size=config.page_size,
        )
    elif config.selector in ("teacher_output", "teacher_influence"):
        assert exact_key is not None
        assert decoder is not None
        page_scores = _teacher_decoded_page_scores(
            query,
            exact_key,
            c1_value,
            decoder,
            valid,
            bias,
            page_size=config.page_size,
            normalization_aware=config.selector == "teacher_influence",
        )
    elif config.selector == "quest_minmax":
        page_scores = _quest_page_scores(
            query,
            landmarks,
            valid,
            bias,
            support=config.quest_support,
        )
    elif config.selector in ("quest_k", "quest_c1_latent", "quest_c1_output"):
        page_scores = _c1_aware_quest_page_scores(
            query,
            landmarks,
            c1_page_metadata,
            valid,
            bias,
            selector=config.selector,
            aggregation=config.query_head_aggregation,
        )
    elif config.selector == "kq_svd":
        assert routing_sidecar is not None
        assert routing_query_projector is not None
        page_scores, routing_selected, routing_selection_statistics = (
            _kq_svd_page_selection(
                query,
                routing_sidecar,
                routing_query_projector,
                valid,
                bias,
                physical_valid,
                page_valid,
                config,
                forced_page_mask,
            )
        )
    else:
        page_scores = _landmark_page_scores(
            query,
            landmarks,
            valid,
            bias,
            use_radius=config.selector == "centroid_radius",
        )
    if routing_selected is None:
        page_scores.masked_fill_(~selection_page_valid, -torch.inf)
        selected = _select_pages(
            page_scores,
            selection_valid,
            selection_page_valid,
            config,
            selection_forced_page_mask,
        )
    else:
        selected = routing_selected
    selected_tokens = _selected_token_mask(
        selected, valid, page_size=config.page_size, kv_heads=kv_heads
    )
    if torch.any(selected_tokens.sum(dim=-1) == 0):
        raise ValueError(
            "Reverse ShadowKV requires at least one selected valid token per query head"
        )

    if vectorized_reference:
        if exact_key is None:
            raise ValueError("vectorized reference attention requires exact K")
        head_to_kv = _head_to_kv(query_heads, kv_heads, query.device)
        expanded_key = exact_key.index_select(1, head_to_kv).float()
        exact_scores = torch.einsum(
            "bhd,bhsd->bhs", query[:, :, 0].float(), expanded_key
        )
        exact_scores.mul_(head_dim**-0.5).add_(bias)
        exact_scores.masked_fill_(~selected_tokens, -torch.inf)
        probability = torch.softmax(exact_scores, dim=-1, dtype=torch.float32)
        heads_per_group = query_heads // kv_heads
        grouped_probability = probability.reshape(
            batch, kv_heads, heads_per_group, sequence
        )
        grouped_output = torch.einsum(
            "bghs,bgsv->bghv", grouped_probability, c1_value.float()
        )
        output = grouped_output.reshape(batch, query_heads, value_rank)
        selected_token_count, statistics = _logical_statistics(
            config=config,
            landmarks=landmarks,
            c1_value=c1_value,
            selected=selected,
            selected_token_mask=selected_tokens,
            valid=valid,
            physical_valid=physical_valid,
            query_heads=query_heads,
            head_dim=head_dim,
            exact_element_size=exact_key.element_size(),
            c1_page_metadata=c1_page_metadata,
            routing_sidecar=routing_sidecar,
        )
        statistics.update(routing_selection_statistics)
        return ReverseShadowResult(
            output=output[:, :, None].to(c1_value.dtype),
            selected_page_ids=_selected_page_ids(selected),
            selected_page_mask=selected,
            selected_token_count=selected_token_count,
            page_scores=page_scores,
            running_lse=torch.logsumexp(exact_scores, dim=-1),
            statistics=statistics,
        )

    physical_selected = _physical_page_union(
        selected, query_heads=query_heads, kv_heads=kv_heads
    )
    requests, page_ids = _page_requests(physical_selected)
    exact_pages = page_store.get_pages(
        layer_idx=layer_idx,
        batch_indices=requests[:, 0],
        kv_head_indices=requests[:, 1],
        page_ids=page_ids,
        page_size=config.page_size,
        device=query.device,
    )
    if tuple(exact_pages.shape) != (len(requests), config.page_size, head_dim):
        raise ValueError("exact-Key page store returned an incompatible tensor")

    running_max = torch.full(
        (batch, query_heads), -torch.inf, dtype=torch.float32, device=query.device
    )
    running_sum = torch.zeros_like(running_max)
    numerator = torch.zeros(
        batch, query_heads, value_rank, dtype=torch.float32, device=query.device
    )
    heads_per_group = query_heads // kv_heads
    for request, exact_page in zip(requests, exact_pages):
        batch_index, group, page = map(int, request.tolist())
        start = page * config.page_size
        stop = min(start + config.page_size, sequence)
        head_start = group * heads_per_group
        head_stop = head_start + heads_per_group
        head_slice = slice(head_start, head_stop)
        block_valid = valid[batch_index, head_slice, start:stop]
        if config.quest_support == "per_query_head":
            head_page_selected = selected[batch_index, head_slice, page]
            block_valid = block_valid & head_page_selected[:, None]
        scores = torch.einsum(
            "hd,sd->hs",
            query[batch_index, head_slice, 0].float(),
            exact_page[: stop - start].float(),
        )
        scores.mul_(head_dim**-0.5).add_(
            bias[batch_index, head_slice, start:stop]
        )
        scores.masked_fill_(~block_valid, -torch.inf)
        block_max = scores.amax(dim=-1)
        finite_block = torch.isfinite(block_max)
        safe_block_max = torch.where(
            finite_block, block_max, torch.zeros_like(block_max)
        )
        weights = torch.exp(scores - safe_block_max[:, None])
        weights = torch.where(block_valid, weights, torch.zeros_like(weights))
        block_sum = weights.sum(dim=-1)
        block_value = c1_value[batch_index, group, start:stop].float()
        block_numerator = torch.einsum("hs,sv->hv", weights, block_value)

        old_max = running_max[batch_index, head_slice].clone()
        old_sum = running_sum[batch_index, head_slice].clone()
        old_numerator = numerator[batch_index, head_slice].clone()
        next_max = torch.maximum(old_max, block_max)
        finite_old = torch.isfinite(old_max)
        finite_next = torch.isfinite(next_max)
        old_scale = torch.where(
            finite_old & finite_next,
            torch.exp(old_max - next_max),
            torch.zeros_like(next_max),
        )
        block_scale = torch.where(
            finite_block & finite_next,
            torch.exp(block_max - next_max),
            torch.zeros_like(next_max),
        )
        running_max[batch_index, head_slice] = next_max
        running_sum[batch_index, head_slice] = (
            old_scale * old_sum + block_scale * block_sum
        )
        numerator[batch_index, head_slice] = (
            old_scale[:, None] * old_numerator
            + block_scale[:, None] * block_numerator
        )
    if torch.any(running_sum == 0):
        raise ValueError("every query head must have at least one valid attention token")
    output = (numerator / running_sum[:, :, None])[:, :, None].to(c1_value.dtype)
    running_lse = running_max + torch.log(running_sum)
    selected_token_count, statistics = _logical_statistics(
        config=config,
        landmarks=landmarks,
        c1_value=c1_value,
        selected=selected,
        selected_token_mask=selected_tokens,
        valid=valid,
        physical_valid=physical_valid,
        query_heads=query_heads,
        head_dim=head_dim,
        exact_element_size=(
            exact_key.element_size() if exact_key is not None else exact_pages.element_size()
        ),
        c1_page_metadata=c1_page_metadata,
        routing_sidecar=routing_sidecar,
    )
    statistics.update(routing_selection_statistics)
    return ReverseShadowResult(
        output=output,
        selected_page_ids=_selected_page_ids(selected),
        selected_page_mask=selected,
        selected_token_count=selected_token_count,
        page_scores=page_scores,
        running_lse=running_lse,
        statistics=statistics,
    )


def _relative_l2(observed: Tensor, expected: Tensor) -> float:
    numerator = torch.linalg.vector_norm((observed - expected).double())
    denominator = torch.linalg.vector_norm(expected.double()).clamp_min(1.0e-300)
    return float(numerator / denominator)


def reverse_shadow_quality_statistics(
    full_exact: ReverseShadowResult,
    candidate: ReverseShadowResult,
    *,
    query: Tensor,
    exact_key: Tensor,
    config: ReverseShadowConfig,
    attention_mask: Tensor | None = None,
    decoder: Tensor | None = None,
    attention_top_k: int = 10,
) -> dict[str, float]:
    """Measure selected exact mass and C1 output error against full attention."""

    batch, query_heads, _, head_dim = map(int, query.shape)
    if exact_key.ndim != 4 or int(exact_key.shape[0]) != batch:
        raise ValueError("exact Key geometry is incompatible with query")
    kv_heads, sequence = map(int, exact_key.shape[1:3])
    head_to_kv = _head_to_kv(query_heads, kv_heads, query.device)
    valid, bias = _attention_mask(
        attention_mask,
        batch=batch,
        query_heads=query_heads,
        sequence=sequence,
        device=query.device,
    )
    expanded_key = exact_key.index_select(1, head_to_kv).float()
    exact_scores = torch.einsum(
        "bhd,bhsd->bhs", query[:, :, 0].float(), expanded_key
    )
    exact_scores.mul_(head_dim**-0.5).add_(bias)
    exact_scores.masked_fill_(~valid, -torch.inf)
    selected_tokens = _selected_token_mask(
        candidate.selected_page_mask,
        valid,
        page_size=config.page_size,
        kv_heads=kv_heads,
    )
    if torch.any(selected_tokens.sum(dim=-1) == 0):
        raise ValueError("quality statistics require nonempty sparse support")
    sparse_scores = exact_scores.masked_fill(~selected_tokens, -torch.inf)
    exact_probability = torch.softmax(exact_scores, dim=-1, dtype=torch.float32)
    sparse_probability = torch.softmax(sparse_scores, dim=-1, dtype=torch.float32)
    safe_exact = exact_probability.clamp_min(1.0e-30)
    sparse_positive = sparse_probability > 0
    reverse_kl_terms = torch.where(
        sparse_positive,
        sparse_probability
        * (sparse_probability.clamp_min(1.0e-30).log() - safe_exact.log()),
        torch.zeros_like(sparse_probability),
    )
    metrics = {
        "exact_attention_mass_selected": float(
            (exact_probability * selected_tokens).sum()
            / exact_probability.sum().clamp_min(1.0e-30)
        ),
        "attention_kl_sparse_to_exact": float(
            reverse_kl_terms.sum(dim=-1).mean()
        ),
        "attention_probability_l1": float(
            (sparse_probability - exact_probability).abs().sum(dim=-1).mean()
        ),
        "attention_max_probability_error": float(
            (sparse_probability - exact_probability).abs().max()
        ),
        "c1_latent_relative_l2": _relative_l2(candidate.output, full_exact.output),
    }

    page_recalls: list[float] = []
    mass_page_recalls: list[float] = []
    token_recalls: list[float] = []
    top_k_overlaps: list[float] = []
    if int(candidate.selected_page_mask.shape[1]) == query_heads:
        page_valid = _query_page_validity(valid, config.page_size)
        page_count = int(page_valid.shape[-1])
        padding = page_count * config.page_size - sequence
        padded_scores = exact_scores
        padded_probability = exact_probability
        if padding:
            padded_scores = torch.cat(
                (
                    padded_scores,
                    torch.full(
                        (batch, query_heads, padding),
                        -torch.inf,
                        dtype=padded_scores.dtype,
                        device=padded_scores.device,
                    ),
                ),
                dim=-1,
            )
            padded_probability = torch.cat(
                (
                    padded_probability,
                    torch.zeros(
                        batch,
                        query_heads,
                        padding,
                        dtype=padded_probability.dtype,
                        device=padded_probability.device,
                    ),
                ),
                dim=-1,
            )
        exact_page_scores = padded_scores.reshape(
            batch, query_heads, page_count, config.page_size
        ).amax(dim=-1)
        exact_mass_page_scores = padded_probability.reshape(
            batch, query_heads, page_count, config.page_size
        ).sum(dim=-1)
    else:
        exact_page_scores = _teacher_page_scores(
            query, exact_key, valid, bias, page_size=config.page_size
        )
        exact_mass_page_scores = _teacher_mass_page_scores(
            query, exact_key, valid, bias, page_size=config.page_size
        )
        _, page_valid = _page_validity(
            valid, kv_heads=kv_heads, page_size=config.page_size
        )

    selection_owners = int(candidate.selected_page_mask.shape[1])
    for batch_index in range(batch):
        for owner in range(selection_owners):
            count = int(candidate.selected_page_mask[batch_index, owner].sum())
            if not count:
                continue
            valid_pages = torch.nonzero(
                page_valid[batch_index, owner], as_tuple=False
            ).flatten()
            order = torch.argsort(
                exact_page_scores[batch_index, owner, valid_pages],
                descending=True,
                stable=True,
            )
            exact_top = valid_pages[order[:count]]
            page_recalls.append(
                float(
                    candidate.selected_page_mask[
                        batch_index, owner, exact_top
                    ].float().mean()
                )
            )
            mass_order = torch.argsort(
                exact_mass_page_scores[batch_index, owner, valid_pages],
                descending=True,
                stable=True,
            )
            exact_mass_top = valid_pages[mass_order[:count]]
            mass_page_recalls.append(
                float(
                    candidate.selected_page_mask[
                        batch_index, owner, exact_mass_top
                    ].float().mean()
                )
            )

        for head in range(query_heads):
            selected_count = int(selected_tokens[batch_index, head].sum())
            valid_positions = torch.nonzero(
                valid[batch_index, head], as_tuple=False
            ).flatten()
            if selected_count:
                order = torch.argsort(
                    exact_scores[batch_index, head, valid_positions],
                    descending=True,
                    stable=True,
                )
                exact_top = valid_positions[order[:selected_count]]
                token_recalls.append(
                    float(selected_tokens[batch_index, head, exact_top].float().mean())
                )
            k = min(attention_top_k, selected_count, len(valid_positions))
            if k:
                exact_top = torch.topk(
                    exact_probability[batch_index, head, valid_positions], k
                ).indices
                sparse_top = torch.topk(
                    sparse_probability[batch_index, head, valid_positions], k
                ).indices
                top_k_overlaps.append(
                    float(torch.isin(exact_top, sparse_top).float().mean())
                )
    metrics["exact_top_page_recall"] = (
        sum(page_recalls) / len(page_recalls) if page_recalls else 0.0
    )
    metrics["exact_top_mass_page_recall"] = (
        sum(mass_page_recalls) / len(mass_page_recalls)
        if mass_page_recalls
        else 0.0
    )
    metrics["exact_top_token_recall"] = (
        sum(token_recalls) / len(token_recalls) if token_recalls else 0.0
    )
    metrics["attention_top_k_overlap"] = (
        sum(top_k_overlaps) / len(top_k_overlaps) if top_k_overlaps else 0.0
    )
    if decoder is not None:
        if decoder.ndim == 4:
            decoder = decoder.reshape(
                query_heads, decoder.shape[-2], decoder.shape[-1]
            )
        value_rank = int(candidate.output.shape[-1])
        if tuple(decoder.shape[:2]) != (query_heads, value_rank):
            raise ValueError("decoder must be [query heads, value rank, output dim]")
        exact_output = torch.einsum(
            "bhqv,hvo->bqo", full_exact.output.float(), decoder.float()
        )
        candidate_output = torch.einsum(
            "bhqv,hvo->bqo", candidate.output.float(), decoder.float()
        )
        decoded_error = torch.linalg.vector_norm(
            (candidate_output - exact_output).double()
        )
        decoded_reference = torch.linalg.vector_norm(exact_output.double())
        metrics["c1_decoded_output_error_l2"] = float(decoded_error)
        metrics["c1_decoded_output_reference_l2"] = float(decoded_reference)
        metrics["c1_decoded_output_relative_l2"] = _relative_l2(
            candidate_output, exact_output
        )
        metrics["c1_decoded_output_max_absolute_error"] = float(
            (candidate_output - exact_output).abs().max()
        )
        metrics["c1_decoded_output_cosine_similarity"] = float(
            torch.nn.functional.cosine_similarity(
                candidate_output.flatten(1), exact_output.flatten(1), dim=-1
            ).mean()
        )
    return metrics
