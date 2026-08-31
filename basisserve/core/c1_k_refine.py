"""C1-KRefine correctness oracles for low-rank Key proxies.

The module deliberately implements attention mathematics, not CPU offload.
Exact post-RoPE Keys remain resident behind :class:`ExactKeyPageStore`; the
interface is the boundary a future CPU-backed store can implement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Protocol

import torch
from torch import Tensor


PageScoreMode = Literal["max", "logsumexp"]
ScorePolicy = Literal[
    "full_exact",
    "proxy_only",
    "proxy_exact_refine",
    "sparse_exact",
]


_PROXY_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class KProxyConfig:
    """Static policy and storage geometry for one KRefine experiment."""

    proxy_rank: int
    page_size: int
    exact_token_budget: int
    recent_exact_window: int = 0
    page_score_mode: PageScoreMode = "max"
    score_policy: ScorePolicy = "proxy_exact_refine"
    proxy_dtype: str = "bfloat16"

    def validate(self, head_dim: int) -> None:
        if not 1 <= self.proxy_rank <= int(head_dim):
            raise ValueError(
                f"proxy rank must be in [1, {head_dim}], got {self.proxy_rank}"
            )
        if self.page_size <= 0:
            raise ValueError("page size must be positive")
        if self.exact_token_budget < 0:
            raise ValueError("exact-token budget must be nonnegative")
        if self.recent_exact_window < 0:
            raise ValueError("recent exact window must be nonnegative")
        if self.page_score_mode not in ("max", "logsumexp"):
            raise ValueError(f"unsupported page-score mode: {self.page_score_mode}")
        if self.score_policy not in (
            "full_exact",
            "proxy_only",
            "proxy_exact_refine",
            "sparse_exact",
        ):
            raise ValueError(f"unsupported score policy: {self.score_policy}")
        if self.proxy_dtype not in _PROXY_DTYPES:
            raise ValueError(f"unsupported proxy dtype: {self.proxy_dtype}")

    @property
    def page_budget(self) -> int:
        """Number of physical pages covered by the requested token budget."""

        if self.exact_token_budget == 0:
            return 0
        return math.ceil(self.exact_token_budget / self.page_size)

    @property
    def torch_proxy_dtype(self) -> torch.dtype:
        try:
            return _PROXY_DTYPES[self.proxy_dtype]
        except KeyError as error:
            raise ValueError(f"unsupported proxy dtype: {self.proxy_dtype}") from error


@dataclass(frozen=True)
class GQAKProxyFactors:
    """Post-RoPE K/Q proxy factors with physical GQA ownership."""

    key_encoder: Tensor
    query_encoders: Tensor

    def validate(
        self,
        *,
        num_kv_heads: int,
        num_query_heads: int,
        head_dim: int,
        proxy_rank: int,
    ) -> None:
        if num_kv_heads <= 0 or num_query_heads % num_kv_heads:
            raise ValueError("query heads must divide evenly across physical KV heads")
        expected_key = (num_kv_heads, head_dim, proxy_rank)
        expected_query = (num_query_heads, head_dim, proxy_rank)
        if tuple(self.key_encoder.shape) != expected_key:
            raise ValueError(
                f"Key encoder must have shape {expected_key}, "
                f"got {tuple(self.key_encoder.shape)}"
            )
        if tuple(self.query_encoders.shape) != expected_query:
            raise ValueError(
                f"Query encoders must have shape {expected_query}, "
                f"got {tuple(self.query_encoders.shape)}"
            )
        if not self.key_encoder.is_floating_point() or not self.query_encoders.is_floating_point():
            raise TypeError("KRefine factors must be floating point")


@dataclass
class KRefineResult:
    output: Tensor
    selected_page_ids: Tensor
    selected_token_count: int
    proxy_scores: Tensor | None
    mixed_scores: Tensor | None
    running_lse: Tensor | None
    statistics: dict[str, float]


def _relative_l2(observed: Tensor, expected: Tensor) -> float:
    numerator = torch.linalg.vector_norm((observed - expected).double())
    denominator = torch.linalg.vector_norm(expected.double()).clamp_min(1.0e-300)
    return float(numerator / denominator)


def _pearson(observed: Tensor, expected: Tensor) -> float:
    observed = observed.double().flatten()
    expected = expected.double().flatten()
    observed = observed - observed.mean()
    expected = expected - expected.mean()
    denominator = (
        torch.linalg.vector_norm(observed) * torch.linalg.vector_norm(expected)
    )
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(observed, expected) else 0.0
    return float(torch.dot(observed, expected) / denominator)


def k_refine_quality_statistics(
    full_exact: KRefineResult,
    candidate: KRefineResult,
    *,
    config: KProxyConfig,
    head_dim: int,
    num_kv_heads: int,
    attention_mask: Tensor | None = None,
    decoder: Tensor | None = None,
    attention_top_k: int = 10,
) -> dict[str, float]:
    """Compare a materialized KRefine result with a full-exact baseline.

    Score RMSE is reported in unscaled raw-QK units. Logical storage/work
    counts remain in ``candidate.statistics`` and are not timing claims.
    """

    if full_exact.mixed_scores is None or candidate.mixed_scores is None:
        raise ValueError("quality statistics require materialized score tensors")
    if candidate.proxy_scores is None:
        raise ValueError("candidate proxy scores are required")
    exact_scores = full_exact.mixed_scores.float()
    mixed_scores = candidate.mixed_scores.float()
    proxy_scores = candidate.proxy_scores.float()
    if exact_scores.shape != mixed_scores.shape or exact_scores.shape != proxy_scores.shape:
        raise ValueError("quality score geometries differ")
    batch, query_heads, sequence = map(int, exact_scores.shape)
    valid, bias = _attention_mask(
        attention_mask,
        batch=batch,
        query_heads=query_heads,
        sequence=sequence,
        device=exact_scores.device,
    )
    raw_exact_scores = exact_scores - bias
    raw_proxy_scores = proxy_scores - bias
    raw_mixed_scores = mixed_scores - bias
    valid_exact = raw_exact_scores[valid]
    valid_proxy = raw_proxy_scores[valid]
    finite_mixed = valid & torch.isfinite(raw_mixed_scores)
    valid_mixed = raw_mixed_scores[finite_mixed]
    mixed_exact = raw_exact_scores[finite_mixed]
    scale_to_raw = math.sqrt(head_dim)

    proxy_error = (valid_proxy - valid_exact) * scale_to_raw
    mixed_error = (valid_mixed - mixed_exact) * scale_to_raw
    exact_raw = valid_exact * scale_to_raw
    exact_norm = torch.linalg.vector_norm(exact_raw.double()).clamp_min(1.0e-300)
    metrics = {
        "proxy_raw_score_rmse": float(proxy_error.double().square().mean().sqrt()),
        "proxy_raw_score_relative_l2": float(
            torch.linalg.vector_norm(proxy_error.double()) / exact_norm
        ),
        "mixed_raw_score_rmse": float(mixed_error.double().square().mean().sqrt()),
        "mixed_raw_score_relative_l2": float(
            torch.linalg.vector_norm(mixed_error.double())
            / torch.linalg.vector_norm(
                (mixed_exact * scale_to_raw).double()
            ).clamp_min(1.0e-300)
        ),
        "mixed_score_evaluated_token_fraction": float(
            finite_mixed.sum() / valid.sum().clamp_min(1)
        ),
        "proxy_score_pearson": _pearson(valid_proxy, valid_exact),
        "mixed_score_pearson": _pearson(valid_mixed, mixed_exact),
    }

    selected = torch.zeros(
        batch,
        num_kv_heads,
        math.ceil(sequence / config.page_size),
        dtype=torch.bool,
        device=exact_scores.device,
    )
    for batch_index in range(batch):
        for group in range(num_kv_heads):
            pages = candidate.selected_page_ids[batch_index, group]
            pages = pages[pages >= 0]
            selected[batch_index, group, pages] = True
    selected_tokens = _selected_token_mask(selected, valid, config.page_size)
    heads_per_group = query_heads // num_kv_heads
    exact_page_scores = _page_statistics(
        exact_scores,
        valid,
        page_size=config.page_size,
        page_score_mode=config.page_score_mode,
    ).reshape(batch, num_kv_heads, heads_per_group, -1).amax(dim=2)
    page_recalls = []
    token_recalls = []
    for batch_index in range(batch):
        for group in range(num_kv_heads):
            count = int(selected[batch_index, group].sum())
            if count:
                finite_pages = torch.isfinite(exact_page_scores[batch_index, group])
                exact_order = torch.argsort(
                    exact_page_scores[batch_index, group], descending=True, stable=True
                )
                exact_top = exact_order[finite_pages[exact_order]][:count]
                page_recalls.append(
                    float(selected[batch_index, group, exact_top].float().mean())
                )
            head_start = group * heads_per_group
            head_stop = head_start + heads_per_group
            for head in range(head_start, head_stop):
                selected_count = int(selected_tokens[batch_index, head].sum())
                if selected_count:
                    valid_positions = torch.nonzero(
                        valid[batch_index, head], as_tuple=False
                    ).flatten()
                    order = torch.argsort(
                        exact_scores[batch_index, head, valid_positions],
                        descending=True,
                        stable=True,
                    )
                    exact_top = valid_positions[order[:selected_count]]
                    token_recalls.append(
                        float(selected_tokens[batch_index, head, exact_top].float().mean())
                    )
    metrics["exact_top_page_recall"] = (
        sum(page_recalls) / len(page_recalls) if page_recalls else 0.0
    )
    metrics["exact_top_token_recall"] = (
        sum(token_recalls) / len(token_recalls) if token_recalls else 0.0
    )

    exact_probability = torch.softmax(exact_scores, dim=-1, dtype=torch.float32)
    mixed_probability = torch.softmax(mixed_scores, dim=-1, dtype=torch.float32)
    metrics["exact_attention_mass_selected"] = float(
        (exact_probability * selected_tokens).sum()
        / exact_probability.sum().clamp_min(1.0e-30)
    )
    safe_exact = exact_probability.clamp_min(1.0e-30)
    safe_mixed = mixed_probability.clamp_min(1.0e-30)
    metrics["attention_kl_exact_to_mixed"] = float(
        (safe_exact * (safe_exact.log() - safe_mixed.log())).sum(dim=-1).mean()
    )
    exact_entropy = -(safe_exact * safe_exact.log()).sum(dim=-1)
    mixed_entropy = -(safe_mixed * safe_mixed.log()).sum(dim=-1)
    metrics["attention_entropy_difference"] = float(
        (mixed_entropy - exact_entropy).mean()
    )
    metrics["attention_max_probability_error"] = float(
        (mixed_probability - exact_probability).abs().max()
    )
    overlaps = []
    for batch_index in range(batch):
        for head in range(query_heads):
            positions = torch.nonzero(valid[batch_index, head], as_tuple=False).flatten()
            k = min(int(attention_top_k), len(positions))
            if k:
                exact_top = positions[
                    torch.topk(exact_probability[batch_index, head, positions], k).indices
                ]
                mixed_top = positions[
                    torch.topk(mixed_probability[batch_index, head, positions], k).indices
                ]
                overlaps.append(
                    float(torch.isin(exact_top, mixed_top).float().mean())
                )
    metrics["attention_top_k_overlap"] = sum(overlaps) / len(overlaps)
    metrics["c1_latent_relative_l2"] = _relative_l2(
        candidate.output, full_exact.output
    )
    if decoder is not None:
        if decoder.ndim == 4:
            decoder = decoder.reshape(query_heads, decoder.shape[-2], decoder.shape[-1])
        if tuple(decoder.shape[:2]) != (query_heads, int(candidate.output.shape[-1])):
            raise ValueError("decoder must be [query heads, value rank, output dim]")
        exact_output = torch.einsum(
            "bhqv,hvo->bqo", full_exact.output.float(), decoder.float()
        )
        candidate_output = torch.einsum(
            "bhqv,hvo->bqo", candidate.output.float(), decoder.float()
        )
        metrics["c1_decoded_output_relative_l2"] = _relative_l2(
            candidate_output, exact_output
        )
        metrics["c1_decoded_output_max_absolute_error"] = float(
            (candidate_output - exact_output).abs().max()
        )
        cosine = torch.nn.functional.cosine_similarity(
            candidate_output.flatten(1), exact_output.flatten(1), dim=-1
        )
        metrics["c1_decoded_output_cosine_similarity"] = float(cosine.mean())
    return metrics


class ExactKeyPageStore(Protocol):
    """Storage boundary for exact post-RoPE Key pages."""

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
        """Return pages in request order with shape ``[requests, page, D]``."""


class GPUExactKeyPageStore:
    """Exact-Key store backed by one GPU or CPU tensor for oracle testing."""

    def __init__(self, exact_key: Tensor, *, layer_idx: int = 0) -> None:
        if exact_key.ndim != 4:
            raise ValueError("exact Key must be [batch, KV heads, sequence, head dim]")
        if not exact_key.is_floating_point():
            raise TypeError("exact Key must be floating point")
        self.exact_key = exact_key
        self.layer_idx = int(layer_idx)
        self.last_request_count = 0
        self.last_unique_request_count = 0

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
        if page_size <= 0:
            raise ValueError("page size must be positive")
        if not (
            batch_indices.ndim == kv_head_indices.ndim == page_ids.ndim == 1
            and len(batch_indices) == len(kv_head_indices) == len(page_ids)
        ):
            raise ValueError("page request indices must be equal-length vectors")
        requests = [
            (int(batch), int(head), int(page))
            for batch, head, page in zip(
                batch_indices.detach().cpu().tolist(),
                kv_head_indices.detach().cpu().tolist(),
                page_ids.detach().cpu().tolist(),
            )
        ]
        self.last_request_count = len(requests)
        unique: dict[tuple[int, int, int], int] = {}
        pages: list[Tensor] = []
        request_to_unique = []
        batch_count, head_count, sequence, head_dim = map(int, self.exact_key.shape)
        page_count = math.ceil(sequence / page_size)
        for request in requests:
            batch, head, page = request
            if not (0 <= batch < batch_count and 0 <= head < head_count):
                raise IndexError(f"invalid exact-Key page owner: {request}")
            if not 0 <= page < page_count:
                raise IndexError(f"invalid exact-Key page id: {page}")
            if request not in unique:
                unique[request] = len(pages)
                start = page * page_size
                stop = min(start + page_size, sequence)
                value = self.exact_key[batch, head, start:stop]
                if stop - start != page_size:
                    value = torch.nn.functional.pad(
                        value,
                        (0, 0, 0, page_size - (stop - start)),
                    )
                pages.append(value.to(device=device))
            request_to_unique.append(unique[request])
        self.last_unique_request_count = len(pages)
        if not requests:
            return torch.empty(
                0,
                page_size,
                head_dim,
                dtype=self.exact_key.dtype,
                device=device,
            )
        bank = torch.stack(pages)
        order = torch.tensor(request_to_unique, dtype=torch.long, device=bank.device)
        return bank.index_select(0, order)


def project_proxy_key(exact_key: Tensor, factors: GQAKProxyFactors) -> Tensor:
    """Project exact post-RoPE Keys once per physical KV head."""

    if exact_key.ndim != 4:
        raise ValueError("exact Key must be [batch, KV heads, sequence, head dim]")
    if tuple(factors.key_encoder.shape[:2]) != (
        int(exact_key.shape[1]),
        int(exact_key.shape[-1]),
    ):
        raise ValueError("Key encoder geometry is incompatible with exact Key")
    return torch.einsum(
        "bgsd,gdr->bgsr",
        exact_key,
        factors.key_encoder.to(device=exact_key.device, dtype=exact_key.dtype),
    )


def _validate_attention_geometry(
    query: Tensor,
    proxy_key: Tensor,
    c1_value: Tensor,
    factors: GQAKProxyFactors,
    config: KProxyConfig,
) -> tuple[int, int, int, int, int, int, int]:
    if query.ndim != 4 or int(query.shape[2]) != 1:
        raise ValueError("query must have shape [batch, query heads, 1, head dim]")
    if proxy_key.ndim != 4 or c1_value.ndim != 4:
        raise ValueError("proxy Key and C1 Value must be rank-four tensors")
    batch, query_heads, _, head_dim = map(int, query.shape)
    key_batch, kv_heads, sequence, proxy_rank = map(int, proxy_key.shape)
    value_rank = int(c1_value.shape[-1])
    if tuple(c1_value.shape[:3]) != (batch, kv_heads, sequence):
        raise ValueError("proxy Key and C1 Value geometry differs")
    if key_batch != batch:
        raise ValueError("query and cache batch dimensions differ")
    if sequence <= 0 or value_rank <= 0:
        raise ValueError("KRefine attention requires nonempty cache dimensions")
    config.validate(head_dim)
    if proxy_rank != config.proxy_rank:
        raise ValueError("proxy Key width differs from configured proxy rank")
    factors.validate(
        num_kv_heads=kv_heads,
        num_query_heads=query_heads,
        head_dim=head_dim,
        proxy_rank=proxy_rank,
    )
    if not (
        query.device == proxy_key.device == c1_value.device
        and factors.key_encoder.device == factors.query_encoders.device == query.device
    ):
        raise ValueError("query, proxy cache, C1 Value, and factors must share a device")
    return batch, query_heads, kv_heads, sequence, head_dim, proxy_rank, value_rank


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
        return valid, torch.zeros(batch, query_heads, sequence, device=device)
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


def _head_geometry(query_heads: int, kv_heads: int, device: torch.device) -> Tensor:
    heads_per_group = query_heads // kv_heads
    return torch.arange(kv_heads, device=device).repeat_interleave(heads_per_group)


def _proxy_query(
    query: Tensor,
    factors: GQAKProxyFactors,
    proxy_dtype: torch.dtype,
) -> Tensor:
    return torch.einsum(
        "bhd,hdr->bhr",
        query[:, :, 0].to(proxy_dtype),
        factors.query_encoders.to(dtype=proxy_dtype),
    )


def _page_statistics(
    proxy_scores: Tensor,
    valid: Tensor,
    *,
    page_size: int,
    page_score_mode: PageScoreMode,
) -> Tensor:
    batch, heads, sequence = map(int, proxy_scores.shape)
    page_count = math.ceil(sequence / page_size)
    statistics = torch.full(
        (batch, heads, page_count),
        -torch.inf,
        dtype=torch.float32,
        device=proxy_scores.device,
    )
    for page in range(page_count):
        start = page * page_size
        stop = min(start + page_size, sequence)
        scores = proxy_scores[:, :, start:stop].float().masked_fill(
            ~valid[:, :, start:stop], -torch.inf
        )
        if page_score_mode == "max":
            statistics[:, :, page] = scores.amax(dim=-1)
        else:
            statistics[:, :, page] = torch.logsumexp(scores, dim=-1)
    return statistics


def _select_pages(
    per_head_page_scores: Tensor,
    valid: Tensor,
    *,
    kv_heads: int,
    config: KProxyConfig,
) -> Tensor:
    batch, query_heads, page_count = map(int, per_head_page_scores.shape)
    sequence = int(valid.shape[-1])
    heads_per_group = query_heads // kv_heads
    grouped_scores = per_head_page_scores.reshape(
        batch, kv_heads, heads_per_group, page_count
    ).amax(dim=2)
    grouped_valid = valid.reshape(
        batch, kv_heads, heads_per_group, sequence
    ).any(dim=2)
    page_valid = torch.zeros(
        batch, kv_heads, page_count, dtype=torch.bool, device=valid.device
    )
    for page in range(page_count):
        start = page * config.page_size
        stop = min(start + config.page_size, sequence)
        page_valid[:, :, page] = grouped_valid[:, :, start:stop].any(dim=-1)

    selected = torch.zeros_like(page_valid)
    if config.score_policy == "proxy_only":
        return selected
    if config.score_policy == "full_exact":
        return page_valid

    for batch_index in range(batch):
        for group in range(kv_heads):
            valid_pages = torch.nonzero(page_valid[batch_index, group], as_tuple=False).flatten()
            if len(valid_pages) == 0:
                continue
            forced = torch.empty(0, dtype=torch.long, device=valid.device)
            if config.recent_exact_window:
                valid_tokens = torch.nonzero(
                    grouped_valid[batch_index, group], as_tuple=False
                ).flatten()
                last = int(valid_tokens[-1])
                threshold = max(0, last - config.recent_exact_window + 1)
                forced = torch.unique(
                    valid_tokens[valid_tokens >= threshold] // config.page_size,
                    sorted=True,
                )
                selected[batch_index, group, forced] = True
            remaining = max(config.page_budget - len(forced), 0)
            if remaining:
                candidates = valid_pages[~selected[batch_index, group, valid_pages]]
                if len(candidates):
                    candidate_scores = grouped_scores[batch_index, group, candidates]
                    # Stable sorting gives deterministic lower-page tie breaking.
                    order = torch.argsort(candidate_scores, descending=True, stable=True)
                    chosen = candidates[order[:remaining]]
                    selected[batch_index, group, chosen] = True
    return selected


def _selected_page_ids(selected: Tensor) -> Tensor:
    counts = selected.sum(dim=-1)
    width = int(counts.max().item()) if selected.numel() else 0
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


def _selected_token_mask(selected: Tensor, valid: Tensor, page_size: int) -> Tensor:
    batch, query_heads, sequence = map(int, valid.shape)
    kv_heads = int(selected.shape[1])
    head_to_kv = _head_geometry(query_heads, kv_heads, valid.device)
    page_ids = torch.arange(sequence, device=valid.device) // page_size
    expanded = selected.index_select(1, head_to_kv).gather(
        2, page_ids.view(1, 1, sequence).expand(batch, query_heads, sequence)
    )
    return expanded & valid


def _page_requests(selected: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    coordinates = torch.nonzero(selected, as_tuple=False)
    if len(coordinates) == 0:
        empty = torch.empty(0, dtype=torch.long, device=selected.device)
        return empty, empty, empty
    return coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]


def _exact_pages(
    page_store: ExactKeyPageStore,
    selected: Tensor,
    *,
    layer_idx: int,
    page_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    batches, groups, pages = _page_requests(selected)
    values = page_store.get_pages(
        layer_idx=layer_idx,
        batch_indices=batches,
        kv_head_indices=groups,
        page_ids=pages,
        page_size=page_size,
        device=device,
    )
    return torch.stack((batches, groups, pages), dim=-1), values


def _logical_statistics(
    *,
    proxy_key: Tensor,
    c1_value: Tensor,
    selected: Tensor,
    valid: Tensor,
    head_dim: int,
    query_heads: int,
) -> dict[str, float]:
    batch, kv_heads, _ = map(int, selected.shape)
    sequence = int(valid.shape[-1])
    physical_valid = valid.reshape(
        batch, kv_heads, query_heads // kv_heads, sequence
    ).any(dim=2)
    valid_tokens = int(physical_valid.sum().item())
    return {
        "resident_c1_value_bytes": float(c1_value.numel() * c1_value.element_size()),
        "resident_proxy_key_bytes": float(proxy_key.numel() * proxy_key.element_size()),
        "physical_valid_tokens": float(valid_tokens),
        "selected_pages": float(selected.sum().item()),
        "selected_tokens": 0.0,
        "selected_token_fraction": 0.0,
        "exact_key_bytes_consulted": 0.0,
        "proxy_qk_flops": float(2 * batch * query_heads * sequence * proxy_key.shape[-1]),
        "exact_refinement_qk_flops": 0.0,
        "full_exact_qk_flops": float(2 * batch * query_heads * sequence * head_dim),
        "estimated_key_bytes_avoided": 0.0,
    }


def _result_statistics(
    *,
    proxy_key: Tensor,
    c1_value: Tensor,
    selected: Tensor,
    valid: Tensor,
    page_size: int,
    head_dim: int,
    query_heads: int,
    exact_element_size: int,
) -> tuple[int, dict[str, float]]:
    stats = _logical_statistics(
        proxy_key=proxy_key,
        c1_value=c1_value,
        selected=selected,
        valid=valid,
        head_dim=head_dim,
        query_heads=query_heads,
    )
    batch, kv_heads, _ = map(int, selected.shape)
    sequence = int(valid.shape[-1])
    physical_valid = valid.reshape(
        batch, kv_heads, query_heads // kv_heads, sequence
    ).any(dim=2)
    physical_selected = torch.zeros_like(physical_valid)
    for page in range(int(selected.shape[-1])):
        start = page * page_size
        stop = min(start + page_size, sequence)
        physical_selected[:, :, start:stop] = selected[:, :, page].unsqueeze(-1)
    selected_tokens = int((physical_selected & physical_valid).sum().item())
    valid_tokens = max(int(physical_valid.sum().item()), 1)
    exact_bytes = selected_tokens * head_dim * exact_element_size
    full_exact_bytes = int(physical_valid.sum().item()) * head_dim * exact_element_size
    stats.update(
        {
            "selected_tokens": float(selected_tokens),
            "selected_token_fraction": selected_tokens / valid_tokens,
            "exact_key_bytes_consulted": float(exact_bytes),
            "exact_refinement_qk_flops": float(
                2 * selected_tokens * (query_heads // kv_heads) * head_dim
            ),
            "estimated_key_bytes_avoided": float(max(full_exact_bytes - exact_bytes, 0)),
        }
    )
    return selected_tokens, stats


def c1_k_refine_attention_reference(
    query: Tensor,
    exact_key: Tensor | None,
    proxy_key: Tensor,
    c1_value: Tensor,
    factors: GQAKProxyFactors,
    config: KProxyConfig,
    attention_mask: Tensor | None = None,
    *,
    page_store: ExactKeyPageStore | None = None,
    layer_idx: int = 0,
) -> KRefineResult:
    """Materialized query-length-one C1-KRefine correctness oracle."""

    batch, query_heads, kv_heads, sequence, head_dim, _, _ = (
        _validate_attention_geometry(query, proxy_key, c1_value, factors, config)
    )
    if page_store is None:
        if exact_key is None:
            raise ValueError("exact Key or an exact-Key page store is required")
        page_store = GPUExactKeyPageStore(exact_key, layer_idx=layer_idx)
    if exact_key is not None and tuple(exact_key.shape) != (
        batch,
        kv_heads,
        sequence,
        head_dim,
    ):
        raise ValueError("exact Key geometry is incompatible")
    proxy_key = proxy_key.to(config.torch_proxy_dtype)
    valid, bias = _attention_mask(
        attention_mask,
        batch=batch,
        query_heads=query_heads,
        sequence=sequence,
        device=query.device,
    )
    head_to_kv = _head_geometry(query_heads, kv_heads, query.device)
    projected_query = _proxy_query(query, factors, config.torch_proxy_dtype).float()
    expanded_proxy_key = proxy_key.index_select(1, head_to_kv).float()
    proxy_scores = torch.einsum("bhr,bhsr->bhs", projected_query, expanded_proxy_key)
    proxy_scores.mul_(head_dim**-0.5).add_(bias)
    page_scores = _page_statistics(
        proxy_scores,
        valid,
        page_size=config.page_size,
        page_score_mode=config.page_score_mode,
    )
    selected = _select_pages(page_scores, valid, kv_heads=kv_heads, config=config)
    selected_tokens = _selected_token_mask(selected, valid, config.page_size)

    if config.score_policy == "sparse_exact":
        selected_per_head = selected_tokens.sum(dim=-1)
        if torch.any(selected_per_head == 0):
            raise ValueError(
                "sparse_exact requires at least one selected valid token per query head"
            )

    mixed_scores = proxy_scores.clone()
    requests, pages = _exact_pages(
        page_store,
        selected,
        layer_idx=layer_idx,
        page_size=config.page_size,
        device=query.device,
    )
    heads_per_group = query_heads // kv_heads
    for request, exact_page in zip(requests, pages):
        batch_index, group, page = map(int, request.tolist())
        start = page * config.page_size
        stop = min(start + config.page_size, sequence)
        head_start = group * heads_per_group
        head_stop = head_start + heads_per_group
        exact_scores = torch.einsum(
            "hd,sd->hs",
            query[batch_index, head_start:head_stop, 0].float(),
            exact_page[: stop - start].float(),
        ).mul_(head_dim**-0.5)
        exact_scores.add_(bias[batch_index, head_start:head_stop, start:stop])
        mixed_scores[batch_index, head_start:head_stop, start:stop] = exact_scores
    if config.score_policy == "sparse_exact":
        mixed_scores = mixed_scores.masked_fill(~selected_tokens, -torch.inf)
    else:
        mixed_scores = mixed_scores.masked_fill(~valid, -torch.inf)
    masked_proxy = proxy_scores.masked_fill(~valid, -torch.inf)
    probabilities = torch.softmax(mixed_scores, dim=-1, dtype=torch.float32)
    expanded_value = c1_value.index_select(1, head_to_kv).float()
    output = torch.einsum("bhs,bhsv->bhv", probabilities, expanded_value)
    output = output[:, :, None, :].to(c1_value.dtype)
    selected_token_count, statistics = _result_statistics(
        proxy_key=proxy_key,
        c1_value=c1_value,
        selected=selected,
        valid=valid,
        page_size=config.page_size,
        head_dim=head_dim,
        query_heads=query_heads,
        exact_element_size=(exact_key.element_size() if exact_key is not None else query.element_size()),
    )
    return KRefineResult(
        output=output,
        selected_page_ids=_selected_page_ids(selected),
        selected_token_count=selected_token_count,
        proxy_scores=masked_proxy,
        mixed_scores=mixed_scores,
        running_lse=None,
        statistics=statistics,
    )


def c1_k_refine_attention_streaming(
    query: Tensor,
    exact_key: Tensor | None,
    proxy_key: Tensor,
    c1_value: Tensor,
    factors: GQAKProxyFactors,
    config: KProxyConfig,
    attention_mask: Tensor | None = None,
    *,
    page_store: ExactKeyPageStore | None = None,
    layer_idx: int = 0,
) -> KRefineResult:
    """Two-pass page-streaming oracle with FP32 online softmax state."""

    batch, query_heads, kv_heads, sequence, head_dim, _, value_rank = (
        _validate_attention_geometry(query, proxy_key, c1_value, factors, config)
    )
    if page_store is None:
        if exact_key is None:
            raise ValueError("exact Key or an exact-Key page store is required")
        page_store = GPUExactKeyPageStore(exact_key, layer_idx=layer_idx)
    if exact_key is not None and tuple(exact_key.shape) != (
        batch,
        kv_heads,
        sequence,
        head_dim,
    ):
        raise ValueError("exact Key geometry is incompatible")
    proxy_key = proxy_key.to(config.torch_proxy_dtype)
    valid, bias = _attention_mask(
        attention_mask,
        batch=batch,
        query_heads=query_heads,
        sequence=sequence,
        device=query.device,
    )
    head_to_kv = _head_geometry(query_heads, kv_heads, query.device)
    projected_query = _proxy_query(query, factors, config.torch_proxy_dtype).float()
    page_count = math.ceil(sequence / config.page_size)

    # Pass 1 retains only one scalar per query head and physical page.
    page_statistics = torch.full(
        (batch, query_heads, page_count),
        -torch.inf,
        dtype=torch.float32,
        device=query.device,
    )
    for page in range(page_count):
        start = page * config.page_size
        stop = min(start + config.page_size, sequence)
        expanded_key = proxy_key[:, :, start:stop].index_select(1, head_to_kv).float()
        scores = torch.einsum("bhr,bhsr->bhs", projected_query, expanded_key)
        scores.mul_(head_dim**-0.5).add_(bias[:, :, start:stop])
        scores.masked_fill_(~valid[:, :, start:stop], -torch.inf)
        if config.page_score_mode == "max":
            page_statistics[:, :, page] = scores.amax(dim=-1)
        else:
            page_statistics[:, :, page] = torch.logsumexp(scores, dim=-1)
    selected = _select_pages(page_statistics, valid, kv_heads=kv_heads, config=config)
    requests, exact_pages = _exact_pages(
        page_store,
        selected,
        layer_idx=layer_idx,
        page_size=config.page_size,
        device=query.device,
    )
    exact_lookup = {
        (int(request[0]), int(request[1]), int(request[2])): exact_page
        for request, exact_page in zip(requests, exact_pages)
    }

    running_max = torch.full(
        (batch, query_heads), -torch.inf, dtype=torch.float32, device=query.device
    )
    running_sum = torch.zeros_like(running_max)
    numerator = torch.zeros(
        batch, query_heads, value_rank, dtype=torch.float32, device=query.device
    )
    heads_per_group = query_heads // kv_heads
    # Pass 2 retains at most one score page.
    for page in range(page_count):
        start = page * config.page_size
        stop = min(start + config.page_size, sequence)
        expanded_key = proxy_key[:, :, start:stop].index_select(1, head_to_kv).float()
        scores = torch.einsum("bhr,bhsr->bhs", projected_query, expanded_key)
        scores.mul_(head_dim**-0.5).add_(bias[:, :, start:stop])
        for batch_index in range(batch):
            for group in range(kv_heads):
                if not bool(selected[batch_index, group, page]):
                    continue
                exact_page = exact_lookup[(batch_index, group, page)]
                head_start = group * heads_per_group
                head_stop = head_start + heads_per_group
                exact_scores = torch.einsum(
                    "hd,sd->hs",
                    query[batch_index, head_start:head_stop, 0].float(),
                    exact_page[: stop - start].float(),
                ).mul_(head_dim**-0.5)
                exact_scores.add_(
                    bias[batch_index, head_start:head_stop, start:stop]
                )
                scores[batch_index, head_start:head_stop] = exact_scores
        block_valid = valid[:, :, start:stop]
        if config.score_policy == "sparse_exact":
            selected_heads = selected[:, :, page].index_select(1, head_to_kv)
            block_valid = block_valid & selected_heads.unsqueeze(-1)
        scores.masked_fill_(~block_valid, -torch.inf)
        block_max = scores.amax(dim=-1)
        finite_block = torch.isfinite(block_max)
        safe_block_max = torch.where(finite_block, block_max, torch.zeros_like(block_max))
        weights = torch.exp(scores - safe_block_max.unsqueeze(-1))
        weights = torch.where(block_valid, weights, torch.zeros_like(weights))
        block_sum = weights.sum(dim=-1)
        expanded_value = c1_value[:, :, start:stop].index_select(1, head_to_kv).float()
        block_numerator = torch.einsum("bhs,bhsv->bhv", weights, expanded_value)

        next_max = torch.maximum(running_max, block_max)
        finite_old = torch.isfinite(running_max)
        finite_next = torch.isfinite(next_max)
        old_scale = torch.where(
            finite_old & finite_next,
            torch.exp(running_max - next_max),
            torch.zeros_like(next_max),
        )
        block_scale = torch.where(
            finite_block & finite_next,
            torch.exp(block_max - next_max),
            torch.zeros_like(next_max),
        )
        running_sum = old_scale * running_sum + block_scale * block_sum
        numerator = (
            old_scale.unsqueeze(-1) * numerator
            + block_scale.unsqueeze(-1) * block_numerator
        )
        running_max = next_max
    if torch.any(running_sum == 0):
        raise ValueError("every query head must have at least one valid attention token")
    output = (numerator / running_sum.unsqueeze(-1))[:, :, None, :].to(c1_value.dtype)
    running_lse = running_max + torch.log(running_sum)
    selected_token_count, statistics = _result_statistics(
        proxy_key=proxy_key,
        c1_value=c1_value,
        selected=selected,
        valid=valid,
        page_size=config.page_size,
        head_dim=head_dim,
        query_heads=query_heads,
        exact_element_size=(exact_key.element_size() if exact_key is not None else query.element_size()),
    )
    return KRefineResult(
        output=output,
        selected_page_ids=_selected_page_ids(selected),
        selected_token_count=selected_token_count,
        proxy_scores=None,
        mixed_scores=None,
        running_lse=running_lse,
        statistics=statistics,
    )
