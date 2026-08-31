"""Training-free raw-score fitting for post-RoPE C1 Key proxies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from basisserve.core.c1_k_output_closure import (
    canonicalize_c1k_factors,
    conjugate_gradient,
)
from basisserve.core.c1_k_refine import GQAKProxyFactors
from basisserve.core.kq_svd import key_svd_projector


Initialization = Literal["pca_shared", "als"]


@dataclass(frozen=True)
class KProxyPairSamples:
    """Independent valid Q/K pairs used by raw-score regression."""

    query: Tensor
    key: Tensor
    query_head: Tensor
    weight: Tensor | None = None

    def validate(self, *, num_query_heads: int) -> None:
        if self.query.ndim != 2 or self.key.shape != self.query.shape:
            raise ValueError("sampled Query and Key must have identical [pairs, D] shapes")
        if self.query_head.shape != (len(self.query),):
            raise ValueError("query-head ids must have one entry per pair")
        if len(self.query) == 0:
            raise ValueError("at least one valid Q/K pair is required")
        if not self.query.is_floating_point() or not self.key.is_floating_point():
            raise TypeError("sampled Query and Key must be floating point")
        if self.query.device != self.key.device or self.query_head.device != self.query.device:
            raise ValueError("pair samples must share a device")
        if self.weight is not None:
            if self.weight.shape != (len(self.query),) or self.weight.device != self.query.device:
                raise ValueError("sample weights must have shape [pairs] on the sample device")
            if not self.weight.is_floating_point() or torch.any(self.weight < 0):
                raise ValueError("sample weights must be nonnegative floating-point values")
        if torch.any(self.query_head < 0) or torch.any(self.query_head >= num_query_heads):
            raise ValueError("sample query-head id is out of range")


@dataclass(frozen=True)
class KProxyFitResult:
    factors: GQAKProxyFactors
    objective_history: tuple[float, ...]
    cg_diagnostics: tuple[dict[str, float | int | bool | str], ...]


@dataclass(frozen=True)
class CausalPairCapture:
    """Sampled pair values plus provenance needed to audit causality."""

    samples: KProxyPairSamples
    sequence_id: Tensor
    query_position: Tensor
    key_position: Tensor


def sample_valid_causal_pairs(
    query: Tensor,
    key: Tensor,
    *,
    attention_mask: Tensor | None = None,
    queries_per_sequence: int,
    keys_per_query: int,
    recent_pair_fraction: float = 0.25,
    top_score_pair_fraction: float = 0.25,
    random_seed: int = 0,
) -> CausalPairCapture:
    """Sample a reproducible recent/top/uniform mixture of valid causal pairs.

    ``query`` and ``key`` are post-RoPE tensors with shapes ``[B,Hq,S,D]`` and
    ``[B,Hkv,S,D]``. Query positions are sampled uniformly; for every selected
    position and query head, Keys are drawn from its valid causal prefix.
    """

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("post-RoPE Query and Key must be rank-four tensors")
    batch, query_heads, sequence, head_dim = map(int, query.shape)
    if tuple(key.shape[:1] + key.shape[2:]) != (batch, sequence, head_dim):
        raise ValueError("post-RoPE Query and Key geometry differs")
    kv_heads = int(key.shape[1])
    if kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query heads must divide evenly across physical KV heads")
    if queries_per_sequence <= 0 or keys_per_query <= 0:
        raise ValueError("pair-sampling counts must be positive")
    if (
        not 0 <= recent_pair_fraction <= 1
        or not 0 <= top_score_pair_fraction <= 1
        or recent_pair_fraction + top_score_pair_fraction > 1
    ):
        raise ValueError("recent/top pair fractions must be nonnegative and sum to at most one")
    if query.device != key.device or not query.is_floating_point() or not key.is_floating_point():
        raise ValueError("post-RoPE Query and Key must be floating point on one device")
    if attention_mask is None:
        valid = torch.ones(batch, sequence, dtype=torch.bool, device=query.device)
    else:
        valid = attention_mask.to(device=query.device)
        if valid.ndim == 4:
            valid = valid[:, 0, 0]
        elif valid.ndim == 3:
            valid = valid[:, 0]
        if valid.shape != (batch, sequence):
            raise ValueError("sampling mask must reduce to [batch, sequence]")
        if valid.dtype != torch.bool:
            valid = torch.isfinite(valid) & (valid > -1.0e20)

    generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
    heads_per_group = query_heads // kv_heads
    sampled_query: list[Tensor] = []
    sampled_key: list[Tensor] = []
    sampled_head: list[int] = []
    sampled_sequence: list[int] = []
    sampled_query_position: list[int] = []
    sampled_key_position: list[int] = []
    for batch_index in range(batch):
        valid_positions = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
        if len(valid_positions) == 0:
            continue
        order = torch.randperm(len(valid_positions), generator=generator)
        query_positions = valid_positions.detach().cpu()[
            order[: min(queries_per_sequence, len(valid_positions))]
        ].tolist()
        for query_position in query_positions:
            candidates = torch.nonzero(
                valid[batch_index, : query_position + 1], as_tuple=False
            ).flatten()
            total = min(keys_per_query, len(candidates))
            recent_count = min(round(total * recent_pair_fraction), total)
            top_count = min(round(total * top_score_pair_fraction), total - recent_count)
            for query_head in range(query_heads):
                group = query_head // heads_per_group
                chosen: list[int] = []
                if recent_count:
                    chosen.extend(
                        candidates[-recent_count:].detach().cpu().tolist()
                    )
                remaining = candidates[
                    ~torch.isin(
                        candidates,
                        torch.tensor(
                            chosen, dtype=candidates.dtype, device=candidates.device
                        ),
                    )
                ]
                if top_count:
                    scores = key[batch_index, group, remaining].float() @ query[
                        batch_index, query_head, query_position
                    ].float()
                    top = remaining[torch.argsort(scores, descending=True, stable=True)[:top_count]]
                    chosen.extend(top.detach().cpu().tolist())
                remaining_count = total - len(chosen)
                if remaining_count:
                    remaining = candidates[
                        ~torch.isin(
                            candidates,
                            torch.tensor(
                                chosen, dtype=candidates.dtype, device=candidates.device
                            ),
                        )
                    ]
                    random_order = torch.randperm(len(remaining), generator=generator)
                    chosen.extend(
                        remaining.detach().cpu()[random_order[:remaining_count]].tolist()
                    )
                for key_position in sorted(chosen):
                    sampled_query.append(query[batch_index, query_head, query_position])
                    sampled_key.append(key[batch_index, group, key_position])
                    sampled_head.append(query_head)
                    sampled_sequence.append(batch_index)
                    sampled_query_position.append(query_position)
                    sampled_key_position.append(key_position)
    if not sampled_query:
        raise ValueError("sampling found no valid causal Q/K pairs")
    index_device = query.device
    return CausalPairCapture(
        samples=KProxyPairSamples(
            query=torch.stack(sampled_query),
            key=torch.stack(sampled_key),
            query_head=torch.tensor(sampled_head, dtype=torch.long, device=index_device),
        ),
        sequence_id=torch.tensor(sampled_sequence, dtype=torch.long, device=index_device),
        query_position=torch.tensor(
            sampled_query_position, dtype=torch.long, device=index_device
        ),
        key_position=torch.tensor(sampled_key_position, dtype=torch.long, device=index_device),
    )


def raw_score_objective(
    samples: KProxyPairSamples,
    factors: GQAKProxyFactors,
    *,
    num_kv_heads: int,
    ridge: float = 0.0,
) -> float:
    """Evaluate the weighted raw-QK score objective in FP64."""

    num_query_heads = int(factors.query_encoders.shape[0])
    samples.validate(num_query_heads=num_query_heads)
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads:
        raise ValueError("invalid GQA ownership")
    heads_per_group = num_query_heads // num_kv_heads
    group = torch.div(samples.query_head, heads_per_group, rounding_mode="floor")
    query_factor = factors.query_encoders.index_select(0, samples.query_head).double()
    key_factor = factors.key_encoder.index_select(0, group).double()
    projected_query = torch.einsum("nd,ndr->nr", samples.query.double(), query_factor)
    projected_key = torch.einsum("nd,ndr->nr", samples.key.double(), key_factor)
    target = (samples.query.double() * samples.key.double()).sum(dim=-1)
    residual = target - (projected_query * projected_key).sum(dim=-1)
    weight = (
        torch.ones_like(residual)
        if samples.weight is None
        else samples.weight.double()
    )
    value = (weight * residual.square()).sum()
    if ridge:
        value = value + ridge * (
            factors.key_encoder.double().square().sum()
            + factors.query_encoders.double().square().sum()
        )
    return float(value)


def canonicalize_gqa_k_proxy_factors(
    factors: GQAKProxyFactors,
) -> GQAKProxyFactors:
    """QR-fix the Key gauge while preserving every proxy QK score."""

    kv_heads = int(factors.key_encoder.shape[0])
    query_heads = int(factors.query_encoders.shape[0])
    if query_heads % kv_heads:
        raise ValueError("invalid GQA ownership")
    heads_per_group = query_heads // kv_heads
    grouped_query = factors.query_encoders.reshape(
        kv_heads,
        heads_per_group,
        *factors.query_encoders.shape[1:],
    )
    key, query = canonicalize_c1k_factors(factors.key_encoder, grouped_query)
    return GQAKProxyFactors(
        key.to(factors.key_encoder.dtype),
        query.to(factors.query_encoders.dtype).reshape_as(factors.query_encoders),
    )


def _initialize_factors(
    samples: KProxyPairSamples,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    proxy_rank: int,
    work_dtype: torch.dtype,
) -> GQAKProxyFactors:
    heads_per_group = num_query_heads // num_kv_heads
    groups = torch.div(samples.query_head, heads_per_group, rounding_mode="floor")
    key_encoders = []
    for group in range(num_kv_heads):
        group_key = samples.key[groups == group].to(work_dtype)
        if len(group_key) == 0:
            raise ValueError(f"KV group {group} has no sampled Keys")
        gram = group_key.mT @ group_key
        encoder, _ = key_svd_projector(gram, proxy_rank)
        key_encoders.append(encoder.to(work_dtype))
    key_encoder = torch.stack(key_encoders)
    query_encoder = key_encoder.repeat_interleave(heads_per_group, dim=0).clone()
    return GQAKProxyFactors(key_encoder, query_encoder)


def _solve_query_factor(
    query: Tensor,
    key: Tensor,
    weight: Tensor,
    key_encoder: Tensor,
    *,
    ridge: float,
    max_iterations: int,
    relative_tolerance: float,
) -> tuple[Tensor, dict[str, float | int | bool]]:
    z = key @ key_encoder
    target = (query * key).sum(dim=-1)
    right_hand_side = torch.einsum("n,n,nd,nr->dr", weight, target, query, z)

    def operator(direction: Tensor) -> Tensor:
        prediction = torch.einsum("nd,dr,nr->n", query, direction, z)
        image = torch.einsum("n,n,nd,nr->dr", weight, prediction, query, z)
        if ridge:
            image.add_(direction, alpha=ridge)
        return image

    return conjugate_gradient(
        operator,
        right_hand_side,
        max_iterations=max_iterations,
        relative_tolerance=relative_tolerance,
    )


def _solve_key_factor(
    query: Tensor,
    key: Tensor,
    query_factor: Tensor,
    weight: Tensor,
    *,
    ridge: float,
    max_iterations: int,
    relative_tolerance: float,
) -> tuple[Tensor, dict[str, float | int | bool]]:
    a = torch.einsum("nd,ndr->nr", query, query_factor)
    target = (query * key).sum(dim=-1)
    right_hand_side = torch.einsum("n,n,nd,nr->dr", weight, target, key, a)

    def operator(direction: Tensor) -> Tensor:
        prediction = torch.einsum("nd,dr,nr->n", key, direction, a)
        image = torch.einsum("n,n,nd,nr->dr", weight, prediction, key, a)
        if ridge:
            image.add_(direction, alpha=ridge)
        return image

    return conjugate_gradient(
        operator,
        right_hand_side,
        max_iterations=max_iterations,
        relative_tolerance=relative_tolerance,
    )


def fit_gqa_k_proxy(
    samples: KProxyPairSamples,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    proxy_rank: int,
    initialization: Initialization = "als",
    als_sweeps: int = 3,
    ridge: float = 1.0e-5,
    cg_iterations: int = 32,
    cg_tolerance: float = 1.0e-6,
    accumulation_dtype: torch.dtype = torch.float32,
) -> KProxyFitResult:
    """Fit one shared Key factor per KV head and one Query factor per Q head."""

    samples.validate(num_query_heads=num_query_heads)
    head_dim = int(samples.query.shape[-1])
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads:
        raise ValueError("query heads must divide evenly across physical KV heads")
    if not 1 <= proxy_rank <= head_dim:
        raise ValueError(f"proxy rank must be in [1, {head_dim}]")
    if initialization not in ("pca_shared", "als"):
        raise ValueError(f"unsupported fitting mode: {initialization}")
    if als_sweeps < 0 or ridge < 0 or cg_iterations <= 0 or cg_tolerance < 0:
        raise ValueError("invalid ALS/CG controls")
    if accumulation_dtype not in (torch.float32, torch.float64):
        raise ValueError("fitting accumulation dtype must be float32 or float64")
    samples = KProxyPairSamples(
        query=samples.query.to(accumulation_dtype),
        key=samples.key.to(accumulation_dtype),
        query_head=samples.query_head,
        weight=None if samples.weight is None else samples.weight.to(accumulation_dtype),
    )
    weights = (
        torch.ones(len(samples.query), dtype=accumulation_dtype, device=samples.query.device)
        if samples.weight is None
        else samples.weight
    )
    factors = _initialize_factors(
        samples,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        proxy_rank=proxy_rank,
        work_dtype=accumulation_dtype,
    )
    history = [
        raw_score_objective(samples, factors, num_kv_heads=num_kv_heads, ridge=ridge)
    ]
    diagnostics: list[dict[str, float | int | bool | str]] = []
    if initialization == "pca_shared":
        return KProxyFitResult(factors, tuple(history), tuple(diagnostics))

    heads_per_group = num_query_heads // num_kv_heads
    group_ids = torch.div(samples.query_head, heads_per_group, rounding_mode="floor")
    for sweep in range(als_sweeps):
        query_encoders = factors.query_encoders.clone()
        for head in range(num_query_heads):
            keep = samples.query_head == head
            if not torch.any(keep):
                raise ValueError(f"query head {head} has no sampled pairs")
            group = head // heads_per_group
            candidate, cg = _solve_query_factor(
                samples.query[keep],
                samples.key[keep],
                weights[keep],
                factors.key_encoder[group],
                ridge=ridge,
                max_iterations=cg_iterations,
                relative_tolerance=cg_tolerance,
            )
            query_encoders[head] = candidate
            diagnostics.append({"sweep": sweep, "block": f"query:{head}", **cg})
        candidate_factors = GQAKProxyFactors(factors.key_encoder, query_encoders)
        candidate_objective = raw_score_objective(
            samples, candidate_factors, num_kv_heads=num_kv_heads, ridge=ridge
        )
        if candidate_objective <= history[-1] * (1.0 + 1.0e-10):
            factors = candidate_factors
            history.append(candidate_objective)
        else:
            history.append(history[-1])

        key_encoders = factors.key_encoder.clone()
        selected_query_factors = factors.query_encoders.index_select(
            0, samples.query_head
        )
        for group in range(num_kv_heads):
            keep = group_ids == group
            candidate, cg = _solve_key_factor(
                samples.query[keep],
                samples.key[keep],
                selected_query_factors[keep],
                weights[keep],
                ridge=ridge,
                max_iterations=cg_iterations,
                relative_tolerance=cg_tolerance,
            )
            key_encoders[group] = candidate
            diagnostics.append({"sweep": sweep, "block": f"key:{group}", **cg})
        candidate_factors = canonicalize_gqa_k_proxy_factors(
            GQAKProxyFactors(key_encoders, factors.query_encoders)
        )
        candidate_objective = raw_score_objective(
            samples, candidate_factors, num_kv_heads=num_kv_heads, ridge=ridge
        )
        if candidate_objective <= history[-1] * (1.0 + 1.0e-10):
            factors = candidate_factors
            history.append(candidate_objective)
        else:
            history.append(history[-1])
    return KProxyFitResult(factors, tuple(history), tuple(diagnostics))
