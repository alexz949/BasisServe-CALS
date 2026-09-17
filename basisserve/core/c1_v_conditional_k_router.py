"""V-conditioned predictive-base and residual-Key routing primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AffineReducedRankMap:
    """Affine map ``x @ left @ right + bias`` with an explicit rank bound."""

    left: torch.Tensor
    right: torch.Tensor
    bias: torch.Tensor

    @property
    def weight(self) -> torch.Tensor:
        return self.left @ self.right

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        return values @ self.left @ self.right + self.bias


@dataclass(frozen=True)
class AffinePredictiveSpectrum:
    """Centered affine C-to-K predictability from sufficient statistics.

    ``predictive_singular_values`` carry the units of the centered target and
    give the exact reduced-rank-regression energy captured at every rank.
    ``canonical_correlations`` are dimensionless CCA correlations and describe
    cross-subspace alignment independently of target scale.
    """

    canonical_correlations: torch.Tensor
    predictive_singular_values: torch.Tensor
    target_centered_energy: float
    predictable_energy: float
    unrestricted_residual_energy: float

    def captured_energy(self, rank: int) -> float:
        selected = min(int(rank), int(self.predictive_singular_values.numel()))
        return float(self.predictive_singular_values[:selected].square().sum())

    def residual_energy(self, rank: int) -> float:
        return max(0.0, self.target_centered_energy - self.captured_energy(rank))


def conditional_routing_query_projector(
    residual_query_projector: torch.Tensor,
) -> torch.Tensor:
    """Compose exact-base and residual-query coordinates.

    The corresponding token sidecar is ``[predicted post-RoPE K, residual
    code]``.  Projecting a query through this matrix therefore produces
    ``[q, q @ U_residual]`` and one dot product evaluates the complete
    predictive-base plus residual proxy score.
    """

    query_heads, head_dim, residual_rank = map(int, residual_query_projector.shape)
    identity = torch.eye(
        head_dim,
        device=residual_query_projector.device,
        dtype=residual_query_projector.dtype,
    ).expand(query_heads, head_dim, head_dim)
    return torch.cat((identity, residual_query_projector), dim=-1).contiguous()


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    half = int(values.shape[-1]) // 2
    return torch.cat((-values[..., half:], values[..., :half]), dim=-1)


def build_conditional_routing_sidecar(
    value_codes: torch.Tensor,
    exact_post_rope_key: torch.Tensor,
    *,
    base_left: torch.Tensor,
    base_right: torch.Tensor,
    base_bias: torch.Tensor,
    residual_encoder: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Materialize a correctness-oracle ``[base K, residual code]`` sidecar.

    ``value_codes`` and ``exact_post_rope_key`` use cache-major geometry
    ``[batch, KV heads, tokens, width]``.  The predictive map acts on the
    resident C1 Value coordinates before RoPE.  Only the residual rank is
    additional persistent information in the deployable formulation; this
    materialized base is intentionally a small-experiment quality oracle.
    """

    dtype = value_codes.dtype
    device = value_codes.device
    predicted_pre_rope = torch.einsum(
        "bhtv,hvr,hrd->bhtd",
        value_codes,
        base_left.to(device=device, dtype=dtype),
        base_right.to(device=device, dtype=dtype),
    )
    predicted_pre_rope.add_(base_bias.to(device=device, dtype=dtype)[None, :, None, :])
    rotary_cos = cos.to(device=device, dtype=dtype).unsqueeze(1)
    rotary_sin = sin.to(device=device, dtype=dtype).unsqueeze(1)
    predicted_post_rope = (
        predicted_pre_rope * rotary_cos + _rotate_half(predicted_pre_rope) * rotary_sin
    )
    residual = exact_post_rope_key - predicted_post_rope
    residual_code = torch.einsum(
        "bhtd,hdr->bhtr",
        residual,
        residual_encoder.to(device=device, dtype=dtype),
    )
    return torch.cat((predicted_post_rope, residual_code), dim=-1).contiguous()


def fit_affine_reduced_rank_map(
    *,
    row_count: int,
    input_sum: torch.Tensor,
    target_sum: torch.Tensor,
    input_gram: torch.Tensor,
    input_target_gram: torch.Tensor,
    rank: int,
    fit_bias: bool = True,
) -> AffineReducedRankMap:
    """Fit unregularized reduced-rank regression, optionally without an intercept.

    The centered input covariance is whitened with its numerical Moore-Penrose
    inverse square root.  Truncating the SVD of the whitened cross moment gives
    the exact least-squares rank-constrained solution.
    With ``fit_bias=False``, use raw moments and constrain the bias to zero;
    this refits the weight rather than removing an already fitted intercept.
    """

    count = float(row_count)
    mean_input = input_sum / count
    mean_target = target_sum / count
    covariance = input_gram
    cross = input_target_gram
    if fit_bias:
        covariance = covariance - count * torch.outer(mean_input, mean_input)
        cross = cross - count * torch.outer(mean_input, mean_target)
    covariance = 0.5 * (covariance + covariance.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    cutoff = (
        torch.finfo(covariance.dtype).eps
        * covariance.shape[0]
        * eigenvalues.abs().amax()
    )
    inverse_roots = torch.where(
        eigenvalues > cutoff,
        torch.rsqrt(eigenvalues.clamp_min(torch.finfo(covariance.dtype).tiny)),
        torch.zeros_like(eigenvalues),
    )
    inverse_square_root = (eigenvectors * inverse_roots.unsqueeze(0)) @ eigenvectors.mT
    whitened_cross = inverse_square_root @ cross
    left_singular, singular_values, right_singular = torch.linalg.svd(
        whitened_cross,
        full_matrices=False,
    )
    selected = min(int(rank), int(singular_values.numel()))
    left = inverse_square_root @ left_singular[:, :selected]
    right = singular_values[:selected, None] * right_singular[:selected]
    bias = mean_target - mean_input @ left @ right if fit_bias else torch.zeros_like(mean_target)
    return AffineReducedRankMap(left=left, right=right, bias=bias)


def fit_affine_metric_reduced_rank_map(
    *,
    row_count: int,
    input_sum: torch.Tensor,
    target_sum: torch.Tensor,
    input_gram: torch.Tensor,
    input_target_gram: torch.Tensor,
    target_metric: torch.Tensor,
    rank: int,
) -> AffineReducedRankMap:
    """Fit affine reduced-rank regression in a target-side PSD metric.

    For row residuals ``E = K - C M - b``, this solves

    ``min rank(M)<=r ||E H^(1/2)||_F^2``

    for the supplied positive-semidefinite target metric ``H``.  The target
    is transformed by the symmetric square root of ``H``, fitted with the
    ordinary exact RRR solver, and mapped back with the numerical
    Moore-Penrose inverse square root.  A query covariance can therefore
    reorder the retained predictive directions without changing the
    unrestricted linear conditional mean when the metric is nonsingular.
    """

    metric = 0.5 * (target_metric + target_metric.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    scale = eigenvalues.abs().amax()
    cutoff = torch.finfo(metric.dtype).eps * metric.shape[0] * scale
    positive = eigenvalues > cutoff
    roots = torch.where(
        positive,
        torch.sqrt(eigenvalues.clamp_min(0)),
        torch.zeros_like(eigenvalues),
    )
    inverse_roots = torch.where(
        positive,
        torch.rsqrt(eigenvalues.clamp_min(torch.finfo(metric.dtype).tiny)),
        torch.zeros_like(eigenvalues),
    )
    square_root = (eigenvectors * roots.unsqueeze(0)) @ eigenvectors.mT
    inverse_square_root = (
        eigenvectors * inverse_roots.unsqueeze(0)
    ) @ eigenvectors.mT
    transformed = fit_affine_reduced_rank_map(
        row_count=row_count,
        input_sum=input_sum,
        target_sum=target_sum @ square_root,
        input_gram=input_gram,
        input_target_gram=input_target_gram @ square_root,
        rank=rank,
    )
    return AffineReducedRankMap(
        left=transformed.left,
        right=transformed.right @ inverse_square_root,
        bias=transformed.bias @ inverse_square_root,
    )


def affine_predictive_spectrum(
    *,
    row_count: int,
    input_sum: torch.Tensor,
    target_sum: torch.Tensor,
    input_gram: torch.Tensor,
    input_target_gram: torch.Tensor,
    target_gram: torch.Tensor,
) -> AffinePredictiveSpectrum:
    """Compute affine RRR and CCA spectra from unnormalized row moments.

    For centered input rows ``C`` and target rows ``K``, the singular values of
    ``G_CC^{-1/2} G_CK`` give the target energy captured by the optimal
    rank-constrained affine predictor.  Whitening the target as well gives the
    canonical correlations.  Numerical null spaces are handled with the same
    Moore-Penrose cutoff used by :func:`fit_affine_reduced_rank_map`.
    """

    count = float(row_count)
    mean_input = input_sum / count
    mean_target = target_sum / count
    input_covariance = input_gram - count * torch.outer(mean_input, mean_input)
    target_covariance = target_gram - count * torch.outer(mean_target, mean_target)
    cross_covariance = input_target_gram - count * torch.outer(mean_input, mean_target)
    input_covariance = 0.5 * (input_covariance + input_covariance.mT)
    target_covariance = 0.5 * (target_covariance + target_covariance.mT)

    def inverse_square_root(covariance: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        cutoff = (
            torch.finfo(covariance.dtype).eps
            * covariance.shape[0]
            * eigenvalues.abs().amax()
        )
        inverse_roots = torch.where(
            eigenvalues > cutoff,
            torch.rsqrt(eigenvalues.clamp_min(torch.finfo(covariance.dtype).tiny)),
            torch.zeros_like(eigenvalues),
        )
        return (eigenvectors * inverse_roots.unsqueeze(0)) @ eigenvectors.mT

    input_inverse_root = inverse_square_root(input_covariance)
    target_inverse_root = inverse_square_root(target_covariance)
    predictive = input_inverse_root @ cross_covariance
    canonical = predictive @ target_inverse_root
    predictive_singular_values = torch.linalg.svdvals(predictive)
    canonical_correlations = torch.linalg.svdvals(canonical).clamp(0.0, 1.0)
    target_energy = float(torch.trace(target_covariance))
    predictable_energy = min(
        target_energy,
        float(predictive_singular_values.square().sum()),
    )
    return AffinePredictiveSpectrum(
        canonical_correlations=canonical_correlations,
        predictive_singular_values=predictive_singular_values,
        target_centered_energy=target_energy,
        predictable_energy=predictable_energy,
        unrestricted_residual_energy=max(0.0, target_energy - predictable_energy),
    )


def residual_page_fisher_gram(
    queries: torch.Tensor,
    exact_key_rows: torch.Tensor,
    residual_rows: torch.Tensor,
    *,
    scaling: float,
    page_size: int,
    excluded_prefix_pages: int,
) -> tuple[torch.Tensor, float]:
    """Build exact-teacher Page-Fisher Grams for residual-Key features.

    Page masses and within-page conditional weights always come from the exact
    Key. Callers pass only the causal historical prefix, excluding any recent
    tokens that are always retained. Prefix pages pinned in the physical cache
    are removed before the candidate distribution is normalized. Only the page
    representative uses the residual feature. The returned energy is the Page-Fisher energy
    of the exact residual score, so a zero-dimensional residual router has
    normalized loss one.
    """

    first_token = int(excluded_prefix_pages) * int(page_size)
    exact_key_rows = exact_key_rows[first_token:]
    residual_rows = residual_rows[first_token:]
    tokens = int(exact_key_rows.shape[0])
    if tokens == 0:
        return queries.new_zeros((queries.shape[0], queries.shape[1], queries.shape[1])), 0.0
    pages = (tokens + int(page_size) - 1) // int(page_size)
    padded_tokens = pages * int(page_size)
    padding = padded_tokens - tokens
    scores = float(scaling) * queries @ exact_key_rows.mT
    probabilities = torch.softmax(scores, dim=-1)
    if padding:
        probabilities = torch.nn.functional.pad(probabilities, (0, padding))
        residual_rows = torch.nn.functional.pad(
            residual_rows,
            (0, 0, 0, padding),
        )
    probabilities_by_page = probabilities.reshape(
        queries.shape[0],
        pages,
        int(page_size),
    )
    residual_by_page = residual_rows.reshape(
        pages,
        int(page_size),
        residual_rows.shape[-1],
    )
    page_mass = probabilities_by_page.sum(dim=-1)
    page_numerator = torch.einsum(
        "hps,psd->hpd",
        probabilities_by_page,
        residual_by_page,
    )
    page_rows = page_numerator / page_mass.clamp_min(
        torch.finfo(page_mass.dtype).tiny
    ).unsqueeze(-1)
    page_mean = torch.einsum("hp,hpd->hd", page_mass, page_rows)
    centered = page_rows - page_mean.unsqueeze(1)
    weighted = centered * torch.sqrt(page_mass).unsqueeze(-1)
    grams = torch.bmm(weighted.mT, weighted)
    grams = 0.5 * (grams + grams.mT)
    page_scores = float(scaling) * torch.einsum(
        "hd,hpd->hp",
        queries,
        page_rows,
    )
    score_mean = torch.sum(page_mass * page_scores, dim=-1, keepdim=True)
    energy = float(0.5 * torch.sum(page_mass * (page_scores - score_mean).square()))
    return grams, energy


__all__ = [
    "AffinePredictiveSpectrum",
    "AffineReducedRankMap",
    "affine_predictive_spectrum",
    "build_conditional_routing_sidecar",
    "conditional_routing_query_projector",
    "fit_affine_metric_reduced_rank_map",
    "fit_affine_reduced_rank_map",
    "residual_page_fisher_gram",
]
