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


def conditional_routing_query_projector(
    residual_query_projector: torch.Tensor,
) -> torch.Tensor:
    """Compose exact-base and residual-query coordinates.

    The corresponding token sidecar is ``[predicted post-RoPE K, residual
    code]``.  Projecting a query through this matrix therefore produces
    ``[q, q @ U_residual]`` and one dot product evaluates the complete
    predictive-base plus residual proxy score.
    """

    query_heads, head_dim, residual_rank = map(
        int, residual_query_projector.shape
    )
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
    predicted_pre_rope.add_(
        base_bias.to(device=device, dtype=dtype)[None, :, None, :]
    )
    rotary_cos = cos.to(device=device, dtype=dtype).unsqueeze(1)
    rotary_sin = sin.to(device=device, dtype=dtype).unsqueeze(1)
    predicted_post_rope = (
        predicted_pre_rope * rotary_cos
        + _rotate_half(predicted_pre_rope) * rotary_sin
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
) -> AffineReducedRankMap:
    """Fit an unregularized affine reduced-rank regression from moments.

    The centered input covariance is whitened with its numerical Moore-Penrose
    inverse square root.  Truncating the SVD of the whitened cross moment gives
    the exact least-squares rank-constrained solution.
    """

    count = float(row_count)
    mean_input = input_sum / count
    mean_target = target_sum / count
    covariance = input_gram - count * torch.outer(mean_input, mean_input)
    cross = input_target_gram - count * torch.outer(mean_input, mean_target)
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
    inverse_square_root = (
        eigenvectors * inverse_roots.unsqueeze(0)
    ) @ eigenvectors.mT
    whitened_cross = inverse_square_root @ cross
    left_singular, singular_values, right_singular = torch.linalg.svd(
        whitened_cross,
        full_matrices=False,
    )
    selected = min(int(rank), int(singular_values.numel()))
    left = inverse_square_root @ left_singular[:, :selected]
    right = singular_values[:selected, None] * right_singular[:selected]
    bias = mean_target - mean_input @ left @ right
    return AffineReducedRankMap(left=left, right=right, bias=bias)


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
    Key.  Prefix pages that are pinned in the physical cache are removed before
    the non-sink distribution is normalized.  Only the page representative
    uses the residual feature.  The returned energy is the Page-Fisher energy
    of the exact residual score, so a zero-dimensional residual router has
    normalized loss one.
    """

    first_token = int(excluded_prefix_pages) * int(page_size)
    exact_key_rows = exact_key_rows[first_token:]
    residual_rows = residual_rows[first_token:]
    tokens = int(exact_key_rows.shape[0])
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
    energy = float(
        0.5 * torch.sum(page_mass * (page_scores - score_mean).square())
    )
    return grams, energy


__all__ = [
    "AffineReducedRankMap",
    "build_conditional_routing_sidecar",
    "conditional_routing_query_projector",
    "fit_affine_reduced_rank_map",
    "residual_page_fisher_gram",
]
