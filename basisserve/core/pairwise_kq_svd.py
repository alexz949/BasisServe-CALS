"""Closed-form activation-aware KQ-SVD for adjacent-layer shared Key codes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PairwiseKQSVDResult:
    """One pair's shared Key encoder and two layer-specific Query maps."""

    key_projector: Tensor
    query_projector: Tensor
    spectrum: Tensor


def _validate(
    pair_key_gram: Tensor,
    query_gram: Tensor,
    rank: int,
) -> tuple[int, int]:
    if pair_key_gram.ndim < 2 or pair_key_gram.shape[-1] != pair_key_gram.shape[-2]:
        raise ValueError("pair Key Gram must contain square matrices")
    pair_width = int(pair_key_gram.shape[-1])
    if pair_width % 2:
        raise ValueError("pair Key width must be twice one head dimension")
    head_dim = pair_width // 2
    expected_query_shape = (2, *pair_key_gram.shape[:-2], head_dim, head_dim)
    if tuple(query_gram.shape) != expected_query_shape:
        raise ValueError(
            f"Query Grams must have shape {expected_query_shape}, "
            f"got {tuple(query_gram.shape)}"
        )
    if not 0 < rank <= pair_width:
        raise ValueError(f"rank must lie in [1, {pair_width}]")
    if pair_key_gram.dtype not in (torch.float32, torch.float64):
        raise TypeError("pairwise KQ-SVD statistics must be float32 or float64")
    if query_gram.dtype != pair_key_gram.dtype:
        raise TypeError("pair Key and Query Grams must use the same dtype")
    return head_dim, pair_width


def _cholesky(name: str, gram: Tensor) -> Tensor:
    symmetric = 0.5 * (gram + gram.mT)
    factor, info = torch.linalg.cholesky_ex(symmetric)
    if torch.any(info != 0):
        failures = int(torch.count_nonzero(info).item())
        raise RuntimeError(f"{name} is not positive definite for {failures} groups")
    return factor


@torch.no_grad()
def fit_pairwise_kq_svd(
    pair_key_gram: Tensor,
    query_gram: Tensor,
    rank: int,
) -> PairwiseKQSVDResult:
    """Solve the joint two-layer score-Frobenius objective.

    For ``X=[K0,K1]``, this minimizes

    ``sum_l ||K_l Q_l.T - (X E) (Q_l F_l).T||_F^2``

    with one shared rank-``rank`` ``E`` and one ``F_l`` per layer.  The solve
    is reduced-rank regression after Cholesky whitening of both Query metrics.
    """

    head_dim, _ = _validate(pair_key_gram, query_gram, rank)
    pair_key_gram = 0.5 * (pair_key_gram + pair_key_gram.mT)
    query_gram = 0.5 * (query_gram + query_gram.mT)
    key_cholesky = _cholesky("pair Key Gram", pair_key_gram)
    query_cholesky = _cholesky("Query Gram", query_gram)

    cross_target = torch.cat(
        (
            pair_key_gram[..., :, :head_dim] @ query_cholesky[0],
            pair_key_gram[..., :, head_dim:] @ query_cholesky[1],
        ),
        dim=-1,
    )
    whitened_cross = torch.linalg.solve_triangular(
        key_cholesky,
        cross_target,
        upper=False,
    )
    left, spectrum, _ = torch.linalg.svd(
        whitened_cross,
        full_matrices=False,
    )
    left = left[..., :rank]
    raw_key = torch.linalg.solve_triangular(
        key_cholesky.mT,
        left,
        upper=True,
    )
    raw_readout = left.mT @ whitened_cross
    key_projector, triangular = torch.linalg.qr(raw_key, mode="reduced")
    readout = triangular @ raw_readout

    query_projectors = []
    for layer_slot in (0, 1):
        start = layer_slot * head_dim
        stop = start + head_dim
        layer_readout = readout[..., start:stop]
        query_projectors.append(
            torch.linalg.solve_triangular(
                query_cholesky[layer_slot].mT,
                layer_readout.mT,
                upper=True,
            )
        )
    return PairwiseKQSVDResult(
        key_projector=key_projector.contiguous(),
        query_projector=torch.stack(query_projectors).contiguous(),
        spectrum=spectrum.contiguous(),
    )


def pairwise_score_squared_errors(
    pair_key_gram: Tensor,
    query_gram: Tensor,
    key_projector: Tensor,
    query_projector: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-layer/group squared error, energy, and relative error."""

    head_dim, pair_width = _validate(
        pair_key_gram,
        query_gram,
        int(key_projector.shape[-1]),
    )
    expected_key = (*pair_key_gram.shape[:-2], pair_width, key_projector.shape[-1])
    expected_query = (
        2,
        *pair_key_gram.shape[:-2],
        head_dim,
        key_projector.shape[-1],
    )
    if tuple(key_projector.shape) != expected_key:
        raise ValueError("pair Key projector geometry differs from its Gram")
    if tuple(query_projector.shape) != expected_query:
        raise ValueError("pair Query projector geometry differs from its Grams")
    errors = []
    energies = []
    for layer_slot in (0, 1):
        selector = torch.zeros(
            pair_width,
            head_dim,
            dtype=pair_key_gram.dtype,
            device=pair_key_gram.device,
        )
        start = layer_slot * head_dim
        selector[start : start + head_dim] = torch.eye(
            head_dim,
            dtype=selector.dtype,
            device=selector.device,
        )
        residual = selector - key_projector @ query_projector[layer_slot].mT
        weighted = pair_key_gram @ residual @ query_gram[layer_slot]
        error = (weighted * residual).sum(dim=(-2, -1)).clamp_min(0.0)
        layer_key_gram = pair_key_gram[..., start : start + head_dim, start : start + head_dim]
        energy = torch.diagonal(
            layer_key_gram @ query_gram[layer_slot],
            dim1=-2,
            dim2=-1,
        ).sum(dim=-1)
        errors.append(error)
        energies.append(energy)
    error = torch.stack(errors)
    energy = torch.stack(energies)
    return error, energy, error / energy.clamp_min(torch.finfo(energy.dtype).tiny)


def block_diagonal_pair_factors(
    independent_key_projector: Tensor,
    independent_query_projector: Tensor,
) -> tuple[Tensor, Tensor]:
    """Embed independent rank-r codecs into an exactly equal pair-rank-2r codec."""

    if independent_key_projector.ndim != 4:
        raise ValueError(
            "independent Key factors must have shape [2, groups, head_dim, rank]"
        )
    if independent_query_projector.ndim != 4:
        raise ValueError(
            "independent Query factors must have shape [2, groups, head_dim, rank]"
        )
    layers, groups, head_dim, rank = map(int, independent_key_projector.shape)
    if layers != 2 or tuple(independent_query_projector.shape) != (
        layers,
        groups,
        head_dim,
        rank,
    ):
        raise ValueError("independent Key and Query factor geometry differs")
    pair_key = independent_key_projector.new_zeros(
        groups,
        2 * head_dim,
        2 * rank,
    )
    pair_query = independent_query_projector.new_zeros(
        layers,
        groups,
        head_dim,
        2 * rank,
    )
    pair_key[:, :head_dim, :rank] = independent_key_projector[0]
    pair_key[:, head_dim:, rank:] = independent_key_projector[1]
    pair_query[0, ..., :rank] = independent_query_projector[0]
    pair_query[1, ..., rank:] = independent_query_projector[1]
    return pair_key, pair_query


__all__ = [
    "PairwiseKQSVDResult",
    "block_diagonal_pair_factors",
    "fit_pairwise_kq_svd",
    "pairwise_score_squared_errors",
]
