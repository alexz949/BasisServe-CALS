"""Closed-form post-RoPE Key/Query compression for grouped-query attention."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_grams(
    key_gram: Tensor,
    query_gram: Tensor | None,
    rank: int,
) -> int:
    if key_gram.ndim < 2 or key_gram.shape[-1] != key_gram.shape[-2]:
        raise ValueError("key Gram must contain square matrices")
    head_dim = int(key_gram.shape[-1])
    if not 0 < rank <= head_dim:
        raise ValueError(f"rank must be in [1, {head_dim}], got {rank}")
    if query_gram is not None and query_gram.shape != key_gram.shape:
        raise ValueError("Key and Query Grams must have identical shapes")
    if not key_gram.is_floating_point() or (
        query_gram is not None and not query_gram.is_floating_point()
    ):
        raise TypeError("KQ-SVD Grams must be floating point")
    return head_dim


def key_svd_projector(key_gram: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    """Return the rank-``rank`` PCA basis and descending Key spectrum."""

    _validate_grams(key_gram, None, rank)
    gram = (key_gram + key_gram.mT) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.arange(
        eigenvalues.shape[-1] - 1,
        -1,
        -1,
        device=eigenvalues.device,
    )
    eigenvalues = eigenvalues.index_select(-1, order)
    eigenvectors = eigenvectors.index_select(-1, order)
    return eigenvectors[..., :rank].contiguous(), eigenvalues.contiguous()


def kq_svd_projectors(
    key_gram: Tensor,
    query_gram: Tensor,
    rank: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Solve the KQ-SVD score-fidelity objective from sufficient statistics.

    The returned matrices ``A`` and ``B`` satisfy

    ``(K @ A) @ (Q @ B).T = K @ (A @ B.T) @ Q.T``.

    This is the covariance-only equivalent of applying QR to the complete
    post-RoPE Key and grouped-Query activation matrices and truncating the SVD
    of ``R_K @ R_Q.T``.  Column rescaling balances the two deployed factors
    without changing their product or the KQ-SVD optimum.
    """

    _validate_grams(key_gram, query_gram, rank)
    key_gram = (key_gram + key_gram.mT) * 0.5
    query_gram = (query_gram + query_gram.mT) * 0.5
    key_cholesky, key_info = torch.linalg.cholesky_ex(key_gram)
    query_cholesky, query_info = torch.linalg.cholesky_ex(query_gram)
    if torch.any(key_info != 0) or torch.any(query_info != 0):
        bad_key = int(torch.count_nonzero(key_info).item())
        bad_query = int(torch.count_nonzero(query_info).item())
        raise RuntimeError(
            "KQ-SVD requires positive-definite undamped Grams; "
            f"failed Key matrices={bad_key}, Query matrices={bad_query}"
        )

    key_r = key_cholesky.mT
    query_r = query_cholesky.mT
    left, spectrum, _ = torch.linalg.svd(
        key_r @ query_r.mT,
        full_matrices=False,
    )
    left = left[..., :rank]
    key_projector = torch.linalg.solve_triangular(
        key_r,
        left,
        upper=True,
    )
    query_projector = key_r.mT @ left

    key_norm = torch.linalg.vector_norm(key_projector, dim=-2)
    query_norm = torch.linalg.vector_norm(query_projector, dim=-2)
    scale = torch.sqrt(query_norm / key_norm)
    key_projector = key_projector * scale.unsqueeze(-2)
    query_projector = query_projector / scale.unsqueeze(-2)
    return (
        key_projector.contiguous(),
        query_projector.contiguous(),
        spectrum.contiguous(),
    )


def relative_score_frobenius_error(
    key_gram: Tensor,
    query_gram: Tensor,
    key_projector: Tensor,
    query_projector: Tensor,
) -> Tensor:
    """Evaluate relative squared score error without materializing score matrices."""

    head_dim = _validate_grams(key_gram, query_gram, key_projector.shape[-1])
    expected = (*key_gram.shape[:-1], key_projector.shape[-1])
    if key_projector.shape != expected or query_projector.shape != expected:
        raise ValueError("K/Q projector geometry does not match the Grams")
    identity = torch.eye(head_dim, device=key_gram.device, dtype=key_gram.dtype)
    residual = identity - key_projector @ query_projector.mT
    numerator = torch.diagonal(
        key_gram @ residual @ query_gram @ residual.mT,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)
    denominator = torch.diagonal(
        key_gram @ query_gram,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)
    return numerator / denominator


def project_grouped_queries_and_keys(
    query: Tensor,
    key: Tensor,
    key_projector: Tensor,
    query_projector: Tensor,
) -> tuple[Tensor, Tensor]:
    """Project post-RoPE GQA tensors using one KQ pair per physical KV head."""

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("GQA Query and Key tensors must be rank four")
    if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError("GQA Query and Key batch/sequence dimensions differ")
    kv_heads = int(key.shape[1])
    query_heads = int(query.shape[1])
    if kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("Query heads must be divisible by physical KV heads")
    expected_prefix = (kv_heads, int(key.shape[-1]))
    if (
        key_projector.ndim != 3
        or query_projector.ndim != 3
        or tuple(key_projector.shape[:2]) != expected_prefix
        or tuple(query_projector.shape[:2]) != expected_prefix
        or key_projector.shape[-1] != query_projector.shape[-1]
        or query.shape[-1] != key.shape[-1]
    ):
        raise ValueError("GQA KQ projector geometry is incompatible")
    groups = query_heads // kv_heads
    expanded_query_projector = query_projector.repeat_interleave(groups, dim=0)
    compressed_query = torch.einsum(
        "bhld,hdr->bhlr",
        query,
        expanded_query_projector,
    )
    compressed_key = torch.einsum(
        "bhld,hdr->bhlr",
        key,
        key_projector,
    )
    return compressed_query, compressed_key
