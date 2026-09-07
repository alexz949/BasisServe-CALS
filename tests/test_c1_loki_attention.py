from __future__ import annotations

import math

import torch

from basisserve.core.c1_loki_attention import c1_loki_pca_topk_attention


def _paper_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    projector: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    batch, query_heads, sequence, head_dim = query.shape
    kv_heads = int(key.shape[1])
    heads_per_group = query_heads // kv_heads
    head_to_kv = torch.arange(kv_heads).repeat_interleave(heads_per_group)
    expanded_key = key.index_select(1, head_to_kv)
    expanded_value = value.index_select(1, head_to_kv)
    expanded_projector = projector.index_select(0, head_to_kv)
    query_code = torch.einsum("bhqd,hdr->bhqr", query, expanded_projector)
    key_code = torch.einsum("bhsd,hdr->bhsr", expanded_key, expanded_projector)
    scale = head_dim**-0.5
    approximate = torch.matmul(query_code, key_code.transpose(-1, -2)) * scale
    exact = torch.matmul(query, expanded_key.transpose(-1, -2)) * scale
    causal = torch.ones(sequence, sequence, dtype=torch.bool).tril()[None, None]
    approximate.masked_fill_(~causal, -torch.inf)
    exact.masked_fill_(~causal, -torch.inf)
    indices = torch.topk(
        approximate,
        min(top_k, sequence),
        dim=-1,
        sorted=False,
    ).indices
    selected = torch.full_like(exact, -torch.inf)
    selected.scatter_(-1, indices, exact.gather(-1, indices))
    probability = torch.softmax(selected.float(), dim=-1).to(query.dtype)
    return torch.matmul(probability, expanded_value)


def test_c1_loki_matches_repository_full_mask_semantics() -> None:
    generator = torch.Generator().manual_seed(20260903)
    query = torch.randn(1, 4, 7, 4, generator=generator, dtype=torch.float64)
    key = torch.randn(1, 2, 7, 4, generator=generator, dtype=torch.float64)
    value = torch.randn(1, 2, 7, 3, generator=generator, dtype=torch.float64)
    projector, _ = torch.linalg.qr(
        torch.randn(2, 4, 2, generator=generator, dtype=torch.float64)
    )
    expected = _paper_reference(query, key, value, projector, top_k=3)
    observed = c1_loki_pca_topk_attention(
        query,
        key,
        value,
        projector,
        projector,
        top_k=3,
        scale=1 / math.sqrt(4),
        query_block_size=2,
    )
    torch.testing.assert_close(observed.output, expected)
    assert observed.statistics["queries"] == 7
    assert observed.statistics["query_selected_tokens"] == 72


def test_c1_loki_query_tiling_is_exact() -> None:
    generator = torch.Generator().manual_seed(20260904)
    query = torch.randn(2, 4, 9, 8, generator=generator)
    key = torch.randn(2, 2, 9, 8, generator=generator)
    value = torch.randn(2, 2, 9, 5, generator=generator)
    projector, _ = torch.linalg.qr(torch.randn(2, 8, 4, generator=generator))
    full = c1_loki_pca_topk_attention(
        query,
        key,
        value,
        projector,
        projector,
        top_k=4,
        scale=1 / math.sqrt(8),
        query_block_size=9,
    )
    tiled = c1_loki_pca_topk_attention(
        query,
        key,
        value,
        projector,
        projector,
        top_k=4,
        scale=1 / math.sqrt(8),
        query_block_size=3,
    )
    torch.testing.assert_close(tiled.output, full.output)
    assert tiled.statistics == full.statistics
