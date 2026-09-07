from __future__ import annotations

import math

import pytest
import torch

from basisserve.kernels.indexed_sparse_decode_attention import (
    gqa_indexed_sparse_decode_attention_triton,
    gqa_page32_log_mass_triton,
    gqa_proxy_scores_triton,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton sparse-decode tests",
)


@pytest.mark.parametrize("routing_rank", (32, 136))
def test_gqa_proxy_kernels_match_materialized_reference(
    routing_rank: int,
) -> None:
    torch.manual_seed(31 + routing_rank)
    device = torch.device("cuda")
    query_code = torch.randn(
        1,
        8,
        routing_rank,
        device=device,
        dtype=torch.bfloat16,
    )
    sidecar = torch.randn(
        1,
        2,
        257,
        routing_rank,
        device=device,
        dtype=torch.bfloat16,
    )
    head_to_kv = torch.arange(2, device=device).repeat_interleave(4)
    expanded = sidecar.index_select(1, head_to_kv)
    scale = 128**-0.5
    expected_scores = torch.matmul(
        query_code[:, :, None],
        expanded.transpose(-1, -2),
    ).mul_(scale)
    observed_scores = gqa_proxy_scores_triton(
        query_code,
        sidecar,
        scale=scale,
    )
    torch.testing.assert_close(
        observed_scores,
        expected_scores,
        atol=1.6e-2,
        rtol=1.6e-2,
    )

    padded_scores = torch.nn.functional.pad(
        expected_scores[:, :, 0].float(),
        (0, 31),
        value=-torch.inf,
    )
    expected_pages = torch.logsumexp(
        padded_scores.reshape(1, 8, 9, 32),
        dim=-1,
    ).reshape(1, 2, 4, 9)
    observed_pages = gqa_page32_log_mass_triton(
        query_code,
        sidecar,
        scale=scale,
    )
    torch.testing.assert_close(
        observed_pages,
        expected_pages,
        atol=2.5e-2,
        rtol=1.6e-2,
    )


@pytest.mark.parametrize("value_dim", (80, 128))
def test_indexed_attention_matches_gather_reference(value_dim: int) -> None:
    torch.manual_seed(67 + value_dim)
    device = torch.device("cuda")
    query = torch.randn(1, 8, 1, 128, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 2, 257, 128, device=device, dtype=torch.bfloat16)
    value = torch.randn(
        1,
        2,
        257,
        value_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    selected = torch.stack(
        [torch.randperm(257, device=device)[:73] for _ in range(8)]
    ).unsqueeze(0)
    selected[:, :, -3:] = -1
    scale = 128**-0.5
    observed = gqa_indexed_sparse_decode_attention_triton(
        query,
        key,
        value,
        selected,
        scale=scale,
    )

    head_to_kv = torch.arange(2, device=device).repeat_interleave(4)
    expanded_key = key.index_select(1, head_to_kv)
    expanded_value = value.index_select(1, head_to_kv)
    safe_selected = selected.clamp_min(0)
    selected_key = torch.gather(
        expanded_key,
        2,
        safe_selected[..., None].expand(1, 8, 73, 128),
    )
    scores = torch.einsum("bhqd,bhkd->bhqk", query, selected_key) * scale
    valid = selected[:, :, None] >= 0
    probabilities = torch.softmax(
        scores.float().masked_fill(~valid, -torch.inf),
        dim=-1,
    )
    selected_value = torch.gather(
        expanded_value,
        2,
        safe_selected[..., None].expand(1, 8, 73, value_dim),
    )
    expected = torch.einsum(
        "bhqk,bhkv->bhqv",
        probabilities,
        selected_value.float(),
    ).to(torch.bfloat16)
    torch.testing.assert_close(observed, expected, atol=2.5e-2, rtol=2.5e-2)


def test_indexed_attention_uses_declared_scale() -> None:
    torch.manual_seed(101)
    device = torch.device("cuda")
    query = torch.randn(1, 4, 1, 128, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 96, 128, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 96, 80, device=device, dtype=torch.bfloat16)
    selected = torch.arange(64, device=device).reshape(1, 1, 64).expand(1, 4, 64)
    first = gqa_indexed_sparse_decode_attention_triton(
        query,
        key,
        value,
        selected,
        scale=1 / math.sqrt(128),
    )
    second = gqa_indexed_sparse_decode_attention_triton(
        query,
        key,
        value,
        selected,
        scale=0.5 / math.sqrt(128),
    )
    assert not torch.equal(first, second)
