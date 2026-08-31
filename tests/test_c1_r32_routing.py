from __future__ import annotations

import pytest
import torch

from basisserve.kernels.c1_r32_routing import (
    c1_r32_page_lse_cuda,
    c1_r32_topk_gqa_union_cuda,
    reference_c1_r32_page_lse,
    reference_c1_r32_topk_gqa_union,
)


def test_r32_page_lse_cuda_rejects_cpu_tensors() -> None:
    query = torch.randn(1, 32, 1, 128, dtype=torch.bfloat16)
    sidecar = torch.randn(1, 8, 65, 32, dtype=torch.bfloat16)
    projector = torch.randn(8, 128, 32, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="requires CUDA"):
        c1_r32_page_lse_cuda(query, sidecar, projector)


def test_r32_topk_reference_emits_sorted_compact_union() -> None:
    scores = torch.full((1, 8, 6), -10.0)
    scores[0, 0, [1, 4]] = torch.tensor([8.0, 7.0])
    scores[0, 1, [2, 4]] = torch.tensor([9.0, 6.0])
    scores[0, 2, [3, 5]] = torch.tensor([8.5, 5.0])
    scores[0, 3, [0, 3]] = torch.tensor([9.5, 4.0])
    scores[0, 4:, 0] = 8.0
    scores[0, 4:, 1] = 7.0

    selected, counts = reference_c1_r32_topk_gqa_union(
        scores,
        pages_per_query_head=2,
    )

    assert counts.tolist() == [[6, 2]]
    assert selected[0, 0].tolist() == [0, 1, 2, 3, 4, 5]
    assert selected[0, 1].tolist() == [0, 1, -1, -1, -1, -1]


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((8, 0), (8, 9), (9, 0)),
    reason="requires SM80, SM89, or SM90",
)
@pytest.mark.parametrize("projector_heads", [8, 32])
def test_r32_page_lse_cuda_matches_materialized_reference(
    projector_heads: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        20260901 + projector_heads
    )
    query = torch.randn(
        2,
        32,
        1,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    sidecar = (
        torch.randn(
            2,
            8,
            193,
            32,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.1
    )
    projector = (
        torch.randn(
            projector_heads,
            128,
            32,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.05
    ).contiguous()

    observed = c1_r32_page_lse_cuda(query, sidecar, projector)
    expected = reference_c1_r32_page_lse(query, sidecar, projector)

    assert observed.shape == (2, 32, 4)
    assert observed.dtype == torch.bfloat16
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((8, 0), (8, 9), (9, 0)),
    reason="requires SM80, SM89, or SM90",
)
@pytest.mark.parametrize("pages_per_query_head", [1, 5, 32])
def test_r32_topk_gqa_union_cuda_matches_reference(
    pages_per_query_head: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        20260911 + pages_per_query_head
    )
    scores = torch.randn(
        2,
        32,
        67,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    scores += (
        torch.arange(67, dtype=torch.bfloat16, device="cuda")[None, None]
        * 0.015625
    )

    observed_ids, observed_counts = c1_r32_topk_gqa_union_cuda(
        scores,
        pages_per_query_head=pages_per_query_head,
    )
    expected_ids, expected_counts = reference_c1_r32_topk_gqa_union(
        scores,
        pages_per_query_head=pages_per_query_head,
    )

    torch.testing.assert_close(observed_counts, expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(observed_ids, expected_ids, rtol=0, atol=0)
