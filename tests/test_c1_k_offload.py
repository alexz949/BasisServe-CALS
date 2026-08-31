from __future__ import annotations

import math

import torch

from basisserve.core.c1_k_offload import (
    PinnedCPUExactKeyPageStore,
    dense_exact_k_c1_attention,
    fetch_and_pack_exact_key_pages,
    kq_svd_gqa_page_selection,
    offloaded_kq_svd_c1_attention,
    sparse_exact_k_c1_attention,
)


def test_cpu_page_store_preserves_request_order_and_partial_pages() -> None:
    exact_key = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(1, 2, 5, 3)
    store = PinnedCPUExactKeyPageStore(exact_key, layer_idx=4)

    observed = store.get_pages(
        layer_idx=4,
        batch_indices=torch.tensor([0, 0, 0]),
        kv_head_indices=torch.tensor([1, 0, 1]),
        page_ids=torch.tensor([2, 0, 0]),
        page_size=2,
        device=torch.device("cpu"),
    )

    expected = torch.stack(
        (
            torch.cat((exact_key[0, 1, 4:], torch.zeros(1, 3))),
            exact_key[0, 0, :2],
            exact_key[0, 1, :2],
        )
    )
    torch.testing.assert_close(observed, expected)
    assert store.last_request_count == 3
    assert store.last_requested_bytes == 3 * 2 * 3 * 4


def test_routing_unions_different_query_head_pages() -> None:
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    sidecar = torch.tensor(
        [[[[10.0, 0.0], [9.0, 0.0], [0.0, 8.0], [0.0, 7.0]]]]
    )
    projector = torch.eye(2).unsqueeze(0)

    selected = kq_svd_gqa_page_selection(
        query,
        sidecar,
        projector,
        page_size=2,
        pages_per_query_head=1,
    )

    assert selected.page_mask.tolist() == [[[True, True]]]


def test_adaptive_routing_expands_only_uncertain_query_head() -> None:
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    sidecar = torch.tensor(
        [[[[10.0, 0.0], [10.0, 0.0], [9.8, 0.0], [9.8, 0.0],
           [0.0, 10.0], [0.0, 10.0], [0.0, 0.0], [0.0, 0.0]]]]
    )
    projector = torch.eye(2).unsqueeze(0)

    selected = kq_svd_gqa_page_selection(
        query,
        sidecar,
        projector,
        page_size=2,
        pages_per_query_head=1,
        adaptive_max_pages_per_query_head=2,
        adaptive_tail_mass_ratio_threshold=0.5,
    )

    assert selected.refined_query_mask is not None
    assert selected.refined_query_mask.tolist() == [[True, False]]
    assert selected.page_mask.tolist() == [[[True, True, True, False]]]


def test_all_selected_pages_match_dense_exact_k_c1_attention() -> None:
    generator = torch.Generator().manual_seed(20260830)
    query = torch.randn(1, 4, 1, 4, generator=generator)
    exact_key = torch.randn(1, 2, 7, 4, generator=generator)
    c1_value = torch.randn(1, 2, 7, 3, generator=generator)
    store = PinnedCPUExactKeyPageStore(exact_key)
    page_mask = torch.ones(1, 2, math.ceil(7 / 3), dtype=torch.bool)
    pages = fetch_and_pack_exact_key_pages(
        store,
        page_mask,
        layer_idx=0,
        page_size=3,
        head_dim=4,
        device=torch.device("cpu"),
    )

    observed = sparse_exact_k_c1_attention(
        query, c1_value, pages, page_size=3
    )
    expected = dense_exact_k_c1_attention(query, exact_key, c1_value)

    torch.testing.assert_close(observed, expected, rtol=1.0e-5, atol=1.0e-6)


def test_end_to_end_offload_uses_exact_k_after_lossless_routing() -> None:
    generator = torch.Generator().manual_seed(20260831)
    query = torch.randn(1, 4, 1, 4, generator=generator)
    exact_key = torch.randn(1, 2, 8, 4, generator=generator)
    c1_value = torch.randn(1, 2, 8, 3, generator=generator)
    projector = torch.eye(4).expand(2, -1, -1).clone()
    sidecar = torch.einsum("bgtd,gdr->bgtr", exact_key, projector)
    store = PinnedCPUExactKeyPageStore(exact_key, layer_idx=7)

    observed = offloaded_kq_svd_c1_attention(
        query,
        sidecar,
        projector,
        c1_value,
        store,
        layer_idx=7,
        page_size=2,
        pages_per_query_head=4,
    )
    expected = dense_exact_k_c1_attention(query, exact_key, c1_value)

    torch.testing.assert_close(
        observed.output, expected, rtol=1.0e-5, atol=1.0e-6
    )
    assert observed.pages.requested_pages == 8
    assert observed.pages.requested_bytes == exact_key.numel() * exact_key.element_size()
