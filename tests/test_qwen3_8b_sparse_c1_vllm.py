from __future__ import annotations

import torch

from basisserve.kernels.vllm_sparse_decode_attention import (
    physical_union_count,
    select_group_shared_pages,
    selected_page_tokens,
)
from basisserve.vllm.qwen3_8b_sparse_c1 import (
    HEAD_DIM,
    NUM_KV_HEADS,
    VALUE_RANK,
    fold_value_projection,
)
from evaluation.eval_qwen3_8b_sparse_c1_ruler_vllm import _merge_statistics


def test_value_projection_folding_matches_per_head_product() -> None:
    generator = torch.Generator().manual_seed(71)
    hidden_size = 19
    dense = torch.randn(
        NUM_KV_HEADS * HEAD_DIM,
        hidden_size,
        generator=generator,
    )
    encoders = torch.randn(
        NUM_KV_HEADS,
        HEAD_DIM,
        VALUE_RANK,
        generator=generator,
    )
    observed = fold_value_projection(dense, encoders)
    assert tuple(observed.shape) == (NUM_KV_HEADS, VALUE_RANK, hidden_size)
    for head in range(NUM_KV_HEADS):
        expected = (
            encoders[head].T.float()
            @ dense[head * HEAD_DIM : (head + 1) * HEAD_DIM].float()
        )
        torch.testing.assert_close(observed[head], expected)


def test_group_shared_page_selection_pins_prefix_and_expands_tokens() -> None:
    page_log_mass = torch.tensor(
        [
            [
                [
                    [9.0, 0.0, 5.0, 1.0, 2.0],
                    [9.0, 0.0, 4.0, 1.0, 2.0],
                    [9.0, 0.0, 3.0, 1.0, 2.0],
                    [9.0, 0.0, 2.0, 1.0, 2.0],
                ]
            ]
        ]
    )
    selected = select_group_shared_pages(
        page_log_mass,
        torch.tensor([5 * 32]),
        pages_per_kv_head=2,
        pinned_prefix_pages=1,
    )
    assert set(selected[0, 0].tolist()) == {0, 2}
    tokens = selected_page_tokens(selected, query_heads=4)
    assert tuple(tokens.shape) == (1, 4, 64)
    assert torch.equal(tokens[:, 0], tokens[:, 3])
    assert set((tokens[0, 0] // 32).tolist()) == {0, 2}


def test_loki_physical_union_counts_distinct_tokens_per_gqa_group() -> None:
    selected = torch.tensor(
        [
            [
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
                [0, 4, 5],
                [8, 9, 10],
                [8, 10, 11],
                [9, 11, 12],
                [8, 12, 13],
            ]
        ]
    )
    assert (
        physical_union_count(
            selected,
            kv_heads=2,
            maximum_sequence=16,
        ).item()
        == 12
    )


def test_shard_routing_statistics_are_summed_before_normalization() -> None:
    shards = [
        {
            "runtime": {
                "routing_statistics": {
                    "layers": [
                        {"queries": 2, "physical_tokens": 20, "logical_tokens": 40},
                        {"queries": 2, "physical_tokens": 24, "logical_tokens": 40},
                    ]
                }
            }
        },
        {
            "runtime": {
                "routing_statistics": {
                    "layers": [
                        {"queries": 3, "physical_tokens": 36, "logical_tokens": 60},
                        {"queries": 3, "physical_tokens": 40, "logical_tokens": 60},
                    ]
                }
            }
        },
    ]
    merged = _merge_statistics(shards)
    assert merged["layers"][0] == {
        "queries": 5,
        "physical_tokens": 56,
        "logical_tokens": 100,
    }
    assert merged["totals"]["physical_tokens_per_layer_query"] == 120 / 10
    assert merged["totals"]["logical_tokens_per_layer_query"] == 200 / 10
