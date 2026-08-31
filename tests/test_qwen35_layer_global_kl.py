from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation.eval_qwen35_hybrid_private_ag_ppl import _private_metadata
from evaluation.run_qwen35_9b_c1_layer_global_kl import (
    ANCHOR_RANK,
    CANDIDATE_RANKS,
    NUM_LAYERS,
    _accounting,
    _allocate,
    _parse_ranks,
)


def _record(layer: int, rank: int, cost: float) -> dict[str, object]:
    return {
        "layer": layer,
        "candidate_rank": rank,
        "terminal_kl_delta": {"mean": cost},
    }


def test_parse_ranks_requires_sorted_grid_and_anchor() -> None:
    raw = ",".join(map(str, CANDIDATE_RANKS))
    assert _parse_ranks(raw, anchor_rank=ANCHOR_RANK) == CANDIDATE_RANKS

    with pytest.raises(ValueError, match="distinct increasing"):
        _parse_ranks("192,128,256", anchor_rank=ANCHOR_RANK)
    with pytest.raises(ValueError, match="include the anchor"):
        _parse_ranks("128,256", anchor_rank=ANCHOR_RANK)


def test_allocate_preserves_exact_average_rank() -> None:
    ranks = (128, 192, 256)
    records = []
    for layer in range(NUM_LAYERS):
        # A balanced transfer is strongly preferred over the uniform anchor:
        # layer 0 receives 64 rank while layer 1 donates 64 rank.
        records.append(_record(layer, 128, -2.0 if layer == 1 else 1.0))
        records.append(_record(layer, 256, -3.0 if layer == 0 else 1.0))

    schedule, allocation = _allocate(
        records,
        ranks=ranks,
        anchor_rank=ANCHOR_RANK,
    )

    assert len(schedule) == NUM_LAYERS
    assert sum(schedule) == NUM_LAYERS * ANCHOR_RANK
    assert schedule[0] == 256
    assert schedule[1] == 128
    assert allocation["changed_layers"] == 2
    assert allocation["predicted_additive_kl_delta"] == pytest.approx(-5.0)


def test_accounting_matches_uniform_r192_communication() -> None:
    accounting = _accounting(
        [ANCHOR_RANK] * NUM_LAYERS,
        anchor_rank=ANCHOR_RANK,
    )

    assert accounting["average_local_rank"] == 192
    assert accounting["communication_fraction_of_dense_allreduce"] == 0.1875
    assert accounting["communication_reduction_fraction"] == 0.8125
    assert accounting["rank_histogram"] == {"192": NUM_LAYERS}

    with pytest.raises(ValueError, match="exact average-rank budget"):
        _accounting([128] * NUM_LAYERS, anchor_rank=ANCHOR_RANK)


def test_private_metadata_supports_ragged_layer_ranks() -> None:
    records = (
        SimpleNamespace(
            layer_index=3,
            tp_size=8,
            local_width=512,
            local_rank=128,
            total_private_rank=1024,
        ),
        SimpleNamespace(
            layer_index=1,
            tp_size=8,
            local_width=512,
            local_rank=256,
            total_private_rank=2048,
        ),
    )

    metadata = _private_metadata(records)

    assert metadata["layer_indices"] == [1, 3]
    assert metadata["local_rank_schedule"] == {"1": 256, "3": 128}
    assert metadata["local_rank_histogram"] == {"128": 1, "256": 1}
    assert metadata["average_local_rank"] == 192
    assert metadata["communication_fraction_of_dense_allreduce"] == 0.1875
