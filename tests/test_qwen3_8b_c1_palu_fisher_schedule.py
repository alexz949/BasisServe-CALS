from __future__ import annotations

import pytest

from evaluation.build_qwen3_8b_c1_palu_fisher_schedule import (
    FISHER_FORMAT,
    NUM_LAYERS,
    fisher_weighted_c1_costs,
    palu_layer_importances,
)


def _fisher() -> dict:
    return {
        "format": FISHER_FORMAT,
        "status": "complete",
        "fisher": {
            "target": "v_proj_only",
            "scalars": {
                f"model.layers.{layer}.self_attn.v_proj": float(layer + 1)
                for layer in range(NUM_LAYERS)
            },
        },
    }


def test_palu_layer_importances_are_ordered_and_mean_normalized() -> None:
    values = palu_layer_importances(_fisher())
    assert len(values) == NUM_LAYERS
    assert sum(values) / len(values) == pytest.approx(1.0)
    assert values[0] < values[-1]


def test_fisher_weighted_costs_anchor_at_zero() -> None:
    local_errors = tuple(
        {32: 4.0, 64: 2.0, 96: 1.0, 128: 0.0}
        for _ in range(NUM_LAYERS)
    )
    importances = tuple(float(layer + 1) for layer in range(NUM_LAYERS))
    costs = fisher_weighted_c1_costs(
        local_errors,
        importances,
        candidate_ranks=(32, 64, 96, 128),
        anchor_rank=64,
        exponent=1.0,
    )
    assert costs[0] == {32: 2.0, 64: 0.0, 96: -1.0, 128: -2.0}
    assert costs[1][32] == pytest.approx(4.0)
    assert costs[1][128] == pytest.approx(-4.0)
