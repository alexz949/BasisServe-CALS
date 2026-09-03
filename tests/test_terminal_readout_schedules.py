from __future__ import annotations

import pytest

from evaluation.build_qwen3_8b_c1_terminal_readout_schedules import (
    full_readout_costs,
    probe_readout_costs,
)


def _source() -> dict:
    records = []
    for layer in range(36):
        for rank, kl_delta, nll_delta in (
            (32, 0.4, 0.8),
            (96, -0.1, -0.2),
            (128, -0.2, -0.4),
        ):
            records.append(
                {
                    "layer": layer,
                    "candidate_rank": rank,
                    "terminal_kl_delta": {"mean": kl_delta},
                    "nll_delta": {"mean": nll_delta},
                }
            )
    return {"profile": {"records": records}}


def test_full_nll_costs_preserve_recorded_interventions() -> None:
    costs = full_readout_costs(
        _source(),
        readout="nll",
        candidate_ranks=(32, 64, 96, 128),
        anchor_rank=64,
    )

    assert len(costs) == 36
    assert costs[0] == {32: 0.8, 64: 0.0, 96: -0.2, 128: -0.4}


def test_probe_nll_uses_same_local_curve_with_readout_specific_sensitivity() -> None:
    local_errors = tuple(
        {32: 0.4, 64: 0.2, 96: 0.1, 128: 0.0}
        for _ in range(36)
    )
    costs, sensitivities = probe_readout_costs(
        _source(),
        local_errors,
        readout="nll",
        candidate_ranks=(32, 64, 96, 128),
        anchor_rank=64,
        probe_rank=96,
        exponent=1.0,
    )

    assert sensitivities == pytest.approx((2.0,) * 36)
    assert costs[0] == pytest.approx({32: 0.4, 64: 0.0, 96: -0.2, 128: -0.4})
