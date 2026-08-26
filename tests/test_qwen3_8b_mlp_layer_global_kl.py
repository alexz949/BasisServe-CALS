from __future__ import annotations

import pytest

from evaluation import run_qwen3_8b_mlp_down_c1_layer_global_kl as global_kl


def _synthetic_records() -> list[dict[str, object]]:
    records = []
    for layer in range(global_kl.NUM_LAYERS):
        for rank in global_kl.CANDIDATE_RANKS:
            if rank == global_kl.ANCHOR_RANK:
                continue
            cost = 10.0
            if layer == 0 and rank == 1536:
                cost = -5.0
            if layer == 1 and rank == 3584:
                cost = -4.0
            records.append(
                {
                    "layer": layer,
                    "candidate_rank": rank,
                    "terminal_kl_delta": {
                        "mean": cost,
                        "one_standard_error_ucb": cost + 0.1,
                    },
                }
            )
    return records


def test_parse_ranks_requires_sorted_multiple_of_256_grid() -> None:
    raw = ",".join(map(str, global_kl.CANDIDATE_RANKS))
    assert global_kl._parse_ranks(raw, anchor_rank=2560) == global_kl.CANDIDATE_RANKS
    with pytest.raises(ValueError):
        global_kl._parse_ranks("1536,2560,2048,3584", anchor_rank=2560)
    with pytest.raises(ValueError):
        global_kl._parse_ranks("1536,2048,2560,3500", anchor_rank=2560)
    with pytest.raises(ValueError):
        global_kl._parse_ranks("2560,2816,3072", anchor_rank=2560)


def test_exact_budget_dp_moves_one_donor_receiver_pair() -> None:
    schedule, diagnostics = global_kl._allocate(
        _synthetic_records(),
        ranks=global_kl.CANDIDATE_RANKS,
        anchor_rank=global_kl.ANCHOR_RANK,
        cost_key="mean",
    )
    assert schedule[:2] == (1536, 3584)
    assert schedule[2:] == (global_kl.ANCHOR_RANK,) * (global_kl.NUM_LAYERS - 2)
    assert sum(schedule) == global_kl.NUM_LAYERS * global_kl.ANCHOR_RANK
    assert diagnostics["changed_layers"] == 2
    assert diagnostics["predicted_additive_kl_delta"] == pytest.approx(-9.0)


def test_accounting_preserves_uniform_communication_budget() -> None:
    schedule = (global_kl.ANCHOR_RANK,) * global_kl.NUM_LAYERS
    accounting = global_kl._accounting(
        schedule, anchor_rank=global_kl.ANCHOR_RANK
    )
    assert accounting["average_rank"] == global_kl.ANCHOR_RANK
    assert accounting["communication_fraction_of_dense_allreduce"] == pytest.approx(
        0.625
    )
    assert accounting["communication_reduction_fraction"] == pytest.approx(0.375)
