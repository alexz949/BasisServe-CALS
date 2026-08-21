from __future__ import annotations

import json

import pytest
import torch

from basisserve.core.gqa_routed_ov_joint import quadratic_from_target

from evaluation.allocate_llama2_mha_c1_tp_source_global_kl import (
    CKA_AUDIT_FORMAT,
    NUM_LAYERS,
    _allocate,
    _grouped_collective_cost,
    _head_ranks_from_source_ranks,
    _load_head_groups,
    _fit_fixed_ragged_schedule,
    _require_zero_sweep_factor_banks,
    _schedule_accounting,
    _source_ordered_encoder_groups,
)
from evaluation.pilot_llama2_mha_c1_per_head_fisher_kl import TP_SIZE


def _record(layer: int, source: int, rank: int, cost: float) -> dict:
    return {
        "layer": layer,
        "source": source,
        "candidate_rank": rank,
        "terminal_kl_delta": {
            "mean": cost,
            "one_standard_error_ucb": cost + 0.01,
        },
    }


def test_exact_source_rank_dp_preserves_uniform_v64_budget() -> None:
    records = []
    for layer in range(NUM_LAYERS):
        for source in range(TP_SIZE):
            low_cost = 10.0
            high_cost = 10.0
            if (layer, source) == (0, 0):
                high_cost = -2.0
            if (layer, source) == (0, 1):
                low_cost = 0.1
            records.extend(
                (
                    _record(layer, source, 48, low_cost),
                    _record(layer, source, 80, high_cost),
                )
            )

    schedule, cost, contributions = _allocate(
        records,
        candidate_ranks=(48, 64, 80),
        anchor_rank=64,
        cost_key="mean",
    )

    assert schedule[0][:2] == [80, 48]
    assert all(rank == 64 for rank in schedule[0][2:])
    assert all(rank == 64 for layer in schedule[1:] for rank in layer)
    assert cost == pytest.approx(-1.9)
    assert sum(sum(layer) for layer in schedule) == NUM_LAYERS * TP_SIZE * 64
    assert len(contributions) == NUM_LAYERS * TP_SIZE


def test_exact_source_rank_dp_supports_expanded_rank_grid() -> None:
    records = []
    for layer in range(NUM_LAYERS):
        for source in range(TP_SIZE):
            costs = {32: 10.0, 48: 10.0, 80: 10.0, 96: 10.0}
            if (layer, source) == (0, 0):
                costs[96] = -3.0
            if (layer, source) == (0, 1):
                costs[32] = 0.25
            records.extend(
                _record(layer, source, rank, cost)
                for rank, cost in costs.items()
            )

    schedule, cost, contributions = _allocate(
        records,
        candidate_ranks=(32, 48, 64, 80, 96),
        anchor_rank=64,
        cost_key="mean",
    )

    assert schedule[0][:2] == [96, 32]
    assert all(rank == 64 for rank in schedule[0][2:])
    assert all(rank == 64 for layer in schedule[1:] for rank in layer)
    assert cost == pytest.approx(-2.75)
    assert sum(sum(layer) for layer in schedule) == NUM_LAYERS * TP_SIZE * 64
    assert len(contributions) == NUM_LAYERS * TP_SIZE


def test_schedule_accounting_separates_variable_and_padded_wire() -> None:
    uniform = [[64] * TP_SIZE for _ in range(NUM_LAYERS)]
    adaptive = [list(layer) for layer in uniform]
    adaptive[0][0] = 80
    adaptive[0][1] = 48

    uniform_cost = _schedule_accounting(uniform)
    adaptive_cost = _schedule_accounting(adaptive)

    assert uniform_cost["source_rank_sum"] == NUM_LAYERS * TP_SIZE * 64
    assert uniform_cost["ideal_variable_allgather_total_width"] == 65536
    assert uniform_cost["padded_rectangular_allgather_total_width"] == 65536
    assert adaptive_cost["ideal_variable_allgather_total_width"] == 65536
    assert adaptive_cost["padded_rectangular_allgather_total_width"] == 66048
    assert adaptive_cost["changed_sources_from_uniform_64"] == 2
    assert adaptive_cost["layers_with_padded_overhead"] == 1
    assert adaptive_cost["padded_overhead_vs_uniform_v64"] == pytest.approx(
        512 / 65536
    )


def test_exact_source_rank_dp_preserves_uniform_v96_budget() -> None:
    records = []
    for layer in range(NUM_LAYERS):
        for source in range(TP_SIZE):
            costs = {64: 10.0, 80: 10.0, 112: 10.0}
            if (layer, source) == (0, 0):
                costs[112] = -2.0
            if (layer, source) == (0, 1):
                costs[80] = 0.1
            records.extend(
                _record(layer, source, rank, cost)
                for rank, cost in costs.items()
            )

    schedule, cost, _ = _allocate(
        records,
        candidate_ranks=(64, 80, 96, 112),
        anchor_rank=96,
        cost_key="mean",
    )

    assert schedule[0][:2] == [112, 80]
    assert sum(sum(layer) for layer in schedule) == NUM_LAYERS * TP_SIZE * 96
    assert cost == pytest.approx(-1.9)
    accounting = _schedule_accounting(schedule, anchor_rank=96)
    assert accounting["changed_sources_from_anchor"] == 2
    assert accounting["uniform_anchor_total_width"] == NUM_LAYERS * 32 * 96


def test_noncontiguous_groups_map_one_rank_to_every_head_in_source() -> None:
    groups = tuple(
        tuple(source + 8 * offset for offset in range(4))
        for source in range(8)
    )
    source_ranks = (64, 80, 96, 112, 64, 80, 96, 112)

    head_ranks = _head_ranks_from_source_ranks(source_ranks, groups)
    cost = _grouped_collective_cost(head_ranks, groups)

    for source, group in enumerate(groups):
        assert {head_ranks[head] for head in group} == {source_ranks[source]}
    assert cost["source_widths"] == [4 * rank for rank in source_ranks]
    assert cost["ideal_allgather_width"] == 4 * sum(source_ranks)
    assert cost["padded_allgather_width"] == 8 * 4 * max(source_ranks)


def test_load_cka_groups_requires_all_layers_and_preserves_provenance(tmp_path) -> None:
    groups = [
        [source + 8 * offset for offset in range(4)]
        for source in range(8)
    ]
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "format": CKA_AUDIT_FORMAT,
                "status": "complete",
                "configuration": {"tp_size": 8},
                "aggregate": {"relative_cka_gain": 1.0},
                "records": [
                    {"layer": layer, "cka_groups": groups}
                    for layer in range(NUM_LAYERS)
                ],
            }
        )
    )

    loaded, provenance = _load_head_groups(path)

    assert loaded[0] == tuple(tuple(group) for group in groups)
    assert loaded[-1] == loaded[0]
    assert provenance["method"] == "train_activation_cka_balanced_tp_groups"
    assert provenance["audit_validation_used_for_group_selection"] is False


def _zero_sweep_result(rank: int) -> dict:
    return {
        "fit_config": {
            "cache_rank_per_head": rank,
            "encoder_sweeps": 0,
            "minimum_encoder_sweeps": 0,
            "snapshot_dir": "/tmp/fit",
            "fit_windows": 128,
            "validation_snapshot_dir": "/tmp/heldout",
            "validation_window_start": 384,
            "validation_windows": 64,
            "validation_row_start": 49152,
            "validation_rows": 8192,
        }
    }


def test_post_allocation_refit_requires_zero_sweep_rank_banks() -> None:
    banks = {rank: _zero_sweep_result(rank) for rank in (64, 80, 96, 112)}
    metadata = _require_zero_sweep_factor_banks(banks)

    assert metadata["verified_zero_sweep"] is True
    assert metadata["fit_windows"] == 128
    assert metadata["validation_rows"] == 8192
    assert set(metadata["per_rank"]) == {"64", "80", "96", "112"}

    banks[80]["fit_config"]["encoder_sweeps"] = 10
    banks[80]["fit_config"]["minimum_encoder_sweeps"] = 2
    with pytest.raises(ValueError, match="requires zero-sweep"):
        _require_zero_sweep_factor_banks(banks)


def test_source_ordered_encoder_groups_preserve_cka_ownership_order() -> None:
    groups = ((0, 2), (1, 3))
    assert _source_ordered_encoder_groups(groups, num_heads=4) == (0, 2, 1, 3)
    with pytest.raises(ValueError, match="partition"):
        _source_ordered_encoder_groups(((0, 2), (2, 3)), num_heads=4)


def test_fixed_ragged_post_allocation_sweep_uses_decoder_closed_selection() -> None:
    generator = torch.Generator().manual_seed(17)
    heads, head_dim, hidden, samples = 4, 3, 5, 48
    activations = torch.randn(
        samples,
        heads,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    covariance = torch.einsum(
        "nhd,nke->hkde", activations, activations
    ) / samples
    target = torch.randn(
        heads,
        head_dim,
        hidden,
        generator=generator,
        dtype=torch.float64,
    )
    objective = quadratic_from_target(
        covariance=covariance,
        target=target,
        name="toy_fixed_tp_source_schedule",
        trace_normalize=False,
    )
    ranks = (1, 2, 1, 2)
    maximum_rank = max(ranks)
    initial_A = torch.zeros(heads, head_dim, maximum_rank, dtype=torch.float64)
    initial_D = torch.zeros(heads, maximum_rank, hidden, dtype=torch.float64)
    for head, rank in enumerate(ranks):
        basis, _ = torch.linalg.qr(
            torch.randn(
                head_dim,
                rank,
                generator=generator,
                dtype=torch.float64,
            ),
            mode="reduced",
        )
        initial_A[head, :, :rank] = basis

    selected_A, selected_D, diagnostics = _fit_fixed_ragged_schedule(
        fit_objective=objective,
        validation_objective=objective,
        initial_A=initial_A,
        initial_D=initial_D,
        head_ranks=ranks,
        head_groups=((0, 2), (1, 3)),
        maximum_sweeps=2,
        minimum_sweeps=2,
        relative_objective_tolerance=0.0,
        patience=2,
        decoder_relative_jitter=0.0,
        encoder_relative_damping=1.0e-8,
        cg_relative_tolerance=1.0e-10,
        cg_iterations=16,
        maximum_backtracks=10,
    )

    assert diagnostics["executed_sweeps"] == 2
    assert diagnostics["encoder_group_update_order"] == [0, 2, 1, 3]
    assert all(
        row["encoder_group_order"] == [0, 2, 1, 3]
        for row in diagnostics["sweeps"]
    )
    assert diagnostics["selection"]["boundary"] in {
        "decoder_only",
        "after_redecoder",
    }
    assert all(
        row["selection_eligible"]
        == (row["boundary"] in {"decoder_only", "after_redecoder"})
        for row in diagnostics["selection"]["checkpoints"]
    )
    assert diagnostics["fit"]["selected_loss"] <= (
        diagnostics["fit"]["decoder_only_loss"] + 1.0e-9
    )
    assert torch.count_nonzero(selected_A[0, :, 1:]) == 0
    assert torch.count_nonzero(selected_D[0, 1:]) == 0
