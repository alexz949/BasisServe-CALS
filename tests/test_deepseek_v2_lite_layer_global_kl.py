from __future__ import annotations

import pytest

from evaluation.run_deepseek_v2_lite_c1_layer_global_kl import (
    NUM_LAYERS,
    _allocate_layer_ranks,
    _cuda_device_indices,
    _parse_factor_dirs,
    _parse_ranks,
    _schedule_accounting,
)


def test_cuda_device_probe_ignores_nvml_only_ordinals(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)

    def get_device_properties(index: int) -> object:
        if index >= 2:
            raise RuntimeError("Invalid device argument")
        return object()

    monkeypatch.setattr("torch.cuda.get_device_properties", get_device_properties)
    assert _cuda_device_indices() == (0, 1)


def _synthetic_records() -> list[dict[str, object]]:
    records = []
    for layer in range(NUM_LAYERS):
        for rank in (64, 96, 160, 192):
            cost = 100.0
            if layer == 0 and rank == 160:
                cost = -10.0
            elif layer == 1 and rank == 96:
                cost = -1.0
            records.append(
                {
                    "layer": layer,
                    "candidate_rank": rank,
                    "terminal_kl_delta": {
                        "mean": cost,
                        "one_standard_error_ucb": cost,
                    },
                }
            )
    return records


def test_exact_dp_preserves_average_rank128_and_moves_one_pair() -> None:
    schedule, total_cost, contributions = _allocate_layer_ranks(
        _synthetic_records(),
        candidate_ranks=(64, 96, 128, 160, 192),
        anchor_rank=128,
        cost_key="mean",
    )

    assert schedule == [160, 96] + [128] * (NUM_LAYERS - 2)
    assert sum(schedule) == NUM_LAYERS * 128
    assert total_cost == -11.0
    assert [row["rank"] for row in contributions[:2]] == [160, 96]


def test_schedule_accounting_reports_fifty_percent_allgather_reduction() -> None:
    schedule = [160, 96] + [128] * (NUM_LAYERS - 2)
    accounting = _schedule_accounting(schedule, anchor_rank=128)

    assert accounting["average_source_rank"] == 128
    assert accounting["retained_ratio_vs_dense_allgather"] == 0.5
    assert accounting["reduction_vs_dense_allgather"] == 0.5
    assert accounting["kv_cache_compression"] == "none"
    assert accounting["ragged_sources_within_layer"] is False


def test_rank_and_factor_directory_parsing_is_strict() -> None:
    assert _parse_ranks("192,64,128,96,160") == (64, 96, 128, 160, 192)
    assert set(_parse_factor_dirs(("64=/tmp/r64", "96=/tmp/r96"))) == {64, 96}
    with pytest.raises(ValueError, match="source width"):
        _parse_ranks("0,128")
    with pytest.raises(ValueError, match="duplicate"):
        _parse_factor_dirs(("64=/tmp/a", "64=/tmp/b"))
