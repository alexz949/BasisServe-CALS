from __future__ import annotations

import pytest

from evaluation.benchmark_qwen3_8b_wo_collectives_tp4 import (
    collective_geometry,
    comparison_metrics,
    parse_positive_csv,
)


def test_collective_grid_parser_rejects_duplicates_and_nonpositive_values() -> None:
    assert parse_positive_csv("512,640,1024", label="ranks") == (512, 640, 1024)
    with pytest.raises(ValueError, match="nonempty and unique"):
        parse_positive_csv("512,512", label="ranks")
    with pytest.raises(ValueError, match="positive"):
        parse_positive_csv("0,512", label="ranks")


def test_tp4_c1_and_lr_geometry_is_exactly_wire_matched() -> None:
    geometry = collective_geometry(source_rank=512, rows=1)

    assert geometry["wire_matched_lr_rank"] == 1024
    assert geometry["c1_and_wire_lr_ideal_ring_bytes_per_rank"] == 3072.0
    assert geometry["dense_ideal_ring_bytes_per_rank"] == 12288.0
    assert geometry["ideal_ring_communication_reduction_vs_dense"] == 0.75
    assert geometry["wire_matching_passed"] is True


def test_entire_requested_rank_grid_preserves_equal_wire_accounting() -> None:
    for source_rank in (512, 640, 768, 896, 1024):
        for rows in (1, 64, 2048, 32768):
            geometry = collective_geometry(source_rank=source_rank, rows=rows)
            assert geometry["wire_matched_lr_rank"] == 2 * source_rank
            assert geometry["wire_matching_passed"] is True


def test_latency_comparisons_keep_packing_separate_from_collective() -> None:
    arms = {
        "dense_allreduce": {"timing": {"p50_ms": 4.0}},
        "c1_allgather_collective_only": {"timing": {"p50_ms": 1.5}},
        "c1_packing_plus_allgather": {"timing": {"p50_ms": 2.0}},
        "wire_lr_allreduce": {"timing": {"p50_ms": 2.5}},
    }

    result = comparison_metrics(arms)

    assert result["c1_collective_latency_change_vs_wire_lr"] == pytest.approx(-0.4)
    assert result["c1_packed_latency_change_vs_wire_lr"] == pytest.approx(-0.2)
    assert result["c1_packing_overhead_vs_collective_only"] == pytest.approx(1 / 3)
    assert result["c1_packed_latency_reduction_vs_dense"] == pytest.approx(0.5)
