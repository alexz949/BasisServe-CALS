from __future__ import annotations

import pytest

from evaluation.benchmark_qwen3_8b_wo_decoder_gemm import (
    comparison_metrics,
    decoder_geometry,
    gemm_work,
    parse_positive_csv,
    performance_metrics,
)


def test_decoder_grid_parser_rejects_duplicates_and_nonpositive_values() -> None:
    assert parse_positive_csv("1,8,256", label="rows") == (1, 8, 256)
    with pytest.raises(ValueError, match="nonempty and unique"):
        parse_positive_csv("1,1", label="rows")
    with pytest.raises(ValueError, match="positive"):
        parse_positive_csv("0,1", label="rows")


def test_tp4_decoder_geometry_matches_uniform_rank_sweep() -> None:
    geometry = decoder_geometry(local_rank=640, rows=32)

    assert geometry["c1_total_decoder_rank"] == 2560
    assert geometry["wire_matched_lr_decoder_rank"] == 1280
    assert geometry["dense_local_reduction_width"] == 1024
    assert geometry["c1_decoder_flop_ratio_vs_dense_local_o_proj"] == 2.5
    assert geometry["wire_lr_decoder_flop_ratio_vs_dense_local_o_proj"] == 1.25


def test_gemm_work_counts_fma_as_two_flops_and_all_tensors_once() -> None:
    work = gemm_work(rows=2, reduction_width=3, output_width=5, dtype_bytes=2)

    assert work["flops"] == 60
    assert work["minimum_algorithmic_bytes"] == 62
    assert work["arithmetic_intensity_flops_per_minimum_byte"] == pytest.approx(
        60 / 62
    )


def test_performance_metrics_use_p50_kernel_time() -> None:
    metrics = performance_metrics(
        rows=100,
        reduction_width=200,
        output_width=300,
        p50_ms=2.0,
    )

    assert metrics["flops"] == 12_000_000
    assert metrics["tflops_at_p50"] == pytest.approx(0.006)


def test_decoder_comparison_does_not_mix_collective_latency() -> None:
    arms = {
        "dense_local_o_proj_gemm": {"timing": {"p50_ms": 1.0}},
        "c1_feature_major_decoder_gemm": {"timing": {"p50_ms": 2.0}},
        "wire_lr_token_major_decoder_gemm": {"timing": {"p50_ms": 1.5}},
    }

    comparisons = comparison_metrics(arms)

    assert comparisons["c1_latency_change_vs_dense_local_o_proj"] == 1.0
    assert comparisons["wire_lr_latency_change_vs_dense_local_o_proj"] == 0.5
    assert comparisons["c1_latency_change_vs_wire_lr"] == pytest.approx(1 / 3)
