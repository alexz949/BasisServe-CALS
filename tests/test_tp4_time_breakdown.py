from __future__ import annotations

import pytest

from evaluation.benchmark_tp4_interconnect import _parse_byte_csv, _parse_bytes
from evaluation.profile_qwen3_8b_tp4_decode_breakdown import _derive_breakdown
from evaluation.summarize_qwen3_8b_tp4_decode_breakdown import (
    _configuration_summary,
)
from evaluation.summarize_qwen3_8b_tp4_uniform_comparison import _comparison_row


def _profile(**means: float) -> dict:
    return {
        "categories": {
            label: {"mean_ms": value}
            for label, value in means.items()
        }
    }


def test_parse_binary_payload_sizes() -> None:
    assert _parse_bytes("8KB") == 8192
    assert _parse_bytes("1MiB") == 1 << 20
    assert _parse_byte_csv("8KB,64KB,1MB") == (8192, 65536, 1 << 20)
    with pytest.raises(ValueError):
        _parse_byte_csv("8KB,8KB")


def test_dense_breakdown_separates_both_all_reduces() -> None:
    macro = _profile(
        decode_total=10.0,
        attention_total=4.0,
        mlp_total=3.0,
        input_rmsnorm=0.2,
        post_attention_rmsnorm=0.2,
        final_rmsnorm=0.1,
        embedding=0.1,
        rotary_embedding=0.1,
        lm_head=0.5,
        distributed_greedy=0.2,
    )
    attention = _profile(
        attention_q_proj=0.5,
        attention_k_proj=0.2,
        attention_v_proj=0.2,
        attention_q_norm=0.1,
        attention_k_norm=0.1,
        attention_core=1.0,
        attention_output_path=1.5,
    )
    mlp = _profile(mlp_gate_proj=0.5, mlp_up_proj=0.5, mlp_down_proj=1.5)
    collectives = _profile(attention_all_reduce=0.8, mlp_all_reduce=0.9)
    result = _derive_breakdown(
        arm="dense",
        e2e={"mean_ms": 9.5},
        macro=macro,
        attention=attention,
        mlp=mlp,
        collectives=collectives,
        collective_ablations={
            "all_main_collectives_e2e_ms": 1.6,
            "all_main_collectives_fraction_of_e2e": 1.6 / 9.5,
        },
    )
    assert result["collective_ablation"][
        "all_main_collectives_e2e_ms"
    ] == pytest.approx(1.6)
    assert result["instrumented_output_path_detail_ms"][
        "instrumented_dense_o_proj_local_gemm_and_dispatch_ms"
    ] == pytest.approx(0.7)
    assert result["instrumented_output_path_detail_ms"][
        "instrumented_mlp_down_local_gemm_and_dispatch_ms"
    ] == pytest.approx(0.6)


def test_c1_summary_uses_trusted_e2e_latency() -> None:
    common = {
        "batch_size": 8,
        "context_length_including_current_token": 2048,
    }
    dense = {
        **common,
        "e2e": {"mean_ms": 32.0},
        "derived": {
            "macro_stage_ms": {"attention_total": 12.0, "mlp_total": 10.0},
            "collective_ablation": {
                "all_main_collectives_e2e_ms": 7.0,
                "attention_collective_marginal_e2e_ms": 4.0,
                "mlp_collective_marginal_e2e_ms": 3.0,
                "variants": {
                    "without_attention_and_mlp_collectives": {"mean_ms": 25.0}
                },
            },
            "instrumented_output_path_detail_ms": {},
            "instrumentation_overhead_percent": 2.0,
        },
    }
    c1 = {
        **common,
        "e2e": {"mean_ms": 28.0},
        "derived": {
            "macro_stage_ms": {"attention_total": 8.0, "mlp_total": 10.0},
            "collective_ablation": {
                "all_main_collectives_e2e_ms": 4.0,
                "attention_collective_marginal_e2e_ms": 1.0,
                "mlp_collective_marginal_e2e_ms": 3.0,
                "variants": {
                    "without_attention_and_mlp_collectives": {"mean_ms": 24.0}
                },
            },
            "instrumented_output_path_detail_ms": {},
            "instrumentation_overhead_percent": 2.5,
        },
    }
    result = _configuration_summary(dense, c1)
    assert result["speedup"] == pytest.approx(32.0 / 28.0)
    assert result["c1_tokens_per_second"] == pytest.approx(8 * 1000.0 / 28.0)


def test_uniform_comparison_uses_matched_e2e_and_ablation_records() -> None:
    def record(e2e: float, attention_collective: float, all_collectives: float) -> dict:
        return {
            "batch_size": 64,
            "context_length_including_current_token": 4096,
            "e2e": {"mean_ms": e2e},
            "derived": {
                "collective_ablation": {
                    "attention_collective_marginal_e2e_ms": attention_collective,
                    "all_main_collectives_e2e_ms": all_collectives,
                }
            },
        }

    result = _comparison_row(
        record(32.0, 3.0, 7.0),
        record(30.0, 0.6, 4.5),
        record(29.7, 0.5, 4.4),
    )
    assert result["uniform_speedup_vs_dense"] == pytest.approx(32.0 / 29.7)
    assert result["uniform_latency_delta_vs_mean_dp_ms"] == pytest.approx(-0.3)
    assert result["uniform_attention_collective_e2e_ms"] == pytest.approx(0.5)
