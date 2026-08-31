from __future__ import annotations

from types import SimpleNamespace

import pytest

from basisserve.core.c1_e2e_latency import (
    DecoderWave,
    EffectiveNetwork,
    barrier_c1_boundary,
    pipelined_c1_boundary,
    ragged_maximum_received_bytes,
    readiness_waves,
    ring_allreduce_bytes_per_rank,
    validate_decoder_waves,
)
from evaluation.benchmark_qwen3_32b_c1_variable_v_decode import (
    FORMAT as ATTENTION_FORMAT,
)
from evaluation.simulate_qwen3_32b_c1_full_decode import (
    _attention_curves,
    _full_step_for_percentile,
    _scaled_decoder_waves,
)


def test_effective_network_uses_decimal_gigabytes_per_second() -> None:
    network = EffectiveNetwork(latency_us=20.0, bandwidth_gbps=3.125)
    assert network.transfer_ms(256 * 1024) == pytest.approx(
        0.020 + (256 * 1024) / 3.125e6
    )


def test_readiness_waves_are_balanced_and_rank_ordered() -> None:
    ranks = (64, 128, 32, 96, 48, 112, 80, 64)
    assert readiness_waves(ranks, 2) == ((2, 4, 0, 7), (6, 3, 5, 1))
    assert readiness_waves(ranks, 3) == ((2, 4, 0), (7, 6, 3), (5, 1))
    assert readiness_waves(ranks, 8) == (
        (2,),
        (4,),
        (0,),
        (7,),
        (6,),
        (3,),
        (5,),
        (1,),
    )


def test_tp8_byte_accounting_matches_decode_shapes() -> None:
    widths = (512,) * 7 + (1024,)
    assert (
        ragged_maximum_received_bytes(
            widths,
            batch=256,
            element_bytes=2,
        )
        == 2 * 1024 * 1024
    )
    assert (
        ring_allreduce_bytes_per_rank(
            batch=256,
            width=5120,
            world_size=8,
            element_bytes=2,
        )
        == 4_587_520
    )


def test_pipeline_reports_communication_only_and_ideal_overlap_bounds() -> None:
    ready = (0.0, 10.0)
    widths = (10, 20)
    waves = (DecoderWave((0,), 2.0), DecoderWave((1,), 2.0))
    network = EffectiveNetwork(latency_us=1000.0, bandwidth_gbps=1e9)
    barrier = barrier_c1_boundary(
        ready,
        widths,
        batch=1,
        element_bytes=2,
        decoder_ms=4.0,
        network=network,
    )
    communication_only = pipelined_c1_boundary(
        ready,
        widths,
        waves,
        batch=1,
        element_bytes=2,
        network=network,
        overlap_decoder_with_local_attention=False,
    )
    ideal = pipelined_c1_boundary(
        ready,
        widths,
        waves,
        batch=1,
        element_bytes=2,
        network=network,
        overlap_decoder_with_local_attention=True,
    )
    assert ideal.completion_ms < communication_only.completion_ms
    assert communication_only.completion_ms < barrier.completion_ms
    assert barrier.completion_ms == pytest.approx(15.0)


def test_decoder_waves_must_cover_every_source_once() -> None:
    with pytest.raises(ValueError, match="cover every source"):
        validate_decoder_waves(
            (DecoderWave((0, 1), 1.0), DecoderWave((1, 2), 1.0)),
            world_size=3,
        )


def _timing(p50: float, p95: float) -> dict[str, float]:
    return {
        "minimum_ms": p50,
        "p50_ms": p50,
        "p95_ms": p95,
        "maximum_ms": p95,
    }


def test_attention_profile_reduces_sampled_layers_to_rank_curves() -> None:
    records = []
    for layer, compact_ms in ((0, 2.0), (1, 4.0)):
        records.append(
            {
                "layer": layer,
                "source_rank": 64,
                "batch": 256,
                "context_length": 4096,
                "timings": {
                    "compact": {
                        "cache_append": _timing(0.1, 0.2),
                        "attention": _timing(compact_ms, compact_ms + 1.0),
                    },
                    "dense": {
                        "cache_append": _timing(0.2, 0.3),
                        "attention": _timing(5.0, 6.0),
                    },
                },
            }
        )
    reduced = _attention_curves(
        {"format": ATTENTION_FORMAT, "records": records},
        batch=256,
        context_length=4096,
        required_source_ranks=(64,),
    )
    assert reduced["sampled_layers"] == [0, 1]
    assert reduced["curves"]["p50_ms"]["64"]["compact_attention_ms"] == 3.0
    assert reduced["curves"]["p95_ms"]["64"]["compact_attention_ms"] == 4.0


def test_decoder_wave_scaling_matches_measured_full_path() -> None:
    waves = _scaled_decoder_waves(
        {
            "full": _timing(10.0, 20.0),
            "waves": [
                {"sources": [0], "timing": _timing(2.0, 3.0)},
                {"sources": [1], "timing": _timing(3.0, 5.0)},
            ],
        },
        percentile="p50_ms",
    )
    assert [wave.duration_ms for wave in waves] == pytest.approx([4.0, 6.0])
    assert sum(wave.duration_ms for wave in waves) == pytest.approx(10.0)


def test_full_step_compositor_connects_dense_and_all_c1_paths() -> None:
    source_ranks = (64,) * 8
    loader = SimpleNamespace(
        geometry=SimpleNamespace(
            hidden_size=5120,
            tp_size=8,
            query_heads_per_rank=8,
            head_dim=128,
        ),
        schedule=(source_ranks,),
    )
    invariant_names = (
        "input_rmsnorm",
        "qk_headnorm_rope",
        "post_attention_residual_rmsnorm",
        "mlp_full",
        "mlp_residual",
        "dense_o_proj_local",
        "final_rmsnorm",
        "lm_head_local",
        "greedy_local_argmax",
    )
    invariant = {name: _timing(0.1, 0.2) for name in invariant_names}

    def decoder_path(groups: tuple[tuple[int, ...], ...]) -> dict[str, object]:
        return {
            "full": _timing(float(len(groups)), 2.0 * len(groups)),
            "waves": [
                {"sources": list(group), "timing": _timing(1.0, 2.0)}
                for group in groups
            ],
        }

    component = {
        "baseline": {
            "rank_invariant": invariant,
            "fused_qkv_by_source_rank": {
                "64": _timing(0.3, 0.4),
                "128": _timing(0.4, 0.5),
            },
            "scheduler_host": _timing(0.01, 0.02),
            "scheduler_h2d": _timing(0.01, 0.02),
        },
        "decoder_layers": {
            "0": {
                "source_ranks": list(source_ranks),
                "source_widths": [512] * 8,
                "paths": {
                    "big": decoder_path((tuple(range(8)),)),
                    "wave_2": decoder_path((tuple(range(4)), tuple(range(4, 8)))),
                    "wave_3": decoder_path(((0, 1, 2), (3, 4, 5), (6, 7))),
                    "partial_8": decoder_path(tuple((source,) for source in range(8))),
                },
            }
        },
    }
    rank_curve = {
        "64": {"compact_cache_append_ms": 0.1, "compact_attention_ms": 2.0},
        "dense": {"dense_cache_append_ms": 0.1, "dense_attention_ms": 3.0},
    }
    result = _full_step_for_percentile(
        percentile="p50_ms",
        loader=loader,
        attention={"curves": {"p50_ms": rank_curve}},
        component=component,
        network=EffectiveNetwork(latency_us=20.0, bandwidth_gbps=3.125),
        batch=256,
        element_bytes=2,
    )
    paths = {row["path"] for row in result["aggregate"]}
    assert paths == {
        "dense_tp8",
        "barrier_big",
        "barrier_wave_2",
        "barrier_wave_3",
        "barrier_partial_8",
        "comm_overlap_wave_2",
        "comm_overlap_wave_3",
        "comm_overlap_partial_8",
        "ideal_overlap_wave_2",
        "ideal_overlap_wave_3",
        "ideal_overlap_partial_8",
    }
    assert all(row["decode_step_ms"] > 0.0 for row in result["aggregate"])
