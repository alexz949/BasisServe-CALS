from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from basisserve.core.qwen3_8b_wo_tp4 import ARMS
from evaluation.benchmark_qwen3_8b_wo_cuda_graph import (
    _assert_same_tokens,
    graph_decode_step,
    parse_configurations,
    timing_summary,
)
from evaluation.summarize_qwen3_8b_wo_cuda_graph import (
    comparison_rows,
    validate_matched_protocol,
)


def test_wo_cuda_graph_has_five_scientifically_distinct_arms() -> None:
    assert ARMS == (
        "dense",
        "wo_lr_ar_wire",
        "wo_lr_ar_capacity",
        "wo_c1_ag",
        "wo_c1_local_ar",
    )


def test_cuda_graph_configuration_parser() -> None:
    assert parse_configurations("1x512,64x4096") == ((1, 512), (64, 4096))
    with pytest.raises(ValueError, match="nonempty and unique"):
        parse_configurations("1x512,1x512")
    with pytest.raises(ValueError, match="positive"):
        parse_configurations("0x512")


def test_cuda_graph_timing_summary_reports_population_std() -> None:
    summary = timing_summary((1.0, 2.0, 3.0, 4.0))

    assert summary["mean_ms"] == 2.5
    assert summary["median_ms"] == 2.5
    assert summary["p95_ms"] == 4.0
    assert summary["std_ms"] == pytest.approx(1.11803398875)


def test_graph_decode_passes_explicit_resolved_no_mask(monkeypatch) -> None:
    observed = {}

    class FakeBackbone:
        def __call__(self, **kwargs):
            observed.update(kwargs)
            return SimpleNamespace(last_hidden_state=torch.ones(2, 1, 3))

    model = SimpleNamespace(
        model=FakeBackbone(),
        lm_head=SimpleNamespace(weight=torch.ones(5, 3)),
    )
    monkeypatch.setattr(
        "evaluation.benchmark_qwen3_8b_wo_cuda_graph._distributed_greedy",
        lambda logits: logits.argmax(dim=-1),
    )

    tokens = graph_decode_step(model, torch.ones(2, dtype=torch.int64), position=512)

    assert observed["attention_mask"] == {"full_attention": None}
    assert observed["use_cache"] is False
    assert tokens.shape == (2,)


def test_token_gate_reports_the_mismatch() -> None:
    with pytest.raises(
        AssertionError,
        match=r"graph replay: 1/3 tokens differ; expected prefix=\[1, 2, 3\]",
    ):
        _assert_same_tokens(
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 9, 3]),
            label="graph replay",
        )


def _payload(arm: str, latency: float) -> dict:
    return {
        "arm": arm,
        "model": {"config_sha256": "model"},
        "phase1": {"sha256": "factors"},
        "protocol": {
            "configurations": [
                {"batch_size": 8, "fixed_context_length": 512}
            ],
            "warmup": 10,
            "repeats": 50,
            "dtype": "bfloat16",
        },
        "environment": {
            "gpu": "L40S",
            "torch": "2.6",
            "cuda": "12.4",
            "nccl": "2.21.5",
        },
        "records": [
            {
                "batch_size": 8,
                "fixed_context_length": 512,
                "cuda_graph": {
                    "mean_ms": latency,
                    "median_ms": latency,
                    "p95_ms": latency * 1.1,
                    "std_ms": latency * 0.01,
                },
                "cuda_graph_tokens_per_second": 8000.0 / latency,
                "memory": {"peak_allocated_bytes_per_rank": 1024},
            }
        ],
    }


def test_five_arm_summary_rejects_protocol_mismatch() -> None:
    payloads = [_payload(arm, 2.0) for arm in ARMS]
    validate_matched_protocol(payloads)

    payloads[-1]["protocol"]["repeats"] = 49
    with pytest.raises(ValueError, match="protocol mismatch"):
        validate_matched_protocol(payloads)


def test_five_arm_summary_computes_speedup_against_dense() -> None:
    payloads = [
        _payload(arm, 2.0 if arm == "dense" else 1.0)
        for arm in ARMS
    ]

    rows = comparison_rows(payloads)
    c1 = next(row for row in rows if row["arm"] == "wo_c1_ag")

    assert c1["speedup_over_dense"] == 2.0
    assert c1["latency_reduction_vs_dense"] == 0.5
