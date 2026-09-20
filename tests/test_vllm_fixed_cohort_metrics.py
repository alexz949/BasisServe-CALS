"""Metric checks without a GPU engine."""

from types import SimpleNamespace
from unittest.mock import patch

from evaluation.benchmark_vllm_qwen3_8b_c1 import run_batch


def test_request_tpot_includes_interleaved_prefill():
    events = []
    outputs = [SimpleNamespace(
        request_id=str(index),
        metrics=SimpleNamespace(queued_ts=1.0, scheduled_ts=scheduled,
                                first_token_ts=first, last_token_ts=last,
                                is_corrupted=False),
        outputs=[SimpleNamespace(token_ids=[1, 2, 3])],
    ) for index, (scheduled, first, last) in enumerate([(2., 3., 7.), (4., 5., 8.)])]
    llm = SimpleNamespace(
        get_metrics=lambda: [SimpleNamespace(name="vllm:num_preemptions", value=0)],
        sleep=lambda **kwargs: events.append("pause"),
        enqueue=lambda *args, **kwargs: events.append("enqueue") or ["0", "1"],
        wake_up=lambda **kwargs: events.append("resume"),
        wait_for_completion=lambda **kwargs: events.append("complete") or outputs,
    )
    with patch("evaluation.benchmark_vllm_qwen3_8b_c1.time.perf_counter", side_effect=[10., 16.]):
        result = run_batch(llm, [{}, {}], object(), 3)
    assert events == ["pause", "enqueue", "resume", "complete"]
    assert result["wall_seconds"] == 6.
    assert result["preemptions"] == 0
    assert result["output_tokens_per_second"] == 1.
    assert result["ttft_ms"]["mean"] == 3000.
    assert result["tpot_ms"]["mean"] == 1750.
