"""Admission must be inside E2E and every output length must be checked."""
from types import SimpleNamespace
from evaluation.benchmark_kcuu_tp4 import run_batch


def test_e2e_includes_enqueue_and_wakeup(monkeypatch):
    import evaluation.benchmark_kcuu_tp4 as benchmark
    clock = [0.0]
    monkeypatch.setattr(benchmark.time, 'perf_counter', lambda: clock[0])

    class Engine:
        def get_metrics(self):
            return []
        def sleep(self, **kwargs):
            clock[0] += 10  # Pausing itself is preparation, outside E2E.
        def enqueue(self, prompts, sampling, **kwargs):
            clock[0] += 1
            return ['request']
        def wake_up(self, **kwargs):
            clock[0] += 2
        def wait_for_completion(self, **kwargs):
            clock[0] += 4
            return [SimpleNamespace(request_id='request', prompt_token_ids=[1]*8192,
                outputs=[SimpleNamespace(token_ids=[2]*128)],
                metrics=SimpleNamespace(is_corrupted=False,queued_ts=11,scheduled_ts=13,
                                        first_token_ts=14,last_token_ts=17))]

    result = run_batch(Engine(), [{'prompt_token_ids':[1]*8192}], None)
    assert result['wall_seconds'] == 7
    assert result['successful_requests'] == 1 and result['total_output_tokens'] == 128
    assert result['output_tokens_per_second'] == 128/7
    assert result['ttft_ms']['median'] == 3000
