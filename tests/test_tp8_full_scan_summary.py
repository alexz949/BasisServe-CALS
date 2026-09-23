import pytest

from benchmarks.system import summarize_tp8_full_scan as summary


@pytest.mark.parametrize("log,expected", [
    ('torch.OutOfMemoryError: CUDA out of memory.', "gpu_oom_setup"),
    ('{"status": "model_ready"}\ntorch.OutOfMemoryError: CUDA out of memory.', "gpu_oom_prefill"),
    ('{"status": "prefill_complete"}\ntorch.OutOfMemoryError: CUDA out of memory.', "gpu_oom_decode"),
    ('MemoryError: Cannot allocate memory', "host_oom"),
    ('Process killed by signal 9', "process_failure"),
])
def test_failure_classification(log, expected):
    assert summary.classify_failure(log)[0] == expected


def rows():
    result = []
    for arm, scale in zip(summary.ARMS, (4, 3, 2)):
        for cohort, latency in enumerate((1, 2, 9)):
            result.append(dict(arm=arm, prompt_tokens=4096, batch=1, cohort=cohort,
                               status="complete", routing_mode="test",
                               **{metric: scale * latency for metric in summary.METRICS}))
    return result


def test_three_cohort_medians(monkeypatch):
    monkeypatch.setattr(summary, "MATRIX", {4096: (1,)})
    groups = summary.aggregate(rows())
    assert [r["mean_ms"] for r in groups] == [8, 6, 4]
    assert groups[2]["speedup_vs_dense"] == 2
    assert groups[2]["speedup_vs_als_full"] == 1.5


def test_failed_cohort_suppresses_median_and_speedup(monkeypatch):
    monkeypatch.setattr(summary, "MATRIX", {4096: (1,)})
    trials = rows()
    trials[0]["status"] = "gpu_oom_prefill"
    groups = summary.aggregate(trials)
    assert groups[0]["successful_trials"] == 2
    assert all(groups[0][metric] is None for metric in summary.METRICS)
    assert all(row["speedup_vs_dense"] is None for row in groups)
    assert groups[2]["speedup_vs_als_full"] == 1.5
