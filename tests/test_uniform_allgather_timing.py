import pytest
import torch

from benchmarks import bench_uniform_allgather as benchmark


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("explicit_stream", [False, True])
def test_measure_uses_timing_stream_and_restores_caller(monkeypatch, explicit_stream):
    monkeypatch.setattr(benchmark.dist, "barrier", lambda: None)
    monkeypatch.setattr(benchmark.dist, "all_reduce", lambda tensor, op: None)
    device = torch.device("cuda", torch.cuda.current_device())
    caller = torch.cuda.Stream(device=device)
    timing = torch.cuda.Stream(device=device) if explicit_stream else None
    expected = timing if explicit_stream else caller
    output = torch.empty(16, device=device)
    observed = []

    def run_once():
        observed.append(torch.cuda.current_stream(device).cuda_stream)
        output.fill_(len(observed))
        return output

    with torch.cuda.stream(caller):
        result, samples = benchmark.measure(
            run_once, warmup=2, iters=3, device=device, timing_stream=timing
        )
        assert torch.cuda.current_stream(device) == caller
    assert observed == [expected.cuda_stream] * 5
    assert result is output and len(samples) == 3
    assert all(sample >= 0 for sample in samples)
    torch.testing.assert_close(output, torch.full_like(output, 5))
