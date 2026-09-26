"""Synthetic records test validation and request-level summary arithmetic."""

import json

import pytest

from evaluation.summarize_qwen3_nuq4 import summarize


@pytest.fixture
def grid(tmp_path):
    root = tmp_path / "preflight"
    for name, kernel, seconds in (("dense", "flash", 2), ("r64", "splitk", 1), ("r96", "splitk", 4)):
        path = root / name / f"graph_{kernel}_4096.json"
        path.parent.mkdir(parents=True)
        config = dict(model="synthetic", tensor_parallel_size=8, dtype="bfloat16", max_model_len=4224,
            max_num_seqs=256, max_num_batched_tokens=8192, gpu_memory_utilization=0.8, block_size=16,
            enable_prefix_caching=False, enable_chunked_prefill=True, async_scheduling=False,
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"})
        payload = dict(status="complete", phase="preflight", metric_notes={"output_length": 128},
                       configuration=config, batches=[])
        for batch in (1, 256):
            payload["batches"].append(dict(batch=batch, runs=[dict(wall_seconds=seconds,
                output_tokens_per_second=batch*128/seconds, ttft_ms={"median": 10}, tpot_ms={"median": 5},
                preemptions=0, requests=[{"output_token_ids": [0]*128} for _ in range(batch)])],
                workers=[dict(peak_cuda_allocated_bytes=2**30, overflow_layers=[], loaded_value_layers=36,
                              capture_calls=1) for _ in range(8)]))
        path.write_text(json.dumps(payload))
    return root


def test_request_speedup(grid):
    rows, _, _ = summarize(grid)
    assert len(rows) == 6
    assert {row["arm"]: row["request_speedup"] for row in rows} == {"dense": 1, "r64": 2, "r96": 0.5}
    assert all(row["peak_allocated_gib"] == 1 for row in rows)


@pytest.mark.parametrize("fault", ["incomplete", "overflow", "configuration", "output_length"])
def test_reject_invalid_comparison(grid, fault):
    path = grid / "r64/graph_splitk_4096.json"
    payload = json.loads(path.read_text())
    if fault == "incomplete":
        payload["status"] = "running"
    elif fault == "overflow":
        payload["batches"][0]["workers"][0]["overflow_layers"] = [3]
    elif fault == "configuration":
        payload["configuration"]["max_num_batched_tokens"] = 16384
    else:
        payload["batches"][0]["runs"][0]["requests"][0]["output_token_ids"] = [0]*4
    path.write_text(json.dumps(payload))
    with pytest.raises(AssertionError):
        summarize(grid)
