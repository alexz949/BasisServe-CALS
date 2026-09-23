"""CPU-only checks that failed TP8 trials remain reportable."""

from benchmarks.system.run_llama31_8b_tp8_request_grid import failure_details


def test_oom_with_truncated_rank_log_is_retained(tmp_path):
    launcher = tmp_path / "launcher.log"
    launcher.write_text("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")
    (tmp_path / "rank0.log").write_text(
        '{"status": "starting"}\n{"status": "prefill_start"}\n{"status": "svd'
    )
    (tmp_path / "rank1.log").write_text('{"status": "prefill_start"}\n')

    result = failure_details(tmp_path, launcher)

    assert result["oom"] is True
    assert result["allocation_request"] == "2.00 GiB"
    assert result["missing_rank_results"] == list(range(8))
    assert result["last_rank_stages"]["0"] == "prefill_start"
    assert result["last_rank_stages"]["1"] == "prefill_start"
    assert result["last_rank_stages"]["2"] == "not_started"
    assert result["truncated_rank_logs"] == [0]
