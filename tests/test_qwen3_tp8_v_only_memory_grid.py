"""CPU checks for TP8 V-only memory-grid accounting."""

import csv
import json
import sys

from benchmarks.system import summarize_qwen3_8b_tp8_v_only_memory as summary_module

from benchmarks.system.run_qwen3_8b_tp8_v_only_memory_grid import (
    FULL_GRID,
    failure_details,
)
from benchmarks.system.summarize_qwen3_8b_tp8_v_only_memory import (
    group_status,
    plot_memory,
    status_of,
    theoretical_state_bytes,
)


def test_theoretical_state_uses_actual_tp8_placement():
    ranks = [1024 if index in (0, 1, 31) else 512 for index in range(36)]
    basis = theoretical_state_bytes("basis_v64", 4096, 8, None)
    star = theoretical_state_bytes("star_v_adaptive", 4096, 8, ranks)
    assert basis["dense_key_cache"] == star["dense_key_cache"]
    assert basis["value_cache"] == 8 * 4097 * 36 * 64 * 2
    assert star["value_cache"] == 8 * 4097 * (3 * 128 + 33 * 512) * 2


def test_oom_group_never_becomes_complete():
    rows = [{"status": "failed_oom_cache_state_allocation"}]
    assert group_status(rows) == "oom_or_mixed"
    assert group_status([{"status": "complete"}]) == "complete"
    assert status_of({"status": "failed", "failure": {
        "oom": True, "phase": "transient_prefill_workspace"
    }}) == "failed_oom_transient_prefill_workspace"
    assert len(FULL_GRID["arms"]) * len(FULL_GRID["contexts"]) * len(FULL_GRID["batches"]) * len(FULL_GRID["cohorts"]) == 36


def test_failure_keeps_phase_allocation_and_missing_ranks(tmp_path):
    launcher = tmp_path / "launcher.log"
    launcher.write_text("[rank3]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")
    (tmp_path / "rank0.log").write_text(json.dumps({"status": "cache_allocate_layer", "layer": 19}) + "\n")
    details = failure_details(tmp_path, launcher)
    assert details["oom"]
    assert details["phase"] == "unclassified"
    assert details["oom_ranks"] == [3]
    assert details["oom_rank_stages"] == {"3": "not_started"}
    assert details["allocation_request"] == "2.00 GiB"
    assert details["last_rank_stages"]["0"] == "cache_allocate_layer"
    assert details["missing_rank_results"] == list(range(8))


def test_oom_phase_uses_failing_rank_stage(tmp_path):
    launcher = tmp_path / "launcher.log"
    launcher.write_text("[rank3]: torch.OutOfMemoryError: CUDA out of memory\n")
    (tmp_path / "rank3.log").write_text(json.dumps({"status": "cache_allocate_layer"}) + "\n")
    (tmp_path / "rank0.log").write_text(json.dumps({"status": "prefill_chunk_start"}) + "\n")
    details = failure_details(tmp_path, launcher)
    assert details["phase"] == "cache_state_allocation"
    assert details["oom_rank_stages"] == {"3": "cache_allocate_layer"}


def test_plot_marks_oom_without_inventing_a_resident_value(tmp_path):
    rows = [
        {"arm": "basis_v64", "batch": 1, "prompt_tokens": 4096,
         "status": "complete", "decode_ready_nvml_max_rank_gib": 4.0, "oom_cohorts": 0},
        {"arm": "star_v_adaptive", "batch": 8, "prompt_tokens": 130048,
         "status": "oom_or_mixed", "decode_ready_nvml_max_rank_gib": None, "oom_cohorts": 1},
    ]
    assert plot_memory(rows, tmp_path)
    assert (tmp_path / "starkv_tp8_memory.pdf").stat().st_size > 0
    assert (tmp_path / "starkv_tp8_memory.png").stat().st_size > 0


def test_single_cohort_summary_preserves_max_rank_and_oom(tmp_path, monkeypatch):
    root = tmp_path / "run"
    complete_dir = root / "raw" / "basis"
    complete_dir.mkdir(parents=True)
    state = {**theoretical_state_bytes("basis_v64", 4096, 1, None), "value_factors": 1024}
    for rank in range(8):
        memory = {phase: {
            **{field: 4 * 2 ** 30 for field in summary_module.FIELDS},
            "nvml_process_bytes": (4 * 2 ** 30) + rank * (2 ** 30 // 8),
        } for phase in ("cache_allocated", "prefill_complete", "decode_ready", "decode_complete")}
        record = {
            "schema": "basisserve.qwen3_8b.tp8_v_only_memory.v1", "status": "complete",
            "rank": rank, "tp": 8, "dp": 1, "pp": 1, "arm": "basis_v64",
            "prompt_tokens": 4096, "batch": 1, "cohort": 0, "chunk_size": 4096,
            "dense_key": True, "full_attention": True, "key_offload": False,
            "value_offload": False, "dtype": "bfloat16", "generated_token_ids": [[10], [11]],
            "star_value_ranks": None, "state_bytes": state, "memory": memory,
            "metadata": {name: "test" for name in (
                "git_commit", "git_status", "pytorch", "cuda", "nccl",
                "transformers", "flash_attn", "gpu_name")},
        }
        (complete_dir / f"rank{rank}.json").write_text(json.dumps(record))
    failed_dir = root / "raw" / "star"
    failed_dir.mkdir()
    manifest = {
        "grid": FULL_GRID, "status": "partial", "attempts": [
            {"arm": "basis_v64", "prompt_tokens": 4096, "batch": 1, "cohort": 0,
             "attempt": 0, "status": "complete", "output_dir": str(complete_dir)},
            {"arm": "star_v_adaptive", "prompt_tokens": 4096, "batch": 1, "cohort": 0,
             "attempt": 0, "status": "failed", "output_dir": str(failed_dir),
             "launcher_log": str(failed_dir / "launcher.log"), "returncode": 1,
             "failure": {"oom": True, "oom_ranks": [3],
                         "oom_rank_stages": {"3": "cache_allocate_layer"},
                         "phase": "cache_state_allocation"}},
        ],
    }
    (root / "grid_trials.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["summary", "--output-root", str(root)])
    summary_module.main()

    with (root / "summary.csv").open(newline="") as stream:
        trials = list(csv.DictReader(stream))
    with (root / "memory_summary.csv").open(newline="") as stream:
        grouped = list(csv.DictReader(stream))
    with (root / "oom_frontier.csv").open(newline="") as stream:
        frontier = list(csv.DictReader(stream))
    with (root / "failures.csv").open(newline="") as stream:
        failures = list(csv.DictReader(stream))
    assert len(trials) == len(grouped) == 36
    assert [row["status"] for row in trials].count("complete") == 1
    assert [row["status"] for row in trials].count("failed_oom_cache_state_allocation") == 1
    basis = next(row for row in grouped if row["arm"] == "basis_v64" and row["batch"] == "1"
                 and row["prompt_tokens"] == "4096")
    assert float(basis["decode_ready_nvml_max_rank_gib"]) == 4.875
    assert not any("median" in field for field in basis)
    star = next(row for row in grouped if row["arm"] == "star_v_adaptive" and row["batch"] == "1"
                and row["prompt_tokens"] == "4096")
    assert star["decode_ready_nvml_max_rank_gib"] == ""
    assert next(row for row in frontier if row["arm"] == "star_v_adaptive" and row["batch"] == "1")[
        "first_cache_state_oom_context"] == "4096"
    assert len(failures) == 1 and failures[0]["oom_rank_stages"] == '{"3": "cache_allocate_layer"}'
