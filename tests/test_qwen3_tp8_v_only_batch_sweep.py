"""CPU checks for the fixed 16K TP8 V-only batch sweep."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import torch

from benchmarks.system.bench_qwen3_8b_tp8_v_only_memory import select_prompts
from benchmarks.system import summarize_qwen3_8b_tp8_v_only_batch_sweep as summary_module
from benchmarks.system.run_qwen3_8b_tp8_v_only_batch_sweep import GRID, trial_command
from benchmarks.system.summarize_qwen3_8b_tp8_v_only_batch_sweep import (
    expected_cache_bytes,
    load_trial,
    plot_memory,
)


def test_batch_grid_and_command(tmp_path):
    assert len(GRID["arms"]) * len(GRID["batches"]) == 18
    assert GRID["prompt_tokens"] == 16384
    assert GRID["chunk_size"] == 256
    assert GRID["output_tokens"] == GRID["reserve_decode_tokens"] == 128
    cmd = trial_command("basis_v64", 256, tmp_path)
    assert cmd[cmd.index("--batch") + 1] == "256"
    assert cmd[cmd.index("--output-tokens") + 1] == "128"
    assert cmd[cmd.index("--reserve-decode-tokens") + 1] == "128"
    assert "--repeat-prompts" in cmd


def test_grid_script_direct_entrypoint():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "benchmarks/system/run_qwen3_8b_tp8_v_only_batch_sweep.py"),
         "--help"],
        cwd=root, capture_output=True, text=True,
    )
    assert completed.returncode == 0
    assert "--output-root" in completed.stdout


def test_prompt_repeat_preserves_frozen_row_order():
    input_ids = torch.arange(8 * 16).reshape(8, 16)
    selected = select_prompts(input_ids, 2, 16, False)
    repeated = select_prompts(input_ids, 256, 16, True)
    assert torch.equal(selected, input_ids[:2])
    assert repeated.shape == (256, 16)
    assert torch.equal(repeated[:8], input_ids)
    assert torch.equal(repeated[8:16], input_ids)
    assert torch.equal(repeated[-8:], input_ids)


def test_actual_cache_widths_include_128_decode_slots():
    ranks = [512] * 36
    basis = expected_cache_bytes("basis_v64", 128, None)
    star = expected_cache_bytes("star_v_adaptive", 64, ranks)
    assert basis["dense_key_cache"] == 128 * (16384 + 128) * 36 * 128 * 2
    assert basis["value_cache"] == 128 * (16384 + 128) * 36 * 64 * 2
    assert star["value_cache"] == 64 * (16384 + 128) * (3 * 128 + 33 * 512) * 2


def test_eight_rank_128_output_validation_and_summary(tmp_path, monkeypatch):
    folder = tmp_path / "trial"
    folder.mkdir()
    state = {**expected_cache_bytes("basis_v64", 2, None), "value_factors": 1024}
    memory = {phase: {
        "allocated_bytes": 1000, "reserved_bytes": 1200,
        "peak_allocated_bytes": 1300, "peak_reserved_bytes": 1400,
        "nvml_process_bytes": 1500,
    } for phase in ("cache_allocated", "prefill_complete", "decode_ready", "decode_complete")}
    for rank in range(8):
        row = {
            "schema": "basisserve.qwen3_8b.tp8_v_only_memory.v1", "status": "complete",
            "rank": rank, "tp": 8, "dp": 1, "pp": 1, "arm": "basis_v64",
            "prompt_tokens": 16384, "batch": 2, "cohort": 0, "chunk_size": 256,
            "output_tokens": 128, "reserve_decode_tokens": 128,
            "prompt_source_rows": 8, "prompt_repeated": False,
            "dense_key": True, "full_attention": True, "key_offload": False,
            "value_offload": False, "dtype": "bfloat16", "star_value_ranks": None,
            "generated_token_ids": [[index, index + 1] for index in range(128)],
            "state_bytes": state, "memory": memory,
            "metadata": {name: "test" for name in (
                "git_commit", "git_status", "pytorch", "cuda", "nccl",
                "transformers", "flash_attn")},
        }
        (folder / f"rank{rank}.json").write_text(json.dumps(row))
    ranks = load_trial({"output_dir": str(folder), "arm": "basis_v64", "batch": 2})
    assert len(ranks) == 8
    assert len(ranks[0]["generated_token_ids"]) == 128

    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    manifest = {
        "grid": GRID, "status": "partial", "attempts": [
            {"arm": "basis_v64", "prompt_tokens": 16384, "batch": 2, "cohort": 0,
             "attempt": 0, "status": "complete", "output_dir": str(folder)},
            {"arm": "star_v_adaptive", "prompt_tokens": 16384, "batch": 64, "cohort": 0,
             "attempt": 0, "status": "failed", "output_dir": str(failed_dir),
             "launcher_log": str(failed_dir / "launcher.log"), "returncode": 1,
             "failure": {"oom": True, "phase": "cache_state_allocation", "oom_ranks": [3],
                         "oom_rank_stages": {"3": "cache_allocate_layer"}}},
        ],
    }
    (tmp_path / "grid_trials.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["summary", "--output-root", str(tmp_path)])
    summary_module.main()
    with (tmp_path / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (tmp_path / "batch_frontier.csv").open(newline="") as stream:
        frontier = list(csv.DictReader(stream))
    assert len(rows) == 18
    assert [row["status"] for row in rows].count("complete") == 1
    assert [row["status"] for row in rows].count("failed_oom_cache_state_allocation") == 1
    assert next(row for row in frontier if row["arm"] == "star_v_adaptive")["first_oom_batch"] == "64"
    assert (tmp_path / "plots/batch_sweep_16k_memory.pdf").stat().st_size > 0


def test_plot_marks_oom_without_resident_value(tmp_path):
    rows = [
        {"arm": "basis_v64", "batch": 64, "status": "complete",
         "decode_ready_nvml_max_rank_gib": 20.0},
        {"arm": "star_v_adaptive", "batch": 64,
         "status": "failed_oom_cache_state_allocation",
         "decode_ready_nvml_max_rank_gib": None},
    ]
    assert plot_memory(rows, tmp_path)
    assert (tmp_path / "batch_sweep_16k_memory.pdf").stat().st_size > 0
    assert (tmp_path / "batch_sweep_16k_memory.png").stat().st_size > 0
