"""CPU-only checks for the ShadowKV raw-trial summary contract."""

import json

import pytest

from benchmarks.system.summarize_tp8_shadowkv_request import load_validated_trial


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_shadowkv_phase_count_scales_with_batch(tmp_path, batch):
    attempt = {
        "arm": "shadowkv", "mode": "request", "prompt_tokens": 4096,
        "batch": batch, "cohort": 0, "output_dir": str(tmp_path),
    }
    expected = 32 * (2 + 3 * batch) + 1
    replica = {
        **{key: attempt[key] for key in ("arm", "mode", "prompt_tokens", "batch", "cohort")},
        "status": "complete", "successful_ranks": 8,
        "request": {"phase_timeline": [None] * expected,
                    "figure_stack_fields": ["prefill_ms", "representation_build_ms",
                                            "decode_after_representation_ready_ms"],
                    "prefill_ms": 2, "representation_build_ms": 3,
                    "decode_after_representation_ready_ms": 5,
                    "total_request_ms": 10, "representation_ready_ms": 5},
    }
    (tmp_path / "replica.json").write_text(json.dumps(replica))
    for rank in range(8):
        row = {
            **{key: attempt[key] for key in ("arm", "mode", "prompt_tokens", "batch", "cohort")},
            "rank": rank, "metadata": {"git_commit": "test"},
            "status": "complete", "schema": "basisserve.llama31_8b.tp8_request.v1",
            "generated_token_ids": [[0] * 128 for _ in range(batch)],
        }
        (tmp_path / f"rank{rank}.json").write_text(json.dumps(row))

    load_validated_trial(attempt)
    replica["request"]["phase_timeline"].pop()
    (tmp_path / "replica.json").write_text(json.dumps(replica))
    with pytest.raises(AssertionError):
        load_validated_trial(attempt)
