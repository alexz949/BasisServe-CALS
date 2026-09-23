"""CPU-only tests for request phase accounting, not performance measurements."""

from copy import deepcopy

import pytest

from benchmarks.system.tp8_request_timing import summarize_request_timeline


def phase(name, start, end, layer=0, request=0):
    return {"name": name, "layer": layer, "request": request,
            "start_ns": start * 1_000_000, "end_ns": end * 1_000_000}


def records(arm="basis_joint"):
    return [{"rank": rank, "clock_id": "same-host-boot-and-clock", "arm": arm,
             "output_tokens": 128, "request_start_ns": 0,
             "first_output_ns": 100_000_000, "representation_ready_ns": 110_000_000,
             "final_output_ns": 240_000_000, "phases": []}
            for rank in range(8)]


@pytest.mark.parametrize("arm", ["dense", "basis_joint"])
def test_ordinary_cache_work_is_not_free(arm):
    result = summarize_request_timeline(records(arm))
    assert result["prefill_ms"] == 110
    assert result["representation_build_ms"] == 0
    assert result["shadow_build_span_start_ms"] is None
    assert result["total_request_ms"] == 240
    assert sum(result[key] for key in result["figure_stack_fields"]) == 240


def test_nonowner_svd_wait_is_not_counted_twice():
    rows = records("shadowkv")
    for row in rows:
        row["phases"] = [phase("k_gather", 10, 20), phase("svd", 20, 20),
                         phase("factor_redistribution", 20, 80)]
    rows[3]["phases"] = [phase("k_gather", 10, 20), phase("svd", 20, 70),
                         phase("factor_redistribution", 70, 80)]
    result = summarize_request_timeline(rows)
    assert result["k_gather_ms"] == 10
    assert result["svd_ms"] == 50
    assert result["factor_redistribution_ms"] == 10
    assert result["representation_build_ms"] == 70
    assert result["shadow_build_span_start_ms"] == 10
    assert result["shadow_build_span_end_ms"] == 80
    assert result["prefill_ms"] == 40
    assert sum(result[key] for key in result["figure_stack_fields"]) == 240


def test_post_first_token_prepare_is_not_decode_in_stack():
    rows = records("shadowkv")
    for row in rows:
        row["phases"] = [phase("other_prepare", 100, 110, layer=-1, request=-1)]
    result = summarize_request_timeline(rows)
    assert result["first_token_ms"] == 100
    assert result["representation_ready_ms"] == 110
    assert result["decode_after_first_token_ms"] == 140
    assert result["decode_after_representation_ready_ms"] == 130
    assert result["other_prepare_ms"] == 10
    assert result["prefill_ms"] == 100
    assert sum(result[key] for key in result["figure_stack_fields"]) == 240


def test_absolute_clock_origin_and_start_skew():
    rows = records()
    for row in rows:
        for key in ("request_start_ns", "first_output_ns", "representation_ready_ns", "final_output_ns"):
            row[key] += 1_234_567_000_000
    rows[4]["request_start_ns"] += 2_000_000
    rows[7]["final_output_ns"] += 3_000_000
    result = summarize_request_timeline(rows)
    assert result["request_start_skew_ms"] == 2
    assert result["total_request_ms"] == 243
    assert result["decode_after_representation_ready_ms"] == 133


@pytest.mark.parametrize("malformation", ["missing_rank", "duplicate_rank", "clock", "order",
                                         "phase_identity", "wrong_arm", "token_count"])
def test_reject_inconsistent_timelines(malformation):
    rows = records("shadowkv")
    for row in rows:
        row["phases"] = [phase("k_gather", 10, 20), phase("svd", 20, 70)]
    if malformation == "missing_rank":
        rows.pop()
    elif malformation == "duplicate_rank":
        rows[7]["rank"] = 0
    elif malformation == "clock":
        rows[7]["clock_id"] = "different-host"
    elif malformation == "order":
        rows[7]["phases"][1]["start_ns"] = 19_000_000
    elif malformation == "phase_identity":
        rows[7]["phases"][1]["request"] = 1
    elif malformation == "wrong_arm":
        for row in rows:
            row["arm"] = "basis_joint"
    elif malformation == "token_count":
        rows[0]["output_tokens"] = 129
    with pytest.raises(AssertionError):
        summarize_request_timeline(rows)


def test_input_records_are_not_mutated():
    rows = records()
    original = deepcopy(rows)
    summarize_request_timeline(rows)
    assert rows == original
