"""Aggregate nonoverlapping TP8 request phases on one host's monotonic clock."""

from contextlib import contextmanager
from pathlib import Path
import time

import torch

BUILD_PHASES = ("k_gather", "svd", "factor_redistribution", "other_prepare")


def clock_id():
    return f"{Path('/proc/sys/kernel/random/boot_id').read_text().strip()}:{time.get_clock_info('perf_counter').implementation}"


class RequestPhaseRecorder:
    def __init__(self):
        self.phases = []
        self.peak_allocated_bytes = 0
        self.peak_reserved_bytes = 0
        self.owner_temporary_peak_allocated_bytes = 0

    def capture_peak(self):
        self.peak_allocated_bytes = max(self.peak_allocated_bytes, torch.cuda.max_memory_allocated())
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, torch.cuda.max_memory_reserved())

    @contextmanager
    def phase(self, name, layer, request):
        assert name in BUILD_PHASES
        torch.cuda.synchronize()
        self.capture_peak()
        torch.cuda.reset_peak_memory_stats()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        start_ns = time.perf_counter_ns()
        yield
        end.record()
        end.synchronize()
        end_ns = time.perf_counter_ns()
        peak = torch.cuda.max_memory_allocated()
        self.capture_peak()
        if name in ("k_gather", "svd") and layer % 8 == torch.distributed.get_rank():
            self.owner_temporary_peak_allocated_bytes = max(
                self.owner_temporary_peak_allocated_bytes, peak
            )
        self.phases.append({"name": name, "layer": layer, "request": request,
                            "start_ns": start_ns, "end_ns": end_ns,
                            "cuda_event_ms": begin.elapsed_time(end),
                            "peak_allocated_bytes": peak})


def summarize_request_timeline(ranks):
    """Use all-rank boundary envelopes, not sums of local phase maxima.

    Each rank records absolute perf_counter_ns timestamps on the same host.
    Phase boundaries must be CUDA-synchronized and recorded in identical order.
    A component spans the latest start to latest end across ranks. With ordered
    phases these spans do not overlap, including when nonowners wait for SVD
    inside their redistribution call. Instrumentation overhead remains in the
    measured request; these are synchronized, instrumented timings.
    """
    assert len(ranks) == 8
    assert {row["rank"] for row in ranks} == set(range(8))
    assert len({row["clock_id"] for row in ranks}) == 1
    assert all(row["clock_id"] for row in ranks)
    arms = {row["arm"] for row in ranks}
    assert len(arms) == 1 and arms <= {"dense", "basis_joint", "shadowkv"}
    assert all(row["output_tokens"] == 128 for row in ranks)
    count = len(ranks[0]["phases"])
    assert all(len(row["phases"]) == count for row in ranks)
    if ranks[0]["arm"] != "shadowkv":
        assert count == 0

    for row in ranks:
        start = row["request_start_ns"]
        first = row["first_output_ns"]
        ready = row["representation_ready_ns"]
        final = row["final_output_ns"]
        assert start <= first <= ready <= final
        previous_end = start
        for phase in row["phases"]:
            assert phase["name"] in BUILD_PHASES
            assert previous_end <= phase["start_ns"] <= phase["end_ns"] <= ready
            previous_end = phase["end_ns"]

    start = min(row["request_start_ns"] for row in ranks)
    first = max(row["first_output_ns"] for row in ranks)
    ready = max(row["representation_ready_ns"] for row in ranks)
    final = max(row["final_output_ns"] for row in ranks)
    totals = dict.fromkeys(BUILD_PHASES, 0)
    timeline = []
    previous_end = start
    for index in range(count):
        phases = [row["phases"][index] for row in ranks]
        identities = {(phase["name"], phase["layer"], phase["request"]) for phase in phases}
        assert len(identities) == 1
        name, layer, request = identities.pop()
        begin = max(phase["start_ns"] for phase in phases)
        end = max(phase["end_ns"] for phase in phases)
        assert previous_end <= begin <= end <= ready
        totals[name] += end - begin
        timeline.append({"name": name, "layer": layer, "request": request,
                         "start_ms": (begin - start) / 1e6,
                         "end_ms": (end - start) / 1e6,
                         "duration_ms": (end - begin) / 1e6})
        previous_end = end

    build = sum(totals.values())
    prefill = ready - start - build
    decode = final - ready
    total = final - start
    assert prefill >= 0 and prefill + build + decode == total
    return {
        "prefill_ms": prefill / 1e6,
        "representation_build_ms": build / 1e6,
        **{name + "_ms": duration / 1e6 for name, duration in totals.items()},
        "prefill_start_ms": 0.0,
        "inclusive_prefill_end_ms": (first - start) / 1e6,
        "shadow_build_span_start_ms": timeline[0]["start_ms"] if timeline else None,
        "shadow_build_span_end_ms": timeline[-1]["end_ms"] if timeline else None,
        "first_token_ms": (first - start) / 1e6,
        "representation_ready_ms": (ready - start) / 1e6,
        "final_output_ms": total / 1e6,
        "decode_after_first_token_ms": (final - first) / 1e6,
        "decode_after_representation_ready_ms": decode / 1e6,
        "total_request_ms": total / 1e6,
        "request_start_skew_ms": (
            max(row["request_start_ns"] for row in ranks) - start) / 1e6,
        "output_tokens": 128,
        "decode_steps_after_prefill": 127,
        "figure_stack_fields": ["prefill_ms", "representation_build_ms",
                                "decode_after_representation_ready_ms"],
        "phase_timeline": timeline,
        "timing_definition": "same-host all-rank boundary envelopes; CUDA-synchronized phase boundaries",
        "prefill_definition": "ordinary projections, attention, MLP, encoding, cache writes, and unclassified request overhead",
        "build_definition": "ShadowKV-specific gather/SVD/redistribution/preparation only",
        "build_span_warning": "enclosing start/end contains interleaved ordinary prefill; use the sum of disjoint phases for construction cost",
        "warning": "decode_after_first_token_ms overlaps any post-first-token preparation; do not use it as the additive decode stack",
    }
