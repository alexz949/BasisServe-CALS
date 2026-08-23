"""Trace-driven latency algebra for the C1 TP8 decode serving path.

The simulator in this module deliberately does not launch CUDA work.  It
combines measured rank-local operator traces with an explicit effective
network model.  Keeping the timeline algebra separate from the profilers
makes every reported number attributable to either a measurement or a named
communication assumption.

One layer has eight source-ready times.  The barrier path waits for the
slowest source, performs one ragged AllGather, and runs one compact decoder
GEMM.  Pipelined paths let each source arrive independently and execute one
or more decoder waves as their inputs become available.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class EffectiveNetwork:
    """An explicit alpha-beta model for one effective collective data path.

    ``bandwidth_gbps`` uses decimal GB/s, matching hardware/network vendor
    notation.  The model is intentionally effective rather than topology
    inferred: values should eventually be fitted to the target NCCL run.
    """

    latency_us: float
    bandwidth_gbps: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.latency_us) or self.latency_us < 0.0:
            raise ValueError("network latency must be finite and nonnegative")
        if not math.isfinite(self.bandwidth_gbps) or self.bandwidth_gbps <= 0.0:
            raise ValueError("network bandwidth must be finite and positive")

    def transfer_ms(self, byte_count: int) -> float:
        """Return ``alpha + bytes / beta`` in milliseconds."""

        size = int(byte_count)
        if size < 0:
            raise ValueError("communication byte count cannot be negative")
        return self.latency_us / 1000.0 + size / (self.bandwidth_gbps * 1e6)


@dataclass(frozen=True)
class DecoderWave:
    """One decoder GEMM consuming a fixed set of TP source blocks."""

    sources: tuple[int, ...]
    duration_ms: float

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("decoder wave cannot be empty")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("decoder wave cannot contain duplicate sources")
        if not math.isfinite(self.duration_ms) or self.duration_ms < 0.0:
            raise ValueError("decoder wave duration must be finite and nonnegative")


@dataclass(frozen=True)
class C1BoundaryTimeline:
    """Critical-path result for the C1 attention-output boundary."""

    completion_ms: float
    source_ready_ms: tuple[float, ...]
    receiver_completion_ms: tuple[float, ...]
    maximum_received_bytes: int


def readiness_waves(
    source_ranks: Sequence[int],
    wave_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Bucket sources by increasing rank into balanced readiness waves.

    Rank is the stable readiness proxy used by the profiler: smaller Value
    ranks complete compact attention first.  Process rank breaks ties, so the
    partition is deterministic and matches checkpoint source ownership.
    """

    ranks = tuple(int(value) for value in source_ranks)
    if not ranks or any(value <= 0 for value in ranks):
        raise ValueError("source ranks must be a nonempty positive sequence")
    count = int(wave_count)
    if not 1 <= count <= len(ranks):
        raise ValueError("wave count must lie between one and the TP size")
    ordered = tuple(
        sorted(range(len(ranks)), key=lambda source: (ranks[source], source))
    )
    base, extra = divmod(len(ordered), count)
    waves: list[tuple[int, ...]] = []
    start = 0
    for wave_index in range(count):
        size = base + int(wave_index < extra)
        stop = start + size
        waves.append(ordered[start:stop])
        start = stop
    if start != len(ordered):
        raise AssertionError("readiness wave partition is incomplete")
    return tuple(waves)


def validate_decoder_waves(
    waves: Sequence[DecoderWave],
    *,
    world_size: int,
) -> tuple[DecoderWave, ...]:
    """Validate that decoder waves cover every source exactly once."""

    selected = tuple(waves)
    if not selected:
        raise ValueError("decoder schedule cannot be empty")
    flattened = tuple(source for wave in selected for source in wave.sources)
    expected = tuple(range(int(world_size)))
    if tuple(sorted(flattened)) != expected:
        raise ValueError(
            "decoder waves must cover every source exactly once: "
            f"observed={flattened}, expected={expected}"
        )
    return selected


def ragged_maximum_received_bytes(
    source_widths: Sequence[int],
    *,
    batch: int,
    element_bytes: int,
) -> int:
    """Maximum compact bytes received by any rank in a replicated gather."""

    widths = tuple(int(value) for value in source_widths)
    rows = int(batch)
    item_size = int(element_bytes)
    if not widths or any(value <= 0 for value in widths):
        raise ValueError("source widths must be a nonempty positive sequence")
    if rows <= 0 or item_size <= 0:
        raise ValueError("batch and element size must be positive")
    return rows * (sum(widths) - min(widths)) * item_size


def ring_allreduce_bytes_per_rank(
    *,
    batch: int,
    width: int,
    world_size: int,
    element_bytes: int,
) -> int:
    """Ring-equivalent per-rank send volume for ReduceScatter+AllGather."""

    rows = int(batch)
    features = int(width)
    peers = int(world_size)
    item_size = int(element_bytes)
    if min(rows, features, peers, item_size) <= 0:
        raise ValueError("all AllReduce dimensions must be positive")
    payload = rows * features * item_size
    return math.ceil(2 * (peers - 1) * payload / peers)


def barrier_c1_boundary(
    source_ready_ms: Sequence[float],
    source_widths: Sequence[int],
    *,
    batch: int,
    element_bytes: int,
    decoder_ms: float,
    network: EffectiveNetwork,
) -> C1BoundaryTimeline:
    """Simulate gather-then-one-big-GEMM C1 output reconstruction."""

    ready = _validated_ready_times(source_ready_ms, source_widths)
    decode = float(decoder_ms)
    if not math.isfinite(decode) or decode < 0.0:
        raise ValueError("decoder duration must be finite and nonnegative")
    received = ragged_maximum_received_bytes(
        source_widths,
        batch=batch,
        element_bytes=element_bytes,
    )
    completion = max(ready) + network.transfer_ms(received) + decode
    return C1BoundaryTimeline(
        completion_ms=completion,
        source_ready_ms=ready,
        receiver_completion_ms=tuple(completion for _ in ready),
        maximum_received_bytes=received,
    )


def pipelined_c1_boundary(
    source_ready_ms: Sequence[float],
    source_widths: Sequence[int],
    waves: Sequence[DecoderWave],
    *,
    batch: int,
    element_bytes: int,
    network: EffectiveNetwork,
    overlap_decoder_with_local_attention: bool,
) -> C1BoundaryTimeline:
    """Simulate source-wise arrival and serial decoder waves on every receiver.

    Transfers from distinct sources are treated as independently progressing;
    therefore this is an optimistic communication schedule until the effective
    alpha-beta parameters are calibrated to the real collective.  Setting
    ``overlap_decoder_with_local_attention`` to false gives the conservative
    same-GPU policy where decoder work cannot begin before that receiver's own
    attention finishes.  Decoder waves themselves always serialize.
    """

    ready = _validated_ready_times(source_ready_ms, source_widths)
    widths = tuple(int(value) for value in source_widths)
    schedule = validate_decoder_waves(waves, world_size=len(widths))
    rows = int(batch)
    item_size = int(element_bytes)
    if rows <= 0 or item_size <= 0:
        raise ValueError("batch and element size must be positive")

    receiver_finishes: list[float] = []
    for receiver in range(len(widths)):
        previous_finish = 0.0
        local_compute_gate = (
            0.0 if overlap_decoder_with_local_attention else ready[receiver]
        )
        for wave in schedule:
            arrivals = []
            for source in wave.sources:
                if source == receiver:
                    arrivals.append(ready[source])
                else:
                    message_bytes = rows * widths[source] * item_size
                    arrivals.append(ready[source] + network.transfer_ms(message_bytes))
            wave_ready = max(max(arrivals), local_compute_gate)
            previous_finish = max(previous_finish, wave_ready) + wave.duration_ms
        receiver_finishes.append(previous_finish)

    received = ragged_maximum_received_bytes(
        widths,
        batch=rows,
        element_bytes=item_size,
    )
    return C1BoundaryTimeline(
        completion_ms=max(receiver_finishes),
        source_ready_ms=ready,
        receiver_completion_ms=tuple(receiver_finishes),
        maximum_received_bytes=received,
    )


def _validated_ready_times(
    source_ready_ms: Sequence[float],
    source_widths: Sequence[int],
) -> tuple[float, ...]:
    ready = tuple(float(value) for value in source_ready_ms)
    widths = tuple(int(value) for value in source_widths)
    if not ready or len(ready) != len(widths):
        raise ValueError("source-ready times and widths must have equal nonzero length")
    if any(not math.isfinite(value) or value < 0.0 for value in ready):
        raise ValueError("source-ready times must be finite and nonnegative")
    if any(value <= 0 for value in widths):
        raise ValueError("source widths must be positive")
    return ready


__all__ = [
    "C1BoundaryTimeline",
    "DecoderWave",
    "EffectiveNetwork",
    "barrier_c1_boundary",
    "pipelined_c1_boundary",
    "ragged_maximum_received_bytes",
    "readiness_waves",
    "ring_allreduce_bytes_per_rank",
    "validate_decoder_waves",
]
