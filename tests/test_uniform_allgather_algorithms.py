from __future__ import annotations

import pytest


def _initial(world_size: int) -> list[list[int | None]]:
    arenas: list[list[int | None]] = []
    for rank in range(world_size):
        row: list[int | None] = [None] * world_size
        row[rank] = rank
        arenas.append(row)
    return arenas


def _recursive_doubling(world_size: int) -> list[list[int | None]]:
    arenas = _initial(world_size)
    group_size = 1
    while group_size < world_size:
        snapshot = [row.copy() for row in arenas]
        for rank in range(world_size):
            partner = rank ^ group_size
            group_start = rank & ~(group_size - 1)
            arenas[partner][group_start : group_start + group_size] = snapshot[rank][
                group_start : group_start + group_size
            ]
        group_size <<= 1
    return arenas


def _ring(world_size: int) -> list[list[int | None]]:
    arenas = _initial(world_size)
    for phase in range(world_size - 1):
        snapshot = [row.copy() for row in arenas]
        for rank in range(world_size):
            right = (rank + 1) % world_size
            source_rank = (rank - phase + world_size) % world_size
            arenas[right][source_rank] = snapshot[rank][source_rank]
    return arenas


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_recursive_doubling_source_intervals_are_exact(world_size: int) -> None:
    expected = list(range(world_size))
    assert _recursive_doubling(world_size) == [expected] * world_size


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ring_forwarding_order_is_exact(world_size: int) -> None:
    expected = list(range(world_size))
    assert _ring(world_size) == [expected] * world_size


def _partition_bytes(total_bytes: int, channels: int) -> list[tuple[int, int]]:
    vector_units = total_bytes % 16 == 0
    units = total_bytes // 16 if vector_units else total_bytes
    quotient, remainder = divmod(units, channels)
    scale = 16 if vector_units else 1
    result: list[tuple[int, int]] = []
    for channel in range(channels):
        begin_units = quotient * channel + min(channel, remainder)
        channel_units = quotient + int(channel < remainder)
        result.append((begin_units * scale, channel_units * scale))
    return result


@pytest.mark.parametrize("total_bytes", [1, 15, 16, 17, 1024, 1536, 196608, 524288])
@pytest.mark.parametrize("channels", [1, 2, 4, 8])
def test_channel_partitions_are_exact_and_nonoverlapping(
    total_bytes: int,
    channels: int,
) -> None:
    ranges = _partition_bytes(total_bytes, channels)
    cursor = 0
    for offset, length in ranges:
        assert offset == cursor
        assert length >= 0
        cursor += length
    assert cursor == total_bytes
    if total_bytes % 16 == 0:
        assert all(offset % 16 == 0 and length % 16 == 0 for offset, length in ranges)
