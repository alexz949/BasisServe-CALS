from __future__ import annotations

import pytest
import torch

from basisserve.kernels.ragged_allgather import (
    RaggedNcclCommunicator,
    StaticRaggedPlan,
)


def test_static_ragged_plan_from_tp_source_head_ranks() -> None:
    plan = StaticRaggedPlan.from_head_ranks(
        (32, 48, 64, 80, 96, 64, 48, 32),
        heads_per_source=4,
    )

    assert plan.source_widths == (128, 192, 256, 320, 384, 256, 192, 128)
    assert plan.offsets == (0, 128, 320, 576, 896, 1280, 1536, 1728)
    assert plan.total_width == 1856
    assert plan.padded_total_width == 8 * 384
    assert plan.rank_major_offsets(batch=3) == tuple(3 * value for value in plan.offsets)
    assert plan.padding_overhead == pytest.approx(3072 / 1856 - 1.0)


@pytest.mark.parametrize(
    "widths",
    ((), (0, 1), (-1, 2), tuple(1 for _ in range(33))),
)
def test_static_ragged_plan_rejects_invalid_widths(widths: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        StaticRaggedPlan.from_source_widths(widths)


def test_static_ragged_reference_rank_major_to_token_major_layout() -> None:
    plan = StaticRaggedPlan.from_source_widths((2, 3, 1))
    batch = 2
    source_rows = (
        torch.tensor([[10, 11], [12, 13]]),
        torch.tensor([[20, 21, 22], [23, 24, 25]]),
        torch.tensor([[30], [31]]),
    )
    rank_major = torch.cat(tuple(value.flatten() for value in source_rows))
    expected = torch.cat(source_rows, dim=1)

    observed = torch.empty(batch, plan.total_width, dtype=rank_major.dtype)
    for source, width in enumerate(plan.source_widths):
        source_start = batch * plan.offsets[source]
        destination_start = plan.offsets[source]
        block = rank_major[source_start : source_start + batch * width].reshape(
            batch,
            width,
        )
        observed[:, destination_start : destination_start + width].copy_(block)

    torch.testing.assert_close(observed, expected)


def test_batch_one_rank_major_layout_is_already_compact() -> None:
    plan = StaticRaggedPlan.from_source_widths((3, 5, 2, 7))
    rank_major = torch.arange(plan.total_width)
    assert rank_major.view(1, plan.total_width).is_contiguous()
    assert plan.rank_major_offsets(batch=1) == plan.offsets


@pytest.mark.parametrize(
    "algorithm",
    ("direct", "pairwise", "ring", "biring_grouped"),
)
def test_ragged_algorithm_names(algorithm: str) -> None:
    RaggedNcclCommunicator._validate_algorithm(algorithm)


def test_ragged_algorithm_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown ragged algorithm"):
        RaggedNcclCommunicator._validate_algorithm("tree")
