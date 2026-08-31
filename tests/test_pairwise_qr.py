from __future__ import annotations

import pytest
import torch

from basisserve.core.pairwise_qr import (
    StreamingTSQR,
    pack_upper_triangular,
    pairwise_qr_r,
    solve_square_root_least_squares,
    unpack_upper_triangular,
)


def test_pairwise_qr_preserves_gram_without_stacking_all_rows() -> None:
    torch.manual_seed(20260823)
    matrix = torch.randn(48, 7, dtype=torch.float64)
    blocks = (matrix[:16], matrix[16:31], matrix[31:])

    r, diagnostics = pairwise_qr_r(blocks)

    torch.testing.assert_close(r.T @ r, matrix.T @ matrix, rtol=1e-12, atol=1e-12)
    assert torch.all(torch.diagonal(r) >= 0)
    assert diagnostics.column_count == 7
    assert diagnostics.input_row_count == 48
    assert diagnostics.leaf_row_counts == (16, 15, 17)
    assert diagnostics.node_counts_by_level == (3, 2, 1)


@pytest.mark.parametrize("damping", (0.0, 3.0e-4))
def test_pairwise_qr_square_root_augmentation_matches_direct_problem(
    damping: float,
) -> None:
    torch.manual_seed(20260824)
    rows = torch.randn(41, 9, dtype=torch.float64)
    basis = torch.randn(9, 5, dtype=torch.float64)
    target_weight = torch.randn(9, 4, dtype=torch.float64)
    scale = rows.shape[0] ** -0.5

    r_x, _ = pairwise_qr_r((rows[:21] * scale, rows[21:] * scale))
    qr_solution, diagnostics = solve_square_root_least_squares(
        left_factor=r_x,
        basis=basis,
        target_weight=target_weight,
        absolute_product_damping=damping,
        output_chunk_size=2,
    )
    direct_design = rows * scale @ basis
    direct_target = rows * scale @ target_weight
    if damping:
        direct_design = torch.cat(
            (direct_design, damping**0.5 * basis),
            dim=0,
        )
        direct_target = torch.cat(
            (direct_target, damping**0.5 * target_weight),
            dim=0,
        )
    direct_solution = torch.linalg.lstsq(
        direct_design,
        direct_target,
        driver="gels",
    ).solution

    torch.testing.assert_close(qr_solution, direct_solution, rtol=1e-11, atol=1e-11)
    assert diagnostics.design_rows == 9 * (2 if damping else 1)
    assert diagnostics.design_columns == 5
    assert diagnostics.right_hand_side_count == 4
    assert diagnostics.absolute_product_damping == damping


def test_pairwise_qr_rejects_short_or_incompatible_leaves() -> None:
    with pytest.raises(ValueError, match="rows but"):
        pairwise_qr_r((torch.ones(2, 3),))
    with pytest.raises(ValueError, match="same column count"):
        pairwise_qr_r((torch.ones(4, 3), torch.ones(5, 4)))


def test_streaming_tsqr_power_of_two_checkpoints_match_direct_gram() -> None:
    torch.manual_seed(20260825)
    rows = torch.randn(64, 5, dtype=torch.float64)
    accumulator = StreamingTSQR(
        columns=5,
        block_rows=8,
        device="cpu",
        dtype=torch.float64,
    )
    checkpoints = {16, 32, 64}
    observed: dict[int, torch.Tensor] = {}
    for start in range(0, len(rows), 3):
        accumulator.append(rows[start : start + 3])
        if accumulator.total_rows in checkpoints:
            observed[accumulator.total_rows] = accumulator.snapshot_r()

    # Chunk boundaries do not necessarily land on every requested row count;
    # repeat with milestone-aligned appends to exercise all three snapshots.
    accumulator = StreamingTSQR(
        columns=5,
        block_rows=8,
        device="cpu",
        dtype=torch.float64,
    )
    observed.clear()
    for start, stop in ((0, 7), (7, 16), (16, 32), (32, 47), (47, 64)):
        accumulator.append(rows[start:stop])
        if stop in checkpoints:
            observed[stop] = accumulator.snapshot_r()

    assert set(observed) == checkpoints
    for count, r in observed.items():
        torch.testing.assert_close(
            r.T @ r,
            rows[:count].T @ rows[:count],
            rtol=1e-12,
            atol=1e-12,
        )
    diagnostics = accumulator.diagnostics()
    assert diagnostics.total_rows == 64
    assert diagnostics.completed_leaf_count == 8
    assert diagnostics.merge_count == 7
    assert diagnostics.buffered_rows == 0


def test_streaming_tsqr_rejects_unaligned_snapshot() -> None:
    accumulator = StreamingTSQR(
        columns=3,
        device="cpu",
        dtype=torch.float32,
    )
    accumulator.append(torch.ones(4, 3))
    accumulator.append(torch.ones(1, 3))
    with pytest.raises(RuntimeError, match="empty row buffer"):
        accumulator.snapshot_r()


def test_upper_triangular_pack_round_trip() -> None:
    torch.manual_seed(20260826)
    matrix = torch.triu(torch.randn(7, 7, dtype=torch.float64))
    packed = pack_upper_triangular(matrix)
    assert tuple(packed.shape) == (28,)
    restored = unpack_upper_triangular(packed, dimension=7)
    torch.testing.assert_close(restored, matrix)
    with pytest.raises(ValueError, match="28 values"):
        unpack_upper_triangular(packed[:-1], dimension=7)
