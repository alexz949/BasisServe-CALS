from __future__ import annotations

import torch

from basisserve.core.iterative_least_squares import least_squares_matrix


def _operator(matrix: torch.Tensor, shape: tuple[int, ...]):
    def matvec(value: torch.Tensor) -> torch.Tensor:
        return matrix @ value.reshape(-1)

    def rmatvec(value: torch.Tensor) -> torch.Tensor:
        return (matrix.mT @ value).reshape(shape)

    return matvec, rmatvec


def test_lsqr_matches_direct_overdetermined_least_squares() -> None:
    torch.manual_seed(20260910)
    matrix = torch.randn(31, 12, dtype=torch.float64)
    rhs = torch.randn(31, dtype=torch.float64)
    matvec, rmatvec = _operator(matrix, (3, 4))

    actual, diagnostics = least_squares_matrix(
        matvec,
        rmatvec,
        rhs,
        solution_shape=(3, 4),
        relative_tolerance=1e-12,
        max_iterations=40,
    )
    expected = torch.linalg.lstsq(matrix, rhs).solution.reshape(3, 4)

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    assert diagnostics.converged
    assert diagnostics.relative_normal_residual < 1e-10


def test_lsqr_matches_direct_ridge_least_squares() -> None:
    torch.manual_seed(20260911)
    matrix = torch.randn(23, 9, dtype=torch.float64)
    rhs = torch.randn(23, dtype=torch.float64)
    damping = 3e-3
    matvec, rmatvec = _operator(matrix, (3, 3))

    actual, diagnostics = least_squares_matrix(
        matvec,
        rmatvec,
        rhs,
        solution_shape=(3, 3),
        relative_tolerance=1e-12,
        max_iterations=40,
        absolute_damping=damping,
    )
    expected = torch.linalg.solve(
        matrix.mT @ matrix + damping * torch.eye(9, dtype=torch.float64),
        matrix.mT @ rhs,
    ).reshape(3, 3)

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    assert diagnostics.converged
    assert diagnostics.absolute_damping == damping


def test_lsqr_returns_zero_for_zero_rhs() -> None:
    matrix = torch.randn(7, 4, dtype=torch.float64)
    matvec, rmatvec = _operator(matrix, (2, 2))
    solution, diagnostics = least_squares_matrix(
        matvec,
        rmatvec,
        torch.zeros(7, dtype=torch.float64),
        solution_shape=(2, 2),
        relative_tolerance=1e-8,
        max_iterations=10,
    )
    assert torch.count_nonzero(solution) == 0
    assert diagnostics.converged
    assert diagnostics.iterations == 0
