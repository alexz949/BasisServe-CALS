from __future__ import annotations

import torch

from basisserve.core.c1_k_output_closure import (
    c1k_jvp,
    c1k_teacher_and_student_output,
    c1k_vjp,
    canonicalize_c1k_factors,
    conjugate_gradient,
)


def _case():
    generator = torch.Generator().manual_seed(101)
    shape = {
        "batch": 2,
        "groups": 2,
        "heads": 3,
        "sequence": 7,
        "head_dim": 6,
        "value_rank": 4,
        "rank": 3,
        "output": 9,
    }
    rand = lambda *size: torch.randn(*size, generator=generator, dtype=torch.float64)
    query = rand(
        shape["batch"],
        shape["groups"],
        shape["heads"],
        shape["sequence"],
        shape["head_dim"],
    )
    key = rand(shape["batch"], shape["groups"], shape["sequence"], shape["head_dim"])
    value = rand(
        shape["batch"], shape["groups"], shape["sequence"], shape["value_rank"]
    )
    decoder = rand(
        shape["groups"], shape["heads"], shape["value_rank"], shape["output"]
    )
    key_projector = rand(shape["groups"], shape["head_dim"], shape["rank"])
    query_projector = rand(
        shape["groups"], shape["heads"], shape["head_dim"], shape["rank"]
    )
    return query, key, value, decoder, key_projector, query_projector


def test_matrix_free_jvp_and_vjp_are_adjoint() -> None:
    query, key, value, decoder, key_projector, query_projector = _case()
    generator = torch.Generator().manual_seed(103)
    delta_key = torch.randn(
        key_projector.shape, generator=generator, dtype=key_projector.dtype
    )
    delta_query = torch.randn(
        query_projector.shape, generator=generator, dtype=query_projector.dtype
    )
    output_cotangent = torch.randn(
        query.shape[0],
        query.shape[3],
        decoder.shape[-1],
        generator=generator,
        dtype=query.dtype,
    )
    image = c1k_jvp(
        query,
        key,
        value,
        decoder,
        key_projector,
        query_projector,
        delta_key_projector=delta_key,
        delta_query_projector=delta_query,
        query_chunk_size=3,
    )
    transpose_image = c1k_vjp(
        query,
        key,
        value,
        decoder,
        key_projector,
        query_projector,
        output_cotangent,
        query_chunk_size=3,
    )
    left = (image * output_cotangent).sum()
    right = (delta_key * transpose_image.key).sum()
    right += (delta_query * transpose_image.query).sum()
    torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)


def test_jvp_matches_finite_difference() -> None:
    query, key, value, decoder, key_projector, query_projector = _case()
    generator = torch.Generator().manual_seed(107)
    delta_key = torch.randn(
        key_projector.shape, generator=generator, dtype=key_projector.dtype
    )
    delta_query = torch.randn(
        query_projector.shape, generator=generator, dtype=query_projector.dtype
    )
    analytic = c1k_jvp(
        query,
        key,
        value,
        decoder,
        key_projector,
        query_projector,
        delta_key_projector=delta_key,
        delta_query_projector=delta_query,
        query_chunk_size=4,
    )
    epsilon = 1.0e-3
    _, plus, _ = c1k_teacher_and_student_output(
        query,
        key,
        value,
        decoder,
        key_projector + epsilon * delta_key,
        query_projector + epsilon * delta_query,
        query_chunk_size=4,
    )
    _, minus, _ = c1k_teacher_and_student_output(
        query,
        key,
        value,
        decoder,
        key_projector - epsilon * delta_key,
        query_projector - epsilon * delta_query,
        query_chunk_size=4,
    )
    torch.testing.assert_close(analytic, (plus - minus) / (2 * epsilon), rtol=3e-3, atol=3e-3)


def test_qr_canonicalization_preserves_scores() -> None:
    _, _, _, _, key_projector, query_projector = _case()
    canonical_key, canonical_query = canonicalize_c1k_factors(
        key_projector,
        query_projector,
    )
    before = torch.einsum("ghdr,gkr->ghdk", query_projector, key_projector)
    after = torch.einsum("ghdr,gkr->ghdk", canonical_query, canonical_key)
    torch.testing.assert_close(
        before,
        after,
        rtol=2e-5,
        atol=2e-5,
        check_dtype=False,
    )
    identity = torch.eye(key_projector.shape[-1]).expand(key_projector.shape[0], -1, -1)
    torch.testing.assert_close(canonical_key.mT @ canonical_key, identity)


def test_conjugate_gradient_solves_spd_system() -> None:
    matrix = torch.tensor(
        [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
    )
    rhs = torch.tensor([1.0, 2.0, 3.0])
    solution, diagnostics = conjugate_gradient(
        lambda vector: matrix @ vector,
        rhs,
        max_iterations=3,
        relative_tolerance=1.0e-6,
    )
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, rhs))
    assert diagnostics["converged"]
