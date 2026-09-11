"""Page-Fisher router refits used by Store80 capacity-sharing ablations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from basisserve.core.gqa_joint_routing_payload_s80_fisher import (
    S80CompactSoftmaxFisherRouting,
    compact_softmax_fisher_loss,
)
from basisserve.core.gqa_routed_ov_joint import (
    CGDiagnostics,
    conjugate_gradient_matrix,
)


@dataclass(frozen=True)
class PageFisherRouterSweep:
    sweep: int
    loss_before: float
    loss_after_queries: float
    loss_after_encoder: float


@dataclass(frozen=True)
class PageFisherRouterFit:
    routing_encoders: torch.Tensor
    routing_query_factors: torch.Tensor
    sweeps: tuple[PageFisherRouterSweep, ...]
    final_query_diagnostics: tuple[CGDiagnostics, ...]
    encoder_diagnostics: tuple[tuple[CGDiagnostics, ...], ...]


def _selector(statistics: S80CompactSoftmaxFisherRouting) -> torch.Tensor:
    result = statistics.queries_by_head.new_zeros(
        statistics.key_dim,
        statistics.joint_dim,
    )
    result[:, statistics.value_dim :] = torch.eye(
        statistics.key_dim,
        device=result.device,
        dtype=result.dtype,
    )
    return result


def _target_maps(
    statistics: S80CompactSoftmaxFisherRouting,
    target_maps: torch.Tensor | None,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    if target_maps is None:
        return _selector(statistics).to(like).unsqueeze(0).expand(
            statistics.queries_by_head.shape[0],
            -1,
            -1,
        )
    return target_maps.to(like)


def _relative_damping(diagonal: torch.Tensor, relative: float) -> float:
    return float(relative) * max(
        float(diagonal.abs().mean()),
        torch.finfo(diagonal.dtype).tiny,
    )


def _diagonal_preconditioner(
    diagonal: torch.Tensor,
    *,
    damping: float,
):
    shifted = diagonal.clamp_min(0).add(float(damping))
    shifted.clamp_min_(torch.finfo(shifted.dtype).tiny)

    def apply(value: torch.Tensor) -> torch.Tensor:
        return value / shifted

    return apply


def refit_page_fisher_query_factors(
    statistics: S80CompactSoftmaxFisherRouting,
    *,
    routing_encoders: torch.Tensor,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
    target_maps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[CGDiagnostics, ...]]:
    """Solve unrestricted per-head query maps for fixed routing encoders."""

    mapping = statistics.head_to_kv_group.to(
        device=routing_encoders.device,
        dtype=torch.long,
    )
    queries = statistics.queries_by_head.to(routing_encoders)
    grams = statistics.fisher_grams_by_head.to(routing_encoders)
    targets = _target_maps(
        statistics,
        target_maps,
        like=routing_encoders,
    )
    fitted = routing_encoders.new_empty(
        statistics.queries_by_head.shape[0],
        statistics.key_dim,
        routing_encoders.shape[-1],
    )
    diagnostics: list[CGDiagnostics] = []
    for head, group in enumerate(mapping.tolist()):
        query = queries[head]
        gram = grams[head]
        encoder = routing_encoders[group]
        right_factors = torch.einsum(
            "ir,dij,js->drs",
            encoder,
            gram,
            encoder,
        )
        target = query @ targets[head]
        weighted_target = torch.einsum(
            "di,dij,jr->dr",
            target,
            gram,
            encoder,
        )
        rhs = query.mT @ weighted_target

        def operator(value: torch.Tensor) -> torch.Tensor:
            codes = query @ value
            transformed = torch.einsum("dr,drs->ds", codes, right_factors)
            return query.mT @ transformed

        diagonal = torch.einsum(
            "di,dr->ir",
            query.square(),
            right_factors.diagonal(dim1=-2, dim2=-1),
        )
        damping = _relative_damping(diagonal, relative_damping)
        solution, record = conjugate_gradient_matrix(
            operator,
            rhs,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            absolute_damping=damping,
            preconditioner=_diagonal_preconditioner(
                diagonal,
                damping=damping,
            ),
        )
        fitted[head] = solution
        diagnostics.append(record)
    return fitted, tuple(diagnostics)


def refit_page_fisher_routing_encoders(
    statistics: S80CompactSoftmaxFisherRouting,
    *,
    routing_query_factors: torch.Tensor,
    active_joint_rows: torch.Tensor,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
    target_maps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[CGDiagnostics, ...]]:
    """Solve group encoders with support restricted to selected joint rows."""

    queries = statistics.queries_by_head.to(routing_query_factors)
    grams = statistics.fisher_grams_by_head.to(routing_query_factors)
    mapping = statistics.head_to_kv_group.to(
        device=routing_query_factors.device,
        dtype=torch.long,
    )
    active = active_joint_rows.to(
        device=routing_query_factors.device,
        dtype=torch.long,
    )
    targets = _target_maps(
        statistics,
        target_maps,
        like=routing_query_factors,
    )
    encoders = routing_query_factors.new_zeros(
        int(mapping.max()) + 1,
        statistics.joint_dim,
        routing_query_factors.shape[-1],
    )
    diagnostics: list[CGDiagnostics] = []
    for group in range(encoders.shape[0]):
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        group_queries = queries.index_select(0, heads)
        group_grams = grams.index_select(0, heads)
        query_factors = routing_query_factors.index_select(0, heads)
        query_codes = torch.einsum(
            "hdk,hkr->hdr",
            group_queries,
            query_factors,
        )
        active_grams = group_grams.index_select(-2, active).index_select(-1, active)
        active_to_joint = group_grams.index_select(-2, active)
        group_targets = targets.index_select(0, heads)
        target = torch.einsum("hdk,hki->hdi", group_queries, group_targets)
        weighted_target = torch.einsum(
            "hdij,hdj->hdi",
            active_to_joint,
            target,
        )
        rhs = torch.einsum("hdi,hdr->ir", weighted_target, query_codes)

        def operator(value: torch.Tensor) -> torch.Tensor:
            # Associate G @ (E @ u) before the final outer product with u.
            # This avoids a [heads, documents, features, rank] intermediate
            # and applies each Fisher Gram to one vector instead of rank vectors.
            projected = torch.einsum("ir,hdr->hdi", value, query_codes)
            transformed = torch.einsum("hdij,hdj->hdi", active_grams, projected)
            return torch.einsum("hdi,hdr->ir", transformed, query_codes)

        diagonal = torch.einsum(
            "hdi,hdr->ir",
            active_grams.diagonal(dim1=-2, dim2=-1),
            query_codes.square(),
        )
        damping = _relative_damping(diagonal, relative_damping)
        solution, record = conjugate_gradient_matrix(
            operator,
            rhs,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            absolute_damping=damping,
            preconditioner=_diagonal_preconditioner(
                diagonal,
                damping=damping,
            ),
        )
        encoders[group].index_copy_(0, active, solution)
        diagnostics.append(record)
    return encoders, tuple(diagnostics)


def fit_page_fisher_router(
    statistics: S80CompactSoftmaxFisherRouting,
    *,
    initial_routing_encoders: torch.Tensor,
    initial_query_factors: torch.Tensor,
    active_joint_rows: torch.Tensor,
    sweeps: int,
    relative_damping: float,
    relative_tolerance: float,
    max_iterations: int,
    target_maps: torch.Tensor | None = None,
) -> PageFisherRouterFit:
    """Alternately close unrestricted query maps and supported encoders."""

    encoders = initial_routing_encoders.to(
        device=statistics.queries_by_head.device,
        dtype=statistics.queries_by_head.dtype,
    ).clone()
    query_factors = initial_query_factors.to(encoders).clone()
    records: list[PageFisherRouterSweep] = []
    encoder_diagnostics: list[tuple[CGDiagnostics, ...]] = []
    query_diagnostics: tuple[CGDiagnostics, ...] = ()

    def loss() -> float:
        return compact_softmax_fisher_loss(
            statistics,
            routing_payload_encoders=encoders,
            routing_query_factors=query_factors,
            target_maps=target_maps,
        )

    for sweep in range(1, int(sweeps) + 1):
        before = loss()
        query_factors, query_diagnostics = refit_page_fisher_query_factors(
            statistics,
            routing_encoders=encoders,
            relative_damping=relative_damping,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            target_maps=target_maps,
        )
        after_queries = loss()
        encoders, encoder_step = refit_page_fisher_routing_encoders(
            statistics,
            routing_query_factors=query_factors,
            active_joint_rows=active_joint_rows,
            relative_damping=relative_damping,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            target_maps=target_maps,
        )
        encoder_diagnostics.append(encoder_step)
        records.append(
            PageFisherRouterSweep(
                sweep=sweep,
                loss_before=before,
                loss_after_queries=after_queries,
                loss_after_encoder=loss(),
            )
        )
        print(
            f"Page-Fisher sweep {sweep}/{sweeps}: "
            f"loss {before:.6g} -> {records[-1].loss_after_encoder:.6g}",
            flush=True,
        )
    query_factors, query_diagnostics = refit_page_fisher_query_factors(
        statistics,
        routing_encoders=encoders,
        relative_damping=relative_damping,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
        target_maps=target_maps,
    )
    return PageFisherRouterFit(
        routing_encoders=encoders,
        routing_query_factors=query_factors,
        sweeps=tuple(records),
        final_query_diagnostics=query_diagnostics,
        encoder_diagnostics=tuple(encoder_diagnostics),
    )


__all__ = [
    "PageFisherRouterFit",
    "PageFisherRouterSweep",
    "fit_page_fisher_router",
    "refit_page_fisher_query_factors",
    "refit_page_fisher_routing_encoders",
]
