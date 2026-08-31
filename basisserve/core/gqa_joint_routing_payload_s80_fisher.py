"""Forward-only softmax-Fisher routing for the shared Store80 latent."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from basisserve.calibration.gqa_joint_routing_payload_s80_stats import (
    S80DirectResidualData,
)
from basisserve.core.exact_qk_v_offload import gqa_union_page_mass_mask


@dataclass(frozen=True)
class S80SoftmaxFisherRouting:
    queries_by_head: torch.Tensor
    joint_rows_by_group: torch.Tensor
    exact_scores_by_head: torch.Tensor
    probabilities_by_head: torch.Tensor
    head_to_kv_group: torch.Tensor
    value_dim: int
    key_dim: int
    scaling: float
    teacher_fisher_energy: float

    @property
    def documents(self) -> int:
        return int(self.queries_by_head.shape[1])

    @property
    def tokens(self) -> int:
        return int(self.joint_rows_by_group.shape[2])


def softmax_fisher_transform(
    scores: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    """Return a residual whose squared norm is ``0.5 delta^T J_p delta``."""

    mean = torch.sum(probabilities * scores, dim=-1, keepdim=True)
    centered = scores - mean
    return torch.sqrt(0.5 * probabilities) * centered


def softmax_fisher_adjoint(
    residual: torch.Tensor,
    probabilities: torch.Tensor,
) -> torch.Tensor:
    """Apply the transpose of :func:`softmax_fisher_transform`."""

    weighted = torch.sqrt(0.5 * probabilities) * residual
    return weighted - probabilities * weighted.sum(dim=-1, keepdim=True)


def prepare_softmax_fisher_routing(
    direct: S80DirectResidualData,
    *,
    head_to_kv_group: torch.Tensor,
    value_dim: int,
    key_dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> S80SoftmaxFisherRouting:
    """Materialize teacher logits and probabilities from paired direct capture."""

    target_device = torch.device(device)
    queries = (
        direct.routing_queries.permute(1, 0, 2)
        .contiguous()
        .to(device=target_device, dtype=dtype)
    )
    rows = (
        direct.routing_joint_rows.permute(2, 0, 1, 3)
        .contiguous()
        .to(device=target_device, dtype=dtype)
    )
    mapping = head_to_kv_group.to(device=target_device, dtype=torch.long)
    heads = int(queries.shape[0])
    exact = torch.empty(
        heads,
        int(queries.shape[1]),
        int(rows.shape[2]),
        device=target_device,
        dtype=dtype,
    )
    scaling = key_dim**-0.5
    for group in range(int(rows.shape[0])):
        head_indices = torch.nonzero(mapping == group, as_tuple=False).flatten()
        group_scores = torch.einsum(
            "hdk,dtk->hdt",
            queries.index_select(0, head_indices),
            rows[group, ..., value_dim : value_dim + key_dim],
        )
        exact.index_copy_(0, head_indices, scaling * group_scores)
    probabilities = torch.softmax(exact, dim=-1)
    teacher_energy = float(
        softmax_fisher_transform(exact, probabilities).square().sum()
    )
    return S80SoftmaxFisherRouting(
        queries_by_head=queries,
        joint_rows_by_group=rows,
        exact_scores_by_head=exact,
        probabilities_by_head=probabilities,
        head_to_kv_group=mapping,
        value_dim=int(value_dim),
        key_dim=int(key_dim),
        scaling=float(scaling),
        teacher_fisher_energy=teacher_energy,
    )


def softmax_fisher_proxy_scores(
    routing: S80SoftmaxFisherRouting,
    *,
    routing_payload_encoders: torch.Tensor,
    routing_query_factors: torch.Tensor,
) -> torch.Tensor:
    """Compute every Store80 Route32 proxy score in head-major layout."""

    proxy = torch.empty_like(routing.exact_scores_by_head)
    groups = int(routing.joint_rows_by_group.shape[0])
    for group in range(groups):
        head_indices = torch.nonzero(
            routing.head_to_kv_group == group,
            as_tuple=False,
        ).flatten()
        query_code = torch.einsum(
            "hdk,hkr->hdr",
            routing.queries_by_head.index_select(0, head_indices),
            routing_query_factors.index_select(0, head_indices),
        )
        token_code = torch.einsum(
            "dti,ir->dtr",
            routing.joint_rows_by_group[group],
            routing_payload_encoders[group],
        )
        scores = torch.einsum("hdr,dtr->hdt", query_code, token_code)
        proxy.index_copy_(0, head_indices, routing.scaling * scores)
    return proxy


def softmax_fisher_routing_loss(
    routing: S80SoftmaxFisherRouting,
    *,
    routing_payload_encoders: torch.Tensor,
    routing_query_factors: torch.Tensor,
) -> float:
    proxy = softmax_fisher_proxy_scores(
        routing,
        routing_payload_encoders=routing_payload_encoders,
        routing_query_factors=routing_query_factors,
    )
    residual = softmax_fisher_transform(
        proxy - routing.exact_scores_by_head,
        routing.probabilities_by_head,
    )
    return float(residual.square().sum())


def softmax_fisher_encoder_diagonal(
    *,
    joint_rows: torch.Tensor,
    projected_queries: torch.Tensor,
    probabilities: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Return the exact Fisher Hessian diagonal for one Route32 encoder block."""

    mean = torch.einsum("dht,dti->dhi", probabilities, joint_rows)
    second = torch.einsum(
        "dht,dti->dhi",
        probabilities,
        joint_rows.square(),
    )
    variance = torch.clamp(second - mean.square(), min=0)
    return 0.5 * scaling**2 * torch.einsum(
        "dhi,dhr->ir",
        variance,
        projected_queries.square(),
    )


def softmax_fisher_adapter_system(
    routing: S80SoftmaxFisherRouting,
    *,
    routing_payload_encoders: torch.Tensor,
    routing_query_factors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Kronecker factors and normal-equation RHS for adapter-U."""

    heads = int(routing.queries_by_head.shape[0])
    documents = routing.documents
    rank = int(routing_query_factors.shape[-1])
    left = routing_query_factors.new_empty(documents, heads, rank, rank)
    right = routing_query_factors.new_empty(documents, heads, rank, rank)
    gradient = routing_query_factors.new_zeros(heads, rank, rank)
    for group in range(int(routing.joint_rows_by_group.shape[0])):
        head_indices = torch.nonzero(
            routing.head_to_kv_group == group,
            as_tuple=False,
        ).flatten()
        query_code = torch.einsum(
            "hdk,hkr->hdr",
            routing.queries_by_head.index_select(0, head_indices),
            routing_query_factors.index_select(0, head_indices),
        )
        latent = torch.einsum(
            "dti,ir->dtr",
            routing.joint_rows_by_group[group],
            routing_payload_encoders[group],
        )
        exact = routing.exact_scores_by_head.index_select(0, head_indices)
        probabilities = routing.probabilities_by_head.index_select(0, head_indices)
        proxy = routing.scaling * torch.einsum("hdr,dtr->hdt", query_code, latent)
        delta = proxy - exact
        centered = delta - torch.sum(
            probabilities * delta,
            dim=-1,
            keepdim=True,
        )
        fisher_score = 0.5 * probabilities * centered
        weighted_latent = torch.einsum("hdt,dtr->hdr", fisher_score, latent)
        group_gradient = routing.scaling * torch.einsum(
            "hdr,hds->hrs",
            query_code,
            weighted_latent,
        )
        gradient.index_copy_(0, head_indices, group_gradient)
        group_left = torch.einsum("hdr,hds->dhrs", query_code, query_code)
        left[:, head_indices] = group_left
        for local, head in enumerate(head_indices.tolist()):
            probability = probabilities[local]
            mean = torch.einsum("dt,dtr->dr", probability, latent)
            second = torch.einsum(
                "dtr,dt,dts->drs",
                latent,
                probability,
                latent,
            )
            covariance = second - torch.einsum("dr,ds->drs", mean, mean)
            covariance = 0.5 * (covariance + covariance.mT)
            right[:, head] = 0.5 * routing.scaling**2 * covariance
    return left, right, -gradient


def softmax_fisher_routing_diagnostics(
    routing: S80SoftmaxFisherRouting,
    *,
    routing_payload_encoders: torch.Tensor,
    payload_only_encoders: torch.Tensor,
    routing_query_factors: torch.Tensor,
    payload_decoders: torch.Tensor,
    page_size: int,
    exact_token_budget: int,
) -> dict[str, float | int]:
    """Measure score, physical-page, mass, and exact-refined output quality."""

    proxy = softmax_fisher_proxy_scores(
        routing,
        routing_payload_encoders=routing_payload_encoders,
        routing_query_factors=routing_query_factors,
    )
    delta = proxy - routing.exact_scores_by_head
    raw_nmse = float(delta.square().sum() / routing.exact_scores_by_head.square().sum())
    fisher_loss = float(
        softmax_fisher_transform(delta, routing.probabilities_by_head).square().sum()
    )
    joint_encoders = torch.cat(
        (routing_payload_encoders, payload_only_encoders),
        dim=-1,
    )
    page_recalls: list[float] = []
    mass_recalls: list[float] = []
    selected_fractions: list[float] = []
    output_error = 0.0
    output_energy = 0.0
    pages_per_head = math.ceil(exact_token_budget / page_size)
    heads_per_group = int(routing.queries_by_head.shape[0]) // int(
        routing.joint_rows_by_group.shape[0]
    )
    for document in range(routing.documents):
        exact_scores = routing.exact_scores_by_head[:, document]
        proxy_scores = proxy[:, document]
        proxy_tokens, proxy_pages = gqa_union_page_mass_mask(
            proxy_scores,
            num_kv_heads=int(routing.joint_rows_by_group.shape[0]),
            page_size=page_size,
            pages_per_query_head=pages_per_head,
        )
        _, teacher_pages = gqa_union_page_mass_mask(
            exact_scores,
            num_kv_heads=int(routing.joint_rows_by_group.shape[0]),
            page_size=page_size,
            pages_per_query_head=pages_per_head,
        )
        page_recalls.extend(
            (
                (proxy_pages & teacher_pages).sum(dim=-1)
                / teacher_pages.sum(dim=-1).clamp_min(1)
            )
            .tolist()
        )
        query_mask = proxy_tokens.repeat_interleave(heads_per_group, dim=0)
        probabilities = routing.probabilities_by_head[:, document]
        mass_recalls.extend((probabilities * query_mask).sum(dim=-1).tolist())
        selected_fractions.extend(proxy_tokens.float().mean(dim=-1).tolist())

        latent = torch.einsum(
            "gti,gir->gtr",
            routing.joint_rows_by_group[:, document],
            joint_encoders,
        )
        expanded_latent = latent.index_select(0, routing.head_to_kv_group)
        dense_latent = torch.einsum("ht,htr->hr", probabilities, expanded_latent)
        sparse_probability = torch.softmax(
            exact_scores.masked_fill(~query_mask, -torch.inf),
            dim=-1,
        )
        sparse_latent = torch.einsum(
            "ht,htr->hr",
            sparse_probability,
            expanded_latent,
        )
        dense_output = torch.einsum("hr,hro->o", dense_latent, payload_decoders)
        sparse_output = torch.einsum("hr,hro->o", sparse_latent, payload_decoders)
        output_error += float((sparse_output - dense_output).square().sum())
        output_energy += float(dense_output.square().sum())
    return {
        "documents": routing.documents,
        "tokens": routing.tokens,
        "raw_score_nmse": raw_nmse,
        "softmax_fisher_loss": fisher_loss,
        "softmax_fisher_nmse": fisher_loss / routing.teacher_fisher_energy,
        "teacher_fisher_energy": routing.teacher_fisher_energy,
        "physical_page_recall_mean": sum(page_recalls) / len(page_recalls),
        "physical_page_recall_minimum": min(page_recalls),
        "attention_mass_recall_mean": sum(mass_recalls) / len(mass_recalls),
        "attention_mass_recall_minimum": min(mass_recalls),
        "selected_token_fraction_mean": (
            sum(selected_fractions) / len(selected_fractions)
        ),
        "exact_refined_output_relative_mse": output_error / output_energy,
        "page_size": page_size,
        "exact_token_budget": exact_token_budget,
    }


__all__ = [
    "S80SoftmaxFisherRouting",
    "prepare_softmax_fisher_routing",
    "softmax_fisher_adapter_system",
    "softmax_fisher_adjoint",
    "softmax_fisher_encoder_diagonal",
    "softmax_fisher_proxy_scores",
    "softmax_fisher_routing_diagnostics",
    "softmax_fisher_routing_loss",
    "softmax_fisher_transform",
]
