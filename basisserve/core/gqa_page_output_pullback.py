"""C1-aware Page64 output-pullback metrics for K-only routing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PageOutputPullbackGrams:
    """Per-head K-side metrics for one teacher query and causal prefix."""

    output_grams: torch.Tensor
    page_fisher_grams: torch.Tensor
    teacher_output_energy: float
    teacher_page_fisher_energy: float


def page_output_pullback_grams(
    queries: torch.Tensor,
    key_rows: torch.Tensor,
    payload_latents: torch.Tensor,
    decoder_grams: torch.Tensor,
    *,
    scaling: float,
    page_size: int,
) -> PageOutputPullbackGrams:
    """Return Page-output and Page-Fisher K-side Grams.

    ``key_rows`` are post-RoPE keys for one physical KV group.  The resident
    C1 payload is ``payload_latents = V @ E_V``.  ``decoder_grams[h]`` is
    ``D_h @ D_h.T`` for the corresponding query head.  Neither returned Gram
    includes the QK scale; the compact quadratic applies ``0.5 * scaling**2``.
    """

    tokens = int(key_rows.shape[0])
    page_width = int(page_size)
    pages = (tokens + page_width - 1) // page_width
    padding = pages * page_width - tokens

    scores = float(scaling) * queries @ key_rows.mT
    probabilities = torch.softmax(scores, dim=-1)
    if padding:
        probabilities = F.pad(probabilities, (0, padding))
        key_rows = F.pad(key_rows, (0, 0, 0, padding))
        payload_latents = F.pad(payload_latents, (0, 0, 0, padding))

    probabilities_by_page = probabilities.reshape(
        queries.shape[0],
        pages,
        page_width,
    )
    key_by_page = key_rows.reshape(pages, page_width, key_rows.shape[-1])
    payload_by_page = payload_latents.reshape(
        pages,
        page_width,
        payload_latents.shape[-1],
    )
    page_mass = probabilities_by_page.sum(dim=-1)
    inverse_mass = page_mass.clamp_min(
        torch.finfo(page_mass.dtype).tiny
    ).reciprocal()
    page_keys = torch.einsum(
        "hps,psd->hpd",
        probabilities_by_page,
        key_by_page,
    ) * inverse_mass.unsqueeze(-1)
    page_payloads = torch.einsum(
        "hps,psr->hpr",
        probabilities_by_page,
        payload_by_page,
    ) * inverse_mass.unsqueeze(-1)

    mean_keys = torch.einsum("hp,hpd->hd", page_mass, page_keys)
    mean_payloads = torch.einsum("hp,hpr->hr", page_mass, page_payloads)
    centered_keys = page_keys - mean_keys.unsqueeze(1)
    centered_payloads = page_payloads - mean_payloads.unsqueeze(1)

    cross_covariances = torch.einsum(
        "hp,hpd,hpr->hdr",
        page_mass,
        centered_keys,
        centered_payloads,
    )
    output_grams = torch.einsum(
        "hdr,hrs,hks->hdk",
        cross_covariances,
        decoder_grams,
        cross_covariances,
    )
    page_fisher_grams = torch.einsum(
        "hp,hpd,hpk->hdk",
        page_mass,
        centered_keys,
        centered_keys,
    )
    output_grams = 0.5 * (output_grams + output_grams.mT)
    page_fisher_grams = 0.5 * (
        page_fisher_grams + page_fisher_grams.mT
    )

    output_energy = float(
        0.5
        * float(scaling) ** 2
        * torch.einsum("hd,hdk,hk->", queries, output_grams, queries)
    )
    page_fisher_energy = float(
        0.5
        * float(scaling) ** 2
        * torch.einsum("hd,hdk,hk->", queries, page_fisher_grams, queries)
    )
    return PageOutputPullbackGrams(
        output_grams=output_grams,
        page_fisher_grams=page_fisher_grams,
        teacher_output_energy=output_energy,
        teacher_page_fisher_energy=page_fisher_energy,
    )


__all__ = ["PageOutputPullbackGrams", "page_output_pullback_grams"]
