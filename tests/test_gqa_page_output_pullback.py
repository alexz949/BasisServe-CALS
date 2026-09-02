import torch

from basisserve.core.gqa_joint_routing_payload_s80_ablation import (
    fit_page_fisher_router,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (
    pack_symmetric_fisher_grams,
    prepare_compact_softmax_fisher_routing,
)
from basisserve.core.gqa_page_output_pullback import (
    page_output_pullback_grams,
)


def _page_teacher(
    query: torch.Tensor,
    keys: torch.Tensor,
    payloads: torch.Tensor,
    *,
    scaling: float,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(scaling * keys @ query, dim=0)
    pages = keys.shape[0] // page_size
    probabilities = probabilities.reshape(pages, page_size)
    keys = keys.reshape(pages, page_size, -1)
    payloads = payloads.reshape(pages, page_size, -1)
    mass = probabilities.sum(dim=-1)
    page_keys = torch.einsum("ps,psd->pd", probabilities, keys) / mass[:, None]
    page_payloads = (
        torch.einsum("ps,psr->pr", probabilities, payloads) / mass[:, None]
    )
    return mass, page_keys, page_payloads


def test_page_output_gram_matches_explicit_output_perturbation() -> None:
    generator = torch.Generator().manual_seed(101)
    dtype = torch.float64
    heads = 3
    key_dim = 4
    payload_rank = 3
    output_dim = 5
    page_size = 2
    scaling = key_dim**-0.5
    queries = torch.randn(heads, key_dim, generator=generator, dtype=dtype)
    keys = torch.randn(8, key_dim, generator=generator, dtype=dtype)
    payloads = torch.randn(8, payload_rank, generator=generator, dtype=dtype)
    decoders = torch.randn(
        heads,
        payload_rank,
        output_dim,
        generator=generator,
        dtype=dtype,
    )
    delta = torch.randn(
        heads,
        key_dim,
        key_dim,
        generator=generator,
        dtype=dtype,
    )
    result = page_output_pullback_grams(
        queries,
        keys,
        payloads,
        decoders @ decoders.mT,
        scaling=scaling,
        page_size=page_size,
    )

    explicit_output = torch.zeros((), dtype=dtype)
    explicit_fisher = torch.zeros((), dtype=dtype)
    for head in range(heads):
        mass, page_keys, page_payloads = _page_teacher(
            queries[head],
            keys,
            payloads,
            scaling=scaling,
            page_size=page_size,
        )
        jacobian = torch.diag(mass) - mass[:, None] * mass[None, :]
        page_logit_error = scaling * page_keys @ delta[head] @ queries[head]
        output_error = decoders[head].mT @ page_payloads.mT @ jacobian @ page_logit_error
        explicit_output += 0.5 * output_error.square().sum()
        explicit_fisher += 0.5 * page_logit_error @ jacobian @ page_logit_error

    projected = torch.einsum("hij,hj->hi", delta, queries)
    compact_output = 0.5 * scaling**2 * torch.einsum(
        "hi,hij,hj->",
        projected,
        result.output_grams,
        projected,
    )
    compact_fisher = 0.5 * scaling**2 * torch.einsum(
        "hi,hij,hj->",
        projected,
        result.page_fisher_grams,
        projected,
    )
    torch.testing.assert_close(compact_output, explicit_output)
    torch.testing.assert_close(compact_fisher, explicit_fisher)


def test_constant_decoded_payload_has_zero_output_metric() -> None:
    generator = torch.Generator().manual_seed(103)
    dtype = torch.float64
    queries = torch.randn(2, 4, generator=generator, dtype=dtype)
    keys = torch.randn(8, 4, generator=generator, dtype=dtype)
    payload = torch.randn(1, 3, generator=generator, dtype=dtype).expand(8, -1)
    decoders = torch.randn(2, 3, 5, generator=generator, dtype=dtype)
    result = page_output_pullback_grams(
        queries,
        keys,
        payload,
        decoders @ decoders.mT,
        scaling=0.5,
        page_size=2,
    )
    torch.testing.assert_close(
        result.output_grams,
        torch.zeros_like(result.output_grams),
        atol=1e-28,
        rtol=0,
    )
    assert result.teacher_output_energy < 1e-28


def test_output_and_fisher_grams_are_positive_semidefinite() -> None:
    generator = torch.Generator().manual_seed(107)
    dtype = torch.float64
    queries = torch.randn(4, 6, generator=generator, dtype=dtype)
    keys = torch.randn(9, 6, generator=generator, dtype=dtype)
    payloads = torch.randn(9, 3, generator=generator, dtype=dtype)
    decoders = torch.randn(4, 3, 7, generator=generator, dtype=dtype)
    result = page_output_pullback_grams(
        queries,
        keys,
        payloads,
        decoders @ decoders.mT,
        scaling=6**-0.5,
        page_size=4,
    )
    assert torch.linalg.eigvalsh(result.output_grams).min() > -1e-10
    assert torch.linalg.eigvalsh(result.page_fisher_grams).min() > -1e-10


def test_k_only_output_pullback_bcd_is_monotone() -> None:
    generator = torch.Generator().manual_seed(109)
    dtype = torch.float64
    heads = 4
    groups = 2
    observations = 4
    key_dim = 4
    payload_rank = 3
    output_dim = 5
    scaling = key_dim**-0.5
    queries = torch.randn(
        heads,
        observations,
        key_dim,
        generator=generator,
        dtype=dtype,
    )
    decoders = torch.randn(
        heads,
        payload_rank,
        output_dim,
        generator=generator,
        dtype=dtype,
    )
    decoder_grams = decoders @ decoders.mT
    grams = torch.empty(
        heads,
        observations,
        key_dim,
        key_dim,
        dtype=dtype,
    )
    energy = 0.0
    for observation in range(observations):
        for group in range(groups):
            first = group * (heads // groups)
            stop = first + heads // groups
            keys = torch.randn(
                8,
                key_dim,
                generator=generator,
                dtype=dtype,
            )
            payloads = torch.randn(
                8,
                payload_rank,
                generator=generator,
                dtype=dtype,
            )
            result = page_output_pullback_grams(
                queries[first:stop, observation],
                keys,
                payloads,
                decoder_grams[first:stop],
                scaling=scaling,
                page_size=2,
            )
            grams[first:stop, observation] = result.output_grams
            energy += result.teacher_output_energy
    statistics = prepare_compact_softmax_fisher_routing(
        queries_by_head=queries,
        fisher_grams_packed_by_head=pack_symmetric_fisher_grams(grams),
        head_to_kv_group=torch.arange(heads) // (heads // groups),
        value_dim=0,
        key_dim=key_dim,
        scaling=scaling,
        teacher_fisher_energy=energy,
        device="cpu",
        dtype=dtype,
    )
    initial_encoder = torch.randn(
        groups,
        key_dim,
        2,
        generator=generator,
        dtype=dtype,
    )
    initial_query = torch.randn(
        heads,
        key_dim,
        2,
        generator=generator,
        dtype=dtype,
    )
    fitted = fit_page_fisher_router(
        statistics,
        initial_routing_encoders=initial_encoder,
        initial_query_factors=initial_query,
        active_joint_rows=torch.arange(key_dim),
        sweeps=4,
        relative_damping=1e-8,
        relative_tolerance=1e-10,
        max_iterations=100,
    )
    values = []
    for sweep in fitted.sweeps:
        values.extend(
            (sweep.loss_before, sweep.loss_after_queries, sweep.loss_after_encoder)
        )
    assert all(right <= left + 1e-9 for left, right in zip(values, values[1:]))
