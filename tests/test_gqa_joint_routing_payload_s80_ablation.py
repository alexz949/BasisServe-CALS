import torch

from basisserve.core.gqa_joint_routing_payload_s80_ablation import (
    fit_page_fisher_router,
    refit_page_fisher_query_factors,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (
    compact_softmax_fisher_loss,
    pack_symmetric_fisher_grams,
    page_softmax_fisher_gram,
    prepare_compact_softmax_fisher_routing,
)


def _problem():
    generator = torch.Generator().manual_seed(71)
    heads = 2
    observations = 5
    value_dim = 2
    key_dim = 2
    joint_dim = value_dim + key_dim
    queries = torch.randn(
        heads,
        observations,
        key_dim,
        generator=generator,
        dtype=torch.float64,
    )
    rows = torch.randn(
        observations,
        8,
        joint_dim,
        generator=generator,
        dtype=torch.float64,
    )
    grams = torch.empty(heads, observations, joint_dim, joint_dim, dtype=torch.float64)
    energy = 0.0
    for observation in range(observations):
        current, current_energy = page_softmax_fisher_gram(
            queries[:, observation],
            rows[observation],
            value_dim=value_dim,
            scaling=key_dim**-0.5,
            page_size=2,
        )
        grams[:, observation] = current
        energy += current_energy
    statistics = prepare_compact_softmax_fisher_routing(
        queries_by_head=queries,
        fisher_grams_packed_by_head=pack_symmetric_fisher_grams(grams),
        head_to_kv_group=torch.zeros(heads, dtype=torch.long),
        value_dim=value_dim,
        key_dim=key_dim,
        scaling=key_dim**-0.5,
        teacher_fisher_energy=energy,
        device="cpu",
        dtype=torch.float64,
    )
    return statistics


def test_query_refit_recovers_exact_key_router() -> None:
    statistics = _problem()
    encoder = torch.zeros(1, 4, 2, dtype=torch.float64)
    encoder[:, 2:] = torch.eye(2, dtype=torch.float64)
    query_factors, diagnostics = refit_page_fisher_query_factors(
        statistics,
        routing_encoders=encoder,
        relative_damping=1e-10,
        relative_tolerance=1e-10,
        max_iterations=100,
    )
    loss = compact_softmax_fisher_loss(
        statistics,
        routing_payload_encoders=encoder,
        routing_query_factors=query_factors,
    )
    assert loss < 1e-16
    assert len(diagnostics) == 2


def test_k_only_page_fisher_bcd_is_monotone() -> None:
    statistics = _problem()
    generator = torch.Generator().manual_seed(73)
    encoder = torch.zeros(1, 4, 1, dtype=torch.float64)
    encoder[:, 2:] = torch.randn(1, 2, 1, generator=generator, dtype=torch.float64)
    query_factors = torch.randn(2, 2, 1, generator=generator, dtype=torch.float64)
    result = fit_page_fisher_router(
        statistics,
        initial_routing_encoders=encoder,
        initial_query_factors=query_factors,
        active_joint_rows=torch.arange(2, 4),
        sweeps=4,
        relative_damping=1e-8,
        relative_tolerance=1e-10,
        max_iterations=100,
    )
    values = []
    for sweep in result.sweeps:
        values.extend(
            (sweep.loss_before, sweep.loss_after_queries, sweep.loss_after_encoder)
        )
    assert all(right <= left + 1e-10 for left, right in zip(values, values[1:]))
    assert torch.count_nonzero(result.routing_encoders[:, :2]) == 0
