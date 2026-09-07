import math

import torch

from evaluation.eval_qwen3_8b_v80_fisher_base import (
    _build_statistics,
    _evaluate_fresh,
    _fit_predictive_bases,
    _fit_routers,
)


def test_fisher_base_pipeline_runs_end_to_end_on_synthetic_rows() -> None:
    generator = torch.Generator().manual_seed(211)
    documents, tokens = 4, 16
    groups, heads_per_group, head_dim, value_rank = 2, 2, 4, 3
    terminal_queries = torch.randn(
        documents,
        groups * heads_per_group,
        head_dim,
        generator=generator,
    )
    queries = torch.stack((terminal_queries.roll(1, dims=-1), terminal_queries), dim=1)
    query_positions = torch.tensor([7, 15])
    rows = torch.randn(
        documents,
        tokens,
        groups,
        2 * head_dim,
        generator=generator,
    )
    value_encoder = torch.randn(
        groups,
        head_dim,
        value_rank,
        generator=generator,
    )
    output_decoder = torch.randn(
        groups * heads_per_group,
        value_rank,
        5,
        generator=generator,
    )
    cos = torch.ones(1, tokens, head_dim)
    sin = torch.zeros_like(cos)
    bases, _ = _fit_predictive_bases(
        queries[:3],
        rows[:3],
        query_positions=query_positions,
        value_encoder=value_encoder,
        base_rank=2,
        cos=cos,
        sin=sin,
        pinned_prefix_tokens=2,
        position_chunk=4,
        device=torch.device("cpu"),
    )
    fit_statistics = _build_statistics(
        queries[:3],
        rows[:3],
        query_positions=query_positions,
        source_names=("mse16", "q16", "mse80"),
        value_encoder=value_encoder,
        base_maps=bases,
        cos=cos,
        sin=sin,
        page_size=2,
        pinned_prefix_pages=1,
        device=torch.device("cpu"),
    )
    validation_statistics = _build_statistics(
        queries[3:],
        rows[3:],
        query_positions=query_positions,
        source_names=("mse16", "q16", "mse80"),
        value_encoder=value_encoder,
        base_maps=bases,
        cos=cos,
        sin=sin,
        page_size=2,
        pinned_prefix_pages=1,
        device=torch.device("cpu"),
    )
    fisher_base, residual, diagnostics = _fit_routers(
        fit_statistics,
        validation_statistics,
        base_rank=2,
        residual_ranks=(0, 1),
        sweeps=2,
        relative_damping=1e-5,
        tolerance=1e-6,
        max_iterations=50,
        device=torch.device("cpu"),
    )
    metrics = _evaluate_fresh(
        terminal_queries[3:],
        rows[3:],
        value_encoder=value_encoder,
        output_decoder=output_decoder,
        base_maps=bases,
        page_fisher_base=fisher_base["page_fisher"],
        residual_factors=residual,
        residual_ranks=(0, 1),
        cos=cos,
        sin=sin,
        page_size=2,
        page_budget=3,
        pinned_prefix_pages=1,
        device=torch.device("cpu"),
    )

    assert set(metrics) == {
        "mse_r0",
        "mse_r1",
        "q_rrr_r0",
        "q_rrr_r1",
        "page_fisher_r0",
        "page_fisher_r1",
        "mse80_r0",
        "q_rrr80_r0",
    }
    assert set(diagnostics) == {
        "mse_r0",
        "mse_r1",
        "q_rrr_r0",
        "q_rrr_r1",
        "page_fisher_r0",
        "page_fisher_r1",
    }
    assert all(
        math.isfinite(float(value))
        for arm in metrics.values()
        for value in arm.values()
    )
