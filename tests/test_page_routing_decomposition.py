import torch

from basisserve.core.page_routing_decomposition import (
    budget_recovery,
    fisher_partition,
    page_lse_terms,
    top_page_masks,
)


def test_page_lse_linearization_is_exact_for_page_constant_error() -> None:
    exact = torch.tensor([[0.0, 1.0, -1.0, 2.0]], dtype=torch.float64)
    offsets = torch.tensor([[0.4, 0.4, -0.7, -0.7]], dtype=torch.float64)
    terms = page_lse_terms(exact, exact + offsets, page_size=2)
    torch.testing.assert_close(
        terms.true_error,
        torch.tensor([[0.4, -0.7]], dtype=torch.float64),
    )
    torch.testing.assert_close(terms.remainder, torch.zeros_like(terms.remainder))


def test_fisher_partition_sums_to_total() -> None:
    generator = torch.Generator().manual_seed(13)
    page_error = torch.randn(5, 9, generator=generator, dtype=torch.float64)
    mass = torch.softmax(
        torch.randn(5, 9, generator=generator, dtype=torch.float64),
        dim=-1,
    )
    inside = top_page_masks(mass, pages=3)
    result = fisher_partition(page_error, mass, inside)
    torch.testing.assert_close(
        result.total,
        result.inside_inside + result.outside_outside + result.cross,
    )


def test_budget_recovery_counts_only_newly_added_pages() -> None:
    teacher_logits = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0]])
    proxy_logits = torch.tensor([[9.0, 6.0, 8.0, 7.0, 5.0]])
    mass = torch.softmax(teacher_logits, dim=-1)
    result = budget_recovery(
        teacher_logits,
        proxy_logits,
        mass,
        pages=2,
    )
    assert int(result["false_negative_count"].item()) == 1
    assert float(result["false_negative_recovery"].item()) == 1.0
    assert bool(result["added_mask"][0, 1])
