import numpy as np
import torch

from basisserve.core.query_cpqr_audit import audit_queries, compare_bin


def test_cpqr_matches_cholesky_on_well_separated_features():
    rng = np.random.default_rng(31)
    result = compare_bin(rng.normal(size=(19, 40)), list(range(19)), 8)
    assert result['identical_sequence']
    assert result['cpqr_geometry']['numerical_rank'] == 8
    assert result['cpqr_geometry']['residual_energy_fraction'] < 1


def test_rank_deficiency_does_not_claim_well_conditioned_basis():
    result = compare_bin(np.ones((8, 12)), list(range(8)), 4)
    assert result['cpqr_geometry']['numerical_rank'] == 1
    assert result['cpqr_geometry']['condition_number'] is None
    assert result['cpqr_geometry']['residual_energy_fraction'] < 1e-25


def test_full_window_head_geometry_matches_production_sampler():
    generator = torch.Generator().manual_seed(19)
    queries = torch.randn(3, 32, 2, 5, generator=generator).bfloat16()
    queries[..., 4] = queries[..., 3]
    result = audit_queries(queries, list(range(63, 2048, 64)),
                           context_length=2048, queries_per_bin=3)
    assert len(result['bins']) == 4
    assert all(b['production_matches_cpqr'] for b in result['bins'])
    assert all(b['gram_max_absolute_difference'] < 1e-10 for b in result['bins'])
