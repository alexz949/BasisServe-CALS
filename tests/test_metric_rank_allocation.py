from __future__ import annotations

import itertools
import math
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.metric_rank_allocation import (
    MetricRankOption,
    adverse_cvar,
    allocate_metric_rank_exact,
    paired_standard_error,
    same_rank_family_guard,
    scalarize_domain_costs,
)


def _option(
    option_id: str,
    family: str,
    rank: int,
    cost: float,
    *,
    anchor: bool = False,
) -> MetricRankOption:
    return MetricRankOption(
        option_id=option_id,
        source_family=family,
        rank=rank,
        scalar_cost=cost,
        is_anchor=anchor,
    )


def test_exact_dp_matches_brute_force_with_duplicate_ranks() -> None:
    options = [
        (
            _option("a0", "anchor", 64, 0.0, anchor=True),
            _option("k48", "kfac", 48, 0.3),
            _option("s64", "score_bestkron", 64, -0.1),
            _option("k80", "kfac", 80, -0.4),
        ),
        (
            _option("a1", "anchor", 64, 0.0, anchor=True),
            _option("k48", "kfac", 48, -0.2),
            _option("s64", "score_bestkron", 64, -0.05),
            _option("k80", "kfac", 80, 0.25),
        ),
        (
            _option("a2", "anchor", 64, 0.0, anchor=True),
            _option("k48", "kfac", 48, 0.1),
            _option("s64", "score_bestkron", 64, -0.02),
            _option("k80", "kfac", 80, -0.15),
        ),
    ]
    budget = 3 * 64
    actual = allocate_metric_rank_exact(options, total_rank_budget=budget)
    feasible = [
        candidate
        for candidate in itertools.product(*options)
        if sum(item.rank for item in candidate) == budget
    ]
    expected_cost = min(sum(item.scalar_cost for item in candidate) for candidate in feasible)
    assert actual.total_cost == pytest.approx(expected_cost)
    assert sum(option.rank for option in actual.selected_options) == budget


def test_exact_budget_implies_equal_rank48_and_rank80_counts() -> None:
    coordinates = [
        (
            _option(f"a{index}", "anchor", 64, 0.0, anchor=True),
            _option(f"d{index}", "kfac", 48, -index),
            _option(f"u{index}", "kfac", 80, index / 10),
        )
        for index in range(8)
    ]
    result = allocate_metric_rank_exact(
        coordinates,
        total_rank_budget=8 * 64,
    )
    ranks = [option.rank for option in result.selected_options]
    assert ranks.count(48) == ranks.count(80)
    assert sum(ranks) == 8 * 64


def test_deterministic_ties_prefer_anchor_then_kfac() -> None:
    coordinates = [
        (
            _option(f"anchor_{index}", "anchor", 64, 0.0, anchor=True),
            _option(f"kfac_{index}", "kfac", 64, 0.0),
            _option(f"score_{index}", "score_bestkron", 64, 0.0),
        )
        for index in range(4)
    ]
    first = allocate_metric_rank_exact(coordinates, total_rank_budget=256)
    second = allocate_metric_rank_exact(coordinates, total_rank_budget=256)
    assert first == second
    assert [option.option_id for option in first.selected_options] == [
        f"anchor_{index}" for index in range(4)
    ]


def test_churn_constraint_keeps_required_partial_states() -> None:
    coordinates = [
        (
            _option(f"a{index}", "anchor", 64, 0.0, anchor=True),
            _option(f"d{index}", "kfac", 48, -10.0),
            _option(f"u{index}", "kfac", 80, -10.0),
        )
        for index in range(4)
    ]
    result = allocate_metric_rank_exact(
        coordinates,
        total_rank_budget=256,
        max_changed_coordinates=2,
    )
    assert result.changed_coordinates == 2
    assert sum(option.rank for option in result.selected_options) == 256


def test_scalarizations_match_direct_calculation() -> None:
    summaries = {
        "a": {"mean_delta": -2.0, "upper_confidence_bound": -1.0},
        "b": {"mean_delta": 4.0, "upper_confidence_bound": 7.0},
    }
    actual = scalarize_domain_costs(summaries)
    assert actual["equal_domain_mean"] == pytest.approx(1.0)
    assert actual["equal_domain_ucb"] == pytest.approx(3.0)
    assert actual["coordinate_worst_domain_ucb"] == pytest.approx(7.0)


def test_window_standard_error_and_upper_tail_cvar() -> None:
    values = [-2.0, -1.0, 0.5, 1.5]
    expected_se = pytest.approx(1.5545631755148024 / math.sqrt(4))
    assert paired_standard_error(values) == expected_se
    assert adverse_cvar(values, level=0.5) == pytest.approx(1.0)


def _guard(differences_by_domain: dict[str, list[float]]) -> dict:
    zeros = {
        domain: [0.0] * len(values)
        for domain, values in differences_by_domain.items()
    }
    return same_rank_family_guard(
        score_by_domain=differences_by_domain,
        kfac_by_domain=zeros,
        kappa=1.0,
        cvar_level=0.9,
    )


def test_family_guard_failure_modes_and_success() -> None:
    mean_pass_ucb_fail = _guard({"a": [-1.0, -1.0, -1.0, 2.0]})
    assert mean_pass_ucb_fail["domains"]["a"]["mean_delta"] < 0
    assert not mean_pass_ucb_fail["domains"]["a"]["passes_ucb"]

    ucb_pass_cvar_fail = _guard({"a": [-1.0] * 28 + [0.1] * 4})
    assert ucb_pass_cvar_fail["domains"]["a"]["passes_ucb"]
    assert not ucb_pass_cvar_fail["domains"]["a"]["passes_cvar"]

    one_domain_fails = _guard(
        {"a": [-0.2] * 32, "b": [-0.2] * 28 + [0.1] * 4}
    )
    assert one_domain_fails["domains"]["a"]["admissible"]
    assert not one_domain_fails["admissible"]

    passes = _guard({"a": [-0.2] * 32, "b": [-0.1] * 32})
    assert passes["admissible"]


def test_positive_rank48_cost_is_not_subject_to_family_guard() -> None:
    # The exact budget can still select a costly rank-down candidate to fund a
    # more valuable rank-up elsewhere; family guards compare only same-rank
    # Score-BestKron against KFAC.
    result = allocate_metric_rank_exact(
        [
            (
                _option("a0", "anchor", 64, 0.0, anchor=True),
                _option("k48", "kfac", 48, 1.0),
            ),
            (
                _option("a1", "anchor", 64, 0.0, anchor=True),
                _option("k80", "kfac", 80, -3.0),
            ),
        ],
        total_rank_budget=128,
    )
    assert [option.rank for option in result.selected_options] == [48, 80]
