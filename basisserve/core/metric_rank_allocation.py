"""Exact metric-and-rank allocation with paired multi-domain risk controls.

The routines in this module are deliberately model-free.  Candidate factors
are built elsewhere; this module only summarizes paired forward metrics and
performs deterministic discrete selection under an exact physical-rank
budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MetricRankOption:
    """One immutable factor choice for one physical KV group."""

    option_id: str
    source_family: str
    rank: int
    scalar_cost: float
    source_bank: str = ""
    source_factor_hash: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    is_anchor: bool = False


@dataclass(frozen=True)
class MetricRankAllocation:
    """The deterministic result of an exact-budget allocation."""

    selected_options: tuple[MetricRankOption, ...]
    total_rank: int
    total_cost: float
    changed_coordinates: int
    total_absolute_rank_deviation: int
    tie_key: tuple[Any, ...]


@dataclass(frozen=True)
class _Partial:
    options: tuple[MetricRankOption, ...]
    cost: float
    changes: int
    absolute_rank_deviation: int
    score_bestkron_choices: int

    @property
    def tie_key(self) -> tuple[Any, ...]:
        # The final two tuple fields implement coordinate-order deterministic
        # "lower rank, then lexical option id" tie breaking.
        return (
            self.changes,
            self.absolute_rank_deviation,
            self.score_bestkron_choices,
            tuple(option.rank for option in self.options),
            tuple(option.option_id for option in self.options),
        )


def paired_standard_error(values: Sequence[float]) -> float:
    """Return the sample standard error over independent paired windows."""

    checked = _finite_values(values)
    return (
        statistics.stdev(checked) / math.sqrt(len(checked))
        if len(checked) > 1
        else 0.0
    )


def adverse_cvar(values: Sequence[float], *, level: float = 0.9) -> float:
    """Mean of the worst upper-tail loss differences.

    Positive differences are adverse.  The finite-window definition follows
    the experiment preregistration: ``ceil((1-level) * n)`` complete windows,
    with at least one window retained.
    """

    checked = _finite_values(values)
    if not 0.0 < level < 1.0:
        raise ValueError("CVaR level must lie strictly between zero and one")
    tail_count = max(1, math.ceil((1.0 - level) * len(checked)))
    return statistics.fmean(sorted(checked, reverse=True)[:tail_count])


def paired_risk_summary(
    values: Sequence[float],
    *,
    kappa: float = 1.0,
    cvar_level: float = 0.9,
) -> dict[str, Any]:
    """Summarize paired candidate-minus-reference window differences."""

    checked = _finite_values(values)
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    mean = statistics.fmean(checked)
    standard_error = paired_standard_error(checked)
    tail_count = max(1, math.ceil((1.0 - cvar_level) * len(checked)))
    return {
        "windows": len(checked),
        "mean_delta": mean,
        "paired_standard_error": standard_error,
        "upper_confidence_bound": mean + kappa * standard_error,
        "median_delta": statistics.median(checked),
        "improved_windows": sum(value < 0 for value in checked),
        "tied_windows": sum(value == 0 for value in checked),
        "worsened_windows": sum(value > 0 for value in checked),
        "minimum_delta": min(checked),
        "maximum_adverse_delta": max(checked),
        "adverse_cvar_level": cvar_level,
        "adverse_cvar_tail_windows": tail_count,
        "adverse_cvar_mean": adverse_cvar(checked, level=cvar_level),
        "window_deltas": list(checked),
    }


def scalarize_domain_costs(
    domain_summaries: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Compute the three preregistered additive scalar costs.

    ``worst_domain_ucb`` is the sum of coordinate-wise worst-domain costs when
    used by the separable DP.  It is intentionally named precisely: it is not
    the non-separable global ``max_d sum_i`` objective.
    """

    if not domain_summaries:
        raise ValueError("at least one domain summary is required")
    means = []
    upper_bounds = []
    for domain, summary in sorted(domain_summaries.items()):
        try:
            mean = float(summary["mean_delta"])
            upper = float(summary["upper_confidence_bound"])
        except KeyError as exc:
            raise ValueError(f"domain {domain!r} lacks a scalarization field") from exc
        if not math.isfinite(mean) or not math.isfinite(upper):
            raise ValueError(f"domain {domain!r} has a non-finite cost")
        means.append(mean)
        upper_bounds.append(upper)
    return {
        "equal_domain_mean": statistics.fmean(means),
        "equal_domain_ucb": statistics.fmean(upper_bounds),
        "coordinate_worst_domain_ucb": max(upper_bounds),
    }


def same_rank_family_guard(
    *,
    score_by_domain: Mapping[str, Sequence[float]],
    kfac_by_domain: Mapping[str, Sequence[float]],
    kappa: float = 1.0,
    cvar_level: float = 0.9,
    ucb_epsilon: float = 0.0,
    tail_epsilon: float = 0.0,
) -> dict[str, Any]:
    """Test Score-BestKron against KFAC at the same rank on paired windows."""

    if set(score_by_domain) != set(kfac_by_domain) or not score_by_domain:
        raise ValueError("Score-BestKron and KFAC must cover identical domains")
    domains: dict[str, Any] = {}
    for domain in sorted(score_by_domain):
        score = _finite_values(score_by_domain[domain])
        kfac = _finite_values(kfac_by_domain[domain])
        if len(score) != len(kfac):
            raise ValueError(f"family guard windows are not paired for {domain}")
        differences = [left - right for left, right in zip(score, kfac, strict=True)]
        summary = paired_risk_summary(
            differences,
            kappa=kappa,
            cvar_level=cvar_level,
        )
        summary["passes_ucb"] = (
            summary["upper_confidence_bound"] <= ucb_epsilon
        )
        summary["passes_cvar"] = (
            summary["adverse_cvar_mean"] <= tail_epsilon
        )
        summary["admissible"] = summary["passes_ucb"] and summary["passes_cvar"]
        domains[domain] = summary
    return {
        "admissible": all(item["admissible"] for item in domains.values()),
        "comparison": "score_bestkron_minus_kfac_at_same_rank",
        "kappa": kappa,
        "cvar_level": cvar_level,
        "ucb_epsilon": ucb_epsilon,
        "tail_epsilon": tail_epsilon,
        "domains": domains,
    }


def allocate_metric_rank_exact(
    options_by_coordinate: Sequence[Sequence[MetricRankOption]],
    *,
    total_rank_budget: int,
    cost_tolerance: float = 1e-12,
    anchor_rank: int = 64,
    max_changed_coordinates: int | None = None,
) -> MetricRankAllocation:
    """Solve a separable option-valued exact-rank allocation by dynamic programming."""

    if total_rank_budget <= 0:
        raise ValueError("total_rank_budget must be positive")
    if not math.isfinite(cost_tolerance) or cost_tolerance < 0:
        raise ValueError("cost_tolerance must be finite and nonnegative")
    if anchor_rank <= 0:
        raise ValueError("anchor_rank must be positive")
    if max_changed_coordinates is not None and max_changed_coordinates < 0:
        raise ValueError("max_changed_coordinates must be nonnegative")
    normalized = tuple(_validated_options(options) for options in options_by_coordinate)
    if not normalized:
        raise ValueError("at least one allocation coordinate is required")

    # With a churn constraint, change count is part of the state; otherwise a
    # single best state per budget is sufficient.
    states: dict[tuple[int, int], _Partial] = {
        (0, 0): _Partial((), 0.0, 0, 0, 0)
    }
    for coordinate, options in enumerate(normalized):
        next_states: dict[tuple[int, int], _Partial] = {}
        remaining = len(normalized) - coordinate - 1
        minimum_remaining = sum(
            min(option.rank for option in normalized[index])
            for index in range(coordinate + 1, len(normalized))
        )
        maximum_remaining = sum(
            max(option.rank for option in normalized[index])
            for index in range(coordinate + 1, len(normalized))
        )
        for (used_rank, _), partial in states.items():
            for option in options:
                rank = used_rank + option.rank
                if (
                    rank + minimum_remaining > total_rank_budget
                    or rank + maximum_remaining < total_rank_budget
                ):
                    continue
                changed = partial.changes + (0 if option.is_anchor else 1)
                if (
                    max_changed_coordinates is not None
                    and changed > max_changed_coordinates
                ):
                    continue
                candidate = _Partial(
                    options=(*partial.options, option),
                    cost=partial.cost + option.scalar_cost,
                    changes=changed,
                    absolute_rank_deviation=(
                        partial.absolute_rank_deviation
                        + abs(option.rank - anchor_rank)
                    ),
                    score_bestkron_choices=(
                        partial.score_bestkron_choices
                        + int(option.source_family == "score_bestkron")
                    ),
                )
                state_changes = (
                    changed if max_changed_coordinates is not None else 0
                )
                key = (rank, state_changes)
                incumbent = next_states.get(key)
                if incumbent is None or _partial_is_better(
                    candidate,
                    incumbent,
                    cost_tolerance=cost_tolerance,
                ):
                    next_states[key] = candidate
        states = next_states
        if not states:
            raise ValueError(
                f"exact rank budget became infeasible after coordinate {coordinate}; "
                f"remaining={remaining}"
            )

    finals = [
        partial
        for (rank, _), partial in states.items()
        if rank == total_rank_budget
    ]
    if not finals:
        raise ValueError(f"exact rank budget {total_rank_budget} is infeasible")
    best = finals[0]
    for candidate in finals[1:]:
        if _partial_is_better(
            candidate,
            best,
            cost_tolerance=cost_tolerance,
        ):
            best = candidate
    return MetricRankAllocation(
        selected_options=best.options,
        total_rank=total_rank_budget,
        total_cost=best.cost,
        changed_coordinates=best.changes,
        total_absolute_rank_deviation=best.absolute_rank_deviation,
        tie_key=best.tie_key,
    )


def _validated_options(
    options: Sequence[MetricRankOption],
) -> tuple[MetricRankOption, ...]:
    checked = tuple(options)
    if not checked:
        raise ValueError("every coordinate must expose at least one option")
    identifiers = [option.option_id for option in checked]
    if any(not identifier for identifier in identifiers):
        raise ValueError("option_id must be non-empty")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("option_id must be unique within a coordinate")
    for option in checked:
        if not option.source_family:
            raise ValueError(f"option {option.option_id!r} lacks a source family")
        if option.rank <= 0:
            raise ValueError(f"option {option.option_id!r} has a nonpositive rank")
        if not math.isfinite(option.scalar_cost):
            raise ValueError(f"option {option.option_id!r} has a non-finite cost")
    return tuple(sorted(checked, key=lambda option: option.option_id))


def _partial_is_better(
    candidate: _Partial,
    incumbent: _Partial,
    *,
    cost_tolerance: float,
) -> bool:
    if candidate.cost < incumbent.cost - cost_tolerance:
        return True
    if incumbent.cost < candidate.cost - cost_tolerance:
        return False
    return candidate.tie_key < incumbent.tie_key


def _finite_values(values: Sequence[float]) -> list[float]:
    checked = [float(value) for value in values]
    if not checked:
        raise ValueError("at least one paired window is required")
    if not all(math.isfinite(value) for value in checked):
        raise ValueError("paired window values must be finite")
    return checked
