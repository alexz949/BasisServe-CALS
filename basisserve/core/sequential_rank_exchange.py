"""Pure helpers for sequential exact-budget rank-pair exchange.

The model-scale scripts keep proposal construction and full-model evaluation
separate.  This module contains the deterministic pieces shared by both
stages so they can be tested without loading a language model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True, order=True)
class RankMarginal:
    """One current-state-specific rank birth or death marginal."""

    cost: float
    layer_index: int
    group_index: int
    delta_rank: int
    source_rank: int
    target_rank: int

    def as_move(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "group_index": self.group_index,
            "delta_rank": self.delta_rank,
            "source_rank": self.source_rank,
            "target_rank": self.target_rank,
            "predicted_normalized_validation_delta": self.cost,
        }


def collect_rank_marginals(
    layer_records: Iterable[Mapping[str, Any]],
) -> tuple[list[RankMarginal], list[RankMarginal]]:
    """Collect sorted death and birth marginals from closure profiles."""

    deaths: list[RankMarginal] = []
    births: list[RankMarginal] = []
    seen: set[tuple[int, int]] = set()
    for layer_record in layer_records:
        layer = int(layer_record["layer_index"])
        for group_record in layer_record["branches"]:
            group = int(group_record["group_index"])
            key = (layer, group)
            if key in seen:
                raise ValueError(f"duplicate rank marginal group {key}")
            seen.add(key)
            source_rank = int(group_record["source_rank"])
            branches = group_record["branches"]
            for name, delta, destination in (
                ("death", -16, deaths),
                ("birth", +16, births),
            ):
                branch = branches.get(name)
                if branch is None:
                    continue
                target_ranks = tuple(int(item) for item in branch["group_ranks"])
                target_rank = target_ranks[group]
                if target_rank - source_rank != delta:
                    raise ValueError(
                        f"{name} marginal for {key} has inconsistent rank delta"
                    )
                cost = float(
                    branch[
                        "normalized_validation_delta_vs_same_rank_control"
                    ]
                )
                if not math.isfinite(cost):
                    raise FloatingPointError(f"{name} marginal for {key} is non-finite")
                destination.append(
                    RankMarginal(
                        cost=cost,
                        layer_index=layer,
                        group_index=group,
                        delta_rank=delta,
                        source_rank=source_rank,
                        target_rank=target_rank,
                    )
                )
    deaths.sort()
    births.sort()
    return deaths, births


def form_exact_budget_pairs(
    deaths: Sequence[RankMarginal],
    births: Sequence[RankMarginal],
    *,
    top_deaths: int,
    top_births: int,
) -> list[dict[str, Any]]:
    """Form the deterministic Cartesian pool of exact-budget rank pairs."""

    if top_deaths <= 0 or top_births <= 0:
        raise ValueError("top marginal counts must be positive")
    pool: list[dict[str, Any]] = []
    for death in deaths[:top_deaths]:
        for birth in births[:top_births]:
            donor = (death.layer_index, death.group_index)
            recipient = (birth.layer_index, birth.group_index)
            if donor == recipient:
                continue
            if death.delta_rank + birth.delta_rank != 0:
                raise ValueError("rank-pair candidates must preserve the exact budget")
            moves = sorted(
                (death.as_move(), birth.as_move()),
                key=lambda item: (
                    int(item["layer_index"]),
                    int(item["group_index"]),
                ),
            )
            pool.append(
                {
                    "predicted_normalized_validation_delta": (
                        death.cost + birth.cost
                    ),
                    "moves": moves,
                    "layers_changed": sorted(
                        {death.layer_index, birth.layer_index}
                    ),
                    "same_layer": death.layer_index == birth.layer_index,
                    "rank_delta": 0,
                }
            )
    pool.sort(
        key=lambda item: (
            float(item["predicted_normalized_validation_delta"]),
            tuple(
                (
                    int(move["layer_index"]),
                    int(move["group_index"]),
                    int(move["delta_rank"]),
                )
                for move in item["moves"]
            ),
        )
    )
    for index, item in enumerate(pool):
        item["proposal_id"] = f"pair_{index:03d}"
    return pool


def paired_risk_summary(
    differences: Sequence[float],
    *,
    kappa: float = 1.0,
    cvar_level: float = 0.9,
) -> dict[str, Any]:
    """Summarize paired candidate-minus-source differences."""

    values = [float(item) for item in differences]
    if not values or not all(math.isfinite(item) for item in values):
        raise ValueError("paired differences must be finite and non-empty")
    if kappa < 0 or not 0 < cvar_level < 1:
        raise ValueError("invalid UCB/CVaR settings")
    mean = statistics.fmean(values)
    standard_error = (
        statistics.stdev(values) / math.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )
    tail_count = max(1, math.ceil((1.0 - cvar_level) * len(values)))
    adverse = sorted(values, reverse=True)[:tail_count]
    return {
        "windows": len(values),
        "mean_difference": mean,
        "paired_standard_error": standard_error,
        "upper_confidence_bound": mean + kappa * standard_error,
        "adverse_cvar_level": cvar_level,
        "adverse_cvar_tail_windows": tail_count,
        "adverse_cvar_mean": statistics.fmean(adverse),
        "minimum_difference": min(values),
        "maximum_difference": max(values),
        "improved_windows": sum(item < 0 for item in values),
        "worsened_windows": sum(item > 0 for item in values),
        "window_differences": values,
    }


def select_domain_robust_pair(
    candidate_domains: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    kappa: float = 1.0,
    cvar_level: float = 0.9,
) -> dict[str, Any]:
    """Apply the sequential acceptance rule to a frozen candidate pool.

    CVaR is deliberately diagnostic in the first protocol.  Feasibility uses
    per-domain mean + ``kappa`` paired SE, and the winner minimizes the worst
    domain UCB.  If no candidate is feasible, the current state is retained.
    """

    if not candidate_domains:
        raise ValueError("candidate pool is empty")
    summaries: dict[str, dict[str, Any]] = {}
    expected_domains: set[str] | None = None
    for candidate, domains in candidate_domains.items():
        names = set(domains)
        if not names:
            raise ValueError(f"candidate {candidate!r} has no domains")
        if expected_domains is None:
            expected_domains = names
        elif names != expected_domains:
            raise ValueError("candidate domain sets differ")
        domain_summaries = {
            domain: paired_risk_summary(
                differences,
                kappa=kappa,
                cvar_level=cvar_level,
            )
            for domain, differences in domains.items()
        }
        worst_ucb = max(
            item["upper_confidence_bound"] for item in domain_summaries.values()
        )
        summaries[candidate] = {
            "domains": domain_summaries,
            "worst_domain_upper_confidence_bound": worst_ucb,
            "equal_domain_mean_difference": statistics.fmean(
                item["mean_difference"] for item in domain_summaries.values()
            ),
            "passes_every_domain_ucb_guard": all(
                item["upper_confidence_bound"] <= 0
                for item in domain_summaries.values()
            ),
            "passes_every_domain_cvar_diagnostic": all(
                item["adverse_cvar_mean"] <= 0
                for item in domain_summaries.values()
            ),
        }
    feasible = [
        name
        for name, item in summaries.items()
        if item["passes_every_domain_ucb_guard"]
    ]
    selected = (
        None
        if not feasible
        else min(
            feasible,
            key=lambda name: (
                summaries[name]["worst_domain_upper_confidence_bound"],
                summaries[name]["equal_domain_mean_difference"],
                name,
            ),
        )
    )
    return {
        "candidate_summaries": summaries,
        "feasible_candidates": feasible,
        "selected_candidate": selected,
        "decision": "stop_no_robust_pair" if selected is None else "accept_pair",
        "cvar_role": "diagnostic_only",
    }
