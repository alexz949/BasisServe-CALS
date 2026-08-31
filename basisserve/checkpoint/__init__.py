"""In-memory checkpoint conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from basisserve.core.tp_output import (
    FactorizationMethod,
    LowRankAllReduceOutput,
    factorize_output_weight,
    resolve_rank,
)


@dataclass(frozen=True)
class AttentionOutputReplacementRecord:
    """Description of one attention output projection replaced in memory."""

    module_name: str
    projection_name: str
    in_features: int
    out_features: int
    rank: int
    relative_frobenius_error: float


def _is_attention_like(module: nn.Module) -> bool:
    return all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj"))


def replace_attention_output_projections(
    model: nn.Module,
    *,
    rank: int | None = None,
    rank_ratio: float | None = None,
    multiple: int = 1,
    method: FactorizationMethod = "svd",
) -> list[AttentionOutputReplacementRecord]:
    """Replace dense attention ``o_proj`` modules with low-rank runtime modules.

    This is an offline/in-memory conversion: factorization is completed before
    the replacement module is installed, and no decomposition runs in forward.
    Only attention-like modules exposing Q/K/V/O projection attributes are
    considered. Existing non-``nn.Linear`` output projections are left intact.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    records: list[AttentionOutputReplacementRecord] = []
    for module_name, module in tuple(model.named_modules()):
        if not _is_attention_like(module):
            continue
        projection = module.o_proj
        if not isinstance(projection, nn.Linear):
            continue
        selected_rank = resolve_rank(
            projection.out_features,
            projection.in_features,
            rank=rank,
            rank_ratio=rank_ratio,
            multiple=multiple,
        )
        factors = factorize_output_weight(
            projection.weight,
            selected_rank,
            method=method,
        )
        replacement = LowRankAllReduceOutput(
            factors.input_factor,
            factors.output_basis,
            bias=projection.bias,
        )
        replacement.train(projection.training)
        module.o_proj = replacement
        projection_name = f"{module_name}.o_proj" if module_name else "o_proj"
        records.append(
            AttentionOutputReplacementRecord(
                module_name=module_name,
                projection_name=projection_name,
                in_features=int(projection.in_features),
                out_features=int(projection.out_features),
                rank=factors.rank,
                relative_frobenius_error=factors.relative_frobenius_error,
            )
        )
    return records


__all__ = [
    "AttentionOutputReplacementRecord",
    "replace_attention_output_projections",
]
