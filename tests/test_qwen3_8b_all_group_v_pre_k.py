import torch

from evaluation.analyze_qwen3_8b_all_group_v_pre_k import (
    _collect_moments,
    _evaluate_maps,
    _fit_map,
)


def _cross_group_rows(
    documents: int,
    tokens: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    values = torch.randn(documents, tokens, 2, 2, generator=generator)
    keys = torch.stack((values[:, :, 1], values[:, :, 0]), dim=2)
    return torch.cat((values, keys), dim=-1)


def test_all_group_maps_recover_cross_group_key_information() -> None:
    generator = torch.Generator().manual_seed(359)
    tokens = 32
    fit_rows = _cross_group_rows(4, tokens, generator=generator)
    validation_rows = _cross_group_rows(2, tokens, generator=generator)
    value_encoder = torch.eye(2).expand(2, -1, -1).clone()
    cos = torch.ones(1, tokens, 2)
    sin = torch.zeros_like(cos)
    fit_moments = _collect_moments(
        fit_rows,
        value_encoder=value_encoder,
        cos=cos,
        sin=sin,
        sequence_length=tokens,
        device=torch.device("cpu"),
        label="fit",
    )
    validation_moments = _collect_moments(
        validation_rows,
        value_encoder=value_encoder,
        cos=cos,
        sin=sin,
        sequence_length=tokens,
        device=torch.device("cpu"),
        label="validation",
    )
    raw_map, raw_fit = _fit_map(fit_moments, representation="raw")
    c1_map, c1_fit = _fit_map(fit_moments, representation="c1")
    heldout = _evaluate_maps(
        validation_rows,
        {"raw": raw_map, "c1": c1_map},
        validation_moments,
        value_encoder=value_encoder,
        cos=cos,
        sin=sin,
        sequence_length=tokens,
        device=torch.device("cpu"),
    )

    assert raw_fit["aggregate"]["predictable_total_fraction"] > 0.999999
    assert c1_fit["aggregate"]["predictable_total_fraction"] > 0.999999
    assert heldout["raw"]["aggregate"]["key_centered_relative_mse"] < 1e-10
    assert heldout["c1"]["aggregate"]["key_centered_relative_mse"] < 1e-10
