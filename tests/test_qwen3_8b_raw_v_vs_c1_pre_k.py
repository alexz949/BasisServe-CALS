import torch

from evaluation.analyze_qwen3_8b_raw_v_vs_c1_pre_k import (
    _collect_raw_v_moments,
    _evaluate_unrestricted_maps,
    _fit_unrestricted_maps,
)


def _linear_rows(
    documents: int,
    tokens: int,
    groups: int,
    head_dim: int,
    *,
    generator: torch.Generator,
    matrix: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    dense_value = torch.randn(
        documents,
        tokens,
        groups,
        head_dim,
        generator=generator,
    )
    exact_pre_key = torch.einsum("dtgi,gij->dtgj", dense_value, matrix)
    exact_pre_key = exact_pre_key + bias[None, None]
    return torch.cat((dense_value, exact_pre_key), dim=-1)


def test_unrestricted_raw_v_recovers_affine_pre_key_map() -> None:
    generator = torch.Generator().manual_seed(347)
    tokens, groups, head_dim = 16, 2, 4
    matrix = torch.randn(groups, head_dim, head_dim, generator=generator)
    bias = torch.randn(groups, head_dim, generator=generator)
    fit_rows = _linear_rows(
        4,
        tokens,
        groups,
        head_dim,
        generator=generator,
        matrix=matrix,
        bias=bias,
    )
    validation_rows = _linear_rows(
        2,
        tokens,
        groups,
        head_dim,
        generator=generator,
        matrix=matrix,
        bias=bias,
    )
    cos = torch.ones(1, tokens, head_dim)
    sin = torch.zeros_like(cos)
    fit_moments = _collect_raw_v_moments(
        fit_rows,
        cos=cos,
        sin=sin,
        sequence_length=tokens,
        device=torch.device("cpu"),
        label="fit",
    )
    validation_moments = _collect_raw_v_moments(
        validation_rows,
        cos=cos,
        sin=sin,
        sequence_length=tokens,
        device=torch.device("cpu"),
        label="validation",
    )
    maps = _fit_unrestricted_maps(fit_moments)
    evaluation = _evaluate_unrestricted_maps(
        validation_rows,
        maps,
        validation_moments,
        cos=cos,
        sin=sin,
        sequence_length=tokens,
        device=torch.device("cpu"),
    )

    assert evaluation["aggregate"]["key_centered_relative_mse"] < 1e-10
    assert evaluation["aggregate"]["pre_key_cosine"] > 0.999999
