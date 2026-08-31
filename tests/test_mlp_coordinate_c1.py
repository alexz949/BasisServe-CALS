from __future__ import annotations

import pytest
import torch
from torch import nn

from basisserve.core.mlp_coordinate_c1 import (
    CoordinateMLPDecoderRuntime,
    fit_mlp_coordinate_decoder,
)


class _MLP(nn.Module):
    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(input_width, output_width, bias=False)


class _Layer(nn.Module):
    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.mlp = _MLP(input_width, output_width)


class _Backbone(nn.Module):
    def __init__(self, layers: int, input_width: int, output_width: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_Layer(input_width, output_width) for _ in range(layers)]
        )


class _Model(nn.Module):
    def __init__(self, layers: int, input_width: int, output_width: int) -> None:
        super().__init__()
        self.model = _Backbone(layers, input_width, output_width)


def test_coordinate_decoder_fit_matches_direct_least_squares() -> None:
    generator = torch.Generator().manual_seed(20260826)
    inputs = torch.randn(1024, 7, generator=generator, dtype=torch.float64)
    teacher_weight = torch.randn(5, 7, generator=generator, dtype=torch.float64)
    targets = inputs @ teacher_weight.T
    indices = torch.tensor([0, 2, 3, 6])
    selected = inputs[:, indices]
    rows = len(inputs)
    fitted = fit_mlp_coordinate_decoder(
        selected.T @ selected / rows,
        targets.T @ selected / rows,
        targets.square().sum() / rows,
        input_width=inputs.shape[1],
        factor_dtype=torch.float64,
        work_dtype=torch.float64,
    )
    expected = torch.linalg.lstsq(selected, targets).solution.T
    torch.testing.assert_close(
        fitted.decoder_weight,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )
    prediction = selected @ fitted.decoder_weight.T
    direct_relative_mse = float(
        (targets - prediction).square().sum() / targets.square().sum()
    )
    assert fitted.metrics["fit_relative_output_mse"] == pytest.approx(
        direct_relative_mse,
        rel=1e-10,
        abs=1e-10,
    )
    assert fitted.metrics["relative_covariance_damping"] == 0.0


def test_coordinate_runtime_is_tp_balanced_and_restores() -> None:
    generator = torch.Generator().manual_seed(31)
    model = _Model(layers=2, input_width=8, output_width=4)
    originals = [layer.mlp.down_proj for layer in model.model.layers]
    indices = {
        0: torch.tensor([0, 2, 4, 6]),
        1: torch.tensor([1, 3, 5, 7]),
    }
    decoders = {
        layer: torch.randn(4, 4, generator=generator)
        for layer in range(2)
    }
    source = torch.randn(3, 8, generator=generator)
    with CoordinateMLPDecoderRuntime(
        model,
        indices,
        decoders,
        tp_size=2,
    ) as runtime:
        assert runtime.kept_per_source == 2
        for layer_index, layer in enumerate(model.model.layers):
            expected = source[:, indices[layer_index]] @ decoders[layer_index].T
            torch.testing.assert_close(layer.mlp.down_proj(source), expected)
    for layer_index, layer in enumerate(model.model.layers):
        assert layer.mlp.down_proj is originals[layer_index]


def test_coordinate_runtime_rejects_unbalanced_sources() -> None:
    model = _Model(layers=1, input_width=8, output_width=4)
    with pytest.raises(ValueError, match="not TP balanced"):
        CoordinateMLPDecoderRuntime(
            model,
            {0: torch.tensor([0, 1, 2, 4])},
            {0: torch.randn(4, 4)},
            tp_size=2,
        ).install()
