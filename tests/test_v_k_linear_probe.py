from __future__ import annotations

import pytest
import torch

from basisserve.core.v_k_linear_probe import (
    apply_headwise_linear_probe,
    centered_relative_squared_error,
    fit_headwise_linear_probe,
)


def test_headwise_probe_recovers_affine_maps() -> None:
    generator = torch.Generator().manual_seed(20260828)
    source = torch.randn(5, 3, 37, 7, generator=generator, dtype=torch.float64)
    weight = torch.randn(3, 7, 11, generator=generator, dtype=torch.float64)
    bias = torch.randn(3, 11, generator=generator, dtype=torch.float64)
    target = torch.einsum("bhsi,hio->bhso", source, weight) + bias[None, :, None, :]

    probe = fit_headwise_linear_probe(source, target, relative_ridge=1.0e-10)
    prediction = apply_headwise_linear_probe(source, probe)

    torch.testing.assert_close(prediction.double(), target, rtol=2.0e-5, atol=2.0e-5)
    assert centered_relative_squared_error(target, prediction) < 1.0e-10


def test_headwise_probe_rejects_mismatched_geometry() -> None:
    source = torch.randn(2, 3, 5, 7)
    target = torch.randn(2, 4, 5, 9)

    with pytest.raises(ValueError, match="share batch/head/sequence"):
        fit_headwise_linear_probe(source, target)
