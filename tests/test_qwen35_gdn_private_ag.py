from __future__ import annotations

import torch
from torch import nn

from basisserve.core.qwen35_gdn_private_ag import fit_gdn_private_ag_joint_factors
from basisserve.core.qwen35_gdn_private_ag_runtime import (
    Qwen35PrivateAGOutput,
    Qwen35PrivateAGRuntime,
)


def _positive_moment(width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn(4 * width, width, generator=generator)
    return rows.transpose(0, 1) @ rows / rows.shape[0]


def test_full_source_rank_is_an_exact_endpoint() -> None:
    torch.manual_seed(2001)
    weight = torch.randn(5, 6)
    result = fit_gdn_private_ag_joint_factors(
        weight,
        _positive_moment(6, 2003),
        tp_size=2,
        local_rank=3,
        decoder_relative_ridge=3.0,
        factor_dtype=torch.float32,
    )
    assert result.private_encoders.shape == (2, 3, 3)
    assert result.joint_decoder_weight.shape == (5, 6)
    assert result.metrics["independent_relative_output_mse"] < 2e-5
    assert result.metrics["joint_relative_output_mse"] < 2e-5


def test_unregularized_joint_decoder_does_not_increase_fit_objective() -> None:
    torch.manual_seed(2005)
    result = fit_gdn_private_ag_joint_factors(
        torch.randn(7, 8),
        _positive_moment(8, 2007),
        tp_size=2,
        local_rank=2,
        decoder_relative_ridge=0.0,
        factor_dtype=torch.float32,
    )
    assert (
        result.metrics["joint_relative_output_mse"]
        <= result.metrics["independent_relative_output_mse"] + 2e-5
    )


def test_private_ag_output_matches_explicit_source_codes() -> None:
    torch.manual_seed(2009)
    encoders = torch.randn(3, 4, 2)
    decoder_weight = torch.randn(5, 6)
    module = Qwen35PrivateAGOutput(encoders, decoder_weight)
    hidden = torch.randn(2, 7, 12)
    expected_codes = torch.cat(
        [
            hidden[..., source * 4 : (source + 1) * 4] @ encoders[source]
            for source in range(3)
        ],
        dim=-1,
    )
    torch.testing.assert_close(module(hidden), expected_codes @ decoder_weight.T)


def test_private_ag_runtime_installs_and_restores() -> None:
    class DummyGDN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out_proj = nn.Linear(4, 3, bias=False)

    class DummyLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_attn = DummyGDN()

    class DummyLanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([DummyLayer()])

    class DummyInner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = DummyLanguageModel()

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = DummyInner()

    model = DummyModel().eval()
    gdn = model.model.language_model.layers[0].linear_attn
    original = gdn.out_proj
    factors = {
        "format": "basisserve.qwen35.gdn_private_ag_joint_factors.v1",
        "schema_version": 1,
        "layers": [
            {
                "layer_index": 0,
                "private_encoders": torch.eye(2).expand(2, -1, -1).clone(),
                "joint_decoder_weight": original.weight.detach().clone(),
            }
        ],
    }
    inputs = torch.randn(2, 5, 4)
    expected = original(inputs)
    runtime = Qwen35PrivateAGRuntime(model, factors)
    with runtime:
        assert gdn.out_proj is not original
        assert len(runtime.records) == 1
        torch.testing.assert_close(gdn.out_proj(inputs), expected)
    assert gdn.out_proj is original
    assert not hasattr(gdn, "_basisserve_gdn_private_ag")
