from __future__ import annotations

import torch
from torch import nn

from basisserve.core.qwen35_gdn_private_ag import (
    fit_qwen35_private_ag_joint_factors,
)
from basisserve.core.qwen35_gdn_private_ag_runtime import (
    Qwen35PrivateAGOutput,
    Qwen35PrivateAGRuntime,
)
from scripts.build_qwen35_gdn_private_ag_joint_factors import (
    _validate_moment_pair,
)


def _positive_moment(width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn(4 * width, width, generator=generator)
    return rows.transpose(0, 1) @ rows / rows.shape[0]


def test_full_source_rank_is_an_exact_endpoint() -> None:
    torch.manual_seed(2001)
    weight = torch.randn(5, 6)
    result = fit_qwen35_private_ag_joint_factors(
        weight,
        _positive_moment(6, 2003),
        _positive_moment(6, 2004),
        tp_size=2,
        local_rank=3,
        factor_dtype=torch.float32,
    )
    assert result.private_encoders.shape == (2, 3, 3)
    assert result.joint_decoder_weight.shape == (5, 6)
    assert result.metrics["fit_relative_output_mse"] == 0.0
    assert result.metrics["heldout_relative_output_mse"] == 0.0
    assert result.metrics["selected_boundary"] == "identity"


def test_decoder_closed_als_runs_encoder_and_redecoder_sweeps() -> None:
    torch.manual_seed(2005)
    result = fit_qwen35_private_ag_joint_factors(
        torch.randn(7, 8),
        _positive_moment(8, 2007),
        _positive_moment(8, 2008),
        tp_size=2,
        local_rank=2,
        encoder_sweeps=2,
        minimum_encoder_sweeps=1,
        factor_dtype=torch.float32,
    )
    metrics = result.metrics
    assert metrics["selected_boundary"] in {"decoder_only", "after_redecoder"}
    assert 0 <= metrics["selected_sweep"] <= 2
    assert metrics["diagnostics"]["encoder_sweeps_completed"] >= 1
    assert any(row["boundary"] == "after_encoder" for row in metrics["checkpoints"])
    assert any(row["boundary"] == "after_redecoder" for row in metrics["checkpoints"])
    assert all(
        row["selection_eligible"]
        == (row["boundary"] in {"decoder_only", "after_redecoder"})
        for row in metrics["checkpoints"]
    )
    assert torch.isfinite(result.private_encoders).all()
    assert torch.isfinite(result.joint_decoder_weight).all()


def test_zero_encoder_sweeps_returns_decoder_only_activation_aware_factors() -> None:
    torch.manual_seed(2006)
    result = fit_qwen35_private_ag_joint_factors(
        torch.randn(7, 8),
        _positive_moment(8, 2009),
        _positive_moment(8, 2010),
        tp_size=2,
        local_rank=2,
        encoder_sweeps=0,
        minimum_encoder_sweeps=0,
        factor_dtype=torch.float32,
    )

    assert result.metrics["selected_boundary"] == "decoder_only"
    assert result.metrics["selected_sweep"] == 0
    assert [row["boundary"] for row in result.metrics["checkpoints"]] == [
        "anchor",
        "decoder_only",
    ]


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
        "format": "basisserve.qwen35.gdn_private_ag_joint_factors.v2",
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


def test_als_moment_pair_requires_disjoint_calibration_windows() -> None:
    def payload(indices: tuple[int, ...]) -> dict[str, object]:
        return {
            "model": {"source": "qwen35", "commit_hash": "abc"},
            "geometry": {"wire_input_width": 8, "hidden_size": 8},
            "collection": {
                "records": [{"sample_index": index} for index in indices],
            },
            "layers": [{"layer_index": 0}, {"layer_index": 1}],
        }

    _validate_moment_pair(payload((0, 1)), payload((2, 3)))
    try:
        _validate_moment_pair(payload((0, 1)), payload((1, 2)))
    except ValueError as error:
        assert "reuse calibration windows" in str(error)
    else:
        raise AssertionError("overlapping ALS calibration splits were accepted")
