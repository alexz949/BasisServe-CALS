from __future__ import annotations

import torch
from torch import nn

from basisserve.core.qwen35_fp8_private_ag import (
    FP8_E4M3_MAX,
    Qwen35FP8PrivateAGOutput,
    Qwen35FP8PrivateAGRuntime,
)


def _identity_factors() -> tuple[torch.Tensor, torch.Tensor]:
    encoders = torch.eye(2).expand(2, -1, -1).clone()
    decoder = torch.eye(4)
    return encoders, decoder


def test_passthrough_matches_explicit_private_codes() -> None:
    encoders, decoder = _identity_factors()
    module = Qwen35FP8PrivateAGOutput(
        encoders,
        decoder,
        mode="passthrough",
    )
    hidden = torch.randn(3, 5, 4)
    torch.testing.assert_close(module(hidden), hidden)


def test_observe_produces_one_static_scale_per_source() -> None:
    encoders, decoder = _identity_factors()
    module = Qwen35FP8PrivateAGOutput(encoders, decoder, mode="observe")
    hidden = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]])
    torch.testing.assert_close(module(hidden), hidden)
    torch.testing.assert_close(module.source_amax, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(
        module.calibrated_scale(),
        torch.tensor([2.0, 4.0]) / FP8_E4M3_MAX,
    )


def test_static_fp8_quantization_tracks_clipping() -> None:
    encoders, decoder = _identity_factors()
    scales = torch.tensor([1.0, 1.0]) / FP8_E4M3_MAX
    module = Qwen35FP8PrivateAGOutput(
        encoders,
        decoder,
        mode="quantize",
        source_scales=scales,
    )
    hidden = torch.tensor([[[0.5, -1.0, 2.0, -0.25]]])
    output = module(hidden)
    assert torch.isfinite(output).all()
    assert int(module.wire_elements) == 4
    assert int(module.clipped_elements) == 1
    assert output[0, 0, 2] == 1.0


def test_runtime_installs_both_block_types_and_restores() -> None:
    class DummyGDN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out_proj = nn.Linear(4, 3, bias=False)

    class DummyAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.o_proj = nn.Linear(4, 3, bias=False)

    class DummyLayer(nn.Module):
        def __init__(self, block_type: str) -> None:
            super().__init__()
            if block_type == "linear_attention":
                self.linear_attn = DummyGDN()
            else:
                self.self_attn = DummyAttention()

    class DummyLanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [DummyLayer("linear_attention"), DummyLayer("full_attention")]
            )

    class DummyInner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = DummyLanguageModel()

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = DummyInner()

    model = DummyModel().eval()
    layers = model.model.language_model.layers
    originals = (layers[0].linear_attn.out_proj, layers[1].self_attn.o_proj)
    encoders, _ = _identity_factors()
    gdn_factors = {
        "layers": [
            {
                "layer_index": 0,
                "private_encoders": encoders,
                "joint_decoder_weight": originals[0].weight.detach().clone(),
            }
        ]
    }
    full_factors = {
        "layers": [
            {
                "layer_index": 1,
                "private_encoders": encoders,
                "joint_decoder_weight": originals[1].weight.detach().clone(),
            }
        ]
    }
    runtime = Qwen35FP8PrivateAGRuntime(
        model,
        gdn_factors,
        full_factors,
        mode="passthrough",
    )
    hidden = torch.randn(2, 4)
    expected = (originals[0](hidden), originals[1](hidden))
    with runtime:
        assert len(runtime.records) == 2
        torch.testing.assert_close(layers[0].linear_attn.out_proj(hidden), expected[0])
        torch.testing.assert_close(layers[1].self_attn.o_proj(hidden), expected[1])
    assert layers[0].linear_attn.out_proj is originals[0]
    assert layers[1].self_attn.o_proj is originals[1]
