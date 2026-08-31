from __future__ import annotations

import torch
from torch import nn

from basisserve.core.qwen35_full_attention_private_ag_runtime import (
    Qwen35FullAttentionPrivateAGRuntime,
)


def test_full_attention_private_ag_runtime_installs_and_restores() -> None:
    class DummyAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.o_proj = nn.Linear(4, 3, bias=False)

    class DummyLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = DummyAttention()

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
    attention = model.model.language_model.layers[0].self_attn
    original = attention.o_proj
    factors = {
        "format": "basisserve.qwen35.full_attention_private_ag_joint_factors.v2",
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
    runtime = Qwen35FullAttentionPrivateAGRuntime(model, factors)
    with runtime:
        assert attention.o_proj is not original
        assert len(runtime.records) == 1
        torch.testing.assert_close(attention.o_proj(inputs), expected)
    assert attention.o_proj is original
    assert not hasattr(attention, "_basisserve_full_private_ag")


def test_full_attention_runtime_rejects_non_attention_layer() -> None:
    class DummyLayer(nn.Module):
        pass

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

    factors = {
        "format": "basisserve.qwen35.full_attention_private_ag_joint_factors.v2",
        "schema_version": 1,
        "layers": [
            {
                "layer_index": 0,
                "private_encoders": torch.eye(2).expand(2, -1, -1).clone(),
                "joint_decoder_weight": torch.randn(3, 4),
            }
        ],
    }
    runtime = Qwen35FullAttentionPrivateAGRuntime(DummyModel(), factors)
    try:
        runtime.install()
    except ValueError as error:
        assert "not full attention" in str(error)
    else:
        raise AssertionError("missing full attention should have been rejected")
