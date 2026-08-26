from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from evaluation import build_llama31_8b_palu_m_checkpoint as builder
from evaluation.build_llama31_8b_palu_m_checkpoint import (
    HEAD_DIM,
    NUM_KV_HEADS,
    RANK_PER_KV_HEAD,
    _validate_config,
    factorize_v_projection,
    sequential_cpu_offload_whitening_cholesky,
)


def test_llama31_geometry_is_eight_physical_kv_heads() -> None:
    config = SimpleNamespace(
        model_type="llama",
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=None,
    )
    _validate_config(config)
    assert NUM_KV_HEADS * HEAD_DIM == 1024
    assert NUM_KV_HEADS * RANK_PER_KV_HEAD == 768


def test_wrong_mha_geometry_is_rejected() -> None:
    config = SimpleNamespace(
        model_type="llama",
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
    )
    with pytest.raises(ValueError, match="GQA geometry"):
        _validate_config(config)


def test_full_rank_grouped_whitened_factorization_is_exact() -> None:
    torch.manual_seed(31)
    groups = 2
    group_dim = 4
    input_dim = 7
    weight = torch.randn(groups * group_dim, input_dim, dtype=torch.float64)
    raw = torch.randn(input_dim, input_dim, dtype=torch.float64)
    covariance = raw @ raw.T + 0.5 * torch.eye(input_dim, dtype=torch.float64)
    cholesky = torch.linalg.cholesky(covariance)

    writer, decoder, diagnostics = factorize_v_projection(
        weight,
        cholesky,
        ranks=[group_dim] * groups,
        output_dtype=torch.float64,
    )

    reconstructed = torch.cat(
        [
            decoder[group] @ writer[group * group_dim : (group + 1) * group_dim]
            for group in range(groups)
        ]
    )
    torch.testing.assert_close(reconstructed, weight, rtol=1e-10, atol=1e-10)
    assert diagnostics["relative_frobenius_error"] < 1e-10
    assert diagnostics["relative_activation_weighted_error"] < 1e-10


def test_factor_shapes_preserve_group_boundaries() -> None:
    torch.manual_seed(37)
    weight = torch.randn(12, 9, dtype=torch.float32)
    writer, decoder, diagnostics = factorize_v_projection(
        weight,
        torch.eye(9),
        ranks=[2, 2, 2],
        output_dtype=torch.float32,
    )
    assert writer.shape == (6, 9)
    assert decoder.shape == (3, 4, 2)
    assert 0.0 < diagnostics["relative_frobenius_error"] < 1.0
    assert diagnostics["relative_activation_weighted_error"] == pytest.approx(
        diagnostics["relative_frobenius_error"], rel=1e-6
    )


def test_build_cli_accepts_uniform_rank_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_llama31_8b_palu_m_checkpoint.py",
            "build",
            "--model",
            "/model",
            "--output-dir",
            "/output",
            "--whitening-dir",
            "/whitening",
            "--rank-per-kv-head",
            "64",
        ],
    )
    args = builder.parse_args()
    assert args.rank_per_kv_head == 64


def test_qwen3_profile_uses_the_matched_gqa_geometry() -> None:
    try:
        builder.activate_model_profile("qwen3_8b")
        config = SimpleNamespace(
            model_type="qwen3",
            num_hidden_layers=36,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        )
        builder._validate_config(config)
        assert builder.MODEL_REVISION == "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
        assert builder.CHECKPOINT_FORMAT == "basisserve.qwen3_8b.palu_m_v_only.v1"
    finally:
        builder.activate_model_profile("llama31_8b")


@pytest.mark.parametrize(
    ("profile", "model_type", "layers", "hidden_size", "query_heads"),
    [
        ("qwen3_32b", "qwen3", 64, 5120, 64),
        ("llama31_70b", "llama", 80, 8192, 64),
    ],
)
def test_large_model_profiles_match_downloaded_geometry(
    profile: str,
    model_type: str,
    layers: int,
    hidden_size: int,
    query_heads: int,
) -> None:
    try:
        builder.activate_model_profile(profile)
        builder._validate_config(
            SimpleNamespace(
                model_type=model_type,
                num_hidden_layers=layers,
                hidden_size=hidden_size,
                num_attention_heads=query_heads,
                num_key_value_heads=8,
                head_dim=128,
            )
        )
        assert builder.NUM_KV_HEADS == 8
        assert builder.RANK_PER_KV_HEAD == 96
    finally:
        builder.activate_model_profile("llama31_8b")


class _ToyAttention(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.k_proj = torch.nn.Linear(
            hidden_size, hidden_size, bias=False, dtype=torch.float64
        )


class _ToyLayer(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = _ToyAttention(hidden_size)

    def forward(self, hidden_states: torch.Tensor, **_: object) -> torch.Tensor:
        return torch.tanh(self.self_attn.k_proj(hidden_states))


class _ToyBackbone(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(17, hidden_size, dtype=torch.float64)
        self.layers = torch.nn.ModuleList([_ToyLayer(hidden_size), _ToyLayer(hidden_size)])


class _ToyModel(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.model = _ToyBackbone(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> torch.Tensor:
        del use_cache
        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


def test_sequential_cpu_offload_whitening_runs_layer_by_layer() -> None:
    torch.manual_seed(47)
    model = _ToyModel(hidden_size=4)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    windows = [
        {
            "input_ids": row.unsqueeze(0),
            "attention_mask": torch.ones_like(row).unsqueeze(0),
        }
        for row in input_ids
    ]
    factors = sequential_cpu_offload_whitening_cholesky(
        model,
        windows,
        seqlen=3,
        device=torch.device("cpu"),
    )
    assert len(factors) == 2
    for factor in factors:
        assert factor.shape == (4, 4)
        assert factor.dtype == torch.float32
        assert torch.isfinite(factor).all()
        assert torch.all(factor.diagonal() > 0)
