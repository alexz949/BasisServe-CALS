from __future__ import annotations

import torch
from torch import nn

from basisserve.core.qwen35_global_aa_svd import (
    FACTOR_FORMAT,
    Qwen35GlobalAASVDOutput,
    Qwen35GlobalAASVDRuntime,
    fit_global_activation_aware_svd,
)
from basisserve.core.tp_output import factorize_output_weight


def _positive_moment(width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn(4 * width, width, generator=generator, dtype=torch.float64)
    return rows.T @ rows / rows.shape[0]


def _relative_mse(
    weight: torch.Tensor,
    approximation: torch.Tensor,
    covariance: torch.Tensor,
) -> float:
    residual = weight - approximation
    return float(
        ((residual @ covariance) * residual).sum()
        / ((weight @ covariance) * weight).sum()
    )


def test_full_rank_global_aa_svd_reconstructs_weight() -> None:
    torch.manual_seed(2101)
    weight = torch.randn(8, 8, dtype=torch.float64)
    covariance = _positive_moment(8, 2102)

    factors = fit_global_activation_aware_svd(
        weight,
        covariance,
        covariance,
        rank=8,
        tp_size=2,
        covariance_damping=1.0e-5,
        work_dtype=torch.float64,
        factor_dtype=torch.float64,
    )

    reconstructed = factors.output_basis @ factors.input_factor.T
    torch.testing.assert_close(reconstructed, weight, rtol=1e-10, atol=1e-10)
    assert factors.metrics["heldout_relative_output_mse"] < 1e-20


def test_global_aa_svd_beats_weight_only_svd_in_activation_metric() -> None:
    torch.manual_seed(2103)
    weight = torch.randn(9, 9, dtype=torch.float64)
    covariance = torch.diag(
        torch.tensor(
            [100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01],
            dtype=torch.float64,
        )
    )
    factors = fit_global_activation_aware_svd(
        weight,
        covariance,
        covariance,
        rank=3,
        tp_size=3,
        covariance_damping=0.0,
        work_dtype=torch.float64,
        factor_dtype=torch.float64,
    )
    weight_only = factorize_output_weight(weight, rank=3, method="svd")

    aa_weight = factors.output_basis @ factors.input_factor.T
    svd_weight = weight_only.reconstruct_weight()
    assert _relative_mse(weight, aa_weight, covariance) <= _relative_mse(
        weight, svd_weight, covariance
    ) + 1e-12


def test_global_aa_svd_output_matches_explicit_tp_source_sum() -> None:
    torch.manual_seed(2104)
    input_factor = torch.randn(12, 5)
    output_basis = torch.randn(7, 5)
    hidden = torch.randn(2, 4, 12)
    module = Qwen35GlobalAASVDOutput(input_factor, output_basis, tp_size=3)

    components = module.latent_components(hidden)
    expected_latent = hidden @ input_factor
    expected_output = expected_latent @ output_basis.T
    torch.testing.assert_close(components.sum(dim=-2), expected_latent)
    torch.testing.assert_close(module(hidden), expected_output)


def test_global_aa_svd_runtime_installs_both_qwen35_block_types() -> None:
    class DummyGDN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out_proj = nn.Linear(6, 5, bias=False)

    class DummyAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.o_proj = nn.Linear(6, 5, bias=False)

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

    model = DummyModel()
    layers = model.model.language_model.layers
    originals = (layers[0].linear_attn.out_proj, layers[1].self_attn.o_proj)
    factors = {
        "format": FACTOR_FORMAT,
        "schema_version": 1,
        "tp_size": 2,
        "layers": [
            {
                "layer_index": index,
                "block_type": block_type,
                "input_factor": torch.randn(6, 3),
                "output_basis": torch.randn(5, 3),
            }
            for index, block_type in enumerate(
                ("linear_attention", "full_attention")
            )
        ],
    }
    runtime = Qwen35GlobalAASVDRuntime(model, factors)

    with runtime:
        assert len(runtime.records) == 2
        assert isinstance(layers[0].linear_attn.out_proj, Qwen35GlobalAASVDOutput)
        assert isinstance(layers[1].self_attn.o_proj, Qwen35GlobalAASVDOutput)
    assert layers[0].linear_attn.out_proj is originals[0]
    assert layers[1].self_attn.o_proj is originals[1]


def test_rank768_tp8_communication_matches_c1_local192() -> None:
    torch.manual_seed(2105)
    factors = fit_global_activation_aware_svd(
        torch.randn(8, 8),
        torch.eye(8),
        torch.eye(8),
        rank=3,
        tp_size=2,
    )
    metrics = factors.metrics
    assert metrics["compressed_allreduce_ring_elements_per_token_per_rank"] == 3
    assert metrics["dense_allreduce_ring_elements_per_token_per_rank"] == 8
    assert metrics["communication_fraction_of_dense_allreduce"] == 3 / 8

    tp_size = 8
    global_rank = 768
    c1_local_rank = 192
    allreduce = 2 * (tp_size - 1) / tp_size * global_rank
    private_allgather = (tp_size - 1) * c1_local_rank
    assert allreduce == private_allgather == 1344
