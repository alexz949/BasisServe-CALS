from __future__ import annotations

import torch

from basisserve.core import (
    AttentionTPLayout,
    HadamardLowRankAllReduceOutput,
    LowRankAllReduceOutput,
    PrivateAllGatherOutput,
    factorize_output_weight_with_hadamard_basis,
    factorize_output_weight,
    resolve_rank,
)


def test_full_rank_factorization_matches_dense_row_parallel_sum() -> None:
    torch.manual_seed(7)
    tokens = 5
    d_in = 12
    d_out = 10
    tp_size = 3

    weight = torch.randn(d_out, d_in, dtype=torch.float64)
    hidden = torch.randn(tokens, d_in, dtype=torch.float64)
    factors = factorize_output_weight(weight, rank=min(d_in, d_out), method="svd")

    local_width = d_in // tp_size
    latent_sum = torch.zeros(tokens, factors.rank, dtype=torch.float64)
    for tp_rank in range(tp_size):
        start = tp_rank * local_width
        hidden_i = hidden[:, start : start + local_width]
        factor_i = factors.input_factor[start : start + local_width]
        latent_sum += hidden_i @ factor_i

    low_rank_output = latent_sum @ factors.output_basis.transpose(0, 1)
    dense_output = hidden @ weight.transpose(0, 1)
    torch.testing.assert_close(low_rank_output, dense_output, rtol=1e-9, atol=1e-9)


def test_truncated_factor_reconstructs_reported_weight() -> None:
    torch.manual_seed(11)
    weight = torch.randn(9, 7)
    factors = factorize_output_weight(weight, rank=3, method="gram_eigh")

    reconstructed = factors.reconstruct_weight()
    assert reconstructed.shape == weight.shape
    expected_error = torch.linalg.vector_norm(weight - reconstructed) / torch.linalg.vector_norm(weight)
    assert abs(float(expected_error) - factors.relative_frobenius_error) < 1e-5


def test_module_world_size_one_matches_factorized_weight() -> None:
    torch.manual_seed(19)
    weight = torch.randn(8, 8)
    hidden = torch.randn(4, 8)
    factors = factorize_output_weight(weight, rank=4)
    module = LowRankAllReduceOutput(factors.input_factor, factors.output_basis)

    output, latent = module(hidden, return_reduced_latent=True)
    expected = hidden @ factors.reconstruct_weight().transpose(0, 1)
    torch.testing.assert_close(output, expected)
    assert latent.shape == (4, 4)

    estimate = module.communication_estimate(tokens=4, dtype=torch.float32)
    assert estimate.full_payload_bytes == 4 * 8 * 4
    assert estimate.low_rank_payload_bytes == 4 * 4 * 4
    assert estimate.payload_reduction == 2.0


def test_private_all_gather_world_size_one_matches_projected_weight() -> None:
    torch.manual_seed(21)
    weight = torch.randn(7, 6)
    hidden = torch.randn(5, 6)
    basis = torch.linalg.qr(torch.randn(7, 3)).Q
    input_factor = weight.transpose(0, 1) @ basis
    module = PrivateAllGatherOutput(input_factor, basis)

    output, gathered = module(hidden, return_gathered_latent=True)
    expected_weight = basis @ basis.transpose(0, 1) @ weight
    expected = hidden @ expected_weight.transpose(0, 1)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    assert gathered.shape == (5, 3)


def test_hadamard_module_matches_explicit_projected_weight() -> None:
    torch.manual_seed(23)
    weight = torch.randn(8, 8)
    hidden = torch.randn(4, 8)
    indices = torch.tensor([0, 3, 5, 6])
    factors = factorize_output_weight_with_hadamard_basis(weight, indices)
    module = HadamardLowRankAllReduceOutput(
        factors.input_factor,
        indices,
        out_features=8,
    )

    output, latent = module(hidden, return_reduced_latent=True)
    expected = hidden @ factors.reconstruct_weight().transpose(0, 1)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    assert latent.shape == (4, 4)

    estimate = module.communication_estimate(tokens=4, dtype=torch.float32)
    assert estimate.full_payload_bytes == 4 * 8 * 4
    assert estimate.low_rank_payload_bytes == 4 * 4 * 4
    assert estimate.payload_reduction == 2.0


def test_hadamard_reconstruction_modes_match_explicit_projected_weight() -> None:
    torch.manual_seed(29)
    weight = torch.randn(8, 8)
    hidden = torch.randn(4, 8)
    indices = torch.tensor([0, 2, 4, 7])
    factors = factorize_output_weight_with_hadamard_basis(weight, indices)
    fwht_module = HadamardLowRankAllReduceOutput(
        factors.input_factor,
        indices,
        out_features=8,
        reconstruction="fwht",
    )
    dense_basis_module = HadamardLowRankAllReduceOutput(
        factors.input_factor,
        indices,
        out_features=8,
        reconstruction="dense_basis",
    )
    triton_module = HadamardLowRankAllReduceOutput(
        factors.input_factor,
        indices,
        out_features=8,
        reconstruction="triton_fwht",
    )

    fwht_output = fwht_module(hidden)
    dense_basis_output = dense_basis_module(hidden)
    triton_output = triton_module(hidden)
    expected = hidden @ factors.reconstruct_weight().transpose(0, 1)
    torch.testing.assert_close(fwht_output, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dense_basis_output, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(triton_output, expected, rtol=1e-5, atol=1e-5)


def test_rank_resolution_rounds_for_tensor_core_experiment() -> None:
    assert resolve_rank(8192, 8192, rank_ratio=0.30, multiple=64) == 2496
    assert resolve_rank(512, 512, rank=511, multiple=64) == 512


def test_attention_layout_is_not_mha_specific() -> None:
    mha = AttentionTPLayout(8192, 64, 64, 8)
    gqa = AttentionTPLayout(8192, 64, 8, 8)
    mqa = AttentionTPLayout(8192, 64, 1, 8)

    assert mha.attention_type == "MHA"
    assert mha.kv_partition_mode == "sharded"
    assert gqa.attention_type == "GQA"
    assert gqa.kv_partition_mode == "sharded"
    assert mqa.attention_type == "MQA"
    assert mqa.kv_partition_mode == "replicated"

    for layout in (mha, gqa, mqa):
        assert layout.local_o_input_width == 8192 // 8
        assert layout.supports_low_rank_o_communication


def test_in_memory_o_proj_replacement_on_attention_like_module() -> None:
    from torch import nn

    from basisserve.checkpoint import replace_attention_output_projections

    class DummyAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.k_proj = nn.Linear(8, 8, bias=False)
            self.v_proj = nn.Linear(8, 8, bias=False)
            self.o_proj = nn.Linear(8, 8, bias=False)

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = DummyAttention()

    model = DummyModel()
    hidden = torch.randn(3, 8)
    dense_weight = model.self_attn.o_proj.weight.detach().clone()
    records = replace_attention_output_projections(model, rank=8)

    assert len(records) == 1
    output = model.self_attn.o_proj(hidden)
    expected = hidden @ dense_weight.transpose(0, 1)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
