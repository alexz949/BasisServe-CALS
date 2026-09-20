import torch
import pytest

from basisserve.core.qwen3_8b_vllm_c1 import fold_value_weight


@pytest.mark.parametrize("hidden_size", [4096, 5120])
def test_folded_value_projection_matches_explicit_encoder(hidden_size):
    generator = torch.Generator().manual_seed(102)
    dense = torch.randn(128, hidden_size, generator=generator) / 64
    encoder = torch.randn(128, 64, generator=generator) / 128**0.5
    hidden = torch.randn(3, hidden_size, generator=generator)
    observed = hidden @ fold_value_weight(dense, encoder).T
    expected = (hidden @ dense.T) @ encoder
    torch.testing.assert_close(observed, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("local_heads", [4, 8])
def test_tp8_coordinate_order_matches_sum_of_physical_sources(local_heads):
    generator = torch.Generator().manual_seed(123)
    coordinates = torch.randn(3, 8, local_heads, 64, generator=generator)
    decoders = torch.randn(8 * local_heads, 64, 17, generator=generator)
    arena = torch.cat([coordinates[:, rank].reshape(3, local_heads * 64).T.contiguous() for rank in range(8)])
    observed = arena.T @ decoders.reshape(8 * local_heads * 64, 17)
    expected = torch.einsum("bshr,shrn->bn", coordinates, decoders.reshape(8, local_heads, 64, 17))
    torch.testing.assert_close(observed, expected, rtol=2e-4, atol=5e-5)
