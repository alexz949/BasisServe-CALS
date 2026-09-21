import torch

from basisserve.core.fp8_value_latent import (
    SplitPagedE4M3Cache,
    base_payload_hadamard,
    quantize_paged_e4m3,
)


def test_base_payload_hadamard_is_block_orthogonal():
    matrix = base_payload_hadamard()
    torch.testing.assert_close(matrix.mT @ matrix, torch.eye(96, dtype=torch.float64), rtol=0, atol=1e-12)
    assert torch.count_nonzero(matrix[:16, 16:]) == 0
    assert torch.count_nonzero(matrix[16:, :16]) == 0


def test_page_head_e4m3_storage_and_roundtrip():
    torch.manual_seed(7)
    value = torch.randn(2, 3, 35, 16, dtype=torch.bfloat16)
    packed = quantize_paged_e4m3(value, page_size=32)
    assert packed.codes.shape == (2, 3, 2, 32, 16)
    assert packed.scales.shape == (2, 3, 2, 1, 1)
    assert packed.codes.dtype == torch.float8_e4m3fn
    assert packed.scales.dtype == torch.float32
    assert packed.storage_bytes == packed.codes.numel() + packed.scales.numel() * 4
    restored = packed.dequantize()
    assert restored.shape == value.shape and restored.dtype == torch.bfloat16
    relative_mse = (restored.float() - value.float()).square().sum() / value.float().square().sum()
    assert relative_mse < 0.002


def test_split_cache_appends_with_fixed_page_scale_and_gathers_selected_only():
    torch.manual_seed(11)
    first = torch.randn(1, 2, 35, 96, dtype=torch.bfloat16)
    second = torch.randn(1, 2, 3, 96, dtype=torch.bfloat16)
    cache = SplitPagedE4M3Cache()
    restored_first = cache.append(first)
    first_codes = cache.base.codes.clone()
    restored_second = cache.append(second)
    assert cache.token_count == 38
    assert restored_first.shape == first.shape and restored_second.shape == second.shape
    torch.testing.assert_close(cache.base.codes[:, :, 0], first_codes[:, :, 0], rtol=0, atol=0)
    ids = torch.tensor([[[0, 34, 37], [1, 35, 36]]])
    selected = cache.gather(ids)
    assert selected.shape == (1, 2, 3, 96) and torch.isfinite(selected).all()
    assert cache.storage_bytes == (
        cache.base.codes.numel()
        + cache.payload.codes.numel()
        + 4 * (cache.base.scales.numel() + cache.payload.scales.numel())
    )
