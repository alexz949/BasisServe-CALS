"""Simulated E4M3 KV cache (evaluation/kv_fp8_simulation.py): round-trip error bounds, scale granularity, idempotence."""
import torch

from evaluation.kv_fp8_simulation import fp8_key_roundtrip, fp8_value_roundtrip


def test_key_roundtrip_per_token_scale():
    torch.manual_seed(0)
    key = (torch.randn(1, 8, 37, 128) * torch.rand(1, 8, 37, 1) * 10).bfloat16()
    out = fp8_key_roundtrip(key)
    assert out.shape == key.shape and out.dtype == key.dtype
    err = (out.float() - key.float()).abs()
    assert torch.all(err <= key.float().abs() / 8 + 1e-6)              # E4M3: 3 mantissa bits, relative error <= 2^-4 after rounding, plus bf16
    assert (err / key.float().abs().amax(-1, keepdim=True)).mean() < 0.02
    assert torch.equal(fp8_key_roundtrip(out), out)                     # idempotent


def test_value_roundtrip_page_scale_and_decode_pages():
    torch.manual_seed(0)
    value = (torch.randn(1, 8, 22, 96) * 3).bfloat16()
    out = fp8_value_roundtrip(value, 4)
    assert out.shape == value.shape and out.dtype == value.dtype
    err = (out.float() - value.float()).abs()
    page_max = value.float().abs().reshape(1, 8, -1, 96)
    assert torch.all(err <= value.float().abs() / 8 + 1e-6)
    # a single decode token is its own page: identical to quantizing that token alone
    token = value[:, :, 5:6]
    assert torch.equal(fp8_value_roundtrip(token, 4), fp8_value_roundtrip(token, 1))
    assert torch.equal(fp8_value_roundtrip(out, 4), out)
