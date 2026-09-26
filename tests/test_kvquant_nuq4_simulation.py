"""KVQuant NUQ4 simulation wrappers (evaluation/kvquant_nuq4_simulation.py) against the upstream reference on synthetic data:
outliers are kept exactly, in-range values land on the 16 signposts, and the chunked path equals the whole-tensor call."""
import pytest
import torch

from evaluation import kvquant_nuq4_simulation as S

pytestmark = pytest.mark.skipif(not S.UPSTREAM.exists(), reason='external/KVQuant not cloned')


def synthetic_quantizer(data, qchannel):
    t = 1 - (1 - S.SPARSITY_THRESHOLD) / 2
    hi = torch.quantile(data, t, dim=qchannel)
    lo = torch.quantile(data, 1 - t, dim=qchannel)
    lut = [torch.linspace(-1, 1, 16).double().numpy().reshape(-1, 1)]
    return hi, lo, lut


def test_key_roundtrip_matches_upstream_and_keeps_outliers():
    torch.manual_seed(0)
    upstream = S.load_upstream()
    pre = (torch.randn(1, 8, 300, 128) * torch.linspace(0.5, 3, 128)).bfloat16()
    flat = pre.permute(0, 2, 1, 3).reshape(300, 1024).float()
    quantizer = synthetic_quantizer(flat, 0)
    out = S.nuq4_key_roundtrip(pre, quantizer, upstream)
    assert out.shape == pre.shape and out.dtype == pre.dtype
    hi, lo, lut = quantizer
    mask = upstream.get_outliers(flat, channel=0, outlier_threshold_upper=hi, outlier_threshold_lower=lo)
    ref = upstream.quant_fn_nuq_recon(flat, bits=4, qchannel=0, dynamicquantization=False, include_sparse=True, outlier_mask=mask, maxval=hi, minval=lo, lut=lut, first_few_fp16=-1)
    torch.testing.assert_close(out.permute(0, 2, 1, 3).reshape(300, 1024).float(), ref.bfloat16().float())
    assert torch.equal(out.permute(0, 2, 1, 3).reshape(300, 1024)[mask], pre.permute(0, 2, 1, 3).reshape(300, 1024)[mask])   # outliers exact
    S.CHUNK, chunk = 64, S.CHUNK
    chunked = S.nuq4_key_roundtrip(pre, quantizer, upstream)
    S.CHUNK = chunk
    assert torch.equal(chunked, out)


def test_value_roundtrip_dynamic_per_token():
    torch.manual_seed(1)
    upstream = S.load_upstream()
    value = (torch.randn(1, 8, 100, 96) * 2).bfloat16()
    flat = value.permute(0, 2, 1, 3).reshape(100, 768).float()
    quantizer = synthetic_quantizer(flat, -1)
    out = S.nuq4_value_roundtrip(value, quantizer, upstream)
    hi, lo, lut = quantizer
    mask = upstream.get_outliers_dynamic(flat, channel=-1, thresh=S.SPARSITY_THRESHOLD)
    ref = upstream.quant_fn_nuq_recon(flat, bits=4, qchannel=-1, dynamicquantization=True, include_sparse=True, outlier_mask=mask, maxval=hi, minval=lo, lut=lut, first_few_fp16=-1)
    torch.testing.assert_close(out.permute(0, 2, 1, 3).reshape(100, 768).float(), ref.bfloat16().float())
    err = (out.float() - value.float()).abs().mean() / value.float().abs().mean()
    assert 0.01 < err < 0.2
