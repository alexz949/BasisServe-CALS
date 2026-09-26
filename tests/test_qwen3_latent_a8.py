import pytest
import torch
from torch.nn import functional as F

from basisserve.core.qwen3_latent_a8_quality import LatentDecoder
from basisserve.kernels.fp8_wire import quantize_e4m3_static


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA FP8 GEMM")
def test_a8_and_w8a8_share_scale_and_preserve_weight(monkeypatch):
    torch.manual_seed(19)
    weight = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) / 16
    module = LatentDecoder(weight)
    value = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    actual = torch._scaled_mm
    calls = []

    def observed(*args, **kwargs):
        calls.append((args[0].dtype, args[1].dtype))
        return actual(*args, **kwargs)

    monkeypatch.setattr(torch, "_scaled_mm", observed)
    with torch.inference_mode():
        reference = module(value)
        module.begin_calibration()
        module(value)
        module.end_calibration()
        scale = module.input_scale.clone()
        module.mode = "a8"
        a8 = module(value)
        codes = quantize_e4m3_static(value, scale)
        expected = F.linear((codes.float() * scale).to(torch.bfloat16), weight)
        torch.testing.assert_close(a8, expected, rtol=0, atol=0)
        assert not calls and module.calls == 1
        module.mode = "w8a8"
        w8a8 = module(value)
        assert calls == [(torch.float8_e4m3fn, torch.float8_e4m3fn)]
        assert module.calls == 2 and torch.isfinite(w8a8).all()
        assert (w8a8.float() - reference.float()).norm() / reference.float().norm() < 0.08
        module.mode = "a8"
        module(value * 100)
        assert module.clipped > 0
        torch.testing.assert_close(module.input_scale, scale, rtol=0, atol=0)
        torch.testing.assert_close(module.weight, weight, rtol=0, atol=0)
        module.mode = "bf16"
        torch.testing.assert_close(module(value), reference, rtol=0, atol=0)
