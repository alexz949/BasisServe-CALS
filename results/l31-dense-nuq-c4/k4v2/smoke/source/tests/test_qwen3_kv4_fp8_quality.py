import torch
import pytest

from basisserve.core.qwen3_kv4_fp8_quality import QualityLinear, nuq4_output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FP8 GEMM required")
def test_real_fp8_gemm_and_frozen_scale(monkeypatch):
    torch.manual_seed(7)
    weight = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) / 16
    module = QualityLinear(weight)
    value = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        reference = module(value)
        module.begin_calibration()
        module(value)
        module.end_calibration()
        scale = module.input_scale.clone()
        actual = torch._scaled_mm
        calls = []

        def observed(*args, **kwargs):
            calls.append((args[0].dtype, args[1].dtype, kwargs))
            return actual(*args, **kwargs)

        monkeypatch.setattr(torch, "_scaled_mm", observed)
        module.fp8 = True
        result = module(value)
        assert result.dtype == torch.bfloat16 and result.shape == reference.shape
        assert (result.float() - reference.float()).norm() / reference.float().norm() < 0.08
        assert len(calls) == 1 and calls[0][:2] == (torch.float8_e4m3fn, torch.float8_e4m3fn)
        assert calls[0][2]["use_fast_accum"] is False
        module(value * 100)
        torch.testing.assert_close(module.input_scale, scale, rtol=0, atol=0)
        assert module.clipped > 0 and module.calls == 2


@pytest.mark.parametrize("bits", [4, 2])
def test_nuq4_active_coordinates_preserve_padding(bits):
    class Upstream:
        @staticmethod
        def get_outliers_dynamic(data, **kwargs):
            assert kwargs == dict(channel=-1, thresh=0.99)
            return torch.zeros_like(data, dtype=torch.bool)

        @staticmethod
        def quant_fn_nuq_recon(data, **kwargs):
            assert kwargs["bits"] == bits and kwargs["dynamicquantization"]
            assert kwargs["first_few_fp16"] == -1
            return data + 1

    output = torch.zeros(1, 2, 1024, dtype=torch.bfloat16)
    active = torch.tensor([0, 1, 128, 129])
    quantizers = {"0.v": (torch.ones(4), -torch.ones(4), [torch.zeros(2 ** bits, 1).numpy()])}
    result = nuq4_output(Upstream, quantizers, {"0.v": active}, "0.v", output)
    assert result.shape == output.shape and result.dtype == output.dtype
    assert result.sum() == 8 and output.sum() == 0
    assert (result[..., active] == 1).all()
