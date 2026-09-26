"""Isolate latent wire quantization from decoder weight quantization."""

import torch
from torch.nn import functional as F

from basisserve.core.qwen3_kv4_fp8_quality import QualityLinear
from basisserve.kernels.fp8_wire import FP8_E4M3_MAX, quantize_e4m3_static


class LatentDecoder(QualityLinear):
    """Same static latent scale for BF16-weight A8 and actual W8A8 GEMMs."""

    def __init__(self, weight):
        super().__init__(weight)
        self.mode = "bf16"

    def forward(self, value):
        assert self.mode in ("bf16", "a8", "w8a8")
        self.fp8 = self.mode == "w8a8"
        if self.mode != "a8":
            return super().forward(value)
        assert self.calibrated and not self.observe and not torch.is_grad_enabled()
        flat = value.reshape(-1, value.shape[-1]).contiguous()
        self.clipped.add_((flat.float().abs() > self.input_scale * FP8_E4M3_MAX).sum())
        self.elements += flat.numel()
        self.calls += 1
        codes = quantize_e4m3_static(flat, self.input_scale)
        restored = (codes.float() * self.input_scale).to(torch.bfloat16)
        result = F.linear(restored, self.weight)
        return result.reshape(*value.shape[:-1], self.weight.shape[0])


def install_latent_decoders(model, projections, scales):
    decoders = {}
    for i, layer in enumerate(model.model.layers):
        name = f"{i}.decoder"
        assert not projections[f"{i}.encoder"].fp8
        old = projections[name]
        decoder = LatentDecoder(old.weight)
        selected = scales[name]
        assert selected["input_scale"] > 0
        decoder.input_scale.fill_(selected["input_scale"])
        decoder.observed_amax.fill_(selected["observed_amax"])
        decoder.calibrated = True
        assert float(decoder.weight_scale) == selected["weight_scale"]
        assert torch.equal(decoder.weight, old.weight)
        layer.self_attn.o_proj = decoder
        projections[name] = decoder
        decoders[name] = decoder
    return decoders
