"""Single-GPU NUQ4 cache simulation and real E4M3 encoder/decoder GEMMs."""

import json
from pathlib import Path

from safetensors.torch import load_file
import torch
from torch import nn
from torch.nn import functional as F

from basisserve.kernels.fp8_wire import (
    FP8_E4M3_MAX,
    quantize_e4m3_static,
    quantize_e4m3_tensorwise_col_major,
    scaled_mm_e4m3_static,
)


class QualityLinear(nn.Module):
    """Keep one BF16 reference and its tensorwise E4M3 weight representation."""

    def __init__(self, weight):
        super().__init__()
        assert weight.ndim == 2 and weight.dtype == torch.bfloat16
        self.register_buffer("weight", weight.detach().contiguous())
        codes, scale = quantize_e4m3_tensorwise_col_major(self.weight.T)
        self.register_buffer("fp8_weight", codes)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("input_scale", torch.ones((), device=weight.device))
        self.register_buffer("observed_amax", torch.zeros((), device=weight.device))
        self.register_buffer("clipped", torch.zeros((), dtype=torch.int64, device=weight.device))
        self.fp8 = False
        self.observe = False
        self.calibrated = False
        self.calls = 0
        self.elements = 0

    def forward(self, value):
        if self.observe:
            assert not self.fp8
            self.observed_amax.copy_(torch.maximum(self.observed_amax, value.detach().float().abs().amax()))
        if not self.fp8:
            return F.linear(value, self.weight)
        assert self.calibrated and not torch.is_grad_enabled()
        flat = value.reshape(-1, value.shape[-1]).contiguous()
        self.clipped.add_((flat.float().abs() > self.input_scale * FP8_E4M3_MAX).sum())
        self.elements += flat.numel()
        self.calls += 1
        codes = quantize_e4m3_static(flat, self.input_scale)
        result = scaled_mm_e4m3_static(
            codes, self.fp8_weight, left_scale=self.input_scale,
            right_scale=self.weight_scale, out_dtype=torch.bfloat16,
        )
        return result.reshape(*value.shape[:-1], self.weight.shape[0])

    def begin_calibration(self):
        self.fp8 = False
        self.observe = True
        self.calibrated = False
        self.observed_amax.zero_()

    def end_calibration(self):
        assert torch.isfinite(self.observed_amax) and self.observed_amax > 0
        self.input_scale.copy_((self.observed_amax / FP8_E4M3_MAX).clamp_min(1e-30))
        self.observe = False
        self.calibrated = True

    def reset_stats(self):
        self.calls = self.elements = 0
        self.clipped.zero_()


@torch.no_grad()
def install_factors(model, checkpoint, target_rank):
    """Validate model/factor structure only; fold the same archived C1 factors."""
    checkpoint = Path(checkpoint).resolve()
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    allocation = json.loads((checkpoint / "result.json").read_text())
    config = model.config
    assert (config.model_type, config.hidden_size, config.num_hidden_layers,
            config.num_attention_heads, config.num_key_value_heads, config.head_dim) == (
                "qwen3", 4096, 36, 32, 8, 128)
    assert manifest["status"] == allocation["status"] == "complete"
    assert manifest["model"]["huggingface_repo"] == "Qwen/Qwen3-8B-Base"
    assert manifest["compression"]["equivalent_rank_target"] == target_rank
    schedule = allocation["selection"]["selected_schedule"]
    assert schedule == manifest["compression"]["layer_ranks"] and len(schedule) == 36
    assert sum(sum(r) for r in schedule) == 36 * 8 * target_rank
    artifacts = allocation["selected_artifacts"]
    assert set(artifacts) == set(map(str, range(36)))
    projections, quant_modules, indices = {}, {}, {}
    for i, layer in enumerate(model.model.layers):
        ranks = schedule[i]
        assert len(ranks) == 8 and all(0 < r <= 128 and r % 16 == 0 for r in ranks)
        path = (checkpoint / artifacts[str(i)]["file"]).resolve()
        assert path.is_relative_to(checkpoint)
        factors = load_file(str(path))
        assert set(factors) == {"value_coordinate_encoders", "head_output_decoders", "source_ranks"}
        assert factors["source_ranks"].tolist() == ranks
        enc, dec = factors["value_coordinate_encoders"], factors["head_output_decoders"]
        assert enc.shape == (8, 128, max(ranks)) and dec.shape == (32, max(ranks), 4096)
        assert torch.isfinite(enc).all() and torch.isfinite(dec).all()
        attn = layer.self_attn
        assert attn.v_proj.bias is None and attn.o_proj.bias is None
        assert attn.v_proj.weight.shape == (1024, 4096)
        assert attn.o_proj.weight.shape == (4096, 4096)
        device = attn.v_proj.weight.device
        dense_v = attn.v_proj.weight.detach().float()
        enc, dec = enc.to(device).float(), dec.to(device).float()
        folded_v, folded_o = torch.zeros_like(dense_v), torch.zeros_like(attn.o_proj.weight).float()
        for h, rank in enumerate(ranks):
            folded_v[h * 128:h * 128 + rank] = enc[h, :, :rank].T @ dense_v[h * 128:(h + 1) * 128]
            for q in range(h * 4, (h + 1) * 4):
                folded_o[:, q * 128:q * 128 + rank] = dec[q, :rank].T
        attn.v_proj = QualityLinear(folded_v.to(torch.bfloat16))
        attn.o_proj = QualityLinear(folded_o.to(torch.bfloat16))
        projections[f"{i}.encoder"] = attn.v_proj
        projections[f"{i}.decoder"] = attn.o_proj
        quant_modules[f"{i}.k"] = attn.k_norm
        quant_modules[f"{i}.v"] = attn.v_proj
        indices[f"{i}.k"] = torch.arange(1024, device=device)
        indices[f"{i}.v"] = torch.tensor([h * 128 + j for h, r in enumerate(ranks) for j in range(r)], device=device)
    return projections, quant_modules, indices, schedule


def nuq4_output(upstream, quantizers, indices, name, output):
    flat = output.reshape(-1, 1024)
    active = indices[name]
    data = flat.index_select(-1, active).float()
    hi, lo, lut = quantizers[name]
    hi, lo = hi.flatten().to(data.device), lo.flatten().to(data.device)
    dynamic = name.endswith(".v")
    mask = (upstream.get_outliers_dynamic(data, channel=-1, thresh=0.99) if dynamic else
            upstream.get_outliers(data, channel=0, outlier_threshold_upper=hi, outlier_threshold_lower=lo))
    quantized = upstream.quant_fn_nuq_recon(
        data, bits=4, qchannel=-1 if dynamic else 0, dynamicquantization=dynamic,
        include_sparse=True, outlier_mask=mask, maxval=hi, minval=lo, lut=lut, first_few_fp16=-1,
    )
    assert torch.isfinite(quantized).all()
    result = flat.clone()
    result[:, active] = quantized.to(output.dtype)
    return result.reshape(output.shape)


def install_nuq4_hooks(upstream, quantizers, modules, indices):
    assert set(quantizers) == set(modules)
    return [module.register_forward_hook(
        lambda m, x, y, name=name: nuq4_output(upstream, quantizers, indices, name, y)
    ) for name, module in modules.items()]
