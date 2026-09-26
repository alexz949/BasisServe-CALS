"""Simulated KVQuant NUQ4 KV cache for the routing evaluators (quantize -> dequantize, BF16 storage).
Uses the official simulation code of SqueezeAILab/KVQuant (external/KVQuant, quant/kvquant/simquant_module_quantizer.py):
Keys are quantized pre-RoPE with static per-channel non-uniform 4-bit signposts and calibrated 1% dense-and-sparse outlier
thresholds (RoPE is applied after dequantization, as in KVQuant); the V96 Value latent is quantized per token across the
KV heads with dynamic ranges and 1% outliers. Quantizers come from evaluation/calibrate_llama_v96_kvquant.py."""
import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'external/KVQuant/quant/kvquant/simquant_module_quantizer.py'
CHUNK = 8192
BITS = 4
SPARSITY_THRESHOLD = 0.99


def load_upstream():
    assert UPSTREAM.exists(), UPSTREAM
    spec = importlib.util.spec_from_file_location('kvquant_official', UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_quantizers(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    quantizers = torch.load(directory / 'quantizers.pt', map_location='cpu', weights_only=False)
    return quantizers, manifest


def _flatten(tensor):
    batch, heads, tokens, width = tensor.shape
    assert batch == 1
    return tensor.permute(0, 2, 1, 3).reshape(tokens, heads * width).float(), (heads, width)


def _unflatten(flat, layout, like):
    heads, width = layout
    return flat.reshape(1, flat.shape[0], heads, width).permute(0, 2, 1, 3).to(like.dtype)


def nuq4_key_roundtrip(pre_key, quantizer, upstream):
    """pre_key: [1, kv_heads, tokens, head_dim] pre-RoPE keys -> static per-channel NUQ4 with calibrated outlier thresholds."""
    flat, layout = _flatten(pre_key)
    hi, lo, lut = quantizer
    hi, lo = hi.flatten().to(flat.device), lo.flatten().to(flat.device)
    out = torch.empty_like(flat)
    for start in range(0, flat.shape[0], CHUNK):
        x = flat[start:start + CHUNK]
        mask = upstream.get_outliers(x, channel=0, outlier_threshold_upper=hi, outlier_threshold_lower=lo)
        out[start:start + CHUNK] = upstream.quant_fn_nuq_recon(x, bits=BITS, qchannel=0, dynamicquantization=False, include_sparse=True,
                                                               outlier_mask=mask, maxval=hi, minval=lo, lut=lut, first_few_fp16=-1)
    assert torch.isfinite(out).all()
    return _unflatten(out, layout, pre_key)


def nuq4_value_roundtrip(value, quantizer, upstream):
    """value: [1, kv_heads, tokens, rank] V latent -> dynamic per-token NUQ4 across the KV heads with 1% outliers."""
    flat, layout = _flatten(value)
    hi, lo, lut = quantizer
    hi, lo = hi.flatten().to(flat.device), lo.flatten().to(flat.device)
    out = torch.empty_like(flat)
    for start in range(0, flat.shape[0], CHUNK):
        x = flat[start:start + CHUNK]
        mask = upstream.get_outliers_dynamic(x, channel=-1, thresh=SPARSITY_THRESHOLD)
        out[start:start + CHUNK] = upstream.quant_fn_nuq_recon(x, bits=BITS, qchannel=-1, dynamicquantization=True, include_sparse=True,
                                                               outlier_mask=mask, maxval=hi, minval=lo, lut=lut, first_few_fp16=-1)
    assert torch.isfinite(out).all()
    return _unflatten(out, layout, value)
