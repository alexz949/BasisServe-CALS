"""Shared uniform C1 checkpoints and norm-preserving attention installation."""
from pathlib import Path
import torch
from safetensors.torch import load_file
from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention
from evaluation.v96kl_common import ROOT, MODEL, read_json, sha256
from evaluation.eval_llama31_shadow import MODEL as LLAMA_MODEL


def paths(family):
    qwen = family == 'qwen3'
    return dict(model=MODEL if qwen else LLAMA_MODEL,
        checkpoint=ROOT / f"results/hf/ICLR-results/{family}-8b/checkpoints/{'Q3-8B' if qwen else 'L31-8B'}-C1U-R96",
        calibration=ROOT / ('results/calibration/v96kl_64x32k' if qwen else 'results/calibration/llama31_64x32k'),
        bank=ROOT / f'results/checkpoints/{family}_uniform96_b16r16',
        output=ROOT / f'results/evaluation/{family}_uniform96_compare',
        shadow=ROOT / f'results/evaluation/{family}_uniform96_shadow')


def checkpoint_manifest(checkpoint, model):
    manifest = read_json(checkpoint / 'manifest.json')
    assert manifest['status'] == 'complete'
    assert manifest['compression']['allocation'] == 'uniform_per_layer_per_head'
    assert manifest['model']['config_sha256'] == sha256(model / 'config.json')
    assert manifest['model']['safetensors_index_sha256'] == sha256(model / 'model.safetensors.index.json')
    assert sha256(checkpoint / manifest['artifact']['file']) == manifest['artifact']['sha256']
    for i, record in enumerate(manifest['layers']):
        assert record['layer'] == i and record['ranks'] == [96] * 8
        assert sha256(checkpoint / record['file']) == record['sha256']
    return manifest


def llama_attention_interface(module):
    """Llama uses identity Q/K normalization and full attention in this adapter."""
    module.q_norm = torch.nn.Identity()
    module.k_norm = torch.nn.Identity()
    module.num_key_value_groups = module.config.num_attention_heads // module.config.num_key_value_heads
    module.sliding_window = None


@torch.inference_mode()
def install(model, checkpoint, manifest, family):
    for layer, record in zip(model.model.layers, manifest['layers'], strict=True):
        m = layer.self_attn
        tensors = load_file(str(checkpoint / record['file']))
        encoder, decoder = tensors['value_coordinate_encoders'], tensors['head_output_decoders']
        assert encoder.shape == (8, 128, 96) and decoder.shape == (32, 96, 4096)
        assert encoder.dtype == decoder.dtype == torch.bfloat16
        assert torch.isfinite(encoder).all() and torch.isfinite(decoder).all()
        assert m.v_proj.bias is None and m.o_proj.bias is None
        if family == 'llama31':
            llama_attention_interface(m)
        device = m.v_proj.weight.device
        weight = torch.bmm(encoder.to(device).float().transpose(1, 2), m.v_proj.weight.float().reshape(8, 128, 4096)).reshape(768, 4096)
        output = decoder.to(device).float().permute(2, 0, 1).reshape(4096, 3072)
        layer.self_attn = GQATiedVOQwen3Attention(m, v_proj_compressed_weight=weight,
            o_decoder_weight=output, attention_backend='triton', value_coordinate_encoder=encoder)
        assert layer.self_attn.q_norm is m.q_norm and layer.self_attn.k_norm is m.k_norm
        assert torch.equal(layer.self_attn.v_proj.weight, weight.bfloat16())
        assert torch.equal(layer.self_attn.o_proj.weight, output.bfloat16())
    return model
