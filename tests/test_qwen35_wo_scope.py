"""Ablating one Wo family must preserve the other family's native operator."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from basisserve.core.qwen35_vllm_hybrid import install_vllm_hybrid_layers, VLLMPrivateAGOutput


@pytest.mark.parametrize('scope,expected', [('all', (True, True)), ('full_attention', (True, False)), ('gdn', (False, True))])
def test_wo_family_selection(scope, expected):
    full = SimpleNamespace(attn_output_gate=True, num_kv_heads=1, num_heads=1,
                           head_dim=2, q_size=2, kv_size=2,
                           qkv_proj=nn.Linear(2, 8), attn=nn.Identity(), o_proj=nn.Linear(2, 2))
    gdn = SimpleNamespace(out_proj=nn.Linear(2, 2))
    layers = [SimpleNamespace(layer_type='full_attention', self_attn=full),
              SimpleNamespace(layer_type='linear_attention', linear_attn=gdn)]
    originals = full.o_proj, gdn.out_proj
    bank = {'factor_sha256': 'test', 'layers': {0: {'E_V': torch.eye(2)[None], 'R_V': torch.eye(2)[None]}}}
    wo = {'upstream_v_factor_sha256': 'test', 'tp_size': 4}
    for i, family in enumerate(('full_attention', 'gdn')):
        wo[family] = {'layers': [{'layer_index': i, 'layer_type': family,
                                  'private_encoders': torch.ones(2, 1, 1),
                                  'joint_decoder_weight': torch.eye(2)}]}
    install_vllm_hybrid_layers(layers, bank, wo, scope)
    for operator, original, replaced in zip((full.o_proj, gdn.out_proj), originals, expected):
        if replaced:
            assert isinstance(operator, VLLMPrivateAGOutput)
        else:
            assert operator is original


def test_native_v_endpoint_preserves_attention_and_writer_with_wo():
    full = SimpleNamespace(qkv_proj=nn.Linear(2, 8), attn=nn.Identity(), o_proj=nn.Linear(2, 2))
    gdn = SimpleNamespace(out_proj=nn.Linear(2, 2))
    layers = [SimpleNamespace(layer_type='full_attention', self_attn=full),
              SimpleNamespace(layer_type='linear_attention', linear_attn=gdn)]
    writer, attention = full.qkv_proj, full.attn
    weight, bias = writer.weight.detach().clone(), writer.bias.detach().clone()
    bank = {'factor_sha256': 'dense_test', 'layers': {}, 'schedule': {0: 256}}
    wo = {'upstream_v_factor_sha256': 'dense_test', 'tp_size': 4}
    for i, family in enumerate(('full_attention', 'gdn')):
        wo[family] = {'layers': [{'layer_index': i, 'layer_type': family,
                                  'private_encoders': torch.ones(2, 1, 1),
                                  'joint_decoder_weight': torch.eye(2)}]}
    install_vllm_hybrid_layers(layers, bank, wo)
    assert full.qkv_proj is writer and full.attn is attention
    torch.testing.assert_close(writer.weight, weight, rtol=0, atol=0)
    torch.testing.assert_close(writer.bias, bias, rtol=0, atol=0)
    assert isinstance(full.o_proj, VLLMPrivateAGOutput)
    assert isinstance(gdn.out_proj, VLLMPrivateAGOutput)
