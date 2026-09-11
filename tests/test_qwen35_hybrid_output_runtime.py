import copy

import pytest
import torch

from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from basisserve.core.qwen35_gated_v_runtime import GatedVRuntime, factor_hash
from basisserve.core.qwen35_hybrid_output_runtime import HybridOutputRuntime
from basisserve.core.qwen35_gdn_private_ag_runtime import FACTOR_FORMAT as GDN_FORMAT
from basisserve.core.qwen35_full_attention_private_ag_runtime import FACTOR_FORMAT as FULL_FORMAT


def setup():
    torch.manual_seed(19)
    config = Qwen3_5TextConfig(hidden_size=18, intermediate_size=32, vocab_size=41,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_hidden_layers=2,
        layer_types=['linear_attention', 'full_attention'], linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=4, linear_value_head_dim=4)
    config._attn_implementation = 'sdpa'
    model = Qwen3_5ForCausalLM(config).eval().requires_grad_(False)
    e = torch.linalg.qr(torch.randn(2, 8, 4)).Q
    v = {1: {'E_V': e, 'R_V': e.mT.contiguous()}}
    def identity(index, projection, format):
        local_width = projection.in_features // 2
        return {'format': format, 'schema_version': 1, 'layers': [{'layer_index': index,
            'private_encoders': torch.eye(local_width).repeat(2, 1, 1),
            'joint_decoder_weight': projection.weight.detach().clone()}]}
    output = {'format': 'basisserve.qwen35.hybrid_output_bank.v1', 'upstream_v_factor_sha256': factor_hash(v),
        'gdn': identity(0, model.model.layers[0].linear_attn.out_proj, GDN_FORMAT),
        'full_attention': identity(1, model.model.layers[1].self_attn.o_proj, FULL_FORMAT)}
    return model, v, output


@torch.no_grad()
def test_composed_identity_targets_frozen_v_and_restores():
    model, v, output = setup()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    native_attention = model.model.layers[1].self_attn
    native_gdn_output = model.model.layers[0].linear_attn.out_proj
    dense = model.model(tokens, use_cache=False).last_hidden_state
    with GatedVRuntime(model, v):
        expected = model.model(tokens, use_cache=False).last_hidden_state
        reference = model.model(tokens, use_cache=True).past_key_values
        state_shapes = (reference.conv_states[0].shape, reference.recurrent_states[0].shape)
    assert float((expected - dense).norm()) > 0
    with HybridOutputRuntime(model, v, output):
        actual = model.model(tokens, use_cache=False).last_hidden_state
        cache = model.model(tokens, use_cache=True).past_key_values
        assert (cache.conv_states[0].shape, cache.recurrent_states[0].shape) == state_shapes
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert model.model.layers[1].self_attn is native_attention
    assert model.model.layers[0].linear_attn.out_proj is native_gdn_output
    assert factor_hash(v) == output['upstream_v_factor_sha256']


def test_hash_rejection_and_failed_install_rollback():
    model, v, output = setup()
    original = model.model.layers[1].self_attn
    malformed = copy.deepcopy(output)
    malformed['upstream_v_factor_sha256'] = 'incorrect'
    with pytest.raises(AssertionError):
        HybridOutputRuntime(model, v, malformed)
    malformed = copy.deepcopy(output)
    malformed['full_attention']['layers'][0]['joint_decoder_weight'] = torch.zeros(18, 1)
    with pytest.raises(ValueError):
        with HybridOutputRuntime(model, v, malformed):
            assert False
    assert model.model.layers[1].self_attn is original


@torch.no_grad()
def test_native_capture_preserves_padded_model_forward():
    from evaluation.run_qwen35_hybrid import NativeCapture
    model, _, _ = setup()
    tokens = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 4, 5]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    expected = model.model(tokens, attention_mask=mask, use_cache=False).last_hidden_state
    with NativeCapture(model) as collector:
        actual = model.model(tokens, attention_mask=mask, use_cache=False).last_hidden_state
        assert collector.rows[1]['z'].shape == (2, 5, 4, 8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
