import copy

import pytest
import torch
from transformers import DynamicCache, NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHAttention

from basisserve.checkpoint.gqa_vo_nemotron_h import nemotron_h_c1_attention


@pytest.mark.parametrize('rank', [4, 8])
@torch.inference_mode()
def test_native_unrotated_attention_matches_compressed_coordinates_and_cache(rank):
    torch.manual_seed(72)
    config = NemotronHConfig(hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        hybrid_override_pattern='*', attention_dropout=0.0)
    config._attn_implementation = 'sdpa'
    native = NemotronHAttention(config, 0).eval()
    source = copy.deepcopy(native)
    encoder = torch.stack([torch.linalg.qr(torch.randn(8, 8)).Q[:, :rank] for _ in range(2)])
    value_weight = source.v_proj.weight.reshape(2, 8, 32)
    compressed_weight = torch.bmm(encoder.mT, value_weight).reshape(2*rank, 32)
    output_heads = source.o_proj.weight.T.reshape(4, 8, 32)
    decoder = torch.bmm(encoder.repeat_interleave(2, 0).mT, output_heads)
    compressed = nemotron_h_c1_attention(source,
        v_proj_compressed_weight=compressed_weight,
        o_decoder_weight=decoder.permute(2, 0, 1).reshape(32, 4*rank),
        value_coordinate_encoder=encoder).eval()
    # Dense reference stays on the actual native implementation, with the
    # same low-rank V projection folded back into its original head width.
    native.v_proj.weight.copy_(torch.bmm(encoder, torch.bmm(encoder.mT, value_weight)).reshape(16, 32))
    native_cache, compressed_cache = DynamicCache(config=config), DynamicCache(config=config)
    for length in (23, 1, 1):
        hidden = torch.randn(2, length, 32)
        expected, _ = native(hidden, past_key_values=native_cache)
        actual, _ = compressed(hidden, attention_mask=None, past_key_values=compressed_cache)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(compressed_cache.layers[0].keys, native_cache.layers[0].keys,
            atol=0, rtol=0)
        assert compressed_cache.layers[0].values.shape[-1] == rank
        projected = torch.einsum('bgtd,gdr->bgtr', native_cache.layers[0].values, encoder)
        torch.testing.assert_close(compressed_cache.layers[0].values, projected, atol=2e-6, rtol=2e-5)
