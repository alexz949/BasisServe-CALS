from types import SimpleNamespace

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

from evaluation.capture_qwen35_k_routing import RoutingCapture


@torch.inference_mode()
def test_capture_matches_native_sdpa_inputs():
    torch.manual_seed(23)
    config = Qwen3_5TextConfig(hidden_size=18, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, num_hidden_layers=1, layer_types=['full_attention'], attention_bias=True)
    config._attn_implementation = 'sdpa'
    attention = Qwen3_5Attention(config, 0).eval()
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]))
    hidden = torch.randn(1, 9, 18)
    angles = torch.randn(1, 9, 2).repeat(1, 1, 2)
    kwargs = dict(hidden_states=hidden, position_embeddings=(angles.cos(), angles.sin()), attention_mask=None)
    reference = attention(**kwargs)[0]
    checked = []
    with RoutingCapture(model, [3, 7]) as capture:
        def verify(module, query, key, value, mask, **options):
            row = capture.rows[0]
            torch.testing.assert_close(row['rows'][..., :8], value.transpose(1, 2), rtol=0, atol=0)
            torch.testing.assert_close(row['rows'][..., 8:], key.transpose(1, 2), rtol=0, atol=0)
            torch.testing.assert_close(row['candidate_queries'], query[:, :, [3, 7]].transpose(1, 2), rtol=0, atol=0)
            torch.testing.assert_close(row['pre_rope_keys'][..., 4:], key.transpose(1, 2)[..., 4:], rtol=0, atol=0)
            checked.append(True)
            return sdpa_attention_forward(module, query, key, value, mask, **options)
        ALL_ATTENTION_FUNCTIONS.register('q35_capture_test', verify)
        config._attn_implementation = 'q35_capture_test'
        actual = attention(**kwargs)[0]
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert checked == [True] and not attention._forward_pre_hooks
