import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

from basisserve.core.qwen35_gated_v_runtime import GatedVAttention
from basisserve.core.qwen35_k_routing_runtime import Qwen35RoutingAttention


@pytest.mark.skipif(not torch.cuda.is_available(), reason='All routing arms require CUDA validation')
@pytest.mark.parametrize('arm', ['full', 'exact_sparse', 'b16r16', 'b32r32', 'loki', 'lrqk', 'shadowkv'])
@torch.inference_mode()
def test_shared_prefill_and_compact_decode_for_every_arm(arm):
    torch.manual_seed(47)
    config = Qwen3_5TextConfig(hidden_size=384, num_attention_heads=8, num_key_value_heads=2,
        head_dim=256, num_hidden_layers=1, layer_types=['full_attention'], attention_bias=False)
    config._attn_implementation = 'sdpa'
    native = Qwen3_5Attention(config, 0).cuda().bfloat16().eval()
    encoder = torch.linalg.qr(torch.randn(2, 256, 192, device='cuda')).Q.bfloat16()
    decoder = encoder.mT.contiguous()
    length = 4096
    hidden = torch.randn(1, length+1, 384, device='cuda', dtype=torch.bfloat16)
    angles = torch.randn(1, length+1, 32, device='cuda').repeat(1, 1, 2)
    pos = (angles.cos().bfloat16(), angles.sin().bfloat16())
    prefix_pos = tuple(value[:, :length] for value in pos)
    reference = GatedVAttention(native, encoder, decoder)(hidden[:, :length], prefix_pos)[0]
    factors = None
    if arm in ('b16r16', 'b32r32'):
        rank = 16 if arm == 'b16r16' else 32
        shapes = {f'base_left_b{rank}': (2, 192, rank), f'base_right_b{rank}': (2, rank, 256),
            f'base_bias_b{rank}': (2, 256), f'residual_encoder_b{rank}_r{rank}': (2, 256, rank),
            f'residual_query_b{rank}_r{rank}': (8, 256, rank)}
        factors = {name: torch.randn(shape, device='cuda')*0.01 for name, shape in shapes.items()}
    basis = torch.linalg.qr(torch.randn(2, 256, 32, device='cuda')).Q.bfloat16() if arm == 'loki' else None
    adapter = Qwen35RoutingAttention(native, encoder, decoder, arm=arm, factors=factors, loki_basis=basis)
    cache = DynamicCache(config=config)
    actual = adapter(hidden[:, :length], prefix_pos, past_key_values=cache)[0]
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    output = adapter(hidden[:, length:], tuple(value[:, length:] for value in pos), past_key_values=cache)[0]
    assert output.shape == (1, 1, 384) and torch.isfinite(output).all()
    assert cache.layers[0].values.shape == (1, 2, length+1, 192)
    assert cache.layers[0].keys.shape == (1, 2, length+1, 256)
    assert adapter.o_proj is native.o_proj
