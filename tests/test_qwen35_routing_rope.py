import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from basisserve.core.c1_v_k_index import apply_rotary, invert_rotary


def test_partial_rope_matches_qwen35_native():
    torch.manual_seed(19)
    value = torch.randn(2, 4, 17, 256)
    angles = torch.randn(2, 17, 32).repeat(1, 1, 2)
    cos, sin = angles.cos(), angles.sin()
    expected, _ = apply_rotary_pos_emb(value, value, cos, sin)
    actual = apply_rotary(value, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[..., 64:], value[..., 64:], rtol=0, atol=0)
    torch.testing.assert_close(invert_rotary(actual, cos, sin), value, rtol=1e-5, atol=1e-6)
