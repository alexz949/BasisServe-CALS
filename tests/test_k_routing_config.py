import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from evaluation.k_routing_config import routing_config, routing_position_embeddings
from evaluation.v96kl_common import sha256


def test_static_yarn_matches_calibration_and_evaluation_positions(tmp_path):
    config = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=40960, rope_theta=1000000.0)
    config.save_pretrained(tmp_path)
    original = (tmp_path / 'config.json').read_bytes()
    identity = dict(model=str(tmp_path), model_config_sha256=sha256(tmp_path / 'config.json'),
        hq=4, hkv=2, head_dim=8, hidden_size=32)
    calibration = routing_config(identity, rope='yarn2', sequence_length=32768)
    evaluation = routing_config(identity, rope='yarn2', sequence_length=65536)
    assert calibration.to_dict() == evaluation.to_dict()
    rotary_fit, rotary_eval = Qwen3RotaryEmbedding(calibration), Qwen3RotaryEmbedding(evaluation)
    fit = rotary_fit(torch.empty(1, dtype=torch.float32), torch.arange(32768)[None])
    full = rotary_eval(torch.empty(1, dtype=torch.float32), torch.arange(65536)[None])
    for short, long in zip(fit, full, strict=True):
        assert torch.isfinite(long).all()
        torch.testing.assert_close(short, long[:, :32768], atol=0, rtol=0)
    assert (tmp_path / 'config.json').read_bytes() == original
    with pytest.raises(AssertionError):
        routing_config(identity, rope='native', sequence_length=65536)


def test_nemotron_fitting_preserves_unrotated_keys():
    from transformers import NemotronHConfig
    from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _post_rope_rows

    config = NemotronHConfig(hidden_size=8192, num_attention_heads=64,
        num_key_value_heads=8, max_position_embeddings=131072)
    cos, sin = routing_position_embeddings(config, 32768, 'cpu')
    assert cos.shape == sin.shape == (1, 32768, 128)
    generator = torch.Generator().manual_seed(17)
    keys = torch.randn(257, 8, 128, generator=generator)
    actual = _post_rope_rows(keys, cos[:, :257], sin[:, :257])
    torch.testing.assert_close(actual, keys, atol=0, rtol=0)


def test_position_helper_keeps_qwen_native_phases():
    config = Qwen3Config(hidden_size=32, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8)
    expected = Qwen3RotaryEmbedding(config, device='cpu')(
        torch.empty(1, dtype=torch.float32), torch.arange(257)[None])
    actual = routing_position_embeddings(config, 257, 'cpu')
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
