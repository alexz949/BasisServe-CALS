"""Explicit, bounded runtime position configuration for routing experiments."""
from pathlib import Path

from transformers import AutoConfig

from evaluation.v96kl_common import sha256


def routing_config(identity, *, rope, sequence_length):
    path = Path(identity['model'])
    assert sha256(path / 'config.json') == identity['model_config_sha256']
    config = AutoConfig.from_pretrained(path, trust_remote_code=False, local_files_only=True)
    assert config.model_type in ('llama', 'qwen3', 'nemotron_h')
    assert rope in ('native', 'yarn2')
    if rope == 'yarn2':
        assert config.model_type == 'qwen3'
        assert config.rope_parameters['rope_type'] == 'default'
        config.rope_parameters = dict(rope_type='yarn', factor=2.0,
            original_max_position_embeddings=32768, rope_theta=config.rope_parameters['rope_theta'])
        config.max_position_embeddings = 65536
    assert 0 < sequence_length <= config.max_position_embeddings
    assert config.num_attention_heads == identity['hq']
    assert config.num_key_value_heads == identity['hkv']
    assert config.hidden_size == identity['hidden_size']
    head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    assert head_dim == identity['head_dim']
    return config


def routing_position_embeddings(config, sequence_length, device):
    """FP32 fitting phases; Nemotron has no positional rotation."""
    import torch

    assert 0 < sequence_length <= config.max_position_embeddings
    if config.model_type == 'nemotron_h':
        dim = config.hidden_size // config.num_attention_heads
        shape = (1, sequence_length, dim)
        return torch.ones(shape, device=device), torch.zeros(shape, device=device)
    assert config.model_type in ('llama', 'qwen3')
    if config.model_type == 'llama':
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as Rotary
    else:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding as Rotary
    # Preserve the dense teacher's CPU initialization before device transfer.
    rotary = Rotary(config, device='cpu').to(device)
    return rotary(torch.empty(1, device=device, dtype=torch.float32),
        torch.arange(sequence_length, device=device)[None])
