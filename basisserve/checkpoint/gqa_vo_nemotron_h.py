"""C1 value coordinates at Nemotron-H's unrotated full-attention boundary."""
import torch
from torch import nn

from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention


def _identity_positions(module, positional, kwargs):
    hidden = kwargs['hidden_states'] if 'hidden_states' in kwargs else positional[0]
    assert 'position_embeddings' not in kwargs
    shape = (1, hidden.shape[1], module.head_dim)
    kwargs['position_embeddings'] = (
        torch.ones(shape, device=hidden.device, dtype=hidden.dtype),
        torch.zeros(shape, device=hidden.device, dtype=hidden.dtype),
    )
    return positional, kwargs


def nemotron_h_c1_attention(base_attention, *, v_proj_compressed_weight,
        o_decoder_weight, value_coordinate_encoder, attention_backend='sdpa'):
    """Use shared C1 kernels while preserving native unnormalized, unrotated Q/K.

    The pre-hook also applies when a routing implementation replaces forward.
    Identity rotation lets those implementations share the same sidecar math.
    Mamba modules and recurrent state are outside this adapter's scope.
    """
    assert base_attention.config.model_type == 'nemotron_h'
    assert base_attention.q_proj.bias is None and base_attention.k_proj.bias is None
    assert base_attention.v_proj.bias is None and base_attention.o_proj.bias is None
    assert not hasattr(base_attention, 'q_norm') and not hasattr(base_attention, 'k_norm')
    base_attention.q_norm = nn.Identity()
    base_attention.k_norm = nn.Identity()
    base_attention.sliding_window = None
    attention = GQATiedVOQwen3Attention(base_attention,
        v_proj_compressed_weight=v_proj_compressed_weight,
        o_decoder_weight=o_decoder_weight,
        value_coordinate_encoder=value_coordinate_encoder,
        attention_backend=attention_backend)
    attention.register_forward_pre_hook(_identity_positions, with_kwargs=True)
    return attention
