"""Gated V computation in standard-width cache slots for vLLM quality tests.

Unused V coordinates are zero padding, not reconstructed historical values.
Only current attention outputs are reconstructed, before the native gate.
This module makes no compact-cache or distributed communication claim.
"""

import torch
from torch import nn

from basisserve.core.qwen35_gdn_private_ag_runtime import Qwen35PrivateAGOutput


@torch.no_grad()
def pad_latent_v_writer(weight, bias, encoder):
    groups, width, rank = encoder.shape
    assert 0 < rank <= width and weight.shape[0] == groups * width
    assert torch.isfinite(encoder).all()
    e = encoder.to(weight)
    latent = e.transpose(1, 2) @ weight.reshape(groups, width, -1)
    padded = torch.zeros_like(weight).reshape(groups, width, -1)
    padded[:, :rank] = latent
    padded_bias = None
    if bias is not None:
        assert bias.shape == (groups * width,)
        padded_bias = torch.zeros_like(bias).reshape(groups, width)
        padded_bias[:, :rank] = torch.einsum('gd,gdr->gr', bias.reshape(groups, width), e)
        padded_bias = padded_bias.flatten()
    return padded.reshape_as(weight), padded_bias


class ReconstructBeforeGate(nn.Module):
    def __init__(self, attention, decoder, num_query_heads):
        super().__init__()
        self.attention = attention
        groups, rank, width = decoder.shape
        assert num_query_heads % groups == 0 and 0 < rank <= width
        assert torch.isfinite(decoder).all()
        self.num_query_heads, self.width, self.rank = num_query_heads, width, rank
        mapping = torch.arange(num_query_heads, device=decoder.device) // (num_query_heads // groups)
        self.register_buffer('decoder_by_head', decoder[mapping].contiguous())

    def forward(self, query, key, padded_value):
        padded = self.attention(query, key, padded_value)
        latent = padded.reshape(-1, self.num_query_heads, self.width)[..., :self.rank]
        return torch.einsum('nhr,hrd->nhd', latent, self.decoder_by_head).flatten(1)


class VLLMPrivateAGOutput(nn.Module):
    """Preserve vLLM RowParallelLinear's (output, bias) return contract at TP1."""

    def __init__(self, encoders, decoder_weight, bias=None):
        super().__init__()
        self.output = Qwen35PrivateAGOutput(encoders, decoder_weight, bias=bias)

    def forward(self, hidden_states):
        return self.output(hidden_states), None


@torch.no_grad()
def install_vllm_hybrid_layers(layers, bank, wo=None, wo_scope='all'):
    assert wo_scope in ('all', 'full_attention', 'gdn')
    full = {i for i, layer in enumerate(layers) if layer.layer_type == 'full_attention'}
    if bank['layers']:
        assert set(bank['layers']) == full
    else:
        assert set(bank['schedule']) == full and set(bank['schedule'].values()) == {256}
    for i, factors in bank['layers'].items():
        attention = layers[i].self_attn
        assert attention.attn_output_gate
        e, r = factors['E_V'], factors['R_V']
        assert e.shape[0] == attention.num_kv_heads
        assert e.shape[1] == attention.head_dim and r.shape == (e.shape[0], e.shape[2], e.shape[1])
        offset = 2 * attention.q_size + attention.kv_size
        projection = attention.qkv_proj
        assert projection.weight.shape[0] == offset + attention.kv_size
        bias = projection.bias
        weight, folded_bias = pad_latent_v_writer(projection.weight[offset:], None if bias is None else bias[offset:], e)
        projection.weight[offset:].copy_(weight)
        if bias is not None:
            bias[offset:].copy_(folded_bias)
        attention.attn = ReconstructBeforeGate(attention.attn, r.to(weight), attention.num_heads)
    if wo is not None:
        assert wo['upstream_v_factor_sha256'] == bank['factor_sha256']
        assert wo['tp_size'] == 4
        records = wo['gdn']['layers'] + wo['full_attention']['layers']
        assert sorted(r['layer_index'] for r in records) == list(range(len(layers)))
        for record in records:
            i = record['layer_index']
            is_full = i in full
            assert record['layer_type'] == ('full_attention' if is_full else 'gdn')
            if wo_scope != 'all' and record['layer_type'] != wo_scope:
                continue
            owner = layers[i].self_attn if is_full else layers[i].linear_attn
            name = 'o_proj' if is_full else 'out_proj'
            native = getattr(owner, name)
            e = record['private_encoders'].to(native.weight)
            d = record['joint_decoder_weight'].to(native.weight)
            assert native.weight.shape == (d.shape[0], e.shape[0] * e.shape[1])
            setattr(owner, name, VLLMPrivateAGOutput(e, d, native.bias))
