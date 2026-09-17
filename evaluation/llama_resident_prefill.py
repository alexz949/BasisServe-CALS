"""Memory-bounded Llama 128k prefills with all cache and routing state resident on GPU."""
from types import MethodType

import torch

from basisserve.checkpoint import c1_lrqk_qwen3 as lrqk
from basisserve.core.c1_k_routing_sidecar import build_routing_sidecar
from evaluation import eval_k_routing_ruler as runtime
from evaluation.chunked_prefill_mlp import ChunkedTokenwise, install_chunked_prefill_norms


class InPlaceChunkedMLP(torch.nn.Module):
    """Reuse the normalized MLP input after each token chunk has consumed it."""
    def __init__(self, inner, chunk_size):
        super().__init__()
        self.inner = inner
        self.chunk_size = chunk_size

    def forward(self, hidden_states):
        if hidden_states.shape[1] <= self.chunk_size:
            return self.inner(hidden_states)
        for start in range(0, hidden_states.shape[1], self.chunk_size):
            stop = start + self.chunk_size
            update = self.inner(hidden_states[:, start:stop])
            hidden_states[:, start:stop].copy_(update)
        return hidden_states


def install_resident_workspace(model):
    install_chunked_prefill_norms(model)
    for layer in model.model.layers:
        chunked = layer.mlp
        assert isinstance(chunked, ChunkedTokenwise)
        layer.mlp = InPlaceChunkedMLP(chunked.inner, chunked.chunk_size)


@torch.inference_mode()
def full_or_ours_forward(self, hidden_states, position_embeddings, attention_mask=None,
                         past_key_values=None, **kwargs):
    if past_key_values.get_seq_length(self.layer_idx):
        return runtime.routing_forward(self, hidden_states, position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    batch, length, _ = hidden_states.shape
    assert batch == 1 and attention_mask is None
    assert self._routing_arm in ('full', 'ours')
    q = self.q_norm(self.q_proj(hidden_states).view(
        batch, length, self.num_attention_heads, self.head_dim)).transpose(1, 2)
    pre = self.k_norm(self.k_proj(hidden_states).view(
        batch, length, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(
        batch, length, self.num_key_value_heads, self.value_head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    k = torch.empty_like(pre)
    for start in range(0, length, 1024):
        stop = start + 1024
        qr, kr = apply_rotary_pos_emb(q[:, :, start:stop], pre[:, :, start:stop],
                                     cos[:, start:stop], sin[:, start:stop])
        q[:, :, start:stop], k[:, :, start:stop] = qr, kr
    del pre, qr, kr
    if self._routing_arm == 'ours':
        factors = self._routing_factors
        past_key_values.sidecars[self.layer_idx] = runtime.build_conditional_routing_sidecar(
            v, k, base_left=factors['base_left_b16'], base_right=factors['base_right_b16'],
            base_bias=factors['base_bias_b16'],
            residual_encoder=factors['residual_encoder_b16_r16'], cos=cos, sin=sin)
    k, v = past_key_values.update(k, v, self.layer_idx)
    output = runtime.compressed_v_prefill_attention(q, k, v, scale=self.scaling)
    del q, k, v
    output = output.transpose(1, 2).contiguous().reshape(batch, length, -1)
    return self.o_proj(output), None


@torch.inference_mode()
def baseline_forward(self, hidden_states, position_embeddings, attention_mask=None,
                     past_key_values=None, **kwargs):
    arm = self._baseline_arm
    if past_key_values.get_seq_length(self.layer_idx):
        if arm == 'loki':
            return runtime.routing_forward(self, hidden_states, position_embeddings,
                attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
        return lrqk._forward(self, hidden_states, position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    batch, length, _ = hidden_states.shape
    assert batch == 1 and attention_mask is None and arm in ('loki', 'lrqk')
    q = self.q_norm(self.q_proj(hidden_states).view(
        batch, length, self.num_attention_heads, self.head_dim)).transpose(1, 2)
    pre = self.k_norm(self.k_proj(hidden_states).view(
        batch, length, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(
        batch, length, self.num_key_value_heads, self.value_head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    k = torch.empty_like(pre)
    for start in range(0, length, 1024):
        stop = start + 1024
        qr, kr = apply_rotary_pos_emb(q[:, :, start:stop], pre[:, :, start:stop],
                                     cos[:, start:stop], sin[:, start:stop])
        q[:, :, start:stop], k[:, :, start:stop] = qr, kr
    del pre, qr, kr
    k, v = past_key_values.update(k, v, self.layer_idx)
    if arm == 'loki':
        past_key_values.sidecars[self.layer_idx] = build_routing_sidecar(k, self._loki_projector)
    else:
        past_key_values.lrqk_states[self.layer_idx] = lrqk.LRQKState(
            q, k, self._lrqk_config, self.layer_idx)
    output = runtime.compressed_v_prefill_attention(q, k, v, scale=self.scaling)
    del q, k, v
    output = output.transpose(1, 2).contiguous().reshape(batch, length, -1)
    return self.o_proj(output), None


def install_full_or_ours(model):
    install_resident_workspace(model)
    for layer in model.model.layers:
        layer.self_attn.forward = MethodType(full_or_ours_forward, layer.self_attn)


def install_baseline(model, arm):
    assert arm in ('loki', 'lrqk')
    install_resident_workspace(model)
    for layer in model.model.layers:
        layer.self_attn._baseline_arm = arm
        layer.self_attn.forward = MethodType(baseline_forward, layer.self_attn)
