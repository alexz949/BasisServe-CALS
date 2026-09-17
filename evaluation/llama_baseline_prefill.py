"""Bound baseline prefill memory; restore exact cached tensors before decode."""
from types import MethodType
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from basisserve.checkpoint import c1_lrqk_qwen3 as lrqk
from basisserve.core.c1_k_routing_sidecar import build_routing_sidecar
from evaluation.chunked_prefill_mlp import install_chunked_prefill_norms
from evaluation import eval_k_routing_ruler as runtime


@torch.inference_mode()
def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
    arm = self._baseline_arm
    if past_key_values.get_seq_length(self.layer_idx):
        if arm == 'loki':
            past_key_values.sidecars[self.layer_idx] = past_key_values.sidecars[self.layer_idx].to(hidden_states.device)
            return runtime.routing_forward(self, hidden_states, position_embeddings,
                attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
        state = past_key_values.lrqk_states[self.layer_idx]
        for name, value in tuple(vars(state).items()):
            if isinstance(value, torch.Tensor):
                setattr(state, name, value.to(hidden_states.device))
        return lrqk._forward(self, hidden_states, position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
    batch, length, _ = hidden_states.shape
    assert batch == 1 and attention_mask is None
    q = self.q_norm(self.q_proj(hidden_states).view(batch, length, self.num_attention_heads, self.head_dim)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.value_head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    for start in range(0, length, 1024):
        stop = start + 1024
        qr, kr = apply_rotary_pos_emb(q[:, :, start:stop], k[:, :, start:stop], cos[:, start:stop], sin[:, start:stop])
        q[:, :, start:stop], k[:, :, start:stop] = qr, kr
    del qr, kr
    k, v = past_key_values.update(k, v, self.layer_idx)
    if arm == 'loki':
        past_key_values.sidecars[self.layer_idx] = build_routing_sidecar(k, self._loki_projector).cpu()
    else:
        state = lrqk.LRQKState(q, k, self._lrqk_config, self.layer_idx)
        for name, value in tuple(vars(state).items()):
            if isinstance(value, torch.Tensor):
                setattr(state, name, value.cpu())
        past_key_values.lrqk_states[self.layer_idx] = state
    output = runtime.compressed_v_prefill_attention(q, k, v, scale=self.scaling)
    del q, k, v
    output = output.transpose(1, 2).contiguous().reshape(batch, length, -1)
    return self.o_proj(output), None


def install(model, arm):
    assert arm in ('loki', 'lrqk')
    install_chunked_prefill_norms(model)
    for layer in model.model.layers:
        layer.self_attn._baseline_arm = arm
        layer.self_attn.forward = MethodType(forward, layer.self_attn)
