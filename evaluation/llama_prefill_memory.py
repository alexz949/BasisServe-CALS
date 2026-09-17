"""Bound tokenwise normalization and rotary intermediates for long prefills."""
import torch
from types import MethodType

from evaluation.chunked_prefill_mlp import install_chunked_prefill_norms
from evaluation import eval_k_routing_ruler as runtime


def finite_and_max(value):
    rows = value.reshape(-1, value.shape[-1])
    finite = bool(rows.isfinite().all(dim=-1).all())
    maximum = float(rows.abs().amax(dim=-1).max()) if finite else float('nan')
    return finite, maximum


@torch.inference_mode()
def prefill_forward(self, hidden_states, position_embeddings, attention_mask=None,
                    past_key_values=None, **kwargs):
    if past_key_values.get_seq_length(self.layer_idx):
        if self._routing_arm == 'ours':
            past_key_values.sidecars[self.layer_idx] = past_key_values.sidecars[self.layer_idx].to(hidden_states.device)
        return runtime.routing_forward(self, hidden_states, position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    batch, length, _ = hidden_states.shape
    assert batch == 1 and attention_mask is None
    assert self._routing_arm in ('full', 'ours')
    q = self.q_norm(self.q_proj(hidden_states).view(batch, length,
        self.num_attention_heads, self.head_dim)).transpose(1, 2)
    pre = self.k_norm(self.k_proj(hidden_states).view(batch, length,
        self.num_key_value_heads, self.head_dim)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(batch, length,
        self.num_key_value_heads, self.value_head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    if getattr(self, '_dense_smoke_audit', False):
        for name, value in (('hidden', hidden_states), ('q_pre_rope', q),
                ('k_pre_rope', pre), ('v_pre_rope', v), ('cos', cos), ('sin', sin)):
            finite, maximum = finite_and_max(value)
            print('FINITE', name, self.layer_idx, finite, maximum, flush=True)
            assert finite, (name, self.layer_idx)
    q, k = apply_rotary_pos_emb(q, pre, cos, sin)
    del pre
    if self._routing_arm == 'ours':
        t = self._routing_factors
        past_key_values.sidecars[self.layer_idx] = runtime.build_conditional_routing_sidecar(
            v, k, base_left=t['base_left_b16'], base_right=t['base_right_b16'],
            base_bias=t['base_bias_b16'], residual_encoder=t['residual_encoder_b16_r16'],
            cos=cos, sin=sin).cpu()
    k, v = past_key_values.update(k, v, self.layer_idx)
    output = runtime.compressed_v_prefill_attention(q, k, v, scale=self.scaling)
    del q, k, v
    output = output.transpose(1, 2).contiguous().reshape(batch, length, -1)
    return self.o_proj(output), None


def install(model):
    from transformers.models.llama import modeling_llama

    install_chunked_prefill_norms(model)
    original = modeling_llama.apply_rotary_pos_emb

    @torch.inference_mode()
    def rotary(q, k, cos, sin, unsqueeze_dim=1):
        if q.shape[-2] <= 1024:
            return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        assert unsqueeze_dim == 1
        for start in range(0, q.shape[-2], 1024):
            stop = start + 1024
            part_q, part_k = original(q[:, :, start:stop], k[:, :, start:stop],
                cos[:, start:stop], sin[:, start:stop], unsqueeze_dim=1)
            q[:, :, start:stop] = part_q
            k[:, :, start:stop] = part_k
        return q, k

    modeling_llama.apply_rotary_pos_emb = rotary
    for layer in model.model.layers:
        layer.self_attn.forward = MethodType(prefill_forward, layer.self_attn)
