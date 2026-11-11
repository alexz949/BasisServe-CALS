"""Llama-3.1-8B C1 adapter retaining vLLM's native Llama RoPE and norms."""
import torch
from torch import nn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.llama import LlamaForCausalLM
from basisserve.core.tp4_c1_factors import HEAD_DIM
from basisserve.vllm.diffkv_attention import c1_diffkv_backend
from basisserve.vllm.tp4_c1_model import C1QKVParallelLinear, TP4C1ModelMixin


class LlamaC1Attention(nn.Module):
    def __init__(self, base, encoder, decoder, boundary, config):
        super().__init__()
        assert base.num_heads == 8 and base.num_kv_heads == 2
        assert base.qkv_proj.bias is None and base.o_proj.bias is None
        self.num_heads, self.num_kv_heads = base.num_heads, base.num_kv_heads
        self.value_rank = encoder.shape[-1]
        name = base.attn.layer_name
        assert name.endswith('.attn')
        assert config.compilation_config.static_forward_context.pop(name) is base.attn
        self.qkv_proj = C1QKVParallelLinear(encoder, name[:-5] + '.qkv_proj', 4096, 32)
        # Reuse the actual Llama3-scaled rotary embedding built by vLLM.
        # Llama has no Q/K RMSNorm; the surrounding layer's norms stay intact.
        self.rotary_emb = base.rotary_emb
        self.attn = Attention(
            self.num_heads, HEAD_DIM, base.scaling, num_kv_heads=self.num_kv_heads,
            cache_config=config.cache_config, prefix=name,
            attn_backend=c1_diffkv_backend(self.value_rank), head_size_v=self.value_rank,
            value_rank=self.value_rank,
        )
        object.__setattr__(self.attn, '_basisserve_c1_output', boundary)
        self.register_buffer('decoder', decoder, persistent=False)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((8 * HEAD_DIM, 2 * HEAD_DIM, 2 * self.value_rank), dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        return torch.ops.vllm.basisserve_c1_attention_output(q, k, v, self.decoder, self.attn.layer_name)


class BasisServeLlamaTP4C1ForCausalLM(TP4C1ModelMixin, LlamaForCausalLM):
    model_geometry = (4096, 32, 8, 128, 32)
    attention_class = LlamaC1Attention
