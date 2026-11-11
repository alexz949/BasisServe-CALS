"""Qwen3-32B TP4 compact V64/V96 serving."""
import torch
from torch import Tensor, nn
from basisserve.core.tp4_c1_factors import HEAD_DIM
from basisserve.vllm.tp4_c1_model import C1QKVParallelLinear, TP4C1ModelMixin
from basisserve.vllm.diffkv_attention import c1_diffkv_backend
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

class C1Attention(nn.Module):
    def __init__(self, base, encoder, decoder, boundary, config):
        super().__init__()
        assert base.num_heads == 16 and base.num_kv_heads == 2
        self.num_kv_heads = base.num_kv_heads
        self.value_rank = encoder.shape[-1]
        self.num_heads = base.num_heads
        assert base.dual_chunk_attention_config is None
        name = base.attn.layer_name
        assert name.endswith(".attn")
        assert config.compilation_config.static_forward_context.pop(name) is base.attn
        self.qkv_proj = C1QKVParallelLinear(encoder, name[:-5] + ".qkv_proj",
                                          config.model_config.hf_config.hidden_size,
                                          config.model_config.hf_config.num_attention_heads)
        self.q_norm, self.k_norm, self.rotary_emb = base.q_norm, base.k_norm, base.rotary_emb
        self.attn = Attention(
            self.num_heads, HEAD_DIM, base.scaling, num_kv_heads=self.num_kv_heads,
            cache_config=config.cache_config, prefix=name,
            attn_backend=c1_diffkv_backend(self.value_rank), head_size_v=self.value_rank,
            value_rank=self.value_rank,
        )
        object.__setattr__(self.attn, "_basisserve_c1_output", boundary)
        self.register_buffer("decoder", decoder, persistent=False)

    def forward(self, positions: Tensor, hidden_states: Tensor) -> Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((self.num_heads * HEAD_DIM, self.num_kv_heads * HEAD_DIM, self.num_kv_heads * self.value_rank), dim=-1)
        q = self.q_norm(q.reshape(-1, self.num_heads, HEAD_DIM)).reshape(-1, self.num_heads * HEAD_DIM)
        k = self.k_norm(k.reshape(-1, self.num_kv_heads, HEAD_DIM)).reshape(-1, self.num_kv_heads * HEAD_DIM)
        q, k = self.rotary_emb(positions, q, k)
        return torch.ops.vllm.basisserve_c1_attention_output(
            q, k, v, self.decoder, self.attn.layer_name,
        )


class BasisServeQwen3TP4C1ForCausalLM(TP4C1ModelMixin, Qwen3ForCausalLM):
    model_geometry = (5120, 64, 8, 128, 64)
    attention_class = C1Attention
