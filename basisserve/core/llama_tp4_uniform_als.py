"""Uniform V96 Llama attention with a folded Value projection for TP4/TP8."""

from pathlib import Path

from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.bias import causal_lower_right
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


class LlamaTP4UniformALSAttention(nn.Module):
    def __init__(self, original: nn.Module, factor_path: Path, rank: int) -> None:
        super().__init__()
        self.tp_size = dist.get_world_size()
        assert self.tp_size in (4, 8)
        self.local_kv = 8 // self.tp_size
        self.local_q = 32 // self.tp_size
        assert original.q_proj.weight.to_local().shape == (self.local_q * 128, 4096)
        assert original.k_proj.weight.to_local().shape == (self.local_kv * 128, 4096)
        assert original.v_proj.weight.to_local().shape == (self.local_kv * 128, 4096)
        assert original.o_proj.weight.to_local().shape == (4096, self.local_q * 128)
        assert original.v_proj.bias is None
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.layer_idx = original.layer_idx
        self.scaling = original.scaling

        factors = load_file(str(factor_path), device="cpu")
        assert set(factors) == {
            "value_coordinate_encoders", "head_output_decoders", "source_ranks"
        }
        assert bool(torch.all(factors["source_ranks"] == 96))
        assert factors["value_coordinate_encoders"].shape == (8, 128, 96)
        assert factors["head_output_decoders"].shape == (32, 96, 4096)
        encoder = factors["value_coordinate_encoders"][
            rank * self.local_kv:(rank + 1) * self.local_kv
        ].to(
            device=f"cuda:{rank}", dtype=torch.float32
        )
        source_weight = original.v_proj.weight.detach().to_local().reshape(
            self.local_kv, 128, 4096
        ).float()
        folded = torch.einsum("hdr,hdm->hrm", encoder, source_weight).reshape(
            self.local_kv * 96, 4096
        )
        self.v_proj = nn.Linear(
            4096, self.local_kv * 96, bias=False, device=f"cuda:{rank}", dtype=torch.bfloat16
        )
        self.v_proj.weight = nn.Parameter(folded.bfloat16(), requires_grad=False)
        self.register_buffer(
            "decoder",
            factors["head_output_decoders"].reshape(32 * 96, 4096).to(
                device=f"cuda:{rank}", dtype=torch.bfloat16
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        batch, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, query_length, self.local_q, 128).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, query_length, self.local_kv, 128).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, query_length, self.local_kv, 96).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(
                key, value, self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )
        key_length = key.shape[-2]
        if attention_mask is None and 1 < query_length < key_length:
            attention_mask = causal_lower_right(query_length, key_length)
        is_causal = attention_mask is None and query_length > 1
        coordinates = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, is_causal=is_causal,
            scale=self.scaling, enable_gqa=True,
        )
        return self.project_output(coordinates), None

    def project_output(self, coordinates: torch.Tensor) -> torch.Tensor:
        batch, _, query_length, _ = coordinates.shape
        local = coordinates.transpose(1, 2).contiguous().view(
            batch, query_length, self.local_q * 96
        )
        gathered = [torch.empty_like(local) for _ in range(self.tp_size)]
        dist.all_gather(gathered, local)
        full = torch.cat(gathered, dim=-1)
        return full @ self.decoder


def install_llama_tp4_uniform_als(model: nn.Module, factor_dir: Path, rank: int) -> None:
    assert model.config.model_type == "llama"
    assert model.config.num_hidden_layers == 32
    for layer_index, layer in enumerate(model.model.layers):
        layer.self_attn = LlamaTP4UniformALSAttention(
            layer.self_attn, factor_dir / f"layer_{layer_index:03d}.safetensors", rank
        )
