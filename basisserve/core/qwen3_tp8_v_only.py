"""Qwen3 TP8 full-attention memory placement for STAR-KV V-only and Basis V64."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from safetensors.torch import load_file
from transformers import Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


TP_SIZE = 8
NUM_LAYERS = 36
HEAD_DIM = 128
QUERY_HEADS_PER_RANK = 4
VALUE_RANK = 64
SKIP_LAYERS = (0, 1, 31)


class StarFusedValue(nn.Module):
    """Keep the exported global VS latent and TP-sharded U decoder."""

    def __init__(self, hidden_size: int, latent_rank: int, output_size: int):
        super().__init__()
        self.VS = nn.Linear(hidden_size, latent_rank, bias=False)
        self.U = nn.Linear(latent_rank, output_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.U(self.VS(hidden_states))


class StarQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        ranks = config.star_value_ranks
        assert len(ranks) == NUM_LAYERS
        for index, rank in enumerate(ranks):
            if index not in SKIP_LAYERS:
                self.model.layers[index].self_attn.v_proj = StarFusedValue(
                    config.hidden_size, rank, config.num_key_value_heads * config.head_dim
                )


def star_tp_plan(ranks: list[int]) -> dict[str, str]:
    assert len(ranks) == NUM_LAYERS
    plan = {
        "model.layers.*.self_attn.q_proj": "colwise",
        "model.layers.*.self_attn.k_proj": "colwise",
        "model.layers.*.self_attn.v_proj": "colwise",
        "model.layers.*.self_attn.v_proj.U": "colwise",
        "model.layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "model.layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "model.layers.*.self_attn.o_proj": "rowwise",
        "model.layers.*.mlp.gate_proj": "colwise",
        "model.layers.*.mlp.up_proj": "colwise",
        "model.layers.*.mlp.down_proj": "rowwise",
        "lm_head": "colwise_gather_output",
    }
    return plan


def local_weight(weight: torch.Tensor) -> torch.Tensor:
    return weight.to_local() if hasattr(weight, "to_local") else weight


class VOnlyTP8Attention(nn.Module):
    """Exact dense K; only the resident Value representation differs."""

    def __init__(
        self,
        original: nn.Module,
        *,
        arm: str,
        batch: int,
        capacity: int,
        rank: int,
        factor_root: Path | None,
    ):
        super().__init__()
        assert arm in ("basis_v64", "star_v_adaptive")
        self.arm = arm
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.q_norm = original.q_norm
        self.k_norm = original.k_norm
        self.o_proj = original.o_proj if arm == "star_v_adaptive" else None
        self.scaling = original.scaling
        self.layer_idx = original.layer_idx
        self.length = 0
        self.capacity = capacity
        device = local_weight(self.k_proj.weight).device

        if arm == "basis_v64":
            assert factor_root is not None
            factors = load_file(str(factor_root / f"layer_{self.layer_idx:03d}.safetensors"))
            encoder = factors["value_coordinate_encoders"]
            decoder = factors["head_output_decoders"]
            assert tuple(encoder.shape) == (8, HEAD_DIM, VALUE_RANK)
            assert tuple(decoder.shape) == (32, VALUE_RANK, 4096)
            source = local_weight(original.v_proj.weight)
            assert tuple(source.shape) == (HEAD_DIM, 4096)
            folded = (
                encoder[rank].to(device=device, dtype=torch.float32).T @ source.float()
            ).to(torch.bfloat16)
            self.register_buffer("folded_value_weight", folded.contiguous(), persistent=False)
            local_decoder = decoder[rank * QUERY_HEADS_PER_RANK:(rank + 1) * QUERY_HEADS_PER_RANK]
            self.register_buffer(
                "local_output_decoder",
                local_decoder.reshape(QUERY_HEADS_PER_RANK * VALUE_RANK, 4096)
                .to(device=device, dtype=torch.bfloat16).contiguous(),
                persistent=False,
            )
            value_width = VALUE_RANK
            self.v_proj = None
        else:
            self.v_proj = original.v_proj
            value_width = (
                HEAD_DIM if self.layer_idx in SKIP_LAYERS else self.v_proj.VS.out_features
            )
            self.folded_value_weight = None
            self.local_output_decoder = None

        self.register_buffer(
            "key_cache",
            torch.empty(batch, 1, capacity, HEAD_DIM, device=device, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "value_cache",
            torch.empty(batch, 1, capacity, value_width, device=device, dtype=torch.bfloat16),
            persistent=False,
        )

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask, past_key_values, kwargs
        batch, tokens, _ = hidden_states.shape
        start = self.length
        end = start + tokens
        assert end <= self.capacity
        query = self.q_norm(self.q_proj(hidden_states).view(batch, tokens, -1, HEAD_DIM)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(batch, tokens, 1, HEAD_DIM)).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        self.key_cache[:, :, start:end].copy_(key)

        if self.arm == "basis_v64":
            value = F.linear(hidden_states, self.folded_value_weight)
        elif self.layer_idx in SKIP_LAYERS:
            value = self.v_proj(hidden_states)
        else:
            value = self.v_proj.VS(hidden_states)
        self.value_cache[:, :, start:end].copy_(value[:, None])
        self.length = end

        if self.arm == "star_v_adaptive" and self.layer_idx not in SKIP_LAYERS:
            attended_value = self.v_proj.U(self.value_cache[:, 0, :end]).unsqueeze(1)
        elif self.arm == "basis_v64":
            attended_value = F.pad(self.value_cache[:, :, :end], (0, HEAD_DIM - VALUE_RANK))
        else:
            attended_value = self.value_cache[:, :, :end]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            attended = F.scaled_dot_product_attention(
                query,
                self.key_cache[:, :, :end],
                attended_value,
                attn_mask=causal_lower_right(tokens, end),
                scale=self.scaling,
                enable_gqa=True,
            )
        if self.arm == "basis_v64":
            attended = attended[..., :VALUE_RANK]
        local = attended.transpose(1, 2).reshape(batch, tokens, -1)
        if self.arm == "basis_v64":
            output = F.linear(local, self.local_output_decoder.T)
            dist.all_reduce(output)
        else:
            output = self.o_proj(local)
        return output, None

    def state_bytes(self) -> dict[str, int]:
        factor_tensors = (
            (self.folded_value_weight, self.local_output_decoder)
            if self.arm == "basis_v64"
            else tuple(local_weight(parameter) for parameter in self.v_proj.parameters())
        )
        return {
            "dense_key_cache": self.key_cache.numel() * self.key_cache.element_size(),
            "value_cache": self.value_cache.numel() * self.value_cache.element_size(),
            "value_factors": sum(t.numel() * t.element_size() for t in factor_tensors),
            "attention_workspace": 0,
            "other_persistent_cache": 0,
        }


def install_v_only(
    model: Qwen3ForCausalLM,
    *,
    arm: str,
    batch: int,
    capacity: int,
    rank: int,
    factor_root: Path | None = None,
    on_layer=None,
) -> list[VOnlyTP8Attention]:
    assert len(model.model.layers) == NUM_LAYERS
    result = []
    for index, layer in enumerate(model.model.layers):
        if on_layer is not None:
            on_layer(index)
        attention = VOnlyTP8Attention(
            layer.self_attn,
            arm=arm,
            batch=batch,
            capacity=capacity,
            rank=rank,
            factor_root=factor_root,
        )
        layer.self_attn = attention
        result.append(attention)
    return result
