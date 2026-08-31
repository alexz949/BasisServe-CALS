"""Quality-reference runtime for per-KV-group Llama GQA V/O ranks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from basisserve.checkpoint.gqa_vo_group_rank_qwen3 import (
    GQAVOGroupRankReplacementRecord,
    Qwen3GQAVOGroupRankBankRuntime,
    _replace_module,
)


class RaggedGQATiedVOLlamaAttention(nn.Module):
    """Llama attention with a distinct Value rank per KV group.

    Group latent outputs are concatenated in native head order and decoded by
    one joint output GEMM.  This matches the operation order of native Llama
    attention and avoids numerically fragile sequential FP16 source sums.
    """

    def __init__(
        self,
        base_attention: nn.Module,
        *,
        v_group_weights: Sequence[torch.Tensor],
        o_group_weights: Sequence[torch.Tensor],
        v_group_biases: Sequence[torch.Tensor | None] | None = None,
        o_decoder_bias: torch.Tensor | None = None,
        source_group_size: int = 1,
    ) -> None:
        super().__init__()
        config = base_attention.config
        self.config = config
        self.layer_idx = int(base_attention.layer_idx)
        self.head_dim = int(base_attention.head_dim)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = int(base_attention.num_key_value_groups)
        self.scaling = float(base_attention.scaling)
        self.attention_dropout = float(base_attention.attention_dropout)
        self.is_causal = bool(getattr(base_attention, "is_causal", True))

        if len(v_group_weights) != self.num_key_value_heads:
            raise ValueError("V group count does not match num_key_value_heads")
        if len(o_group_weights) != self.num_key_value_heads:
            raise ValueError("O group count does not match num_key_value_heads")
        if v_group_biases is None:
            v_group_biases = [None] * self.num_key_value_heads
        if len(v_group_biases) != self.num_key_value_heads:
            raise ValueError("V bias group count does not match num_key_value_heads")
        self.source_group_size = int(source_group_size)
        if (
            self.source_group_size <= 0
            or self.num_key_value_heads % self.source_group_size
        ):
            raise ValueError("source group size must divide the KV-head count")

        hidden_size = int(config.hidden_size)
        device = base_attention.q_proj.weight.device
        dtype = base_attention.q_proj.weight.dtype
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        checked_v_weights = []
        checked_v_biases = []
        decoder_weights = []
        value_head_dims = []
        for group_index, (v_weight, o_weight, v_bias) in enumerate(
            zip(v_group_weights, o_group_weights, v_group_biases)
        ):
            if v_weight.ndim != 2 or v_weight.shape[1] != hidden_size:
                raise ValueError(f"invalid V weight for group {group_index}: {v_weight.shape}")
            rank = int(v_weight.shape[0])
            expected_o = (hidden_size, self.num_key_value_groups * rank)
            if tuple(o_weight.shape) != expected_o:
                raise ValueError(
                    f"invalid O weight for group {group_index}: {tuple(o_weight.shape)} "
                    f"vs {expected_o}"
                )
            if v_bias is not None and tuple(v_bias.shape) != (rank,):
                raise ValueError(f"invalid V bias for group {group_index}")
            checked_v_weights.append(v_weight.to(device=device, dtype=dtype))
            checked_v_biases.append(
                None if v_bias is None else v_bias.to(device=device, dtype=dtype)
            )
            decoder_weights.append(o_weight.to(device=device, dtype=dtype))
            value_head_dims.append(rank)
        self.value_head_dims = tuple(value_head_dims)
        self.v_group_projs = nn.ModuleList()
        self.source_value_dims = []
        for first in range(0, self.num_key_value_heads, self.source_group_size):
            stop = first + self.source_group_size
            block_ranks = self.value_head_dims[first:stop]
            if len(set(block_ranks)) != 1:
                raise ValueError("all KV heads inside one source must share one rank")
            rank = block_ranks[0]
            block_biases = checked_v_biases[first:stop]
            if any(bias is None for bias in block_biases) and not all(
                bias is None for bias in block_biases
            ):
                raise ValueError("Value biases must be consistently present inside a source")
            projection = nn.Linear(
                hidden_size,
                self.source_group_size * rank,
                bias=block_biases[0] is not None,
                device=device,
                dtype=dtype,
            )
            projection.weight.data.copy_(
                torch.cat(checked_v_weights[first:stop], dim=0)
            )
            if projection.bias is not None:
                projection.bias.data.copy_(torch.cat(block_biases, dim=0))
            self.v_group_projs.append(projection)
            self.source_value_dims.append(rank)
        self.source_value_dims = tuple(self.source_value_dims)
        self.o_proj = nn.Linear(
            sum(self.num_key_value_groups * rank for rank in self.value_head_dims),
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.o_proj.weight.data.copy_(torch.cat(decoder_weights, dim=1))
        if o_decoder_bias is None:
            self.register_parameter("o_decoder_bias", None)
        else:
            self.o_decoder_bias = nn.Parameter(
                o_decoder_bias.detach().to(device=device, dtype=dtype).clone(),
                requires_grad=False,
            )
        self.train(base_attention.training)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if past_key_values is not None:
            raise NotImplementedError(
                "per-KV-group rank reference attention requires use_cache=False"
            )
        if position_embeddings is None:
            raise ValueError("Llama attention requires precomputed position embeddings")
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape, self.num_attention_heads, self.head_dim)
        key_shape = (*input_shape, self.num_key_value_heads, self.head_dim)
        query_states = self.q_proj(hidden_states).view(query_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(key_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        group_outputs = []
        attention_weights = []
        heads_per_group = self.num_key_value_groups
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.llama.modeling_llama import eager_attention_forward

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        for source_index, (v_proj, value_dim) in enumerate(
            zip(self.v_group_projs, self.source_value_dims, strict=True)
        ):
            first_group = source_index * self.source_group_size
            first_head = first_group * heads_per_group
            source_heads = self.source_group_size * heads_per_group
            group_query = query_states[:, first_head : first_head + source_heads]
            group_key = key_states[
                :, first_group : first_group + self.source_group_size
            ]
            group_value = (
                v_proj(hidden_states)
                .view(*input_shape, self.source_group_size, value_dim)
                .transpose(1, 2)
            )
            group_output, group_weights = attention_interface(
                self,
                group_query,
                group_key,
                group_value,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
            group_outputs.append(group_output.reshape(*input_shape, -1))
            if group_weights is not None:
                attention_weights.append(group_weights)

        if not group_outputs:
            raise RuntimeError("attention has no KV groups")
        latent = torch.cat(group_outputs, dim=-1)
        output = self.o_proj(latent)
        if self.o_decoder_bias is not None:
            output = output + self.o_decoder_bias
        weights = torch.cat(attention_weights, dim=1) if attention_weights else None
        return output, weights


class LlamaGQAVOGroupRankBankRuntime(Qwen3GQAVOGroupRankBankRuntime):
    """Install ragged Llama ranks from an A3 GQA V/O rank bank."""

    def __init__(self, model: nn.Module, rank_bank: str | Path) -> None:
        if getattr(model.config, "model_type", None) != "llama":
            raise ValueError("Llama rank-bank runtime requires a Llama model")
        super().__init__(model, rank_bank)
        if any(
            isinstance(module, RaggedGQATiedVOLlamaAttention)
            for module in self._dense_modules
        ):
            raise ValueError("model already contains compressed Llama attention")

    @torch.no_grad()
    def set_layer_ranks(self, layer_index: int, ranks: Sequence[int]) -> None:
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"layer index out of range: {layer_index}")
        selected = tuple(int(rank) for rank in ranks)
        if len(selected) != self.num_kv_heads:
            raise ValueError("group-rank count does not match num_kv_heads")
        if any(rank not in self.candidate_ranks for rank in selected):
            raise ValueError("group schedule contains a rank absent from the rank bank")
        if tuple(self._current_ranks[layer_index]) == selected:
            return
        dense = self._dense_modules[layer_index]
        if all(rank == self.head_dim for rank in selected):
            replacement = dense
        else:
            groups = [
                self._group_factors(layer_index, group_index, rank)
                for group_index, rank in enumerate(selected)
            ]
            replacement = RaggedGQATiedVOLlamaAttention(
                dense,
                v_group_weights=[item[0] for item in groups],
                o_group_weights=[item[1] for item in groups],
                v_group_biases=[item[2] for item in groups],
                o_decoder_bias=getattr(dense.o_proj, "bias", None),
            )
        _replace_module(self.model, self.module_names[layer_index], replacement)
        self._current_ranks[layer_index] = list(selected)


@torch.no_grad()
def install_llama_gqa_vo_group_rank_schedule(
    model: nn.Module,
    *,
    rank_bank: str | Path,
    schedule_path: str | Path,
) -> list[GQAVOGroupRankReplacementRecord]:
    schedule = json.loads(
        Path(schedule_path).expanduser().resolve().read_text(encoding="utf-8")
    )
    if schedule.get("format") != "basisserve.a3_gqa_vo.group_rank_schedule.v1":
        raise ValueError(f"unsupported group-rank schedule: {schedule_path}")
    runtime = LlamaGQAVOGroupRankBankRuntime(model, rank_bank)
    runtime.apply_schedule(schedule["selected_ranks"])
    records = []
    for layer_index, ranks in enumerate(runtime.current_schedule):
        if all(rank == runtime.head_dim for rank in ranks):
            continue
        records.append(
            GQAVOGroupRankReplacementRecord(
                layer_index=layer_index,
                module_name=runtime.module_names[layer_index],
                head_dim=runtime.head_dim,
                value_head_dims=ranks,
                dense_v_width=runtime.num_kv_heads * runtime.head_dim,
                compressed_v_width=sum(ranks),
                dense_o_width=runtime.num_query_heads * runtime.head_dim,
                compressed_o_width=runtime.heads_per_group * sum(ranks),
            )
        )
    return records
