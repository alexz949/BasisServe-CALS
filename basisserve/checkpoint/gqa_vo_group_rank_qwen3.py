"""Quality-reference runtime for per-KV-group GQA V/O ranks.

The module intentionally evaluates each KV group separately.  It supports
ragged value dimensions for quality experiments without imposing a serving
cache or fused-kernel layout.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch
from torch import nn

from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention
from basisserve.kernels.sageattention import sage_attention_with_compressed_value


def _replace_module(root: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parent_name, child_name = module_name.rsplit(".", 1)
    setattr(root.get_submodule(parent_name), child_name, replacement)


@dataclass(frozen=True)
class GQAVOGroupRankReplacementRecord:
    layer_index: int
    module_name: str
    head_dim: int
    value_head_dims: tuple[int, ...]
    dense_v_width: int
    compressed_v_width: int
    dense_o_width: int
    compressed_o_width: int


class RaggedGQATiedVOQwen3Attention(nn.Module):
    """Qwen3 eager attention with a distinct value rank per KV group."""

    def __init__(
        self,
        base_attention: nn.Module,
        *,
        v_group_weights: Sequence[torch.Tensor],
        o_group_weights: Sequence[torch.Tensor],
        v_group_biases: Sequence[torch.Tensor | None] | None = None,
        o_decoder_bias: torch.Tensor | None = None,
        attention_backend: str = "native",
    ) -> None:
        super().__init__()
        config = base_attention.config
        self.config = config
        self.layer_idx = int(base_attention.layer_idx)
        self.layer_type = getattr(base_attention, "layer_type", None)
        self.head_dim = int(base_attention.head_dim)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = int(base_attention.num_key_value_groups)
        self.scaling = float(base_attention.scaling)
        self.attention_dropout = float(base_attention.attention_dropout)
        self.is_causal = bool(base_attention.is_causal)
        self.sliding_window = getattr(base_attention, "sliding_window", None)
        self.attn_implementation = str(
            getattr(config, "_attn_implementation", "eager") or "eager"
        )
        self.attention_backend = str(attention_backend)
        if self.attention_backend not in {"native", "sage"}:
            raise ValueError(
                "compressed attention backend must be 'native' or 'sage', got "
                f"{self.attention_backend!r}"
            )
        if self.attn_implementation not in {"eager", "sdpa"}:
            raise ValueError(
                "ragged Qwen3 attention supports only eager or sdpa, got "
                f"{self.attn_implementation!r}"
            )

        if len(v_group_weights) != self.num_key_value_heads:
            raise ValueError("V group count does not match num_key_value_heads")
        if len(o_group_weights) != self.num_key_value_heads:
            raise ValueError("O group count does not match num_key_value_heads")
        if v_group_biases is None:
            v_group_biases = [None] * self.num_key_value_heads
        if len(v_group_biases) != self.num_key_value_heads:
            raise ValueError("V bias group count does not match num_key_value_heads")

        hidden_size = int(config.hidden_size)
        device = base_attention.q_proj.weight.device
        dtype = base_attention.q_proj.weight.dtype
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm
        self.v_group_projs = nn.ModuleList()
        self.o_group_projs = nn.ModuleList()
        value_head_dims: list[int] = []
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
            v_proj = nn.Linear(
                hidden_size,
                rank,
                bias=v_bias is not None,
                device=device,
                dtype=dtype,
            )
            o_proj = nn.Linear(
                expected_o[1],
                hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
            )
            v_proj.weight.data.copy_(v_weight.to(device=device, dtype=dtype))
            o_proj.weight.data.copy_(o_weight.to(device=device, dtype=dtype))
            if v_proj.bias is not None:
                v_proj.bias.data.copy_(v_bias.to(device=device, dtype=dtype))
            self.v_group_projs.append(v_proj)
            self.o_group_projs.append(o_proj)
            value_head_dims.append(rank)
        self.value_head_dims = tuple(value_head_dims)
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
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: object | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if past_key_values is not None:
            raise NotImplementedError(
                "per-KV-group rank reference attention requires use_cache=False"
            )
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        if self.attn_implementation == "sdpa":
            from transformers.integrations.sdpa_attention import (
                sdpa_attention_forward as attention_forward,
            )
        else:
            from transformers.models.qwen3.modeling_qwen3 import (
                eager_attention_forward as attention_forward,
            )

        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape, self.num_attention_heads, self.head_dim)
        key_shape = (*input_shape, self.num_key_value_heads, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(query_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(key_shape)).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        output: torch.Tensor | None = None
        attention_weights: list[torch.Tensor] = []
        heads_per_group = self.num_key_value_groups
        for group_index, (v_proj, o_proj) in enumerate(
            zip(self.v_group_projs, self.o_group_projs)
        ):
            first_head = group_index * heads_per_group
            group_query = query_states[:, first_head : first_head + heads_per_group]
            group_key = key_states[:, group_index : group_index + 1]
            group_value = v_proj(hidden_states).unsqueeze(1)
            group_mask = attention_mask
            if (
                group_mask is not None
                and group_mask.ndim >= 4
                and group_mask.shape[1] == self.num_attention_heads
            ):
                group_mask = group_mask[
                    :, first_head : first_head + heads_per_group
                ]
            if self.attention_backend == "sage":
                if self.training and self.attention_dropout:
                    raise NotImplementedError(
                        "SageAttention compressed backend is inference-only"
                    )
                if kwargs.get("output_attentions"):
                    raise NotImplementedError(
                        "SageAttention compressed backend does not return attention weights"
                    )
                if self.sliding_window is not None and group_mask is None:
                    raise NotImplementedError(
                        "SageAttention requires an explicit mask for sliding-window attention"
                    )
                group_output = sage_attention_with_compressed_value(
                    group_query,
                    group_key,
                    group_value,
                    attention_mask=group_mask,
                    is_causal=bool(
                        self.is_causal
                        and group_mask is None
                        and group_query.shape[-2] > 1
                    ),
                    scaling=self.scaling,
                )
                group_weights = None
            else:
                group_output, group_weights = attention_forward(
                    self,
                    group_query,
                    group_key,
                    group_value,
                    group_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    sliding_window=self.sliding_window,
                    **kwargs,
                )
            projected = o_proj(group_output.reshape(*input_shape, -1))
            output = projected if output is None else output + projected
            if group_weights is not None:
                attention_weights.append(group_weights)

        if output is None:
            raise RuntimeError("attention has no KV groups")
        if self.o_decoder_bias is not None:
            output = output + self.o_decoder_bias
        weights = torch.cat(attention_weights, dim=1) if attention_weights else None
        return output, weights


class Qwen3GQAVOGroupRankBankRuntime:
    """Install ragged per-KV-group ranks from an existing A3 rank bank."""

    def __init__(
        self,
        model: nn.Module,
        rank_bank: str | Path,
        *,
        attention_backend: str = "native",
    ) -> None:
        self.model = model
        self.attention_backend = str(attention_backend)
        if self.attention_backend not in {"native", "sage"}:
            raise ValueError(
                "compressed attention backend must be 'native' or 'sage', got "
                f"{self.attention_backend!r}"
            )
        self.rank_bank = Path(rank_bank).expanduser().resolve()
        bank_config = json.loads(
            (self.rank_bank / "config.json").read_text(encoding="utf-8")
        )
        if bank_config.get("format") not in {
            "basisserve.a3_gqa_vo.rank_bank.v1",
            "basisserve.value_interface.composite_ragged_bank.v1",
        }:
            raise ValueError(f"unsupported rank bank: {self.rank_bank}")
        self.full_rank_factor_overrides = bool(
            bank_config.get("full_rank_factor_overrides", False)
        )
        self.profile = json.loads(
            (self.rank_bank / bank_config["profile"]).read_text(encoding="utf-8")
        )
        config = self.profile["model_config"]
        self.num_layers = int(config["num_layers"])
        self.hidden_size = int(config["hidden_size"])
        self.num_query_heads = int(config["num_query_heads"])
        self.num_kv_heads = int(config["num_kv_heads"])
        self.head_dim = int(config["head_dim"])
        self.heads_per_group = self.num_query_heads // self.num_kv_heads
        self.candidate_ranks = tuple(int(rank) for rank in self.profile["candidate_ranks"])
        if self.head_dim not in self.candidate_ranks:
            raise ValueError("rank bank must include full head_dim")

        expected = (
            int(model.config.num_hidden_layers),
            int(model.config.hidden_size),
            int(model.config.num_attention_heads),
            int(model.config.num_key_value_heads),
            int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)),
        )
        actual = (
            self.num_layers,
            self.hidden_size,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        if actual != expected:
            raise ValueError(f"rank-bank/model config mismatch: {actual} vs {expected}")

        self.module_names: list[str] = []
        self._dense_modules: list[nn.Module] = []
        for layer_index in range(self.num_layers):
            module_name = str(self.profile["layers"][str(layer_index)]["module_name"])
            module = model.get_submodule(module_name)
            if isinstance(module, (GQATiedVOQwen3Attention, RaggedGQATiedVOQwen3Attention)):
                raise ValueError(f"model layer is already compressed: {module_name}")
            self.module_names.append(module_name)
            self._dense_modules.append(module)
        self._current_ranks = [
            [self.head_dim] * self.num_kv_heads for _ in range(self.num_layers)
        ]
        self._payload_cache: dict[tuple[int, int], dict] = {}

    @property
    def current_schedule(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(ranks) for ranks in self._current_ranks)

    def _factor_payload(self, layer_index: int, rank: int) -> dict:
        key = (layer_index, rank)
        if key in self._payload_cache:
            return self._payload_cache[key]
        entry = self.profile["layers"][str(layer_index)]["ranks"].get(str(rank))
        if entry is None:
            raise ValueError(f"layer {layer_index} has no rank-{rank} factor")
        payload = torch.load(
            self.rank_bank / entry["factor_path"],
            map_location="cpu",
            weights_only=True,
        )
        if payload.get("format") != "basisserve.gqa_vo_svdllm.layer.v1":
            raise ValueError(f"unsupported factor payload for layer {layer_index}")
        if int(payload.get("rank_per_kv_head", -1)) != rank:
            raise ValueError(f"factor rank mismatch for layer {layer_index}")
        self._payload_cache[key] = payload
        return payload

    def _group_factors(
        self, layer_index: int, group_index: int, rank: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        dense = self._dense_modules[layer_index]
        rank_entry = self.profile["layers"][str(layer_index)]["ranks"].get(
            str(rank)
        )
        use_full_rank_override = bool(
            rank == self.head_dim
            and self.full_rank_factor_overrides
            and rank_entry is not None
        )
        if rank == self.head_dim and not use_full_rank_override:
            v_start = group_index * self.head_dim
            first_head = group_index * self.heads_per_group
            o_start = first_head * self.head_dim
            v_bias = getattr(dense.v_proj, "bias", None)
            return (
                dense.v_proj.weight.detach()[v_start : v_start + self.head_dim],
                dense.o_proj.weight.detach()[
                    :, o_start : o_start + self.heads_per_group * self.head_dim
                ],
                None if v_bias is None else v_bias.detach()[v_start : v_start + self.head_dim],
            )
        payload = self._factor_payload(layer_index, rank)
        v_start = group_index * rank
        first_head = group_index * self.heads_per_group
        o_start = first_head * rank
        v_bias = payload.get("v_proj_compressed_bias")
        return (
            payload["v_proj_compressed_weight"][v_start : v_start + rank],
            payload["o_decoder_weight"][
                :, o_start : o_start + self.heads_per_group * rank
            ],
            None if v_bias is None else v_bias[v_start : v_start + rank],
        )

    def _layer_has_full_rank_override(self, layer_index: int) -> bool:
        return bool(
            self.full_rank_factor_overrides
            and str(self.head_dim)
            in self.profile["layers"][str(layer_index)]["ranks"]
        )

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
            current = self.model.get_submodule(self.module_names[layer_index])
            needs_full_rank_override = bool(
                all(rank == self.head_dim for rank in selected)
                and self._layer_has_full_rank_override(layer_index)
                and not isinstance(
                    current,
                    (GQATiedVOQwen3Attention, RaggedGQATiedVOQwen3Attention),
                )
            )
            if not needs_full_rank_override and (
                self.attention_backend != "sage"
                or isinstance(
                    current,
                    (GQATiedVOQwen3Attention, RaggedGQATiedVOQwen3Attention),
                )
            ):
                return
        dense = self._dense_modules[layer_index]
        if (
            self.attention_backend == "native"
            and all(rank == self.head_dim for rank in selected)
            and not self._layer_has_full_rank_override(layer_index)
        ):
            replacement = dense
        elif len(set(selected)) == 1:
            rank = selected[0]
            payload = self._factor_payload(layer_index, rank)
            replacement = GQATiedVOQwen3Attention(
                dense,
                v_proj_compressed_weight=payload["v_proj_compressed_weight"],
                o_decoder_weight=payload["o_decoder_weight"],
                v_proj_compressed_bias=payload.get("v_proj_compressed_bias"),
                o_decoder_bias=payload.get("o_decoder_bias"),
                attention_backend=self.attention_backend,
            )
        else:
            groups = [
                self._group_factors(layer_index, group_index, rank)
                for group_index, rank in enumerate(selected)
            ]
            replacement = RaggedGQATiedVOQwen3Attention(
                dense,
                v_group_weights=[item[0] for item in groups],
                o_group_weights=[item[1] for item in groups],
                v_group_biases=[item[2] for item in groups],
                o_decoder_bias=getattr(dense.o_proj, "bias", None),
                attention_backend=self.attention_backend,
            )
        _replace_module(self.model, self.module_names[layer_index], replacement)
        self._current_ranks[layer_index] = list(selected)

    @torch.no_grad()
    def apply_schedule(self, ranks: Sequence[Sequence[int]]) -> None:
        selected = tuple(tuple(int(rank) for rank in layer) for layer in ranks)
        if len(selected) != self.num_layers:
            raise ValueError("group schedule layer count does not match the model")
        for layer_index, layer_ranks in enumerate(selected):
            self.set_layer_ranks(layer_index, layer_ranks)

    @torch.no_grad()
    def restore_dense(self) -> None:
        for layer_index, dense in enumerate(self._dense_modules):
            _replace_module(self.model, self.module_names[layer_index], dense)
            self._current_ranks[layer_index] = [self.head_dim] * self.num_kv_heads

    @contextmanager
    @torch.no_grad()
    def temporary_layer_factors(
        self,
        layer_index: int,
        *,
        ranks: Sequence[int],
        v_group_weights: Sequence[torch.Tensor],
        o_group_weights: Sequence[torch.Tensor],
        v_group_biases: Sequence[torch.Tensor | None] | None = None,
        o_decoder_bias: torch.Tensor | None = None,
    ) -> Iterator[None]:
        """Temporarily overlay one explicitly materialized ragged layer.

        This is used by decoder-closed rank-candidate scoring.  The remaining
        layers stay installed from the immutable base rank bank, while the
        affected layer uses candidate-specific folded V factors and the
        jointly solved full-layer decoder.
        """

        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"layer index out of range: {layer_index}")
        selected = tuple(int(rank) for rank in ranks)
        if len(selected) != self.num_kv_heads:
            raise ValueError("group-rank count does not match num_kv_heads")
        if any(not 0 < rank <= self.head_dim for rank in selected):
            raise ValueError("candidate layer uses an invalid Value rank")
        if len(v_group_weights) != self.num_kv_heads:
            raise ValueError("candidate V group count does not match the model")
        if len(o_group_weights) != self.num_kv_heads:
            raise ValueError("candidate O group count does not match the model")
        for group, rank in enumerate(selected):
            if tuple(v_group_weights[group].shape) != (rank, self.hidden_size):
                raise ValueError(f"invalid candidate V shape for group {group}")
            expected_o = (
                self.hidden_size,
                self.heads_per_group * rank,
            )
            if tuple(o_group_weights[group].shape) != expected_o:
                raise ValueError(f"invalid candidate O shape for group {group}")
        if v_group_biases is not None and len(v_group_biases) != self.num_kv_heads:
            raise ValueError("candidate V bias group count does not match the model")

        previous_module = self.model.get_submodule(self.module_names[layer_index])
        previous_ranks = tuple(self._current_ranks[layer_index])
        dense = self._dense_modules[layer_index]
        replacement = RaggedGQATiedVOQwen3Attention(
            dense,
            v_group_weights=v_group_weights,
            o_group_weights=o_group_weights,
            v_group_biases=v_group_biases,
            o_decoder_bias=o_decoder_bias,
            attention_backend=self.attention_backend,
        )
        try:
            _replace_module(self.model, self.module_names[layer_index], replacement)
            self._current_ranks[layer_index] = list(selected)
            yield
        finally:
            _replace_module(self.model, self.module_names[layer_index], previous_module)
            self._current_ranks[layer_index] = list(previous_ranks)

    @contextmanager
    def temporary_group_rank(
        self, layer_index: int, group_index: int, rank: int
    ) -> Iterator[None]:
        if not 0 <= group_index < self.num_kv_heads:
            raise IndexError(f"KV group index out of range: {group_index}")
        previous_ranks = tuple(self._current_ranks[layer_index])
        previous_module = self.model.get_submodule(self.module_names[layer_index])
        updated = list(previous_ranks)
        updated[group_index] = int(rank)
        try:
            self.set_layer_ranks(layer_index, updated)
            yield
        finally:
            _replace_module(self.model, self.module_names[layer_index], previous_module)
            self._current_ranks[layer_index] = list(previous_ranks)

@torch.no_grad()
def install_qwen3_gqa_vo_group_rank_schedule(
    model: nn.Module,
    *,
    rank_bank: str | Path,
    schedule_path: str | Path,
    attention_backend: str = "native",
) -> list[GQAVOGroupRankReplacementRecord]:
    schedule = json.loads(
        Path(schedule_path).expanduser().resolve().read_text(encoding="utf-8")
    )
    if schedule.get("format") not in {
        "basisserve.a3_gqa_vo.group_rank_schedule.v1",
        "basisserve.value_interface.metric_rank_schedule.v1",
    }:
        raise ValueError(f"unsupported group-rank schedule: {schedule_path}")
    runtime = Qwen3GQAVOGroupRankBankRuntime(
        model,
        rank_bank,
        attention_backend=attention_backend,
    )
    selected = schedule["selected_ranks"]
    runtime.apply_schedule(selected)
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
