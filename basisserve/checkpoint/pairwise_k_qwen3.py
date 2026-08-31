"""Reference Qwen3 runtime for pairwise cross-layer historical Key caches.

The reference deliberately leaves Value dense.  Hugging Face ``DynamicCache``
also retains exact Keys so the evaluator can use its normal cache/mask API;
candidate attention ignores those historical exact Keys.  Reported storage is
therefore logical rather than a physical memory benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from basisserve.core.pairwise_k_sparse import (
    PairwiseQuestConfig,
    pairwise_quest_sparse_attention,
)


HistoricalKMode = Literal["independent", "pairwise"]


class PairwiseKCacheState:
    """Completed history shared by one adjacent layer pair."""

    def __init__(self) -> None:
        self.pair_code: Tensor | None = None
        self.independent_code: list[Tensor | None] = [None, None]
        self.pending_layer0: Tensor | None = None

    def reset(self) -> None:
        self.pair_code = None
        self.independent_code = [None, None]
        self.pending_layer0 = None

    @staticmethod
    def _length(code: Tensor | None) -> int:
        return 0 if code is None else int(code.shape[2])

    def history(self, mode: HistoricalKMode, layer_slot: int) -> Tensor | None:
        if layer_slot not in (0, 1):
            raise ValueError("layer_slot must be zero or one")
        return (
            self.pair_code
            if mode == "pairwise"
            else self.independent_code[layer_slot]
        )

    def history_length(self, mode: HistoricalKMode, layer_slot: int) -> int:
        return self._length(self.history(mode, layer_slot))

    @staticmethod
    def _append(previous: Tensor | None, current: Tensor) -> Tensor:
        if current.ndim != 4:
            raise ValueError("compressed Key contribution must be rank four")
        if previous is None:
            return current.detach()
        if (
            previous.shape[:2] != current.shape[:2]
            or previous.shape[3] != current.shape[3]
        ):
            raise ValueError("compressed Key cache geometry changed")
        return torch.cat((previous, current.detach()), dim=2)

    def append(
        self,
        *,
        layer_slot: int,
        independent_contribution: Tensor,
        pair_contribution: Tensor,
    ) -> None:
        if layer_slot not in (0, 1):
            raise ValueError("layer_slot must be zero or one")
        expected_independent = self._length(self.independent_code[layer_slot])
        expected_pair = self._length(self.pair_code)
        if expected_independent != expected_pair:
            raise RuntimeError("pair and independent cache histories diverged")
        self.independent_code[layer_slot] = self._append(
            self.independent_code[layer_slot],
            independent_contribution,
        )
        if layer_slot == 0:
            if self.pending_layer0 is not None:
                raise RuntimeError("layer-0 pair contribution is already pending")
            self.pending_layer0 = pair_contribution.detach()
            return
        if self.pending_layer0 is None:
            raise RuntimeError("layer 1 cannot finalize without layer-0 contribution")
        if self.pending_layer0.shape != pair_contribution.shape:
            raise ValueError("pair layer contributions have different geometry")
        combined = self.pending_layer0 + pair_contribution
        self.pair_code = self._append(self.pair_code, combined)
        self.pending_layer0 = None
        completed = self._length(self.pair_code)
        if any(self._length(code) != completed for code in self.independent_code):
            raise RuntimeError("completed pair and independent cache lengths differ")


class PairwiseKQwen3Attention(nn.Module):
    """Dense-V Qwen3 attention with independent or pairwise historical K."""

    def __init__(
        self,
        base_attention: nn.Module,
        *,
        state: PairwiseKCacheState,
        layer_slot: int,
        independent_key_projector: Tensor,
        independent_query_projector: Tensor,
        pair_key_encoder: Tensor,
        pair_query_projector: Tensor,
    ) -> None:
        super().__init__()
        config = base_attention.config
        self.config = config
        self.layer_idx = int(base_attention.layer_idx)
        self.layer_slot = int(layer_slot)
        self.state = state
        self.head_dim = int(base_attention.head_dim)
        self.value_head_dim = int(
            getattr(base_attention, "value_head_dim", self.head_dim)
        )
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = int(base_attention.num_key_value_groups)
        self.scaling = float(base_attention.scaling)
        self.attention_dropout = float(base_attention.attention_dropout)
        self.is_causal = bool(base_attention.is_causal)
        self.sliding_window = base_attention.sliding_window
        if self.layer_slot not in (0, 1):
            raise ValueError("layer_slot must be zero or one")
        if self.sliding_window is not None:
            raise NotImplementedError("pairwise Key reference does not support sliding window")

        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.v_proj = base_attention.v_proj
        self.o_proj = base_attention.o_proj
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm
        self.runtime_mode: HistoricalKMode = "pairwise"
        self._register_factor(
            "independent_key_projector",
            independent_key_projector,
            (self.num_key_value_heads, self.head_dim),
        )
        self._register_factor(
            "independent_query_projector",
            independent_query_projector,
            (self.num_attention_heads, self.head_dim),
        )
        self._register_factor(
            "pair_key_encoder",
            pair_key_encoder,
            (self.num_key_value_heads, self.head_dim),
        )
        self._register_factor(
            "pair_query_projector",
            pair_query_projector,
            (self.num_attention_heads, self.head_dim),
        )
        if self.independent_key_projector.shape[-1] != self.independent_query_projector.shape[-1]:
            raise ValueError("independent Key and Query ranks differ")
        if self.pair_key_encoder.shape[-1] != self.pair_query_projector.shape[-1]:
            raise ValueError("pair Key and Query ranks differ")
        self.sparse_config: PairwiseQuestConfig | None = None
        self.reset_sparse_statistics()
        self.train(base_attention.training)

    def _register_factor(self, name: str, factor: Tensor, prefix: tuple[int, int]) -> None:
        if factor.ndim != 3 or tuple(factor.shape[:2]) != prefix:
            raise ValueError(f"{name} has incompatible geometry")
        dtype = self.q_proj.weight.dtype
        device = self.q_proj.weight.device
        self.register_buffer(
            name,
            factor.detach().to(device=device, dtype=dtype).contiguous(),
            persistent=False,
        )

    def set_runtime_mode(self, mode: HistoricalKMode) -> None:
        if mode not in ("independent", "pairwise"):
            raise ValueError(f"unknown historical Key mode {mode!r}")
        self.runtime_mode = mode

    def set_sparse_config(self, config: PairwiseQuestConfig | None) -> None:
        if config is not None:
            latent_dim = (
                int(self.pair_query_projector.shape[-1])
                if self.runtime_mode == "pairwise"
                else int(self.independent_query_projector.shape[-1])
            )
            config.validate(latent_dim)
        self.sparse_config = config

    def reset_sparse_statistics(self) -> None:
        self._sparse_totals = {
            "query_positions": 0.0,
            "query_heads": 0.0,
            "selected_page_instances": 0.0,
            "available_page_instances": 0.0,
            "selected_physical_tokens": 0.0,
            "available_physical_tokens": 0.0,
            "selection_bound_flops": 0.0,
            "sparse_historical_qk_flops": 0.0,
            "exact_current_qk_flops": 0.0,
            "sparse_historical_pv_flops": 0.0,
            "exact_current_pv_flops": 0.0,
        }
        self._sparse_maxima = {
            "history_length": 0.0,
            "page_count": 0.0,
            "resident_selector_metadata_bytes": 0.0,
        }

    def _record_sparse_statistics(self, statistics: dict[str, float]) -> None:
        for name in self._sparse_totals:
            self._sparse_totals[name] += float(statistics[name])
        for name in self._sparse_maxima:
            self._sparse_maxima[name] = max(
                self._sparse_maxima[name],
                float(statistics[name]),
            )

    def sparse_statistics(self) -> dict[str, float]:
        selected = self._sparse_totals["selected_physical_tokens"]
        available = self._sparse_totals["available_physical_tokens"]
        return {
            **self._sparse_totals,
            **self._sparse_maxima,
            "selected_token_fraction": selected / available if available else 0.0,
        }

    @staticmethod
    def _repeat_kv(states: Tensor, groups: int) -> Tensor:
        if groups == 1:
            return states
        batch, kv_heads, tokens, width = states.shape
        return (
            states[:, :, None, :, :]
            .expand(batch, kv_heads, groups, tokens, width)
            .reshape(batch, kv_heads * groups, tokens, width)
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        past_key_values: Any | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor | None]:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        if past_key_values is None:
            raise ValueError("pairwise Key attention requires a native exact reference cache")
        input_shape = hidden_states.shape[:-1]
        query_length = int(hidden_states.shape[1])
        query_shape = (*input_shape, self.num_attention_heads, self.head_dim)
        key_shape = (*input_shape, self.num_key_value_heads, self.head_dim)
        value_shape = (
            *input_shape,
            self.num_key_value_heads,
            self.value_head_dim,
        )
        query_states = self.q_norm(self.q_proj(hidden_states).view(query_shape)).transpose(1, 2)
        current_key = self.k_norm(self.k_proj(hidden_states).view(key_shape)).transpose(1, 2)
        current_value = self.v_proj(hidden_states).view(value_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, current_key = apply_rotary_pos_emb(
            query_states,
            current_key,
            cos,
            sin,
        )

        history = self.state.history(self.runtime_mode, self.layer_slot)
        history_length = self.state.history_length(self.runtime_mode, self.layer_slot)
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        exact_key, dense_value = past_key_values.update(
            current_key,
            current_value,
            self.layer_idx,
            cache_kwargs,
        )
        total_length = history_length + query_length
        if int(exact_key.shape[2]) != total_length or int(dense_value.shape[2]) != total_length:
            raise RuntimeError("native cache and pairwise Key history lengths differ")

        probabilities: Tensor | None = None
        query_projector = (
            self.pair_query_projector
            if self.runtime_mode == "pairwise"
            else self.independent_query_projector
        )
        if history is not None and self.sparse_config is not None:
            if self.training:
                raise NotImplementedError("pairwise QUEST reference is inference-only")
            if kwargs.get("output_attentions"):
                raise NotImplementedError(
                    "pairwise QUEST does not materialize dense attention weights"
                )
            projected_query = torch.einsum(
                "bhqd,hdr->bhqr",
                query_states,
                query_projector,
            )
            sparse_result = pairwise_quest_sparse_attention(
                query_states,
                projected_query,
                history,
                current_key,
                dense_value,
                self.sparse_config,
                scaling=self.scaling,
                attention_mask=attention_mask,
            )
            self._record_sparse_statistics(sparse_result.statistics)
            output = sparse_result.output
        else:
            current_repeated_key = self._repeat_kv(
                current_key,
                self.num_key_value_groups,
            )
            current_scores = torch.matmul(
                query_states,
                current_repeated_key.transpose(2, 3),
            ).mul_(self.scaling)
            if history is None:
                scores = current_scores
            else:
                projected_query = torch.einsum(
                    "bhqd,hdr->bhqr",
                    query_states,
                    query_projector,
                )
                repeated_history = self._repeat_kv(
                    history,
                    self.num_key_value_groups,
                )
                historical_scores = torch.matmul(
                    projected_query,
                    repeated_history.transpose(2, 3),
                ).mul_(self.scaling)
                scores = torch.cat((historical_scores, current_scores), dim=-1)
            if attention_mask is not None:
                scores = scores + attention_mask[..., :total_length]
            elif query_length > 1:
                key_positions = torch.arange(total_length, device=scores.device)
                query_positions = history_length + torch.arange(
                    query_length,
                    device=scores.device,
                )
                valid = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                scores = scores.masked_fill(
                    ~valid.view(1, 1, *valid.shape),
                    -torch.inf,
                )
            probabilities = F.softmax(
                scores,
                dim=-1,
                dtype=torch.float32,
            ).to(query_states.dtype)
            if self.attention_dropout and self.training:
                probabilities = F.dropout(
                    probabilities,
                    p=self.attention_dropout,
                    training=True,
                )
            repeated_value = self._repeat_kv(
                dense_value,
                self.num_key_value_groups,
            )
            output = torch.matmul(probabilities, repeated_value)
        output = output.transpose(1, 2).contiguous().reshape(*input_shape, -1)

        independent_contribution = torch.einsum(
            "bgqd,gdr->bgqr",
            current_key,
            self.independent_key_projector,
        )
        pair_contribution = torch.einsum(
            "bgqd,gdr->bgqr",
            current_key,
            self.pair_key_encoder,
        )
        self.state.append(
            layer_slot=self.layer_slot,
            independent_contribution=independent_contribution,
            pair_contribution=pair_contribution,
        )
        return self.o_proj(output), probabilities if kwargs.get("output_attentions") else None


@dataclass(frozen=True)
class PairwiseKReplacement:
    layer_index: int
    pair_index: int
    layer_slot: int
    independent_rank: int
    pair_rank: int


@dataclass
class PairwiseKRuntime:
    """Installed-module controller used by the PPL evaluator."""

    modules: tuple[PairwiseKQwen3Attention, ...]
    states: tuple[PairwiseKCacheState, ...]
    records: tuple[PairwiseKReplacement, ...]

    def reset(self) -> None:
        for state in self.states:
            state.reset()

    def set_mode(self, mode: HistoricalKMode) -> None:
        self.reset()
        for module in self.modules:
            module.set_runtime_mode(mode)

    def set_sparse_policy(
        self,
        config: PairwiseQuestConfig | None,
        *,
        full_layer_indices: tuple[int, ...] = (),
    ) -> None:
        full_layers = set(full_layer_indices)
        installed_layers = {module.layer_idx for module in self.modules}
        unknown = full_layers - installed_layers
        if unknown:
            raise ValueError(f"full-support layers are not installed: {sorted(unknown)}")
        self.reset()
        for module in self.modules:
            module.set_sparse_config(
                None if module.layer_idx in full_layers else config
            )
            module.reset_sparse_statistics()

    def reset_sparse_statistics(self) -> None:
        for module in self.modules:
            module.reset_sparse_statistics()

    def sparse_statistics(self) -> dict[str, Any]:
        sparse_modules = tuple(
            module for module in self.modules if module.sparse_config is not None
        )
        per_layer = {
            str(module.layer_idx): module.sparse_statistics()
            for module in sparse_modules
        }
        total_names = tuple(next(iter(per_layer.values())).keys()) if per_layer else ()
        totals = {
            name: sum(float(layer[name]) for layer in per_layer.values())
            for name in total_names
            if name not in {
                "history_length",
                "page_count",
                "resident_selector_metadata_bytes",
                "selected_token_fraction",
            }
        }
        selected = totals.get("selected_physical_tokens", 0.0)
        available = totals.get("available_physical_tokens", 0.0)
        if sparse_modules and sparse_modules[0].runtime_mode == "pairwise":
            metadata_by_pair: dict[int, float] = {}
            for module in sparse_modules:
                state_id = id(module.state)
                metadata_by_pair[state_id] = max(
                    metadata_by_pair.get(state_id, 0.0),
                    float(per_layer[str(module.layer_idx)][
                        "resident_selector_metadata_bytes"
                    ]),
                )
            resident_metadata_bytes = sum(metadata_by_pair.values())
        else:
            resident_metadata_bytes = sum(
                float(layer["resident_selector_metadata_bytes"])
                for layer in per_layer.values()
            )
        totals.update(
            {
                "selected_token_fraction": selected / available if available else 0.0,
                "maximum_history_length": max(
                    (float(layer["history_length"]) for layer in per_layer.values()),
                    default=0.0,
                ),
                "maximum_page_count": max(
                    (float(layer["page_count"]) for layer in per_layer.values()),
                    default=0.0,
                ),
                "resident_selector_metadata_bytes": resident_metadata_bytes,
            }
        )
        return {
            "sparse_layers": [module.layer_idx for module in sparse_modules],
            "full_support_layers": [
                module.layer_idx
                for module in self.modules
                if module.sparse_config is None
            ],
            "totals": totals,
            "per_layer": per_layer,
        }


def install_qwen3_pairwise_k_runtime(
    model: nn.Module,
    *,
    independent_key_projector: Tensor,
    independent_query_projector: Tensor,
    pair_key_projector: Tensor,
    pair_query_projector: Tensor,
) -> PairwiseKRuntime:
    """Install all adjacent pairs while preserving original dense V and O."""

    layers = model.model.layers
    num_layers = int(model.config.num_hidden_layers)
    query_heads = int(model.config.num_attention_heads)
    kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.head_dim)
    if num_layers % 2:
        raise ValueError("adjacent pairwise Key runtime requires an even layer count")
    independent_rank = int(independent_key_projector.shape[-1])
    pair_rank = int(pair_key_projector.shape[-1])
    expected = {
        "independent_key_projector": (
            num_layers,
            kv_heads,
            head_dim,
            independent_rank,
        ),
        "independent_query_projector": (
            num_layers,
            query_heads,
            head_dim,
            independent_rank,
        ),
        "pair_key_projector": (
            num_layers // 2,
            kv_heads,
            2 * head_dim,
            pair_rank,
        ),
        "pair_query_projector": (
            num_layers,
            query_heads,
            head_dim,
            pair_rank,
        ),
    }
    observed = {
        "independent_key_projector": tuple(independent_key_projector.shape),
        "independent_query_projector": tuple(independent_query_projector.shape),
        "pair_key_projector": tuple(pair_key_projector.shape),
        "pair_query_projector": tuple(pair_query_projector.shape),
    }
    for name, shape in expected.items():
        if observed[name] != shape:
            raise ValueError(f"{name} has shape {observed[name]}, expected {shape}")

    modules: list[PairwiseKQwen3Attention] = []
    states: list[PairwiseKCacheState] = []
    records: list[PairwiseKReplacement] = []
    for pair_index in range(num_layers // 2):
        state = PairwiseKCacheState()
        states.append(state)
        pair_factor = pair_key_projector[pair_index]
        for layer_slot in (0, 1):
            layer_index = 2 * pair_index + layer_slot
            base_attention = layers[layer_index].self_attn
            if isinstance(base_attention, PairwiseKQwen3Attention):
                raise ValueError(f"layer {layer_index} already uses pairwise K")
            start = layer_slot * head_dim
            stop = start + head_dim
            replacement = PairwiseKQwen3Attention(
                base_attention,
                state=state,
                layer_slot=layer_slot,
                independent_key_projector=independent_key_projector[layer_index],
                independent_query_projector=independent_query_projector[layer_index],
                pair_key_encoder=pair_factor[:, start:stop],
                pair_query_projector=pair_query_projector[layer_index],
            )
            layers[layer_index].self_attn = replacement
            modules.append(replacement)
            records.append(
                PairwiseKReplacement(
                    layer_index=layer_index,
                    pair_index=pair_index,
                    layer_slot=layer_slot,
                    independent_rank=independent_rank,
                    pair_rank=pair_rank,
                )
            )
    return PairwiseKRuntime(
        modules=tuple(modules),
        states=tuple(states),
        records=tuple(records),
    )


__all__ = [
    "HistoricalKMode",
    "PairwiseKCacheState",
    "PairwiseKQwen3Attention",
    "PairwiseKReplacement",
    "PairwiseKRuntime",
    "PairwiseQuestConfig",
    "install_qwen3_pairwise_k_runtime",
]
