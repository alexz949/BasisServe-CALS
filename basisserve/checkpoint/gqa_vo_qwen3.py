"""Install folded GQA-tied V/O factors into Hugging Face Qwen3 attention."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
import torch
from torch import nn
from torch.nn import functional as F

from basisserve.core.kq_svd import project_grouped_queries_and_keys
from basisserve.core.c1_k_routing_sidecar import (
    RoutingDynamicCache,
    select_routing_pages,
)
from basisserve.core.c1_v_conditional_k_router import (
    build_conditional_routing_sidecar,
    conditional_routing_query_projector,
)
from basisserve.core.c1_k_reverse_shadow import (
    ReverseShadowConfig,
    c1_k_reverse_shadow_block_attention,
)
from basisserve.core.c1_conditional_page_attention import (
    c1_conditional_page_topk_attention,
)
from basisserve.core.c1_loki_attention import c1_loki_pca_topk_attention
from basisserve.kernels.compressed_v_decode_attention import (
    c1_dense_gqa_v96_decode_attention_cuda,
    c1_pack_exact_key_pages_cuda,
    c1_paged_sparse_decode_attention_cuda,
    compressed_v_decode_attention_triton,
    compressed_v_prefill_attention,
)
from basisserve.kernels.c1_r32_routing import (
    c1_r32_page_lse_cuda,
    c1_r32_topk_gqa_union_cuda,
)
from basisserve.kernels.sageattention import sage_attention_with_compressed_value


UNIFORM_ALS_EXPORT_FORMATS = frozenset(
    {
        "basisserve.qwen3_8b.gqa_c1_joint.v1",
        "basisserve.qwen3_32b.gqa_c1_v96_joint.v1",
    }
)


def _memory_bounded_gqa_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float,
    kv_heads_per_chunk: int = 4,
) -> torch.Tensor:
    """Evaluate exact GQA SDPA without expanding every KV head at once."""

    query_heads = int(query.shape[1])
    kv_heads = int(key.shape[1])
    if int(value.shape[1]) != kv_heads or query_heads % kv_heads:
        raise ValueError("Q/K/V heads are incompatible with grouped attention")
    if kv_heads_per_chunk <= 0:
        raise ValueError("KV-head chunk size must be positive")
    groups = query_heads // kv_heads
    if groups == 1:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    outputs = []
    for kv_start in range(0, kv_heads, kv_heads_per_chunk):
        kv_stop = min(kv_start + kv_heads_per_chunk, kv_heads)
        query_start = kv_start * groups
        query_stop = kv_stop * groups
        chunk_mask = attention_mask
        if attention_mask is not None and attention_mask.ndim == 4:
            mask_heads = int(attention_mask.shape[1])
            if mask_heads == query_heads:
                chunk_mask = attention_mask[:, query_start:query_stop]
            elif mask_heads == kv_heads:
                chunk_mask = attention_mask[:, kv_start:kv_stop].repeat_interleave(
                    groups, dim=1
                )
            elif mask_heads != 1:
                raise ValueError("attention-mask heads are incompatible with GQA")
        outputs.append(
            F.scaled_dot_product_attention(
                query[:, query_start:query_stop],
                key[:, kv_start:kv_stop],
                value[:, kv_start:kv_stop],
                attn_mask=chunk_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=True,
            )
        )
    return torch.cat(outputs, dim=1)


@dataclass(frozen=True)
class GQAVOReplacementRecord:
    layer_index: int
    module_name: str
    head_dim: int
    value_head_dim: int
    dense_v_width: int
    compressed_v_width: int
    dense_o_width: int
    compressed_o_width: int


class GQATiedVOQwen3Attention(nn.Module):
    """Qwen3 eager attention with independent Q/K and compressed V dimensions."""

    def __init__(
        self,
        base_attention: nn.Module,
        *,
        v_proj_compressed_weight: torch.Tensor,
        o_decoder_weight: torch.Tensor,
        v_proj_compressed_bias: torch.Tensor | None = None,
        o_decoder_bias: torch.Tensor | None = None,
        attention_backend: str = "native",
        key_projector: torch.Tensor | None = None,
        query_projector: torch.Tensor | None = None,
        value_coordinate_encoder: torch.Tensor | None = None,
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
        self.is_causal = bool(base_attention.is_causal)
        self.sliding_window = base_attention.sliding_window
        self.attention_backend = str(attention_backend)
        if self.attention_backend not in {
            "native",
            "sage",
            "sdpa",
            "triton",
            "cuda_dense",
            "cuda_sparse",
            "dense_prefill",
        }:
            raise ValueError(
                "compressed attention backend must be 'native', 'sdpa', "
                "'sage', 'triton', 'cuda_dense', 'cuda_sparse', or "
                "'dense_prefill', got "
                f"{self.attention_backend!r}"
            )
        if (
            self.attention_backend == "native"
            and getattr(config, "_attn_implementation", None) != "eager"
        ):
            raise ValueError(
                "native compressed C1 attention requires the model to be loaded "
                "with attn_implementation='eager'; otherwise Transformers may omit "
                "the causal mask expected by eager_attention_forward"
            )

        if v_proj_compressed_weight.ndim != 2:
            raise ValueError("compressed v_proj weight must be a matrix")
        compressed_v_width, hidden_size = v_proj_compressed_weight.shape
        if compressed_v_width % self.num_key_value_heads != 0:
            raise ValueError(
                "compressed V width must be divisible by num_key_value_heads: "
                f"{compressed_v_width} vs {self.num_key_value_heads}"
            )
        self.value_head_dim = compressed_v_width // self.num_key_value_heads
        expected_o_shape = (
            hidden_size,
            self.num_attention_heads * self.value_head_dim,
        )
        if tuple(o_decoder_weight.shape) != expected_o_shape:
            raise ValueError(
                f"o_decoder weight must have shape {expected_o_shape}, "
                f"got {tuple(o_decoder_weight.shape)}"
            )
        if int(config.hidden_size) != hidden_size:
            raise ValueError(
                f"compressed V input width must equal hidden_size={config.hidden_size}, got {hidden_size}"
            )

        device = base_attention.q_proj.weight.device
        dtype = base_attention.q_proj.weight.dtype
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm
        self.dense_v_proj = (
            base_attention.v_proj
            if self.attention_backend == "dense_prefill"
            else None
        )
        self.dense_o_proj = (
            base_attention.o_proj
            if self.attention_backend == "dense_prefill"
            else None
        )
        self.v_proj = nn.Linear(
            hidden_size,
            compressed_v_width,
            bias=v_proj_compressed_bias is not None,
            device=device,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            expected_o_shape[1],
            hidden_size,
            bias=o_decoder_bias is not None,
            device=device,
            dtype=dtype,
        )
        self.v_proj.weight.data.copy_(
            v_proj_compressed_weight.to(device=device, dtype=dtype)
        )
        self.o_proj.weight.data.copy_(o_decoder_weight.to(device=device, dtype=dtype))
        if self.v_proj.bias is not None:
            self.v_proj.bias.data.copy_(
                v_proj_compressed_bias.to(device=device, dtype=dtype)
            )
        if self.o_proj.bias is not None:
            self.o_proj.bias.data.copy_(o_decoder_bias.to(device=device, dtype=dtype))
        self.register_buffer("key_projector", None, persistent=False)
        self.register_buffer("query_projector", None, persistent=False)
        self.register_buffer("routing_key_projector", None, persistent=False)
        self.register_buffer("routing_query_projector", None, persistent=False)
        self.register_buffer("conditional_base_left", None, persistent=False)
        self.register_buffer("conditional_base_right", None, persistent=False)
        self.register_buffer("conditional_base_bias", None, persistent=False)
        self.register_buffer("conditional_residual_encoder", None, persistent=False)
        self.register_buffer(
            "value_coordinate_encoder",
            (
                None
                if value_coordinate_encoder is None
                else value_coordinate_encoder.detach().to(device=device, dtype=dtype)
            ),
            persistent=False,
        )
        self.register_buffer("_cuda_dense_workspace", None, persistent=False)
        self.register_buffer("_cuda_dense_output", None, persistent=False)
        self.register_buffer("_cuda_sparse_key_pages", None, persistent=False)
        self.register_buffer("_cuda_sparse_workspace", None, persistent=False)
        self.register_buffer("_cuda_sparse_output", None, persistent=False)
        self.register_buffer("_cuda_routing_query_code", None, persistent=False)
        self.register_buffer("_cuda_routing_page_scores", None, persistent=False)
        self.register_buffer("_cuda_routing_page_counts", None, persistent=False)
        self.register_buffer(
            "_cuda_sparse_last_selected_page_ids",
            None,
            persistent=False,
        )
        self.cuda_sparse_splits = 32
        self.conditional_page_query_block_size: int | None = None
        self.conditional_page_collect_statistics = True
        self.loki_query_block_size: int | None = None
        self.loki_collect_statistics = True
        self.use_cached_routing_sidecar = True
        self.reverse_shadow_config: ReverseShadowConfig | None = None
        self._reverse_shadow_totals: dict[str, float] = {}
        self.set_qk_projectors(key_projector, query_projector)
        self.train(base_attention.training)

    def set_reverse_shadow_config(self, config: ReverseShadowConfig | None) -> None:
        if config is not None:
            config.validate(self.head_dim)
            if self.attention_backend not in {"native", "cuda_sparse"}:
                raise ValueError(
                    "Reverse ShadowKV requires native or CUDA sparse attention"
                )
            if self.key_projector is not None:
                raise ValueError("Reverse ShadowKV requires exact post-RoPE K")
            if config.selector == "kq_svd" and (
                self.routing_query_projector is None
                or (
                    self.routing_key_projector is None
                    and self.conditional_base_left is None
                )
            ):
                raise ValueError("KQ-SVD Reverse ShadowKV requires routing factors")
        self.reverse_shadow_config = config
        self._cuda_sparse_last_selected_page_ids = None
        self._cuda_routing_page_counts = None
        self.reset_reverse_shadow_statistics()

    def cuda_sparse_last_statistics(self) -> dict[str, float]:
        selected = self._cuda_sparse_last_selected_page_ids
        if selected is None:
            return {
                "selected_pages": 0.0,
                "staging_page_slots": 0.0,
                "staging_exact_key_bytes": 0.0,
            }
        counts = self._cuda_routing_page_counts
        selected_pages = float(
            counts.sum().item()
            if counts is not None
            else (selected >= 0).sum().item()
        )
        staging_page_slots = float(selected.numel())
        return {
            "selected_pages": selected_pages,
            "staging_page_slots": staging_page_slots,
            "staging_exact_key_bytes": float(
                self._cuda_sparse_key_pages.numel()
                * self._cuda_sparse_key_pages.element_size()
            ),
        }

    def reset_reverse_shadow_statistics(self) -> None:
        self._reverse_shadow_totals = {
            "queries": 0.0,
            "physical_valid_tokens": 0.0,
            "query_valid_tokens": 0.0,
            "selected_tokens": 0.0,
            "query_selected_tokens": 0.0,
            "selected_pages": 0.0,
            "logical_selected_pages": 0.0,
            "cpu_exact_key_bytes_fetched": 0.0,
            "selection_qk_flops": 0.0,
            "sparse_exact_qk_flops": 0.0,
            "sparse_c1_value_flops": 0.0,
            "adaptive_eligible_query_heads": 0.0,
            "adaptive_refined_query_heads": 0.0,
            "adaptive_tail_mass_ratio_sum": 0.0,
            "maximum_resident_selector_metadata_bytes": 0.0,
        }

    def set_cached_routing_sidecar_enabled(self, enabled: bool) -> None:
        """Choose cached incremental routing or the exact lazy oracle."""

        self.use_cached_routing_sidecar = bool(enabled)

    def set_loki_query_block_size(
        self,
        query_block_size: int | None,
        *,
        collect_statistics: bool = True,
    ) -> None:
        """Enable the memory-tiled full-query Loki quality path."""

        self.loki_query_block_size = (
            None if query_block_size is None else int(query_block_size)
        )
        self.loki_collect_statistics = bool(collect_statistics)

    def set_conditional_page_query_block_size(
        self,
        query_block_size: int | None,
        *,
        collect_statistics: bool = True,
    ) -> None:
        """Enable memory-tiled full-query conditional Page attention."""

        self.conditional_page_query_block_size = (
            None if query_block_size is None else int(query_block_size)
        )
        self.conditional_page_collect_statistics = bool(collect_statistics)

    def reverse_shadow_statistics(self) -> dict[str, float]:
        totals = dict(self._reverse_shadow_totals)
        valid = totals["physical_valid_tokens"]
        totals["selected_token_fraction"] = (
            totals["selected_tokens"] / valid if valid else 0.0
        )
        query_valid = totals["query_valid_tokens"]
        totals["query_selected_token_fraction"] = (
            totals["query_selected_tokens"] / query_valid if query_valid else 0.0
        )
        return totals

    def _record_reverse_shadow_results(self, results: tuple[Any, ...]) -> None:
        for result in results:
            statistics = result.statistics
            self._reverse_shadow_totals["queries"] += float(result.output.shape[0])
            for name in (
                "physical_valid_tokens",
                "query_valid_tokens",
                "selected_tokens",
                "query_selected_tokens",
                "selected_pages",
                "logical_selected_pages",
                "selection_qk_flops",
                "sparse_exact_qk_flops",
                "sparse_c1_value_flops",
                "adaptive_eligible_query_heads",
                "adaptive_refined_query_heads",
                "adaptive_tail_mass_ratio_sum",
            ):
                self._reverse_shadow_totals[name] += float(statistics[name])
            self._reverse_shadow_totals["cpu_exact_key_bytes_fetched"] += float(
                statistics["oracle_page_store_key_bytes_read"]
            )
            self._reverse_shadow_totals["maximum_resident_selector_metadata_bytes"] = (
                max(
                    self._reverse_shadow_totals[
                        "maximum_resident_selector_metadata_bytes"
                    ],
                    float(statistics["resident_selector_metadata_bytes"]),
                )
            )

    def _record_loki_statistics(self, statistics: dict[str, float]) -> None:
        self._reverse_shadow_totals["queries"] += statistics["queries"]
        for name in (
            "physical_valid_tokens",
            "query_valid_tokens",
            "selected_tokens",
            "query_selected_tokens",
            "selected_pages",
            "logical_selected_pages",
            "selection_qk_flops",
            "sparse_exact_qk_flops",
            "sparse_c1_value_flops",
            "adaptive_eligible_query_heads",
            "adaptive_refined_query_heads",
            "adaptive_tail_mass_ratio_sum",
        ):
            self._reverse_shadow_totals[name] += statistics[name]
        self._reverse_shadow_totals["cpu_exact_key_bytes_fetched"] += statistics[
            "oracle_page_store_key_bytes_read"
        ]
        self._reverse_shadow_totals["maximum_resident_selector_metadata_bytes"] = (
            max(
                self._reverse_shadow_totals[
                    "maximum_resident_selector_metadata_bytes"
                ],
                statistics["resident_selector_metadata_bytes"],
            )
        )

    @torch.no_grad()
    def set_qk_projectors(
        self,
        key_projector: torch.Tensor | None,
        query_projector: torch.Tensor | None,
    ) -> None:
        """Select dense Q/K or one post-RoPE low-rank projector pair.

        K is shared by every Query head in a physical GQA group.  Q may use
        either the same group-shared geometry as KQ-SVD or one projector per
        Query head for output-aware C1-K closure.
        """

        if key_projector is None or query_projector is None:
            if key_projector is not None or query_projector is not None:
                raise ValueError("Key and Query projectors must be set together")
            self.key_projector = None
            self.query_projector = None
            return
        if self.attention_backend == "sage":
            raise ValueError("SageAttention does not support compressed Q/K")
        query_heads = int(query_projector.shape[0]) if query_projector.ndim == 3 else -1
        if (
            key_projector.ndim != 3
            or query_projector.ndim != 3
            or tuple(key_projector.shape[:2])
            != (self.num_key_value_heads, self.head_dim)
            or query_heads not in {self.num_key_value_heads, self.num_attention_heads}
            or int(query_projector.shape[1]) != self.head_dim
            or key_projector.shape[-1] != query_projector.shape[-1]
        ):
            raise ValueError(
                "post-RoPE K must have shape [physical_kv_heads, head_dim, rank]; "
                "Q must have shape [physical_kv_heads or query_heads, head_dim, rank]"
            )
        device = self.q_proj.weight.device
        dtype = self.q_proj.weight.dtype
        self.key_projector = key_projector.detach().to(device=device, dtype=dtype)
        self.query_projector = query_projector.detach().to(device=device, dtype=dtype)

    @torch.no_grad()
    def set_routing_projectors(
        self,
        key_projector: torch.Tensor | None,
        query_projector: torch.Tensor | None,
    ) -> None:
        """Set routing-only factors without changing exact attention Q/K."""

        if key_projector is None or query_projector is None:
            if key_projector is not None or query_projector is not None:
                raise ValueError("routing Key and Query factors must be set together")
            self.routing_key_projector = None
            self.routing_query_projector = None
            return
        query_heads = int(query_projector.shape[0]) if query_projector.ndim == 3 else -1
        if (
            key_projector.ndim != 3
            or query_projector.ndim != 3
            or tuple(key_projector.shape[:2])
            != (self.num_key_value_heads, self.head_dim)
            or query_heads not in {self.num_key_value_heads, self.num_attention_heads}
            or int(query_projector.shape[1]) != self.head_dim
            or key_projector.shape[-1] != query_projector.shape[-1]
            or not 0 < int(key_projector.shape[-1]) <= self.head_dim
        ):
            raise ValueError(
                "routing K/Q factors must be [heads, head_dim, rank] with "
                "matching rank no larger than head_dim"
            )
        device = self.q_proj.weight.device
        dtype = self.q_proj.weight.dtype
        self.routing_key_projector = key_projector.detach().to(
            device=device, dtype=dtype
        )
        self.routing_query_projector = query_projector.detach().to(
            device=device, dtype=dtype
        )

    @torch.no_grad()
    def set_conditional_routing_factors(
        self,
        *,
        base_left: torch.Tensor,
        base_right: torch.Tensor,
        base_bias: torch.Tensor,
        residual_encoder: torch.Tensor,
        residual_query_projector: torch.Tensor,
    ) -> None:
        """Install a V-conditioned predictive base plus residual-Key router."""

        device = self.q_proj.weight.device
        dtype = self.q_proj.weight.dtype
        self.conditional_base_left = base_left.detach().to(
            device=device, dtype=dtype
        )
        self.conditional_base_right = base_right.detach().to(
            device=device, dtype=dtype
        )
        self.conditional_base_bias = base_bias.detach().to(
            device=device, dtype=dtype
        )
        self.conditional_residual_encoder = residual_encoder.detach().to(
            device=device, dtype=dtype
        )
        self.routing_key_projector = None
        self.routing_query_projector = conditional_routing_query_projector(
            residual_query_projector.detach().to(device=device, dtype=dtype)
        )

    def _cuda_sparse_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        routing_sidecar: torch.Tensor,
    ) -> torch.Tensor:
        config = self.reverse_shadow_config
        batch = int(query_states.shape[0])
        sequence_length = int(key_states.shape[2])
        page_count = (
            sequence_length + config.page_size - 1
        ) // config.page_size
        maximum_page_slots = min(
            page_count,
            config.page_budget * self.num_key_value_groups,
        )
        use_fused_r32_routing = (
            query_states.is_cuda
            and routing_sidecar.is_cuda
            and self.routing_query_projector is not None
            and self.routing_query_projector.is_cuda
            and query_states.dtype in (torch.float16, torch.bfloat16)
            and routing_sidecar.dtype == query_states.dtype
            and self.routing_query_projector.dtype == query_states.dtype
            and self.head_dim == 128
            and self.num_key_value_groups == 4
            and int(routing_sidecar.shape[-1]) == 32
            and config.page_size == 64
            and config.page_budget <= 32
            and page_count <= 2048
        )
        if use_fused_r32_routing:
            expected_query_code = (
                batch,
                self.num_attention_heads,
                32,
            )
            if (
                self._cuda_routing_query_code is None
                or tuple(self._cuda_routing_query_code.shape)
                != expected_query_code
                or self._cuda_routing_query_code.dtype != query_states.dtype
                or self._cuda_routing_query_code.device != query_states.device
            ):
                self._cuda_routing_query_code = torch.empty(
                    expected_query_code,
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
            expected_page_scores = (
                batch,
                self.num_attention_heads,
                page_count,
            )
            if (
                self._cuda_routing_page_scores is None
                or tuple(self._cuda_routing_page_scores.shape)
                != expected_page_scores
                or self._cuda_routing_page_scores.dtype != query_states.dtype
                or self._cuda_routing_page_scores.device != query_states.device
            ):
                self._cuda_routing_page_scores = torch.empty(
                    expected_page_scores,
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
            page_log_mass = c1_r32_page_lse_cuda(
                query_states,
                routing_sidecar,
                self.routing_query_projector,
                scale=self.head_dim**-0.5,
                query_code=self._cuda_routing_query_code,
                output=self._cuda_routing_page_scores,
            )
            expected_page_ids = (
                batch,
                self.num_key_value_heads,
                maximum_page_slots,
            )
            if (
                self._cuda_sparse_last_selected_page_ids is None
                or tuple(self._cuda_sparse_last_selected_page_ids.shape)
                != expected_page_ids
                or self._cuda_sparse_last_selected_page_ids.dtype
                != torch.int64
                or self._cuda_sparse_last_selected_page_ids.device
                != query_states.device
            ):
                self._cuda_sparse_last_selected_page_ids = torch.empty(
                    expected_page_ids,
                    dtype=torch.int64,
                    device=query_states.device,
                )
            expected_page_counts = (batch, self.num_key_value_heads)
            if (
                self._cuda_routing_page_counts is None
                or tuple(self._cuda_routing_page_counts.shape)
                != expected_page_counts
                or self._cuda_routing_page_counts.dtype != torch.int32
                or self._cuda_routing_page_counts.device != query_states.device
            ):
                self._cuda_routing_page_counts = torch.empty(
                    expected_page_counts,
                    dtype=torch.int32,
                    device=query_states.device,
                )
            selected_page_ids, _ = c1_r32_topk_gqa_union_cuda(
                page_log_mass,
                pages_per_query_head=config.page_budget,
                selected_page_ids=self._cuda_sparse_last_selected_page_ids,
                selected_page_counts=self._cuda_routing_page_counts,
            )
        else:
            page_indices = torch.arange(
                page_count,
                dtype=torch.int64,
                device=query_states.device,
            ).expand(self.num_key_value_heads, page_count)
            selected_batches = []
            for batch_index in range(batch):
                selection = select_routing_pages(
                    query_states[batch_index, :, 0],
                    routing_sidecar[batch_index],
                    self.routing_query_projector,
                    head_dim=self.head_dim,
                    page_size=config.page_size,
                    nominal_token_budget=config.exact_token_budget,
                )
                padded_ids = torch.where(
                    selection.page_mask,
                    page_indices,
                    page_count,
                )
                selected_ids = torch.sort(padded_ids, dim=-1).values[
                    :, :maximum_page_slots
                ]
                selected_batches.append(
                    selected_ids.masked_fill_(
                        selected_ids == page_count,
                        -1,
                    )
                )
            selected_page_ids = torch.stack(selected_batches).contiguous()
            self._cuda_sparse_last_selected_page_ids = selected_page_ids
            self._cuda_routing_page_counts = None

        expected_key_pages = (
            batch,
            self.num_key_value_heads,
            maximum_page_slots,
            config.page_size,
            self.head_dim,
        )
        if (
            self._cuda_sparse_key_pages is None
            or tuple(self._cuda_sparse_key_pages.shape) != expected_key_pages
            or self._cuda_sparse_key_pages.dtype != key_states.dtype
            or self._cuda_sparse_key_pages.device != key_states.device
        ):
            self._cuda_sparse_key_pages = torch.empty(
                expected_key_pages,
                dtype=key_states.dtype,
                device=key_states.device,
            )
        c1_pack_exact_key_pages_cuda(
            key_states,
            selected_page_ids,
            output=self._cuda_sparse_key_pages,
        )

        splits = self.cuda_sparse_splits
        while splits > maximum_page_slots:
            splits //= 2
        expected_workspace = (
            batch * self.num_attention_heads,
            splits,
            self.value_head_dim + 2,
        )
        if (
            self._cuda_sparse_workspace is None
            or tuple(self._cuda_sparse_workspace.shape) != expected_workspace
            or self._cuda_sparse_workspace.device != query_states.device
        ):
            self._cuda_sparse_workspace = torch.empty(
                expected_workspace,
                dtype=torch.float32,
                device=query_states.device,
            )
        expected_output = (
            batch,
            self.num_attention_heads,
            1,
            self.value_head_dim,
        )
        if (
            self._cuda_sparse_output is None
            or tuple(self._cuda_sparse_output.shape) != expected_output
            or self._cuda_sparse_output.dtype != query_states.dtype
            or self._cuda_sparse_output.device != query_states.device
        ):
            self._cuda_sparse_output = torch.empty(
                expected_output,
                dtype=query_states.dtype,
                device=query_states.device,
            )
        return c1_paged_sparse_decode_attention_cuda(
            query_states,
            self._cuda_sparse_key_pages,
            value_states,
            selected_page_ids,
            scale=self.scaling,
            splits=splits,
            workspace=self._cuda_sparse_workspace,
            output=self._cuda_sparse_output,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        input_shape = hidden_states.shape[:-1]
        dense_prefill = self.attention_backend == "dense_prefill"
        query_shape = (*input_shape, self.num_attention_heads, self.head_dim)
        key_shape = (*input_shape, self.num_key_value_heads, self.head_dim)
        active_value_head_dim = self.head_dim if dense_prefill else self.value_head_dim
        value_shape = (*input_shape, self.num_key_value_heads, active_value_head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(query_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(key_shape)).transpose(
            1, 2
        )
        value_projection = self.dense_v_proj if dense_prefill else self.v_proj
        value_states = value_projection(hidden_states).view(value_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if self.key_projector is not None:
            query_states, key_states = project_grouped_queries_and_keys(
                query_states,
                key_states,
                self.key_projector,
                self.query_projector,
            )
        cached_routing_sidecar = None
        if past_key_values is not None:
            new_key_states = key_states
            new_value_states = value_states
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
            if (
                isinstance(past_key_values, RoutingDynamicCache)
                and self.conditional_base_left is not None
            ):
                sidecar_update = build_conditional_routing_sidecar(
                    new_value_states,
                    new_key_states,
                    base_left=self.conditional_base_left,
                    base_right=self.conditional_base_right,
                    base_bias=self.conditional_base_bias,
                    residual_encoder=self.conditional_residual_encoder,
                    cos=cos,
                    sin=sin,
                )
                cached_routing_sidecar = (
                    past_key_values.update_precomputed_routing_sidecar(
                        sidecar_update,
                        self.layer_idx,
                    )
                )
            elif (
                isinstance(past_key_values, RoutingDynamicCache)
                and self.routing_key_projector is not None
            ):
                cached_routing_sidecar = past_key_values.update_routing_sidecar(
                    new_key_states,
                    self.routing_key_projector,
                    self.layer_idx,
                )

        if self.reverse_shadow_config is not None:
            if kwargs.get("output_attentions"):
                raise NotImplementedError(
                    "Reverse ShadowKV does not return attention weights"
                )
            if (
                self.conditional_page_query_block_size is not None
                and self.conditional_base_left is not None
                and self.reverse_shadow_config.selector == "kq_svd"
                and self.reverse_shadow_config.quest_support == "physical_shared"
            ):
                if cached_routing_sidecar is None:
                    assert past_key_values is None
                    cached_routing_sidecar = build_conditional_routing_sidecar(
                        value_states,
                        key_states,
                        base_left=self.conditional_base_left,
                        base_right=self.conditional_base_right,
                        base_bias=self.conditional_base_bias,
                        residual_encoder=self.conditional_residual_encoder,
                        cos=cos,
                        sin=sin,
                    )
                conditional = c1_conditional_page_topk_attention(
                    query_states,
                    key_states,
                    value_states,
                    cached_routing_sidecar,
                    self.routing_query_projector,
                    page_size=self.reverse_shadow_config.page_size,
                    exact_token_budget=(
                        self.reverse_shadow_config.exact_token_budget
                    ),
                    pinned_prefix_pages=(
                        self.reverse_shadow_config.pinned_prefix_pages
                    ),
                    scale=self.scaling,
                    query_block_size=self.conditional_page_query_block_size,
                    attention_mask=attention_mask,
                    collect_statistics=(
                        self.conditional_page_collect_statistics
                    ),
                )
                head_major_output = conditional.output
                if self.conditional_page_collect_statistics:
                    self._record_loki_statistics(conditional.statistics)
            elif (
                self.loki_query_block_size is not None
                and self.reverse_shadow_config.selector == "kq_svd"
                and self.reverse_shadow_config.quest_support == "per_query_head"
                and self.reverse_shadow_config.page_size == 1
            ):
                loki = c1_loki_pca_topk_attention(
                    query_states,
                    key_states,
                    value_states,
                    self.routing_key_projector,
                    self.routing_query_projector,
                    top_k=self.reverse_shadow_config.exact_token_budget,
                    scale=self.scaling,
                    query_block_size=self.loki_query_block_size,
                    attention_mask=attention_mask,
                    routing_sidecar=cached_routing_sidecar,
                    collect_statistics=self.loki_collect_statistics,
                )
                head_major_output = loki.output
                if self.loki_collect_statistics:
                    self._record_loki_statistics(loki.statistics)
            elif self.attention_backend == "cuda_sparse":
                head_major_output = self._cuda_sparse_attention(
                    query_states,
                    key_states,
                    value_states,
                    cached_routing_sidecar,
                )
            else:
                head_major_output, reverse_results = (
                    c1_k_reverse_shadow_block_attention(
                        query_states,
                        key_states,
                        value_states,
                        self.reverse_shadow_config,
                        attention_mask,
                        routing_key_projector=(
                            self.routing_key_projector
                            if self.reverse_shadow_config.selector == "kq_svd"
                            else None
                        ),
                        routing_query_projector=(
                            self.routing_query_projector
                            if self.reverse_shadow_config.selector == "kq_svd"
                            else None
                        ),
                        routing_sidecar=(
                            cached_routing_sidecar
                            if (
                                self.reverse_shadow_config.selector == "kq_svd"
                                and self.use_cached_routing_sidecar
                            )
                            else None
                        ),
                        layer_idx=self.layer_idx,
                    )
                )
                self._record_reverse_shadow_results(reverse_results)
            attn_output = head_major_output.transpose(1, 2).contiguous()
            attn_weights = None
        elif dense_prefill:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                "sdpa",
                eager_attention_forward,
            )
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=(0.0 if not self.training else self.attention_dropout),
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        elif self.attention_backend == "sage":
            if self.training and self.attention_dropout:
                raise NotImplementedError(
                    "SageAttention compressed backend is inference-only"
                )
            if kwargs.get("output_attentions"):
                raise NotImplementedError(
                    "SageAttention compressed backend does not return attention weights"
                )
            if self.sliding_window is not None and attention_mask is None:
                raise NotImplementedError(
                    "SageAttention requires an explicit mask for sliding-window attention"
                )
            attn_output = sage_attention_with_compressed_value(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                is_causal=bool(
                    self.is_causal
                    and attention_mask is None
                    and query_states.shape[-2] > 1
                ),
                scaling=self.scaling,
            )
            attn_weights = None
        elif self.attention_backend == "cuda_dense":
            if kwargs.get("output_attentions"):
                raise NotImplementedError(
                    "dense GQA V96 CUDA does not return attention weights"
                )
            if int(query_states.shape[-2]) != 1:
                raise ValueError(
                    "dense GQA V96 CUDA requires exactly one query token; "
                    "run prefill with the SDPA backend"
                )
            cache_layer = past_key_values.layers[self.layer_idx]
            valid_sequence_length = cache_layer.cumulative_length
            cache_length = int(key_states.shape[-2])
            splits = 128 if cache_length <= 32768 else 64
            while splits > cache_length:
                splits //= 2
            expected_workspace = (
                int(query_states.shape[0]) * self.num_attention_heads,
                splits,
                self.value_head_dim + 2,
            )
            if (
                self._cuda_dense_workspace is None
                or tuple(self._cuda_dense_workspace.shape) != expected_workspace
                or self._cuda_dense_workspace.device != query_states.device
            ):
                self._cuda_dense_workspace = torch.empty(
                    expected_workspace,
                    dtype=torch.float32,
                    device=query_states.device,
                )
            expected_output = (
                int(query_states.shape[0]),
                self.num_attention_heads,
                1,
                self.value_head_dim,
            )
            if (
                self._cuda_dense_output is None
                or tuple(self._cuda_dense_output.shape) != expected_output
                or self._cuda_dense_output.dtype != query_states.dtype
                or self._cuda_dense_output.device != query_states.device
            ):
                self._cuda_dense_output = torch.empty(
                    expected_output,
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
            attn_output = (
                c1_dense_gqa_v96_decode_attention_cuda(
                    query_states,
                    key_states,
                    value_states,
                    valid_sequence_length,
                    scale=self.scaling,
                    splits=splits,
                    workspace=self._cuda_dense_workspace,
                    output=self._cuda_dense_output,
                )
                .transpose(1, 2)
                .contiguous()
            )
            attn_weights = None
        elif self.attention_backend == "triton":
            if kwargs.get("output_attentions"):
                raise NotImplementedError(
                    "Triton compressed decode does not return attention weights"
                )
            if int(query_states.shape[-2]) > 1:
                assert int(query_states.shape[-2]) == int(key_states.shape[-2])
                attn_output = compressed_v_prefill_attention(
                    query_states,
                    key_states,
                    value_states,
                    scale=self.scaling,
                ).transpose(1, 2).contiguous()
            else:
                valid_sequence_length = None
                if past_key_values is not None:
                    cache_layer = past_key_values.layers[self.layer_idx]
                    if hasattr(cache_layer, "max_cache_len"):
                        valid_sequence_length = cache_layer.cumulative_length
                if attention_mask is not None:
                    assert valid_sequence_length is not None
                attn_output = (
                    compressed_v_decode_attention_triton(
                        query_states,
                        key_states,
                        value_states,
                        scale=self.scaling,
                        valid_sequence_length=valid_sequence_length,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
            attn_weights = None
        elif self.attention_backend == "sdpa":
            if kwargs.get("output_attentions"):
                raise NotImplementedError(
                    "SDPA compressed backend does not return attention weights"
                )
            attn_output = (
                _memory_bounded_gqa_sdpa(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask=attention_mask,
                    dropout_p=(0.0 if not self.training else self.attention_dropout),
                    is_causal=bool(
                        self.is_causal
                        and attention_mask is None
                        and query_states.shape[-2] > 1
                    ),
                    scale=self.scaling,
                )
                .transpose(1, 2)
                .contiguous()
            )
            attn_weights = None
        else:
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output_projection = self.dense_o_proj if dense_prefill else self.o_proj
        return output_projection(attn_output), attn_weights


@torch.no_grad()
def transition_qwen3_dense_prefill_cache_to_c1(
    model: nn.Module,
    cache: Any,
    *,
    attention_backend: str = "sdpa",
) -> Any:
    """Project a dense DynamicCache into C1 coordinates and enter decode mode."""

    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        dense_values = cache.layers[layer_index].values
        encoder = attention.value_coordinate_encoder.to(
            device=dense_values.device,
            dtype=dense_values.dtype,
        )
        batch, heads, tokens, dense_head_dim = dense_values.shape
        head_major = dense_values.permute(1, 0, 2, 3).reshape(
            heads,
            batch * tokens,
            dense_head_dim,
        )
        compressed_values = torch.bmm(head_major, encoder)
        cache.layers[layer_index].values = (
            compressed_values.reshape(
                heads,
                batch,
                tokens,
                int(encoder.shape[-1]),
            )
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        attention.dense_v_proj = None
        attention.dense_o_proj = None
        attention.attention_backend = attention_backend
    return cache


class TrainableGQATiedVOQwen3Attention(nn.Module):
    """Training-time Qwen3 attention with explicit tied bases and decoder."""

    def __init__(
        self,
        base_attention: nn.Module,
        *,
        basis_pt: torch.Tensor,
        decoder_weight: torch.Tensor,
        decoder_bias: torch.Tensor | None = None,
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
        self.is_causal = bool(base_attention.is_causal)
        self.sliding_window = base_attention.sliding_window

        expected_basis_prefix = (self.num_key_value_heads,)
        if basis_pt.ndim != 3 or tuple(basis_pt.shape[:1]) != expected_basis_prefix:
            raise ValueError(
                "basis_pt must have shape [num_key_value_heads, rank, head_dim], "
                f"got {tuple(basis_pt.shape)}"
            )
        if basis_pt.shape[2] != self.head_dim:
            raise ValueError(
                f"basis head dimension must be {self.head_dim}, got {basis_pt.shape[2]}"
            )
        self.value_head_dim = int(basis_pt.shape[1])
        hidden_size = int(config.hidden_size)
        expected_decoder_shape = (
            hidden_size,
            self.num_attention_heads * self.value_head_dim,
        )
        if tuple(decoder_weight.shape) != expected_decoder_shape:
            raise ValueError(
                f"decoder_weight must have shape {expected_decoder_shape}, "
                f"got {tuple(decoder_weight.shape)}"
            )

        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm
        self.register_buffer(
            "dense_v_weight",
            base_attention.v_proj.weight.detach(),
            persistent=False,
        )
        self.register_buffer(
            "dense_v_bias",
            None
            if base_attention.v_proj.bias is None
            else base_attention.v_proj.bias.detach(),
            persistent=False,
        )
        self.basis_pt = nn.Parameter(basis_pt.detach().to(torch.float32).clone())
        self.decoder_weight = nn.Parameter(
            decoder_weight.detach().to(torch.float32).clone()
        )
        self.register_buffer(
            "decoder_bias",
            None if decoder_bias is None else decoder_bias.detach().clone(),
            persistent=False,
        )
        self.train(base_attention.training)

    def folded_v_weight(self) -> torch.Tensor:
        dense = self.dense_v_weight.reshape(
            self.num_key_value_heads,
            self.head_dim,
            -1,
        ).to(self.basis_pt.dtype)
        return torch.bmm(self.basis_pt, dense).reshape(
            self.num_key_value_heads * self.value_head_dim,
            -1,
        )

    def folded_v_bias(self) -> torch.Tensor | None:
        if self.dense_v_bias is None:
            return None
        dense = self.dense_v_bias.reshape(
            self.num_key_value_heads, self.head_dim, 1
        ).to(self.basis_pt.dtype)
        return torch.bmm(self.basis_pt, dense).reshape(-1)

    @torch.no_grad()
    def retract_basis_(self) -> None:
        """QR-retract every basis and compensate decoder blocks exactly."""

        rank = self.value_head_dim
        heads_per_group = self.num_key_value_groups
        for group_index in range(self.num_key_value_heads):
            basis_row = self.basis_pt[group_index].transpose(0, 1)
            q_basis, transform = torch.linalg.qr(basis_row, mode="reduced")
            self.basis_pt[group_index].copy_(q_basis.transpose(0, 1))
            first_head = group_index * heads_per_group
            for head_offset in range(heads_per_group):
                head_index = first_head + head_offset
                start = head_index * rank
                block = self.decoder_weight[:, start : start + rank]
                block.copy_(block @ transform.transpose(0, 1))

    @torch.no_grad()
    def export_projection_payload(
        self,
        *,
        dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        if dtype is None:
            dtype = self.dense_v_weight.dtype
        folded_bias = self.folded_v_bias()
        return {
            "format": "basisserve.gqa_vo_svdllm.layer.v1",
            "v_proj_compressed_weight": self.folded_v_weight().detach().to(dtype).cpu(),
            "v_proj_compressed_bias": (
                None if folded_bias is None else folded_bias.detach().to(dtype).cpu()
            ),
            "o_decoder_weight": self.decoder_weight.detach().to(dtype).cpu(),
            "o_decoder_bias": (
                None
                if self.decoder_bias is None
                else self.decoder_bias.detach().to(dtype).cpu()
            ),
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape, self.num_attention_heads, self.head_dim)
        key_shape = (*input_shape, self.num_key_value_heads, self.head_dim)
        value_shape = (*input_shape, self.num_key_value_heads, self.value_head_dim)
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(query_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(key_shape)).transpose(
            1, 2
        )
        folded_weight = self.folded_v_weight().to(hidden_states.dtype)
        folded_bias = self.folded_v_bias()
        if folded_bias is not None:
            folded_bias = folded_bias.to(hidden_states.dtype)
        value_states = (
            F.linear(hidden_states, folded_weight, folded_bias)
            .view(value_shape)
            .transpose(1, 2)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = F.linear(
            attn_output,
            self.decoder_weight.to(attn_output.dtype),
            None
            if self.decoder_bias is None
            else self.decoder_bias.to(attn_output.dtype),
        )
        return output, attn_weights


def _replace_module(root: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    setattr(parent, child_name, replacement)


def _load_projection_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "basisserve.gqa_vo_svdllm.layer.v1":
        raise ValueError(f"unsupported GQA V/O layer payload in {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_uniform_als_manifest(
    model: nn.Module,
    factor_dir: Path,
) -> dict[str, Any]:
    result_path = factor_dir / "results.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("format") not in UNIFORM_ALS_EXPORT_FORMATS
        or result.get("status") != "complete"
    ):
        raise ValueError("uniform C1 ALS result is incomplete or incompatible")
    config = model.config
    fit_config = result.get("fit_config", {})
    expected_geometry = {
        "model_type": str(config.model_type),
        "hidden_size": int(config.hidden_size),
        "num_query_heads": int(config.num_attention_heads),
        "num_physical_kv_heads": int(config.num_key_value_heads),
        "head_dim": int(
            getattr(
                config,
                "head_dim",
                config.hidden_size // config.num_attention_heads,
            )
        ),
        "num_hidden_layers": int(config.num_hidden_layers),
    }
    for name, expected in expected_geometry.items():
        if fit_config.get(name) != expected:
            raise ValueError(
                f"uniform C1 ALS/model {name} mismatch: "
                f"{fit_config.get(name)!r} vs {expected!r}"
            )
    layer_count = expected_geometry["num_hidden_layers"]
    if tuple(map(int, result.get("layers", ()))) != tuple(range(layer_count)):
        raise ValueError("uniform C1 ALS result does not cover every model layer")
    artifacts = result.get("artifacts", {})
    if set(map(int, artifacts)) != set(range(layer_count)):
        raise ValueError("uniform C1 ALS result has incomplete layer artifacts")
    rank = int(fit_config.get("cache_rank_per_head", 0))
    if not 0 < rank <= expected_geometry["head_dim"]:
        raise ValueError(f"uniform C1 ALS rank is invalid: {rank}")
    return result


@torch.no_grad()
def install_qwen3_gqa_vo_als_export(
    model: nn.Module,
    factor_dir: str | Path,
    *,
    attention_backend: str = "native",
) -> list[GQAVOReplacementRecord]:
    """Install the repository's uniform ALS C1 safetensors without padding V."""

    factor_dir = Path(factor_dir).expanduser().resolve()
    result = _validate_uniform_als_manifest(model, factor_dir)
    config = model.config
    hidden_size = int(config.hidden_size)
    query_heads = int(config.num_attention_heads)
    key_value_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", hidden_size // query_heads))
    rank = int(result["fit_config"]["cache_rank_per_head"])
    records: list[GQAVOReplacementRecord] = []
    for layer_index in range(int(config.num_hidden_layers)):
        module_name = f"model.layers.{layer_index}.self_attn"
        base_attention = model.get_submodule(module_name)
        if isinstance(base_attention, GQATiedVOQwen3Attention):
            raise ValueError(f"attention module is already compressed: {module_name}")
        if not isinstance(base_attention.v_proj, nn.Linear):
            raise TypeError(f"C1 ALS V projection is not linear: {module_name}.v_proj")
        if not isinstance(base_attention.o_proj, nn.Linear):
            raise TypeError(f"C1 ALS O projection is not linear: {module_name}.o_proj")

        artifact = result["artifacts"][str(layer_index)]
        path = factor_dir / str(artifact["file"])
        if _file_sha256(path) != artifact.get("sha256"):
            raise ValueError(
                f"uniform C1 ALS artifact hash mismatch at layer {layer_index}"
            )
        payload = load_file(str(path), device="cpu")
        if set(payload) != {"value_coordinate_encoders", "head_output_decoders"}:
            raise ValueError(f"unexpected uniform C1 tensors at layer {layer_index}")
        encoders = payload["value_coordinate_encoders"]
        decoders = payload["head_output_decoders"]
        if tuple(encoders.shape) != (key_value_heads, head_dim, rank):
            raise ValueError(f"unexpected C1 encoder shape at layer {layer_index}")
        if tuple(decoders.shape) != (query_heads, rank, hidden_size):
            raise ValueError(f"unexpected C1 decoder shape at layer {layer_index}")

        device = base_attention.v_proj.weight.device
        compute_dtype = torch.float32
        dense_v = base_attention.v_proj.weight.detach().to(dtype=compute_dtype)
        grouped_dense_v = dense_v.reshape(key_value_heads, head_dim, hidden_size)
        compressed_v = torch.bmm(
            encoders.to(device=device, dtype=compute_dtype).transpose(1, 2),
            grouped_dense_v,
        ).reshape(key_value_heads * rank, hidden_size)
        decoder_weight = (
            decoders.to(device=device, dtype=compute_dtype)
            .permute(2, 0, 1)
            .reshape(hidden_size, query_heads * rank)
        )
        compressed_v_bias = None
        if base_attention.v_proj.bias is not None:
            dense_v_bias = base_attention.v_proj.bias.detach().to(dtype=compute_dtype)
            compressed_v_bias = torch.bmm(
                encoders.to(device=device, dtype=compute_dtype).transpose(1, 2),
                dense_v_bias.reshape(key_value_heads, head_dim, 1),
            ).reshape(key_value_heads * rank)

        replacement = GQATiedVOQwen3Attention(
            base_attention,
            v_proj_compressed_weight=compressed_v,
            o_decoder_weight=decoder_weight,
            v_proj_compressed_bias=compressed_v_bias,
            o_decoder_bias=base_attention.o_proj.bias,
            attention_backend=attention_backend,
            value_coordinate_encoder=encoders,
        )
        _replace_module(model, module_name, replacement)
        records.append(
            GQAVOReplacementRecord(
                layer_index=layer_index,
                module_name=module_name,
                head_dim=head_dim,
                value_head_dim=rank,
                dense_v_width=key_value_heads * head_dim,
                compressed_v_width=key_value_heads * rank,
                dense_o_width=query_heads * head_dim,
                compressed_o_width=query_heads * rank,
            )
        )
    return records


@torch.no_grad()
def install_qwen3_gqa_vo_export(
    model: nn.Module,
    export_dir: str | Path,
) -> list[GQAVOReplacementRecord]:
    """Replace selected Qwen3 attention layers from an offline factor export."""

    export_dir = Path(export_dir).expanduser().resolve()
    config_path = export_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "basisserve.gqa_vo_svdllm.export.v1":
        raise ValueError(f"unsupported GQA V/O export format in {config_path}")
    if metadata.get("model_type") != "qwen3":
        raise ValueError(f"expected qwen3 export, got {metadata.get('model_type')!r}")
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError(
            f"expected Qwen3 model, got {getattr(model.config, 'model_type', None)!r}"
        )

    layout = metadata["layout"]
    expected_config = {
        "hidden_size": int(model.config.hidden_size),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
        "head_dim": int(
            getattr(
                model.config,
                "head_dim",
                model.config.hidden_size // model.config.num_attention_heads,
            )
        ),
    }
    for key, expected in expected_config.items():
        actual = int(layout[key])
        if actual != expected:
            raise ValueError(f"export/model {key} mismatch: {actual} vs {expected}")

    records = []
    for layer_spec in metadata["layers"]:
        layer_index = int(layer_spec["layer_index"])
        module_name = str(layer_spec["module_name"])
        base_attention = model.get_submodule(module_name)
        if isinstance(base_attention, GQATiedVOQwen3Attention):
            raise ValueError(f"attention module is already compressed: {module_name}")
        if bool(layer_spec.get("dense_passthrough", False)):
            rank = int(layer_spec.get("rank_per_kv_head", expected_config["head_dim"]))
            if rank != expected_config["head_dim"]:
                raise ValueError(
                    f"dense passthrough layer {layer_index} has non-dense rank {rank}"
                )
            continue
        payload = _load_projection_payload(
            export_dir / f"layer_{layer_index:03d}" / "compressed_projections.pt"
        )
        replacement = GQATiedVOQwen3Attention(
            base_attention,
            v_proj_compressed_weight=payload["v_proj_compressed_weight"],
            o_decoder_weight=payload["o_decoder_weight"],
            v_proj_compressed_bias=payload.get("v_proj_compressed_bias"),
            o_decoder_bias=payload.get("o_decoder_bias"),
        )
        _replace_module(model, module_name, replacement)
        records.append(
            GQAVOReplacementRecord(
                layer_index=layer_index,
                module_name=module_name,
                head_dim=replacement.head_dim,
                value_head_dim=replacement.value_head_dim,
                dense_v_width=replacement.num_key_value_heads * replacement.head_dim,
                compressed_v_width=replacement.v_proj.out_features,
                dense_o_width=replacement.num_attention_heads * replacement.head_dim,
                compressed_o_width=replacement.o_proj.in_features,
            )
        )
    return records
