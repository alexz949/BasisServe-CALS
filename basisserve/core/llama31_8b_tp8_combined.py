"""Llama-3.1-8B TP8 Joint-ALS V96 with B16R16 exact-K routing."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from torch.utils.cpp_extension import load
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from basisserve.kernels.feature_ragged_allgather import FeatureRaggedCommunicator
from basisserve.kernels.mapped_host_paged_attention import (
    append_mapped_host_key,
    conditional_router_append_decode,
    gpu_paged_attention,
    mapped_host_bf16_empty,
    mapped_host_device_pointer,
)
from basisserve.kernels.ragged_allgather import StaticRaggedPlan
from basisserve.kernels.slot_indexed_attention import slot_indexed_attention
from benchmarks.system.fused_candidates import candidates as fused_candidates
from benchmarks.system.two_stage_router import Metadata
from evaluation.chunked_prefill_mlp import ChunkedTokenwise


TP_SIZE = 8
QUERY_HEADS_PER_RANK = 4
KV_HEADS_PER_RANK = 1
HEAD_DIM = 128
VALUE_RANK = 96
BASE_RANK = 16
RESIDUAL_RANK = 16
PAGE_SIZE = 32
ROUTED_PAGES = 62
RECENT_TOKENS = 64
SUPPORT_TOKENS = ROUTED_PAGES * PAGE_SIZE + RECENT_TOKENS
ATTENTION_SPLITS = 16


class DecodeBreakdownRecorder:
    """Optional CUDA-event attribution for the TP8 decode hot path."""

    def __init__(self) -> None:
        self.enabled = False
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def begin(self, name: str) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
        if not self.enabled:
            return None
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        self.events.setdefault(name, []).append((begin, end))
        return begin, end

    @staticmethod
    def end(pair: tuple[torch.cuda.Event, torch.cuda.Event] | None) -> None:
        if pair is not None:
            pair[1].record()

    def summary(self, measured_steps: int) -> dict[str, dict[str, float | int]]:
        assert measured_steps > 0
        return {
            name: {
                "calls": len(pairs),
                "calls_per_step": len(pairs) / measured_steps,
                "mean_ms_per_step": sum(begin.elapsed_time(end) for begin, end in pairs)
                / measured_steps,
            }
            for name, pairs in sorted(self.events.items())
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_decode_postprocess_extension() -> object:
    source = Path(__file__).resolve().parents[1] / "kernels/csrc/fused_decode_postprocess.cu"
    digest = _sha256(source)[:10]
    return load(
        name=f"basisserve_fused_decode_postprocess_{digest}",
        sources=[str(source)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math"],
    )


def _local_router_factors(
    root: Path, layer: int, rank: int, *, device: torch.device
) -> dict[str, torch.Tensor]:
    tensors = load_file(str(root / f"layer_{layer:03d}.safetensors"))
    selected = {}
    for name, tensor in tensors.items():
        width = QUERY_HEADS_PER_RANK if name.startswith("residual_query") else 1
        selected[name] = tensor.narrow(0, rank * width, width).to(
            device=device, dtype=torch.bfloat16
        ).contiguous()
    assert tuple(selected["base_left_b16"].shape) == (1, VALUE_RANK, BASE_RANK)
    assert tuple(selected["residual_query_b16_r16"].shape) == (
        QUERY_HEADS_PER_RANK,
        HEAD_DIM,
        RESIDUAL_RANK,
    )
    return selected


def _transformed_factors(
    factor_root: Path,
    layer: int,
    rank: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = load_file(str(factor_root / f"layer_{layer:03d}.safetensors"))
    encoder = value["value_coordinate_encoders"]
    decoder = value["head_output_decoders"]
    assert tuple(encoder.shape) == (8, HEAD_DIM, VALUE_RANK)
    assert tuple(decoder.shape) == (32, VALUE_RANK, 4096)
    return (
        encoder[rank].to(device=device, dtype=torch.bfloat16).contiguous(),
        decoder.reshape(32 * VALUE_RANK, 4096)
        .to(device=device, dtype=torch.bfloat16)
        .contiguous(),
    )


class DenseTP8Attention(nn.Module):
    """Original dense K128/V128 attention for the matched TP8 control."""

    def __init__(self, original: nn.Module, *, batch: int, capacity: int):
        super().__init__()
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        self.layer_idx = int(original.layer_idx)
        self.scaling = float(original.scaling)
        self.capacity = int(capacity)
        self.length = 0
        shape = (batch, KV_HEADS_PER_RANK, capacity, HEAD_DIM)
        self.register_buffer(
            "key_cache", torch.empty(*shape, device="cuda", dtype=torch.bfloat16), persistent=False
        )
        self.register_buffer(
            "value_cache", torch.empty(*shape, device="cuda", dtype=torch.bfloat16), persistent=False
        )
        self.page_ids = torch.arange(
            math.ceil(capacity / PAGE_SIZE), device="cuda", dtype=torch.long
        )[None, None].expand(batch, 1, -1).contiguous()
        self.attention_workspace = torch.empty(
            batch * QUERY_HEADS_PER_RANK,
            ATTENTION_SPLITS,
            HEAD_DIM + 2,
            device="cuda",
            dtype=torch.float32,
        )
        self.attention_output = torch.empty(
            batch, QUERY_HEADS_PER_RANK, 1, HEAD_DIM,
            device="cuda", dtype=torch.bfloat16,
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
        assert end <= self.capacity and (start == 0 or tokens == 1)
        cos, sin = position_embeddings
        key = torch.empty(
            batch, KV_HEADS_PER_RANK, tokens, HEAD_DIM,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        value = torch.empty_like(key)
        for left in range(0, tokens, 2048):
            right = min(left + 2048, tokens)
            block = hidden_states[:, left:right]
            raw_key = self.k_proj(block).view(batch, right - left, 1, HEAD_DIM).transpose(1, 2)
            _, key[:, :, left:right] = apply_rotary_pos_emb(
                raw_key, raw_key, cos[:, left:right], sin[:, left:right]
            )
            value[:, :, left:right] = self.v_proj(block).view(
                batch, right - left, 1, HEAD_DIM
            ).transpose(1, 2)
        self.key_cache[:, :, start:end].copy_(key)
        self.value_cache[:, :, start:end].copy_(value)
        self.length = end
        for left in range(0, tokens, 2048):
            right = min(left + 2048, tokens)
            query = self.q_proj(hidden_states[:, left:right]).view(
                batch, right - left, QUERY_HEADS_PER_RANK, HEAD_DIM
            ).transpose(1, 2)
            query, _ = apply_rotary_pos_emb(
                query, query, cos[:, left:right], sin[:, left:right]
            )
            if start == 0:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    attention = F.scaled_dot_product_attention(
                        query,
                        key[:, :, :right],
                        value[:, :, :right],
                        attn_mask=causal_lower_right(right - left, right),
                        scale=self.scaling,
                        enable_gqa=True,
                    )
            else:
                pages = math.ceil(end / PAGE_SIZE)
                attention = gpu_paged_attention(
                    self.key_cache,
                    query,
                    self.value_cache,
                    self.page_ids[:, :, :pages],
                    sequence_length=end,
                    page_size=PAGE_SIZE,
                    splits=ATTENTION_SPLITS,
                    workspace=self.attention_workspace,
                    output=self.attention_output,
                    scale=self.scaling,
                )
            projected = self.o_proj(
                attention.transpose(1, 2).reshape(batch, right - left, -1).contiguous()
            )
            hidden_states[:, left:right].copy_(projected)
        return hidden_states, None


class JointALSTP8Attention(nn.Module):
    """Uniform V96 attention, optionally with two-stage exact-K routing."""

    def __init__(
        self,
        original: nn.Module,
        *,
        arm: str,
        batch: int,
        capacity: int,
        factor_root: Path,
        router_root: Path,
        rank: int,
        communicator: FeatureRaggedCommunicator,
        router_extension: object | None,
        fine_router_extension: object | None,
        postprocess_extension: object | None,
        shared_rope: dict[str, torch.Tensor],
        breakdown: DecodeBreakdownRecorder | None,
    ):
        super().__init__()
        assert arm in ("als_full", "basis_joint")
        self.arm = arm
        self.layer_idx = int(original.layer_idx)
        self.scaling = float(original.scaling)
        self.capacity = int(capacity)
        self.length = 0
        self.shared_rope = shared_rope
        self.router_extension = router_extension
        self.fine_router_extension = fine_router_extension
        self.postprocess_extension = postprocess_extension
        self.breakdown = breakdown
        device = original.q_proj.weight.device

        encoder, decoder = _transformed_factors(
            factor_root, self.layer_idx, rank, device=device
        )
        dense_value_weight = original.v_proj.weight.detach()
        if hasattr(dense_value_weight, "to_local"):
            dense_value_weight = dense_value_weight.to_local()
        assert tuple(dense_value_weight.shape) == (HEAD_DIM, 4096), tuple(
            dense_value_weight.shape
        )
        folded = (encoder.float().T @ dense_value_weight.float()).to(torch.bfloat16)
        query_weight = original.q_proj.weight.detach()
        key_weight = original.k_proj.weight.detach()
        if hasattr(query_weight, "to_local"):
            query_weight = query_weight.to_local()
        if hasattr(key_weight, "to_local"):
            key_weight = key_weight.to_local()
        assert tuple(query_weight.shape) == (QUERY_HEADS_PER_RANK * HEAD_DIM, 4096)
        assert tuple(key_weight.shape) == (KV_HEADS_PER_RANK * HEAD_DIM, 4096)
        self.register_buffer(
            "qkv_weight",
            torch.cat((query_weight, key_weight, folded), dim=0).contiguous(),
            persistent=False,
        )
        self.register_buffer("global_decoder", decoder, persistent=False)

        cache_shape = (batch, KV_HEADS_PER_RANK, capacity)
        self.register_buffer(
            "value_cache",
            torch.empty(*cache_shape, VALUE_RANK, device=device, dtype=torch.bfloat16),
            persistent=False,
        )
        if arm == "als_full":
            self.register_buffer(
                "key_cache",
                torch.empty(*cache_shape, HEAD_DIM, device=device, dtype=torch.bfloat16),
                persistent=False,
            )
            self.host_key = None
            self.page_ids = torch.arange(
                math.ceil(capacity / PAGE_SIZE), device=device, dtype=torch.long
            )[None, None].expand(batch, 1, -1).contiguous()
        else:
            self.key_cache = None
            self.host_key = mapped_host_bf16_empty(
                batch=batch, kv_heads=KV_HEADS_PER_RANK, capacity=capacity
            )
            self.key_pointer = mapped_host_device_pointer(self.host_key)
            self.router_factors = _local_router_factors(
                router_root, self.layer_idx, rank, device=device
            )
            self.register_buffer(
                "residual_cache",
                torch.empty(*cache_shape, RESIDUAL_RANK, device=device, dtype=torch.bfloat16),
                persistent=False,
            )
            coordinate_base_left = torch.zeros(
                1, VALUE_RANK, BASE_RANK, device=device, dtype=torch.bfloat16
            )
            coordinate_base_left[0, :BASE_RANK].copy_(
                torch.eye(BASE_RANK, device=device, dtype=torch.bfloat16)
            )
            self.register_buffer("coordinate_base_left", coordinate_base_left, persistent=False)
            maximum_pages = math.ceil(capacity / PAGE_SIZE)
            self.query_code = torch.empty(
                batch, 1, QUERY_HEADS_PER_RANK, RESIDUAL_RANK,
                device=device, dtype=torch.bfloat16,
            )
            self.candidate_ids = torch.empty(batch, 1, 512, device=device, dtype=torch.long)
            self.fine_router_output = torch.empty(
                batch, 1, QUERY_HEADS_PER_RANK, min(512, maximum_pages),
                device=device, dtype=torch.float32,
            )
            self.selected_pages = torch.empty(
                batch, 1, ROUTED_PAGES, device=device, dtype=torch.long
            )
            self.support_ids = torch.empty(
                batch, 1, SUPPORT_TOKENS, device=device, dtype=torch.long
            )
            self.slot_key_cache = torch.empty(
                batch, 1, SUPPORT_TOKENS, HEAD_DIM, device=device, dtype=torch.bfloat16
            )
            self.slot_resident = torch.full(
                (batch, 1, SUPPORT_TOKENS), -1, device=device, dtype=torch.long
            )
            self.slot_lookup = torch.full(
                (batch, 1, capacity), -1, device=device, dtype=torch.int32
            )
            self.selected_slots = torch.empty_like(self.support_ids)
            self.slot_missing = torch.empty_like(self.support_ids, dtype=torch.int32)
            self.slot_counts = torch.empty(batch, 1, 2, device=device, dtype=torch.int32)
            self.metadata: Metadata | None = None

        self.plan = StaticRaggedPlan.from_source_widths(
            (QUERY_HEADS_PER_RANK * VALUE_RANK,) * TP_SIZE
        )
        self.decode_allgather = communicator.prepare_uniform(
            self.plan, tokens=batch, dtype=torch.bfloat16, backend="uniform_nccl"
        )
        self.decode_output = self.decode_allgather.local_feature_major_view_fast().T.view(
            batch, QUERY_HEADS_PER_RANK, 1, VALUE_RANK
        )
        if arm == "als_full":
            self.attention_workspace = torch.empty(
                batch * QUERY_HEADS_PER_RANK,
                ATTENTION_SPLITS,
                VALUE_RANK + 2,
                device=device,
                dtype=torch.float32,
            )
        else:
            self.slot_workspace = (
                torch.empty(
                    batch, QUERY_HEADS_PER_RANK, ATTENTION_SPLITS, VALUE_RANK,
                    device=device, dtype=torch.float32,
                ),
                torch.empty(
                    batch, QUERY_HEADS_PER_RANK, ATTENTION_SPLITS,
                    device=device, dtype=torch.float32,
                ),
                self.decode_output,
            )

    @property
    def base_cache(self) -> torch.Tensor:
        return self.value_cache[..., :BASE_RANK]

    @property
    def query_weight(self) -> torch.Tensor:
        return self.qkv_weight[: QUERY_HEADS_PER_RANK * HEAD_DIM]

    @property
    def key_weight(self) -> torch.Tensor:
        left = QUERY_HEADS_PER_RANK * HEAD_DIM
        return self.qkv_weight[left : left + KV_HEADS_PER_RANK * HEAD_DIM]

    @property
    def value_weight(self) -> torch.Tensor:
        left = (QUERY_HEADS_PER_RANK + KV_HEADS_PER_RANK) * HEAD_DIM
        return self.qkv_weight[left:]

    def _project_output(self, attention: torch.Tensor, *, decode: bool) -> torch.Tensor:
        batch, heads, tokens, width = attention.shape
        assert heads == QUERY_HEADS_PER_RANK and width == VALUE_RANK
        prepared = self.decode_allgather if decode else self.decode_allgather_for(tokens * batch)
        if not decode:
            prepared.local_feature_major_view_fast().copy_(
                attention.permute(1, 3, 0, 2).reshape(heads * width, batch * tokens)
            )
        collective = self.breakdown.begin("output_allgather") if self.breakdown else None
        full = prepared.gather_inplace_fast()
        DecodeBreakdownRecorder.end(collective)
        decoder = self.breakdown.begin("output_decoder") if self.breakdown else None
        output = (full.T @ self.global_decoder).reshape(batch, tokens, 4096)
        DecodeBreakdownRecorder.end(decoder)
        return output

    def decode_allgather_for(self, tokens: int):
        return self._communicator.prepare_uniform(
            self.plan, tokens=tokens, dtype=torch.bfloat16, backend="uniform_nccl"
        )

    def bind_communicator(self, communicator: FeatureRaggedCommunicator) -> None:
        object.__setattr__(self, "_communicator", communicator)

    def _write_prefill_codes(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        cos_half: torch.Tensor,
        sin_half: torch.Tensor,
        start: int,
    ) -> None:
        stop = start + key.shape[2]
        self.value_cache[:, :, start:stop].copy_(value)
        factors = self.router_factors
        predicted = (
            self.base_cache[:, :, start:stop] @ factors["base_right_b16"]
            + factors["base_bias_b16"][None, :, None]
        ).to(torch.bfloat16)
        first = (
            predicted[..., :64] * cos_half[:, None]
            - predicted[..., 64:] * sin_half[:, None]
        ).to(torch.bfloat16)
        second = (
            predicted[..., 64:] * cos_half[:, None]
            + predicted[..., :64] * sin_half[:, None]
        ).to(torch.bfloat16)
        predicted = torch.cat((first, second), dim=-1)
        self.residual_cache[:, :, start:stop].copy_(
            (key - predicted) @ factors["residual_encoder_b16_r16"]
        )

    def _append_basis_decode(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        cos_half: torch.Tensor,
        sin_half: torch.Tensor,
        start: int,
    ) -> None:
        assert self.metadata is not None
        factors = self.router_factors
        conditional_router_append_decode(
            key,
            value,
            base_left=self.coordinate_base_left,
            base_right=factors["base_right_b16"],
            base_bias=factors["base_bias_b16"],
            residual_encoder=factors["residual_encoder_b16_r16"],
            rope_cos=cos_half,
            rope_sin=sin_half,
            value_cache=self.value_cache,
            base_cache=self.base_cache,
            residual_cache=self.residual_cache,
            rope_cos_cache=cos_half,
            rope_sin_cache=sin_half,
            start=start,
            write_rope=False,
            mapped_host_key=self.host_key,
            mapped_host_key_device_pointer=self.key_pointer,
            metadata_minimum=self.metadata.minimum,
            metadata_maximum=self.metadata.maximum,
            metadata_ring=self.metadata.ring,
            metadata_position=self.metadata.n,
            metadata_slot=self.metadata.slot,
        )
        self.metadata.commit_advance()

    def _basis_decode_attention(self, query: torch.Tensor, end: int) -> torch.Tensor:
        assert end > SUPPORT_TOKENS and self.metadata is not None
        historical = end - RECENT_TOKENS
        factors = self.router_factors
        coarse = self.breakdown.begin("router_coarse") if self.breakdown else None
        coarse_scores = self.metadata.scores(query)
        DecodeBreakdownRecorder.end(coarse)
        selection = self.breakdown.begin("router_candidates") if self.breakdown else None
        candidate_ids = fused_candidates(
            coarse_scores, historical, output=self.candidate_ids
        )
        DecodeBreakdownRecorder.end(selection)
        candidate_count = candidate_ids.shape[-1]
        route_args = (
            query,
            self.base_cache[:, :, :historical],
            self.residual_cache[:, :, :historical],
            factors["base_right_b16"],
            factors["base_bias_b16"],
            factors["residual_query_b16_r16"],
            self.shared_rope["cos"][:historical],
            self.shared_rope["sin"][:historical],
            self.query_code,
        )
        fine = self.breakdown.begin("router_fine") if self.breakdown else None
        self.fine_router_extension.conditional_router_page_lse(
            *route_args,
            self.fine_router_output[:, :, :, :candidate_count],
            self.scaling,
            False,
            candidate_ids,
        )
        DecodeBreakdownRecorder.end(fine)
        postprocess = self.breakdown.begin("router_postprocess") if self.breakdown else None
        self.postprocess_extension.select_pack_refresh(
            self.fine_router_output[:, :, :, :candidate_count],
            candidate_ids,
            self.selected_pages,
            self.support_ids,
            self.key_pointer,
            self.slot_key_cache,
            self.slot_resident,
            self.slot_lookup,
            self.selected_slots,
            self.slot_missing,
            self.slot_counts,
            int(historical),
            int(end),
        )
        DecodeBreakdownRecorder.end(postprocess)
        attention = self.breakdown.begin("sparse_attention") if self.breakdown else None
        output = slot_indexed_attention(
            query,
            self.slot_key_cache,
            self.value_cache,
            self.support_ids,
            self.selected_slots,
            self.slot_workspace,
            scale=self.scaling,
        )
        DecodeBreakdownRecorder.end(attention)
        return output

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
        assert end <= self.capacity and (start == 0 or tokens == 1)
        cos, sin = position_embeddings

        if start != 0:
            qkv = self.breakdown.begin("qkv_projection_rope_append") if self.breakdown else None
            packed = F.linear(hidden_states, self.qkv_weight)
            query_end = QUERY_HEADS_PER_RANK * HEAD_DIM
            key_end = query_end + KV_HEADS_PER_RANK * HEAD_DIM
            raw_query = packed[..., :query_end].view(
                batch, tokens, QUERY_HEADS_PER_RANK, HEAD_DIM
            ).transpose(1, 2)
            raw_key = packed[..., query_end:key_end].view(
                batch, tokens, KV_HEADS_PER_RANK, HEAD_DIM
            ).transpose(1, 2)
            coordinates = packed[..., key_end:].view(
                batch, tokens, KV_HEADS_PER_RANK, VALUE_RANK
            ).transpose(1, 2)
            query, key = apply_rotary_pos_emb(raw_query, raw_key, cos, sin)
            if self.layer_idx == 0:
                self.shared_rope["cos"][start:end].copy_(cos[0, :, : HEAD_DIM // 2])
                self.shared_rope["sin"][start:end].copy_(sin[0, :, : HEAD_DIM // 2])
            if self.arm == "basis_joint":
                self._append_basis_decode(
                    key,
                    coordinates,
                    cos[0, :, : HEAD_DIM // 2].contiguous(),
                    sin[0, :, : HEAD_DIM // 2].contiguous(),
                    start,
                )
            else:
                self.key_cache[:, :, start:end].copy_(key)
                self.value_cache[:, :, start:end].copy_(coordinates)
            self.length = end
            DecodeBreakdownRecorder.end(qkv)
            if self.arm == "als_full":
                pages = math.ceil(end / PAGE_SIZE)
                attention = gpu_paged_attention(
                    self.key_cache,
                    query,
                    self.value_cache,
                    self.page_ids[:, :, :pages],
                    sequence_length=end,
                    page_size=PAGE_SIZE,
                    splits=ATTENTION_SPLITS,
                    workspace=self.attention_workspace,
                    output=self.decode_output,
                    scale=self.scaling,
                )
            else:
                attention = self._basis_decode_attention(query, end)
            projected = self._project_output(attention, decode=True)
            hidden_states.copy_(projected)
            return hidden_states, None

        append = self.breakdown.begin("kv_projection_append") if self.breakdown else None
        key = torch.empty(
            batch, 1, tokens, HEAD_DIM,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        padded_value = torch.zeros(
            batch, 1, tokens, HEAD_DIM,
            device=hidden_states.device, dtype=hidden_states.dtype,
        ) if start == 0 else None
        for left in range(0, tokens, 2048):
            right = min(left + 2048, tokens)
            block = hidden_states[:, left:right]
            raw_key = F.linear(block, self.key_weight).view(
                batch, right - left, 1, HEAD_DIM
            ).transpose(1, 2)
            _, post_key = apply_rotary_pos_emb(
                raw_key, raw_key, cos[:, left:right], sin[:, left:right]
            )
            key[:, :, left:right].copy_(post_key)
            coordinates = F.linear(block, self.value_weight).view(
                batch, right - left, 1, VALUE_RANK
            ).transpose(1, 2)
            cos_rows = cos[:, left:right, : HEAD_DIM // 2]
            sin_rows = sin[:, left:right, : HEAD_DIM // 2]
            if self.arm == "basis_joint":
                self._write_prefill_codes(post_key, coordinates, cos_rows, sin_rows, start + left)
            else:
                self.value_cache[:, :, start + left : start + right].copy_(coordinates)
            if padded_value is not None:
                padded_value[:, :, left:right, :VALUE_RANK].copy_(coordinates)
        if self.layer_idx == 0:
            self.shared_rope["cos"][start:end].copy_(cos[0, :, : HEAD_DIM // 2])
            self.shared_rope["sin"][start:end].copy_(sin[0, :, : HEAD_DIM // 2])
        if self.arm == "basis_joint":
            if start == 0:
                append_mapped_host_key(self.host_key, key, start=0)
                self.metadata = Metadata(key, self.capacity)
            else:
                self._append_basis_decode(
                    key,
                    coordinates,
                    cos[0, :1, : HEAD_DIM // 2].contiguous(),
                    sin[0, :1, : HEAD_DIM // 2].contiguous(),
                    start,
                )
        else:
            self.key_cache[:, :, start:end].copy_(key)
        self.length = end
        DecodeBreakdownRecorder.end(append)

        for left in range(0, tokens, 2048):
            right = min(left + 2048, tokens)
            query_projection = (
                self.breakdown.begin("query_projection_rope") if self.breakdown else None
            )
            query = F.linear(hidden_states[:, left:right], self.query_weight).view(
                batch, right - left, QUERY_HEADS_PER_RANK, HEAD_DIM
            ).transpose(1, 2)
            query, _ = apply_rotary_pos_emb(
                query, query, cos[:, left:right], sin[:, left:right]
            )
            DecodeBreakdownRecorder.end(query_projection)
            if start == 0:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    attention = F.scaled_dot_product_attention(
                        query,
                        key[:, :, :right],
                        padded_value[:, :, :right],
                        attn_mask=causal_lower_right(right - left, right),
                        scale=self.scaling,
                        enable_gqa=True,
                    )[..., :VALUE_RANK]
                projected = self._project_output(attention, decode=False)
            elif self.arm == "als_full":
                pages = math.ceil(end / PAGE_SIZE)
                attention = gpu_paged_attention(
                    self.key_cache,
                    query,
                    self.value_cache,
                    self.page_ids[:, :, :pages],
                    sequence_length=end,
                    page_size=PAGE_SIZE,
                    splits=ATTENTION_SPLITS,
                    workspace=self.attention_workspace,
                    output=self.decode_output,
                    scale=self.scaling,
                )
                projected = self._project_output(attention, decode=True)
            else:
                attention = self._basis_decode_attention(query, end)
                projected = self._project_output(attention, decode=True)
            hidden_states[:, left:right].copy_(projected)
        return hidden_states, None


@torch.inference_mode()
def decoder_layer_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache=False,
    position_embeddings=None,
    **kwargs,
):
    del position_ids, use_cache, kwargs
    breakdown = getattr(self.self_attn, "breakdown", None)
    residual = hidden_states
    pre_norm = breakdown.begin("pre_attention_norm") if breakdown else None
    hidden_states = self.input_layernorm(hidden_states)
    DecodeBreakdownRecorder.end(pre_norm)
    attention = breakdown.begin("attention_total") if breakdown else None
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
    )
    DecodeBreakdownRecorder.end(attention)
    residual_add = breakdown.begin("attention_residual") if breakdown else None
    hidden_states.add_(residual)
    DecodeBreakdownRecorder.end(residual_add)
    mlp = breakdown.begin("mlp_block") if breakdown else None
    for left in range(0, hidden_states.shape[1], 1024):
        block = hidden_states[:, left : left + 1024]
        block.add_(self.mlp(self.post_attention_layernorm(block)))
    DecodeBreakdownRecorder.end(mlp)
    return hidden_states


def install_tp8_combined(
    model: nn.Module,
    *,
    arm: str,
    batch: int,
    capacity: int,
    factor_root: Path | None,
    router_root: Path | None,
    rank: int,
    communicator: FeatureRaggedCommunicator | None,
    router_extension: object | None = None,
    fine_router_extension: object | None = None,
    postprocess_extension: object | None = None,
    breakdown: DecodeBreakdownRecorder | None = None,
) -> list[nn.Module]:
    assert arm in ("dense", "als_full", "basis_joint")
    assert model.config.model_type == "llama" and model.config.num_hidden_layers == 32
    if arm != "dense":
        assert factor_root is not None and router_root is not None and communicator is not None
    if arm == "basis_joint":
        assert all(
            item is not None
            for item in (
                router_extension,
                fine_router_extension,
                postprocess_extension,
            )
        )
    shared_rope = {
        "cos": torch.empty(capacity, 64, device="cuda", dtype=torch.bfloat16),
        "sin": torch.empty(capacity, 64, device="cuda", dtype=torch.bfloat16),
    }
    modules = []
    for layer in model.model.layers:
        if arm == "dense":
            attention = DenseTP8Attention(layer.self_attn, batch=batch, capacity=capacity)
        else:
            attention = JointALSTP8Attention(
                layer.self_attn,
                arm=arm,
                batch=batch,
                capacity=capacity,
                factor_root=factor_root,
                router_root=router_root,
                rank=rank,
                communicator=communicator,
                router_extension=router_extension,
                fine_router_extension=fine_router_extension,
                postprocess_extension=postprocess_extension,
                shared_rope=shared_rope,
                breakdown=breakdown,
            )
            attention.bind_communicator(communicator)
        layer.self_attn = attention
        layer.input_layernorm = ChunkedTokenwise(layer.input_layernorm, chunk_size=2048)
        layer.forward = MethodType(decoder_layer_forward, layer)
        modules.append(attention)
    model.model.norm = ChunkedTokenwise(model.model.norm, chunk_size=2048)
    return modules


__all__ = [
    "BASE_RANK",
    "DecodeBreakdownRecorder",
    "PAGE_SIZE",
    "RECENT_TOKENS",
    "RESIDUAL_RANK",
    "ROUTED_PAGES",
    "SUPPORT_TOKENS",
    "TP_SIZE",
    "VALUE_RANK",
    "install_tp8_combined",
    "load_decode_postprocess_extension",
]
