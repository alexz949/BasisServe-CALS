"""End-to-end TP1 Llama-3.1-8B sparse-decode benchmark.

The prefill is exact Flash SDPA. Decode uses the calibrated B16R16 Page32
router, 1,984 routed tokens plus the exact most-recent 64 tokens, and the
production C1 attention kernel.  Teacher-forced decode tokens make legacy and
optimized runs consume identical model inputs.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from basisserve.kernels.mapped_host_paged_attention import _load_extension
from basisserve.kernels.slot_indexed_attention import slot_indexed_attention
from benchmarks.system.bench_router_append_v6 import _build_shared_router_baseline
from benchmarks.system.fused_candidates import candidates as fused_candidates
from benchmarks.system.two_stage_router import Metadata, compile_fine
from evaluation.chunked_prefill_mlp import ChunkedTokenwise


MODEL_PATH = Path(
    "/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
FACTOR_ROOT = Path("/workspace/runs/l31-router-source/v128-router/ours_b16r16")
TOKENS_PATH = Path("/workspace/runs/l31-cal128/calibration/windows.safetensors")
PAGE_SIZE = 32
ROUTED_PAGES = 62
RECENT_TOKENS = 64
SUPPORT_TOKENS = ROUTED_PAGES * PAGE_SIZE + RECENT_TOKENS
SPLITS = 32
SLOT_SPLITS = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


@lru_cache(maxsize=1)
def _load_slot_extension() -> object:
    source = REPOSITORY_ROOT / "basisserve/kernels/csrc/persistent_key_slots.cu"
    digest = _sha256(source)[:10]
    return load(
        name=f"basisserve_persistent_key_slots_{digest}",
        sources=[str(source)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
    )


@lru_cache(maxsize=1)
def _load_postprocess_extension() -> object:
    source = REPOSITORY_ROOT / "basisserve/kernels/csrc/fused_decode_postprocess.cu"
    digest = _sha256(source)[:10]
    return load(
        name=f"basisserve_fused_decode_postprocess_{digest}",
        sources=[str(source)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math"],
    )


class TP1SparseAttention(torch.nn.Module):
    """Llama attention with exact prefill and native Page32 sparse decode."""

    def __init__(
        self,
        original: torch.nn.Module,
        factors: dict[str, torch.Tensor],
        *,
        capacity: int,
        storage: str,
        mode: str,
        router_extension: object,
        fine_router_extension: object | None,
        attention_extension: object,
        routing: str,
        key_reuse: bool,
        slot_extension: object | None,
        postprocess_extension: object | None,
        shared_rope: dict[str, torch.Tensor],
        profile_components: bool,
        validate: bool,
    ) -> None:
        super().__init__()
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        self.config = original.config
        self.layer_idx = int(original.layer_idx)
        self.head_dim = int(original.head_dim)
        self.scaling = float(original.scaling)
        self.attention_dropout = float(original.attention_dropout)
        self.router_extension = router_extension
        self.fine_router_extension = fine_router_extension
        self.attention_extension = attention_extension
        self.shared_rope = shared_rope
        self.storage = storage
        self.mode = mode
        self.routing = routing
        self.key_reuse = bool(key_reuse)
        self.slot_extension = slot_extension
        self.postprocess_extension = postprocess_extension
        self.capacity = int(capacity)
        self.profile_components = bool(profile_components)
        self.validate = bool(validate)
        self.validated = False
        self.router_validated = False
        self.full_router_page_recall = None
        self.full_router_score_recall = None
        self.length = 0
        self.metadata = None

        self.register_buffer(
            "base_left", factors["base_left_b16"].to("cuda", torch.bfloat16).contiguous()
        )
        self.register_buffer(
            "base_right", factors["base_right_b16"].to("cuda", torch.bfloat16).contiguous()
        )
        self.register_buffer(
            "base_bias", factors["base_bias_b16"].to("cuda", torch.bfloat16).contiguous()
        )
        self.register_buffer(
            "residual_encoder",
            factors["residual_encoder_b16_r16"].to("cuda", torch.bfloat16).contiguous(),
        )
        self.register_buffer(
            "residual_query",
            factors["residual_query_b16_r16"].to("cuda", torch.bfloat16).contiguous(),
        )

        cache_shape = (1, 8, capacity)
        self.value_cache = torch.empty(*cache_shape, 128, device="cuda", dtype=torch.bfloat16)
        self.base_cache = torch.empty(*cache_shape, 16, device="cuda", dtype=torch.bfloat16)
        self.residual_cache = torch.empty(*cache_shape, 16, device="cuda", dtype=torch.bfloat16)
        if storage == "local":
            self.key_cache = torch.empty(
                *cache_shape, 128, device="cuda", dtype=torch.bfloat16
            )
            self.host_key = None
            self.key_pointer = int(self.key_cache.data_ptr())
        else:
            self.key_cache = None
            self.host_key = router_extension.mapped_host_bf16_empty(1, 8, capacity, 128)
            self.key_pointer = int(router_extension.device_pointer(self.host_key))

        maximum_pages = math.ceil(capacity / PAGE_SIZE)
        self.query_code = torch.empty(1, 8, 4, 16, device="cuda", dtype=torch.bfloat16)
        self.router_output = torch.empty(1, 8, 4, maximum_pages, device="cuda")
        self.candidate_ids = torch.empty(1, 8, 512, device="cuda", dtype=torch.long)
        self.fine_router_output = torch.empty(1, 8, 4, 512, device="cuda")
        self.selected_pages = torch.empty(
            1, 8, ROUTED_PAGES, device="cuda", dtype=torch.long
        )
        self.reference_pages = torch.empty_like(self.selected_pages) if self.validate else None
        self.support_ids = torch.empty(
            1, 8, SUPPORT_TOKENS, device="cuda", dtype=torch.long
        )
        self.page_offsets = torch.arange(PAGE_SIZE, device="cuda", dtype=torch.long)
        self.token_range = torch.arange(capacity, device="cuda", dtype=torch.long)
        self.attention_workspace = torch.empty(
            32, SPLITS, 130, device="cuda", dtype=torch.float32
        )
        self.attention_output = torch.empty(
            1, 32, 1, 128, device="cuda", dtype=torch.bfloat16
        )
        if self.key_reuse:
            self.slot_key_cache = torch.empty(
                1, 8, SUPPORT_TOKENS, 128, device="cuda", dtype=torch.bfloat16
            )
            self.slot_resident = torch.full(
                (1, 8, SUPPORT_TOKENS), -1, device="cuda", dtype=torch.long
            )
            self.slot_lookup = torch.full(
                (1, 8, capacity), -1, device="cuda", dtype=torch.int32
            )
            self.selected_slots = torch.empty_like(self.support_ids)
            self.slot_missing = torch.empty_like(self.support_ids, dtype=torch.int32)
            self.slot_counts = torch.empty(1, 8, 2, device="cuda", dtype=torch.int32)
            self.slot_workspace = (
                torch.empty(1, 32, SLOT_SPLITS, 128, device="cuda", dtype=torch.float32),
                torch.empty(1, 32, SLOT_SPLITS, device="cuda", dtype=torch.float32),
                self.attention_output,
            )
        else:
            self.slot_key_cache = None
            self.slot_resident = None
            self.slot_lookup = None
            self.selected_slots = None
            self.slot_missing = None
            self.slot_counts = None
            self.slot_workspace = None
        self.single_rope = torch.empty(1, 64, device="cuda", dtype=torch.bfloat16)
        self.profile_events = {}
        if self.profile_components:
            names = [
                "attention_block",
                "cache_append",
                "router_scan",
                "page_selection",
                "sparse_attention",
            ]
            if self.key_reuse:
                names.append("key_refresh")
            for name in names:
                self.profile_events[name] = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )

    def _start(self, name: str) -> None:
        if self.profile_components:
            self.profile_events[name][0].record()

    def _end(self, name: str) -> None:
        if self.profile_components:
            self.profile_events[name][1].record()

    def component_times(self) -> dict[str, float]:
        if not self.profile_components:
            return {}
        return {
            name: begin.elapsed_time(end)
            for name, (begin, end) in self.profile_events.items()
        }

    def _store_exact_key(self, key: torch.Tensor, start: int) -> None:
        if self.storage == "local":
            self.key_cache[:, :, start : start + key.shape[2]].copy_(key)
        else:
            self.router_extension.append(self.host_key, key, int(start))

    def _write_codes(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        cos_half: torch.Tensor,
        sin_half: torch.Tensor,
        start: int,
    ) -> None:
        stop = start + int(key.shape[2])
        base = torch.einsum("bhtd,hdr->bhtr", value, self.base_left)
        predicted = (
            torch.einsum("bhtr,hrd->bhtd", base, self.base_right)
            + self.base_bias[None, :, None, :]
        ).to(torch.bfloat16)
        cos = cos_half[None, None]
        sin = sin_half[None, None]
        first = (
            (predicted[..., :64] * cos).to(torch.bfloat16)
            - (predicted[..., 64:] * sin).to(torch.bfloat16)
        ).to(torch.bfloat16)
        second = (
            (predicted[..., 64:] * cos).to(torch.bfloat16)
            + (predicted[..., :64] * sin).to(torch.bfloat16)
        ).to(torch.bfloat16)
        predicted_rotary = torch.cat((first, second), dim=-1)
        residual = torch.einsum(
            "bhtd,hdr->bhtr", (key - predicted_rotary).to(torch.bfloat16), self.residual_encoder
        )
        self.value_cache[:, :, start:stop].copy_(value)
        self.base_cache[:, :, start:stop].copy_(base)
        self.residual_cache[:, :, start:stop].copy_(residual)

    def _append_decode(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        cos_half: torch.Tensor,
        sin_half: torch.Tensor,
        start: int,
    ) -> None:
        if self.mode == "legacy":
            self._write_codes(key, value, cos_half, sin_half, start)
            self._store_exact_key(key, start)
            return
        mapped_pointer = self.key_pointer if self.storage == "offload" else 0
        mapped_capacity = self.capacity if self.storage == "offload" else 0
        metadata_minimum = self.metadata.minimum if self.metadata is not None else self.single_rope
        metadata_maximum = self.metadata.maximum if self.metadata is not None else self.single_rope
        metadata_ring = self.metadata.ring if self.metadata is not None else self.single_rope
        metadata_position = self.metadata.n if self.metadata is not None else 0
        metadata_slot = self.metadata.slot if self.metadata is not None else 0
        self.router_extension.conditional_router_append_decode(
            key,
            value,
            self.base_left,
            self.base_right,
            self.base_bias,
            self.residual_encoder,
            cos_half,
            sin_half,
            self.value_cache,
            self.base_cache,
            self.residual_cache,
            self.single_rope,
            self.single_rope,
            int(start),
            False,
            mapped_pointer,
            mapped_capacity,
            metadata_minimum,
            metadata_maximum,
            metadata_ring,
            metadata_position,
            metadata_slot,
            self.metadata is not None,
        )
        if self.storage == "local":
            self.key_cache[:, :, start : start + 1].copy_(key)

    def _validate_attention(self, query: torch.Tensor, end: int) -> None:
        ids = self.support_ids
        valid = ids >= 0
        gather_ids = ids.clamp_min(0)[:, :, :, None].expand(-1, -1, -1, 128)
        if self.storage == "local":
            selected_key = torch.gather(self.key_cache, 2, gather_ids)
        else:
            cpu_ids = gather_ids.cpu()
            selected_key = torch.gather(self.host_key, 2, cpu_ids).to("cuda")
        selected_value = torch.gather(self.value_cache, 2, gather_ids)
        selected_key = selected_key.repeat_interleave(4, dim=1)
        selected_value = selected_value.repeat_interleave(4, dim=1)
        scores = torch.matmul(
            query.float(), selected_key.transpose(-1, -2).float()
        ) * self.scaling
        scores.masked_fill_(~valid.repeat_interleave(4, dim=1)[:, :, None, :], -torch.inf)
        expected = torch.matmul(torch.softmax(scores, dim=-1), selected_value.float())
        torch.testing.assert_close(
            self.attention_output.float(), expected, rtol=0.025, atol=0.025
        )
        assert end <= self.capacity
        self.validated = True

    def _decode_attention(self, query: torch.Tensor, end: int) -> torch.Tensor:
        historical = end - RECENT_TOKENS
        pages = math.ceil(historical / PAGE_SIZE)
        selected_count = min(ROUTED_PAGES, pages)
        assert selected_count == ROUTED_PAGES
        self._start("router_scan")
        route_args = (
            query,
            self.base_cache[:, :, :historical],
            self.residual_cache[:, :, :historical],
            self.base_right,
            self.base_bias,
            self.residual_query,
            self.shared_rope["cos"][:historical],
            self.shared_rope["sin"][:historical],
            self.query_code,
        )
        if self.routing == "full":
            self.router_extension.conditional_router_page_lse(
                *route_args,
                self.router_output[:, :, :, :pages],
                self.scaling,
                False,
            )
        else:
            coarse_scores = self.metadata.scores(query)
            candidate_ids = fused_candidates(
                coarse_scores, historical, output=self.candidate_ids
            )
            candidate_count = candidate_ids.shape[-1]
            self.fine_router_extension.conditional_router_page_lse(
                *route_args,
                self.fine_router_output[:, :, :, :candidate_count],
                self.scaling,
                False,
                candidate_ids,
            )
        self._end("router_scan")
        self._start("page_selection")
        if self.routing == "full":
            self.router_extension.select_fixed_group_max_pages(
                self.router_output[:, :, :, :pages],
                self.selected_pages,
                ROUTED_PAGES,
                1,
                False,
            )
        else:
            self.postprocess_extension.select_and_pack(
                self.fine_router_output[:, :, :, :candidate_count],
                candidate_ids,
                self.selected_pages,
                self.support_ids,
                int(historical),
                int(end),
            )
            if self.validate and not self.router_validated:
                self.router_extension.select_fixed_group_max_pages(
                    self.fine_router_output[:, :, :, :candidate_count],
                    self.reference_pages,
                    ROUTED_PAGES,
                    1,
                    False,
                )
                expected_pages = torch.gather(
                    candidate_ids, 2, self.reference_pages
                )
                torch.testing.assert_close(
                    self.selected_pages, expected_pages, rtol=0, atol=0
                )
                self.router_extension.conditional_router_page_lse(
                    *route_args,
                    self.router_output[:, :, :, :pages],
                    self.scaling,
                    False,
                )
                self.router_extension.select_fixed_group_max_pages(
                    self.router_output[:, :, :, :pages],
                    self.reference_pages,
                    ROUTED_PAGES,
                    1,
                    False,
                )
                self.full_router_page_recall = float(
                    (
                        self.selected_pages[..., None]
                        == self.reference_pages[..., None, :]
                    )
                    .any(dim=-1)
                    .float()
                    .mean()
                )
                full_logits = self.router_output[:, :, :, :pages].clone()
                full_logits[..., 0] = -torch.inf
                full_group_scores = torch.softmax(full_logits, dim=-1).amax(dim=2)
                selected_scores = torch.gather(
                    full_group_scores, 2, self.reference_pages
                )
                covered = (
                    self.reference_pages[..., None]
                    == self.selected_pages[..., None, :]
                ).any(dim=-1)
                self.full_router_score_recall = float(
                    (selected_scores * covered).sum(dim=-1).div(
                        selected_scores.sum(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
                    ).mean()
                )
                expected_routed = (
                    self.selected_pages[..., None] * PAGE_SIZE + self.page_offsets
                ).flatten(-2)
                expected_routed.masked_fill_(expected_routed >= historical, -1)
                expected_support = torch.cat(
                    (
                        expected_routed,
                        self.token_range[historical:end][None, None, :].expand(1, 8, -1),
                    ),
                    dim=-1,
                )
                torch.testing.assert_close(
                    self.support_ids, expected_support, rtol=0, atol=0
                )
                self.router_validated = True
        if self.routing == "full":
            routed = self.support_ids[:, :, : ROUTED_PAGES * PAGE_SIZE].view(
                1, 8, ROUTED_PAGES, PAGE_SIZE
            )
            torch.add(
                self.selected_pages[..., None] * PAGE_SIZE,
                self.page_offsets,
                out=routed,
            )
            routed.masked_fill_(routed >= historical, -1)
            self.support_ids[:, :, ROUTED_PAGES * PAGE_SIZE :].copy_(
                self.token_range[historical:end][None, None, :].expand(1, 8, -1)
            )
        self._end("page_selection")
        if self.key_reuse:
            self._start("key_refresh")
            self.slot_extension.refresh(
                self.key_pointer,
                self.slot_key_cache,
                self.support_ids,
                self.slot_resident,
                self.slot_lookup,
                self.selected_slots,
                self.slot_missing,
                self.slot_counts,
                True,
            )
            self._end("key_refresh")
        self._start("sparse_attention")
        if self.key_reuse:
            slot_indexed_attention(
                query,
                self.slot_key_cache,
                self.value_cache,
                self.support_ids,
                self.selected_slots,
                self.slot_workspace,
                scale=self.scaling,
            )
        else:
            self.attention_extension.attention(
                self.key_pointer,
                self.capacity,
                query,
                self.value_cache,
                self.support_ids,
                self.attention_workspace,
                self.attention_output,
                int(end),
                self.scaling,
                SPLITS,
                self.value_cache,
                0,
            )
        self._end("sparse_attention")
        if self.validate and not self.validated:
            self._validate_attention(query, end)
        return self.attention_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask, past_key_values, kwargs
        batch, tokens, _ = hidden_states.shape
        assert batch == 1 and position_embeddings is not None
        start = self.length
        end = start + tokens
        assert end <= self.capacity
        hidden_shape = (batch, tokens, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        cos_rows = cos[0] if cos.ndim == 3 else cos
        sin_rows = sin[0] if sin.ndim == 3 else sin
        cos_half = cos_rows[:, :64].contiguous()
        sin_half = sin_rows[:, :64].contiguous()
        if self.layer_idx == 0:
            self.shared_rope["cos"][start:end].copy_(cos_half)
            self.shared_rope["sin"][start:end].copy_(sin_half)

        if start == 0:
            self._store_exact_key(key, 0)
            for offset in range(0, tokens, 2048):
                stop = min(tokens, offset + 2048)
                self._write_codes(
                    key[:, :, offset:stop],
                    value[:, :, offset:stop],
                    cos_half[offset:stop],
                    sin_half[offset:stop],
                    offset,
                )
            if self.routing == "two-stage":
                self.metadata = Metadata(key, self.capacity)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attention = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=True,
                    scale=self.scaling,
                    enable_gqa=True,
                )
        else:
            assert tokens == 1 and start >= RECENT_TOKENS
            self._start("attention_block")
            self._start("cache_append")
            self._append_decode(key, value, cos_half, sin_half, start)
            if self.routing == "two-stage":
                self.metadata.commit_advance()
            self._end("cache_append")
            attention = self._decode_attention(query, end)
            self._end("attention_block")

        self.length = end
        output = attention.transpose(1, 2).reshape(batch, tokens, -1).contiguous()
        return self.o_proj(output), None


def _load_tokens(context: int, decode_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    windows = load_file(str(TOKENS_PATH))["input_ids"]
    assert windows.ndim == 2 and windows.shape[0] >= 2
    assert windows.shape[1] >= context and windows.shape[1] >= decode_steps
    return windows[0, :context].clone(), windows[1, :decode_steps].clone()


def _install_sparse_attention(
    model: torch.nn.Module,
    *,
    capacity: int,
    storage: str,
    mode: str,
    router_extension: object,
    fine_router_extension: object | None,
    attention_extension: object,
    routing: str,
    key_reuse: bool,
    slot_extension: object | None,
    postprocess_extension: object | None,
    profile_components: bool,
    validate: bool,
) -> list[TP1SparseAttention]:
    shared_rope = {
        "cos": torch.empty(capacity, 64, device="cuda", dtype=torch.bfloat16),
        "sin": torch.empty(capacity, 64, device="cuda", dtype=torch.bfloat16),
    }
    installed = []
    for layer_index, layer in enumerate(model.model.layers):
        factors = load_file(str(FACTOR_ROOT / f"layer_{layer_index:03d}.safetensors"))
        replacement = TP1SparseAttention(
            layer.self_attn,
            factors,
            capacity=capacity,
            storage=storage,
            mode=mode,
            router_extension=router_extension,
            fine_router_extension=fine_router_extension,
            attention_extension=attention_extension,
            routing=routing,
            key_reuse=key_reuse,
            slot_extension=slot_extension,
            postprocess_extension=postprocess_extension,
            shared_rope=shared_rope,
            profile_components=profile_components,
            validate=validate,
        )
        layer.self_attn = replacement
        layer.mlp = ChunkedTokenwise(layer.mlp, chunk_size=1024)
        layer.input_layernorm = ChunkedTokenwise(layer.input_layernorm, chunk_size=2048)
        layer.post_attention_layernorm = ChunkedTokenwise(
            layer.post_attention_layernorm, chunk_size=2048
        )
        installed.append(replacement)
    model.model.norm = ChunkedTokenwise(model.model.norm, chunk_size=2048)
    return installed


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("legacy", "optimized"), required=True)
    parser.add_argument("--storage", choices=("local", "offload"), required=True)
    parser.add_argument("--routing", choices=("full", "two-stage"), default="full")
    parser.add_argument("--key-reuse", action="store_true")
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--measure-steps", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--tag", default="smoke")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/system_benchmarks/tp1_sparse_full_v6"),
    )
    args = parser.parse_args()
    assert args.length >= 4096 and args.warmup_steps >= 0 and args.measure_steps > 0
    assert args.routing == "full" or args.mode == "optimized"
    assert not args.key_reuse or (args.mode == "optimized" and args.storage == "offload")
    total_steps = args.warmup_steps + args.measure_steps
    capacity = args.length + total_steps
    optimization_suffix = ""
    if args.routing != "full" or args.key_reuse:
        reuse_suffix = "_reuse" if args.key_reuse else ""
        optimization_suffix = f"_{args.routing}{reuse_suffix}"
    output = args.output_root / (
        f"{args.tag}_{args.mode}_{args.storage}{optimization_suffix}_t{args.length}_r{args.repeat}"
    )
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    log_path.write_text("status=running\n", encoding="utf-8")

    def emit(payload: dict) -> None:
        line = json.dumps(payload, sort_keys=True)
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(line + "\n")

    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (8, 9)
    assert MODEL_PATH.is_dir() and FACTOR_ROOT.is_dir() and TOKENS_PATH.is_file()
    torch.manual_seed(20260921)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    emit(
        {
            "status": "starting",
            "mode": args.mode,
            "storage": args.storage,
            "routing": args.routing,
            "key_reuse": args.key_reuse,
            "context": args.length,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "profile_components": args.profile_components,
            "gpu": torch.cuda.get_device_name(),
        }
    )

    optimized_extension = _load_extension(
        value_dim=128,
        queries_per_kv=4,
        page_size=32,
        base_rank=16,
        residual_rank=16,
    )
    router_extension = (
        optimized_extension if args.mode == "optimized" else _build_shared_router_baseline()
    )
    fine_router_extension = None
    if args.routing == "two-stage":
        _, fine_router_extension = compile_fine(
            Path("/tmp/basisserve_tp1_two_stage_v7")
        )
    attention_extension = _load_extension(
        value_dim=128,
        queries_per_kv=4,
        page_size=1,
        base_rank=16,
        residual_rank=16,
    )
    slot_extension = _load_slot_extension() if args.key_reuse else None
    postprocess_extension = (
        _load_postprocess_extension() if args.routing == "two-stage" else None
    )
    emit({"status": "extensions_ready"})

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    layers = _install_sparse_attention(
        model,
        capacity=capacity,
        storage=args.storage,
        mode=args.mode,
        router_extension=router_extension,
        fine_router_extension=fine_router_extension,
        attention_extension=attention_extension,
        routing=args.routing,
        key_reuse=args.key_reuse,
        slot_extension=slot_extension,
        postprocess_extension=postprocess_extension,
        profile_components=args.profile_components,
        validate=args.validate,
    )
    prefill_tokens, decode_tokens = _load_tokens(args.length, total_steps)
    prefill_tokens = prefill_tokens[None].to("cuda")
    decode_tokens = decode_tokens.to("cuda")
    emit(
        {
            "status": "model_ready",
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "host_exact_key_gib": (
                32 * capacity * 8 * 128 * 2 / 2**30 if args.storage == "offload" else 0.0
            ),
        }
    )

    torch.cuda.reset_peak_memory_stats()
    prefill_begin = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    prefill_begin.record()
    outputs = model.model(input_ids=prefill_tokens, use_cache=False)
    last_hidden = outputs.last_hidden_state[:, -1:].contiguous()
    prefill_end.record()
    prefill_end.synchronize()
    prefill_ms = prefill_begin.elapsed_time(prefill_end)
    del outputs, last_hidden, prefill_tokens
    emit(
        {
            "status": "prefill_complete",
            "prefill_ms": prefill_ms,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
    )

    cuda_ms = []
    wall_ms = []
    component_samples = {
        "attention_block_ms": [],
        "cache_append_ms": [],
        "router_scan_ms": [],
        "page_selection_ms": [],
        "key_refresh_ms": [],
        "sparse_attention_ms": [],
    }
    argmax_tokens = []
    finite = []
    for step in range(total_steps):
        input_id = decode_tokens[step].view(1, 1)
        position_id = torch.tensor([[args.length + step]], device="cuda")
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        wall_start = time.perf_counter()
        hidden = model.model(
            input_ids=input_id,
            position_ids=position_id,
            use_cache=False,
        ).last_hidden_state
        logits = model.lm_head(hidden[:, -1:])
        end.record()
        end.synchronize()
        elapsed_cuda = begin.elapsed_time(end)
        elapsed_wall = (time.perf_counter() - wall_start) * 1000.0
        components = {name: 0.0 for name in component_samples}
        if args.profile_components:
            for layer in layers:
                for name, value in layer.component_times().items():
                    components[f"{name}_ms"] += value
        token = int(logits.argmax(dim=-1).item())
        is_finite = bool(torch.isfinite(logits).all().item())
        argmax_tokens.append(token)
        finite.append(is_finite)
        if step >= args.warmup_steps:
            cuda_ms.append(elapsed_cuda)
            wall_ms.append(elapsed_wall)
            if args.profile_components:
                for name, value in components.items():
                    component_samples[name].append(value)
        emit(
            {
                "status": "decode_step",
                "step": step,
                "measured": step >= args.warmup_steps,
                "cuda_ms": elapsed_cuda,
                "wall_ms": elapsed_wall,
                "components": components,
                "argmax": token,
                "finite": is_finite,
            }
        )

    component_summary = {
        name: {
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
        }
        for name, values in component_samples.items()
        if values
    }
    source_paths = [
        REPOSITORY_ROOT / "basisserve/kernels/csrc/conditional_router_page32.cu",
        REPOSITORY_ROOT / "basisserve/kernels/csrc/mapped_host_paged_attention.cu",
        Path(__file__).resolve(),
    ]
    if args.routing == "two-stage":
        source_paths.extend(
            [
                REPOSITORY_ROOT / "benchmarks/system/two_stage_router.py",
                REPOSITORY_ROOT / "benchmarks/system/fused_candidates.py",
                REPOSITORY_ROOT / "basisserve/kernels/csrc/fused_decode_postprocess.cu",
            ]
        )
    slot_hit_fraction = None
    if args.key_reuse:
        source_paths.extend(
            [
                REPOSITORY_ROOT / "basisserve/kernels/csrc/persistent_key_slots.cu",
                REPOSITORY_ROOT / "basisserve/kernels/slot_indexed_attention.py",
            ]
        )
        slot_counts = torch.stack([layer.slot_counts for layer in layers]).sum(dim=(0, 1, 2))
        slot_hit_fraction = float(slot_counts[0].float() / slot_counts[1].clamp_min(1))
    result = {
        "schema": "tp1-sparse-full-v7",
        "mode": args.mode,
        "storage": args.storage,
        "routing": args.routing,
        "key_reuse": args.key_reuse,
        "last_step_key_hit_fraction": slot_hit_fraction,
        "context_tokens": args.length,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measure_steps,
        "repeat": args.repeat,
        "page_size": PAGE_SIZE,
        "routed_pages": ROUTED_PAGES,
        "recent_tokens": RECENT_TOKENS,
        "physical_token_budget": SUPPORT_TOKENS,
        "base_rank": 16,
        "residual_rank": 16,
        "value_dim": 128,
        "prefill_backend": "torch-flash-sdpa",
        "prefill_ms": prefill_ms,
        "decode_cuda_ms": cuda_ms,
        "decode_wall_ms": wall_ms,
        "decode_cuda_median_ms": statistics.median(cuda_ms),
        "decode_cuda_mean_ms": statistics.fmean(cuda_ms),
        "decode_cuda_p95_ms": _percentile(cuda_ms, 0.95),
        "decode_wall_median_ms": statistics.median(wall_ms),
        "throughput_tokens_per_second": 1000.0 / statistics.fmean(wall_ms),
        "profile_components": args.profile_components,
        "components": component_summary,
        "all_logits_finite": all(finite),
        "argmax_tokens": argmax_tokens,
        "validated_layers": sum(int(layer.validated) for layer in layers),
        "full_router_page_recall": (
            {
                "mean": statistics.fmean(
                    layer.full_router_page_recall for layer in layers
                ),
                "minimum": min(layer.full_router_page_recall for layer in layers),
                "per_layer": [layer.full_router_page_recall for layer in layers],
            }
            if args.routing == "two-stage" and args.validate
            else None
        ),
        "full_router_score_recall": (
            {
                "mean": statistics.fmean(
                    layer.full_router_score_recall for layer in layers
                ),
                "minimum": min(layer.full_router_score_recall for layer in layers),
                "per_layer": [layer.full_router_score_recall for layer in layers],
            }
            if args.routing == "two-stage" and args.validate
            else None
        ),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "model_path": str(MODEL_PATH),
        "factor_root": str(FACTOR_ROOT),
        "tokens_path": str(TOKENS_PATH),
        "source_sha256": {str(path.relative_to(REPOSITORY_ROOT)): _sha256(path) for path in source_paths},
    }
    (output / "benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    emit(
        {
            "status": "complete",
            "decode_cuda_median_ms": result["decode_cuda_median_ms"],
            "decode_wall_median_ms": result["decode_wall_median_ms"],
            "validated_layers": result["validated_layers"],
            "last_step_key_hit_fraction": slot_hit_fraction,
            "result": str(output / "benchmark.json"),
        }
    )


if __name__ == "__main__":
    main()
