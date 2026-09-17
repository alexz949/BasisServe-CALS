"""Chunk8 landmark routing with a hard physical-token budget."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from basisserve.core.c1_conditional_page_attention import _selected_pages


CHUNK_SIZE = 8
SINK_TOKENS = 32
RECENT_TOKENS = 64


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    half = int(values.shape[-1]) // 2
    return torch.cat((-values[..., half:], values[..., :half]), dim=-1)


def predict_post_rope_base(
    values: torch.Tensor,
    *,
    base_left: torch.Tensor,
    base_right: torch.Tensor,
    base_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Predict post-RoPE Key from dense Value coordinates with frozen Base16."""

    assert values.ndim == 4
    dtype, device = values.dtype, values.device
    predicted = torch.einsum(
        "bhtv,hvr,hrd->bhtd",
        values,
        base_left.to(device=device, dtype=dtype),
        base_right.to(device=device, dtype=dtype),
    )
    predicted.add_(base_bias.to(device=device, dtype=dtype)[None, :, None, :])
    rotary_cos = cos.to(device=device, dtype=dtype).unsqueeze(1)
    rotary_sin = sin.to(device=device, dtype=dtype).unsqueeze(1)
    return (
        predicted * rotary_cos + _rotate_half(predicted) * rotary_sin
    ).contiguous()


def _complete_means(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, tokens, width = map(int, values.shape)
    complete = tokens // CHUNK_SIZE
    stop = complete * CHUNK_SIZE
    means = (
        values[:, :, :stop]
        .float()
        .reshape(batch, heads, complete, CHUNK_SIZE, width)
        .mean(dim=-2)
        .to(values.dtype)
        .contiguous()
    )
    return means, values[:, :, stop:].contiguous()


@dataclass
class ChunkLandmarkState:
    """Incremental fixed-boundary Chunk8 means for Base and residual codes."""

    base: torch.Tensor
    residual: torch.Tensor | None
    pending_base: torch.Tensor
    pending_residual: torch.Tensor | None
    tokens: int

    @classmethod
    def from_tokens(
        cls,
        base_tokens: torch.Tensor,
        residual_codes: torch.Tensor | None,
    ) -> "ChunkLandmarkState":
        base, pending_base = _complete_means(base_tokens)
        residual = None
        pending_residual = None
        if residual_codes is not None:
            assert residual_codes.shape[:-1] == base_tokens.shape[:-1]
            residual, pending_residual = _complete_means(residual_codes)
        result = cls(
            base=base,
            residual=residual,
            pending_base=pending_base,
            pending_residual=pending_residual,
            tokens=int(base_tokens.shape[2]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        assert self.base.ndim == self.pending_base.ndim == 4
        assert int(self.pending_base.shape[2]) == self.tokens % CHUNK_SIZE
        assert int(self.base.shape[2]) == self.tokens // CHUNK_SIZE
        assert self.base.shape[:2] == self.pending_base.shape[:2]
        assert self.base.shape[-1] == self.pending_base.shape[-1]
        assert (self.residual is None) == (self.pending_residual is None)
        if self.residual is not None:
            assert self.pending_residual is not None
            assert self.residual.shape[:3] == self.base.shape[:3]
            assert self.pending_residual.shape[:3] == self.pending_base.shape[:3]

    def append(
        self,
        base_tokens: torch.Tensor,
        residual_codes: torch.Tensor | None,
    ) -> None:
        assert base_tokens.ndim == 4 and int(base_tokens.shape[2]) > 0
        assert base_tokens.shape[:2] == self.pending_base.shape[:2]
        assert base_tokens.shape[-1] == self.pending_base.shape[-1]
        assert (residual_codes is None) == (self.residual is None)
        combined_base = torch.cat((self.pending_base, base_tokens), dim=2)
        completed_base, self.pending_base = _complete_means(combined_base)
        if int(completed_base.shape[2]):
            self.base = torch.cat((self.base, completed_base), dim=2)
        if residual_codes is not None:
            assert self.pending_residual is not None and self.residual is not None
            combined_residual = torch.cat(
                (self.pending_residual, residual_codes), dim=2
            )
            completed_residual, self.pending_residual = _complete_means(
                combined_residual
            )
            assert completed_residual.shape[:3] == completed_base.shape[:3]
            if int(completed_residual.shape[2]):
                self.residual = torch.cat(
                    (self.residual, completed_residual), dim=2
                )
        self.tokens += int(base_tokens.shape[2])
        self.validate()


def exact_chunk_logits(
    grouped_query: torch.Tensor,
    exact_key: torch.Tensor,
    complete_chunks: int,
    *,
    scale: float,
) -> torch.Tensor:
    """Exact per-token QK log-sum-exp for complete routed Chunk8 candidates."""

    batch, groups, heads, width = map(int, grouped_query.shape)
    assert exact_key.shape[:2] == (batch, groups)
    assert int(exact_key.shape[-1]) == width and complete_chunks > 0
    stop = complete_chunks * CHUNK_SIZE
    token_logits = float(scale) * torch.einsum(
        "bghd,bgtd->bght",
        grouped_query.float(),
        exact_key[:, :, :stop].float(),
    )
    return torch.logsumexp(
        token_logits.reshape(
            batch,
            groups,
            heads,
            complete_chunks,
            CHUNK_SIZE,
        ),
        dim=-1,
    )


def landmark_chunk_logits(
    grouped_query: torch.Tensor,
    state: ChunkLandmarkState,
    complete_chunks: int,
    *,
    scale: float,
    query_factor: torch.Tensor | None,
) -> torch.Tensor:
    """Base16 mean score plus the optional direct Mean-Chunk-R16 correction."""

    batch, groups, heads, width = map(int, grouped_query.shape)
    assert state.base.shape[:2] == (batch, groups)
    assert int(state.base.shape[-1]) == width
    assert 0 < complete_chunks <= int(state.base.shape[2])
    result = float(scale) * torch.einsum(
        "bghd,bgcd->bghc",
        grouped_query.float(),
        state.base[:, :, :complete_chunks].float(),
    )
    result.add_(math.log(CHUNK_SIZE))
    if query_factor is not None:
        assert state.residual is not None
        assert query_factor.shape[:3] == (groups, heads, width)
        query_code = torch.einsum(
            "bghd,ghdr->bghr",
            grouped_query.float(),
            query_factor.float(),
        )
        result.add_(
            float(scale)
            * torch.einsum(
                "bghr,bgcr->bghc",
                query_code,
                state.residual[:, :, :complete_chunks].float(),
            )
        )
    return result


def chunk8_hard_budget_support(
    chunk_logits: torch.Tensor,
    total_tokens: int,
    *,
    budget: int = 2048,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Select 256 physical Chunk8 slots while retaining full sink32/recent64.

    When the decode position is not Chunk8-aligned, the one chunk crossing the
    recent64 boundary is retained with the exact tail.  The last current chunk
    is ragged, so actual token support is 2041--2047 rather than exceeding the
    hard 2048-token ceiling.  Aligned positions retain exactly 2048 tokens.
    """

    batch, groups, heads, complete_chunks = map(int, chunk_logits.shape)
    assert total_tokens > budget
    assert budget % CHUNK_SIZE == 0
    assert SINK_TOKENS % CHUNK_SIZE == 0
    assert RECENT_TOKENS % CHUNK_SIZE == 0
    recent_start = total_tokens - RECENT_TOKENS
    routed_stop = (recent_start // CHUNK_SIZE) * CHUNK_SIZE
    assert complete_chunks == routed_stop // CHUNK_SIZE
    tail_tokens = total_tokens - routed_stop
    historical_chunk_budget = (budget - tail_tokens) // CHUNK_SIZE
    pinned_chunks = SINK_TOKENS // CHUNK_SIZE
    assert historical_chunk_budget >= pinned_chunks
    proxy = chunk_logits.reshape(batch, groups * heads, 1, complete_chunks)
    selected, selected_valid = _selected_pages(
        proxy,
        torch.ones_like(proxy, dtype=torch.bool),
        kv_heads=groups,
        page_size=1,
        page_budget=historical_chunk_budget,
        pinned_prefix_pages=pinned_chunks,
    )
    selected, order = selected.squeeze(-2).sort(dim=-1)
    selected_valid = selected_valid.squeeze(-2).gather(-1, order)
    assert selected_valid.all()
    offsets = torch.arange(CHUNK_SIZE, device=chunk_logits.device)
    historical_ids = (selected[..., None] * CHUNK_SIZE + offsets).flatten(-2)
    tail = torch.arange(
        routed_stop,
        total_tokens,
        device=chunk_logits.device,
    ).expand(batch, groups, -1)
    ids = torch.cat((historical_ids, tail), dim=-1)
    actual_tokens = int(ids.shape[-1])
    assert actual_tokens <= budget
    assert int(ids.min()) == 0 and int(ids.max()) == total_tokens - 1
    assert torch.all(ids[..., 1:] > ids[..., :-1])
    tail_chunks = math.ceil(tail_tokens / CHUNK_SIZE)
    assert historical_chunk_budget + tail_chunks == budget // CHUNK_SIZE
    return ids, {
        "logical_budget_tokens": budget,
        "actual_support_tokens": actual_tokens,
        "equivalent_chunk_slots": budget // CHUNK_SIZE,
        "historical_chunks": historical_chunk_budget,
        "pinned_sink_chunks": pinned_chunks,
        "routed_historical_chunks": historical_chunk_budget - pinned_chunks,
        "tail_chunks": tail_chunks,
        "exact_tail_tokens": tail_tokens,
        "unused_token_capacity": budget - actual_tokens,
    }


__all__ = [
    "CHUNK_SIZE",
    "SINK_TOKENS",
    "RECENT_TOKENS",
    "ChunkLandmarkState",
    "predict_post_rope_base",
    "exact_chunk_logits",
    "landmark_chunk_logits",
    "chunk8_hard_budget_support",
]
