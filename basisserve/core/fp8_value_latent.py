"""Page-scaled E4M3 storage for routing-aligned Value coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from basisserve.kernels.fp8_wire import FP8_E4M3_DTYPE, FP8_E4M3_MAX


@dataclass(frozen=True)
class PagedE4M3:
    codes: Tensor
    scales: Tensor
    token_count: int
    page_size: int

    @property
    def storage_bytes(self) -> int:
        return self.codes.numel() * self.codes.element_size() + self.scales.numel() * self.scales.element_size()

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> Tensor:
        assert dtype in (torch.float16, torch.bfloat16, torch.float32)
        batch, heads, pages, page_size, width = self.codes.shape
        assert page_size == self.page_size
        decoded = self.codes.float() * self.scales.float()
        return decoded.reshape(batch, heads, pages * page_size, width)[:, :, : self.token_count].to(dtype)


class AppendablePagedE4M3:
    """Append-only Page32 cache with fixed scales once a page is created."""

    def __init__(self, *, page_size: int = 32):
        assert page_size > 0
        self.page_size = page_size
        self.codes: Tensor | None = None
        self.scales: Tensor | None = None
        self.token_count = 0

    @property
    def storage_bytes(self) -> int:
        assert self.codes is not None and self.scales is not None
        return self.codes.numel() * self.codes.element_size() + self.scales.numel() * self.scales.element_size()

    def append(self, value: Tensor) -> Tensor:
        assert value.ndim == 4 and value.is_floating_point() and value.shape[2] > 0
        first = self.token_count
        if self.codes is None:
            packed = quantize_paged_e4m3(value, page_size=self.page_size)
            self.codes, self.scales = packed.codes.view(torch.uint8), packed.scales
            self.token_count = int(value.shape[2])
        else:
            assert self.scales is not None
            assert value.shape[:2] == self.codes.shape[:2] and value.shape[-1] == self.codes.shape[-1]
            cursor = 0
            offset = self.token_count % self.page_size
            if offset:
                count = min(self.page_size - offset, int(value.shape[2]))
                scale = self.scales[:, :, -1]
                codes = (value[:, :, :count].float() / scale).clamp(
                    -FP8_E4M3_MAX, FP8_E4M3_MAX
                ).to(FP8_E4M3_DTYPE).view(torch.uint8)
                self.codes[:, :, -1, offset : offset + count].copy_(codes)
                cursor = count
                self.token_count += count
            if cursor < value.shape[2]:
                packed = quantize_paged_e4m3(value[:, :, cursor:], page_size=self.page_size)
                self.codes = torch.cat((self.codes, packed.codes.view(torch.uint8)), dim=2)
                self.scales = torch.cat((self.scales, packed.scales), dim=2)
                self.token_count += int(value.shape[2]) - cursor
        ids = torch.arange(first, self.token_count, device=value.device).view(1, 1, -1)
        return self.gather(ids.expand(value.shape[0], value.shape[1], -1), dtype=value.dtype)

    def gather(self, ids: Tensor, *, dtype: torch.dtype = torch.bfloat16) -> Tensor:
        assert self.codes is not None and self.scales is not None
        assert ids.ndim == 3 and ids.shape[:2] == self.codes.shape[:2]
        assert ids.dtype == torch.long and ids.min() >= 0 and ids.max() < self.token_count
        batch, heads, _, _, width = self.codes.shape
        flat_codes = self.codes.reshape(batch, heads, -1, width)
        selected_bytes = flat_codes.gather(2, ids[..., None].expand(*ids.shape, width))
        selected_codes = selected_bytes.view(FP8_E4M3_DTYPE)
        page_ids = ids // self.page_size
        page_scales = self.scales.reshape(batch, heads, -1).gather(2, page_ids)
        return (selected_codes.float() * page_scales[..., None]).to(dtype)


class SplitPagedE4M3Cache:
    """Separate-scale Base16 and Payload80 storage for routing-aligned V96."""

    def __init__(self, *, page_size: int = 32):
        self.base = AppendablePagedE4M3(page_size=page_size)
        self.payload = AppendablePagedE4M3(page_size=page_size)

    @property
    def token_count(self) -> int:
        assert self.base.token_count == self.payload.token_count
        return self.base.token_count

    @property
    def storage_bytes(self) -> int:
        return self.base.storage_bytes + self.payload.storage_bytes

    def append(self, value: Tensor) -> Tensor:
        assert value.shape[-1] == 96
        base = self.base.append(value[..., :16])
        payload = self.payload.append(value[..., 16:])
        return torch.cat((base, payload), dim=-1)

    def gather(self, ids: Tensor, *, dtype: torch.dtype = torch.bfloat16) -> Tensor:
        return torch.cat(
            (self.base.gather(ids, dtype=dtype), self.payload.gather(ids, dtype=dtype)), dim=-1
        )


def quantize_paged_e4m3(value: Tensor, *, page_size: int = 32) -> PagedE4M3:
    """Physically store one E4M3 block per page and KV head.

    One FP32 dequantization scale is shared by all tokens and coordinates in a
    ``page_size x width`` block.  Base and payload use separate calls, hence
    separate scales.
    """

    assert value.ndim == 4 and value.is_floating_point()
    assert page_size > 0
    batch, heads, tokens, width = value.shape
    pages = math.ceil(tokens / page_size)
    padded = torch.zeros(
        batch,
        heads,
        pages * page_size,
        width,
        device=value.device,
        dtype=value.dtype,
    )
    padded[:, :, :tokens].copy_(value)
    blocked = padded.reshape(batch, heads, pages, page_size, width)
    scales = (blocked.float().abs().amax(dim=(-2, -1), keepdim=True) / FP8_E4M3_MAX).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    codes = (blocked.float() / scales).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_E4M3_DTYPE)
    assert codes.element_size() == 1 and scales.dtype == torch.float32
    return PagedE4M3(codes=codes, scales=scales, token_count=tokens, page_size=page_size)


def normalized_hadamard(size: int, *, device=None, dtype=torch.float64) -> Tensor:
    """Return a deterministic orthonormal Sylvester Hadamard matrix."""

    assert size > 0 and size & (size - 1) == 0
    matrix = torch.ones(1, 1, device=device, dtype=dtype)
    while matrix.shape[0] < size:
        matrix = torch.cat((torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)), dim=0)
    return matrix / math.sqrt(size)


def base_payload_hadamard(*, device=None, dtype=torch.float64) -> Tensor:
    """Block-diagonal H16 + H64 + H16 for routing-aligned V96."""

    blocks = (
        normalized_hadamard(16, device=device, dtype=dtype),
        normalized_hadamard(64, device=device, dtype=dtype),
        normalized_hadamard(16, device=device, dtype=dtype),
    )
    matrix = torch.block_diag(*blocks)
    assert matrix.shape == (96, 96)
    return matrix


__all__ = [
    "AppendablePagedE4M3",
    "PagedE4M3",
    "SplitPagedE4M3Cache",
    "base_payload_hadamard",
    "normalized_hadamard",
    "quantize_paged_e4m3",
]
