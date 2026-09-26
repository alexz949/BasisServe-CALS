"""Packed NUQ4 pages with lossless BF16 exceptions.

K uses static per-channel bounds; V receives bounds computed over all active
KV heads. The finite exception arena is a capacity constraint, not clipping:
call check() outside capture before accepting any result. Each slot is appended
once between page resets. Reading back whole rows is a diagnostic, not serving.
The 32-bit per-token descriptor packs a pool offset and six-bit bitmap word
counts, avoiding a cross-channel prefix scan on every attention cache read.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.constexpr_function
def _base_bits(capacity):
    return capacity.bit_length()


@dataclass(frozen=True)
class NUQ4Layout:
    width: int
    block: int = 16
    exceptions_per_token: int = 8

    def __post_init__(self):
        assert 0 < self.width <= 128 and self.width % 16 == 0
        assert self.block in (16, 32)
        assert 0 < self.exceptions_per_token <= self.width

    @property
    def codes(self):
        return 16

    @property
    def bitmap(self):
        return self.codes + self.block * self.width // 2

    @property
    def bases(self):
        return self.bitmap + self.block * self.width // 8

    @property
    def scales(self):
        return self.bases + self.block * 4

    @property
    def pool(self):
        return self.scales + self.block * 8

    @property
    def capacity(self):
        return self.block * self.exceptions_per_token

    @property
    def page_bytes(self):
        return triton.cdiv(self.pool + self.capacity * 2, 16) * 16

    def constants(self):
        return dict(D=self.width, BLOCK=self.block, CODES=self.codes,
                    BITS=self.bitmap, BASES=self.bases, SCALES=self.scales,
                    POOL=self.pool, CAP=self.capacity, PAGE_BYTES=self.page_bytes,
                    BD=triton.next_power_of_2(self.width))


@triton.jit
def _v_stats(X, Y, S0: tl.constexpr, S1: tl.constexpr, D: tl.constexpr,
             BD: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    x = tl.load(X + row * S0 + d * S1, d < D, other=float("inf")).to(tl.float32)
    ordered = tl.sort(x, descending=False)
    low_index: tl.constexpr = (D - 1) * 0.005
    high_index: tl.constexpr = (D - 1) * 0.995
    low_base: tl.constexpr = int(low_index)
    high_base: tl.constexpr = int(high_index)
    low_a = tl.sum(tl.where(d == low_base, ordered, 0.0), 0)
    low_b = tl.sum(tl.where(d == low_base + 1, ordered, 0.0), 0)
    high_a = tl.sum(tl.where(d == high_base, ordered, 0.0), 0)
    high_b = tl.sum(tl.where(d == high_base + 1, ordered, 0.0), 0)
    lo = low_a + (low_b - low_a) * (low_index - low_base)
    hi = high_a + (high_b - high_a) * (high_index - high_base)
    median = tl.sum(tl.where(d == (D - 1) // 2, ordered, 0.0), 0)
    replaced = tl.where((x <= lo) | (x >= hi), median, x)
    lower = tl.min(tl.where(d < D, replaced, float("inf")), 0)
    upper = tl.max(tl.where(d < D, replaced, -float("inf")), 0)
    tl.store(Y + row * 4, lo)
    tl.store(Y + row * 4 + 1, hi)
    tl.store(Y + row * 4 + 2, (upper - lower) * 0.5)
    tl.store(Y + row * 4 + 3, (upper + lower) * 0.5)


def nuq4_value_stats(all_active_values, output=None):
    """Input is the global active V vector, never just one TP shard."""
    x = all_active_values
    assert x.ndim == 2 and x.is_cuda and x.dtype == torch.bfloat16
    n, d = x.shape
    assert 16 <= d <= 1024
    if output is None:
        output = torch.empty((n, 4), device=x.device, dtype=torch.float32)
    assert output.shape == (n, 4) and output.is_contiguous()
    assert output.dtype == torch.float32 and output.device == x.device
    _v_stats[(n,)](x, output, *x.stride(), d, triton.next_power_of_2(d),
                    enable_fp_fusion=False)
    return output


@triton.jit
def _reset_pages(CACHE, SLOTS, BLOCK: tl.constexpr, PAGE_BYTES: tl.constexpr):
    slot = tl.load(SLOTS + tl.program_id(0))
    if (slot >= 0) & (slot % BLOCK == 0):
        ptr = CACHE + (slot // BLOCK).to(tl.int64) * PAGE_BYTES
        tl.store(ptr.to(tl.pointer_type(tl.int32)), 0)


@triton.jit
def _append(X, SLOTS, CACHE, LOWER, UPPER, LUT, STATS, FAILED,
            S0: tl.constexpr, S1: tl.constexpr, DYNAMIC: tl.constexpr,
            D: tl.constexpr, BLOCK: tl.constexpr, CODES: tl.constexpr,
            BITS: tl.constexpr, BASES: tl.constexpr, SCALES: tl.constexpr,
            POOL: tl.constexpr, CAP: tl.constexpr, PAGE_BYTES: tl.constexpr,
            BD: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(SLOTS + row)
    if slot < 0:
        return
    d = tl.arange(0, BD)
    page = CACHE + (slot // BLOCK).to(tl.int64) * PAGE_BYTES
    token = slot % BLOCK
    x = tl.load(X + row * S0 + d * S1, d < D, 0).to(tl.float32)
    if DYNAMIC:
        lower = tl.load(STATS + row * 4)
        upper = tl.load(STATS + row * 4 + 1)
        scale = tl.load(STATS + row * 4 + 2)
        offset = tl.load(STATS + row * 4 + 3)
        exception = ((x <= lower) | (x >= upper)) & (d < D)
        scalep = (page + SCALES + token * 8).to(tl.pointer_type(tl.float32))
        tl.store(scalep, scale)
        tl.store(scalep + 1, offset)
    else:
        lower = tl.load(LOWER + d, d < D, 0)
        upper = tl.load(UPPER + d, d < D, 0)
        scale = (upper - lower) * 0.5
        offset = (upper + lower) * 0.5
        exception = ((x < lower) | (x > upper)) & (d < D)
    normalized = tl.div_rn(tl.where(exception, 0.0, x - offset), scale)
    best = tl.abs(normalized - tl.load(LUT))
    code = tl.full((BD,), 0, tl.int32)
    for index in tl.static_range(1, 16):
        distance = tl.abs(normalized - tl.load(LUT + index))
        better = distance < best
        code = tl.where(better, index, code)
        best = tl.where(better, distance, best)
    pairs = tl.reshape(code, (BD // 2, 2))
    packed = tl.sum(pairs << (tl.arange(0, 2)[None, :] * 4), 1)
    p = tl.arange(0, BD // 2)
    tl.store(page + CODES + token * (D // 2) + p, packed, p < D // 2)
    bitmap = tl.sum(tl.reshape(exception.to(tl.int32), (BD // 8, 8))
                    << tl.arange(0, 8)[None, :], 1)
    b = tl.arange(0, BD // 8)
    tl.store(page + BITS + token * (D // 8) + b, bitmap, b < D // 8)
    count = tl.sum(exception.to(tl.int32), 0)
    base = tl.atomic_add(page.to(tl.pointer_type(tl.int32)), count)
    BASE_BITS: tl.constexpr = _base_bits(CAP)
    descriptor = base.to(tl.uint32)
    # Three six-bit word counts fit above the pool offset, even at CAP=4096.
    for word in tl.static_range(tl.cdiv(D, 32) - 1):
        word_count = tl.sum(tl.where(d // 32 == word, exception.to(tl.int32), 0), 0)
        descriptor |= word_count.to(tl.uint32) << (BASE_BITS + word * 6)
    tl.store((page + BASES + token * 4).to(tl.pointer_type(tl.uint32)), descriptor)
    if base + count > CAP:
        tl.atomic_max(FAILED, base + count)
    ordinal = tl.cumsum(exception.to(tl.int32), 0) - 1
    dest = (page + POOL).to(tl.pointer_type(tl.bfloat16)) + base + ordinal
    tl.store(dest, x, exception & (base + ordinal < CAP))


@triton.jit
def load_nuq4_tile(CACHE, LOWER, UPPER, LUT, slots, valid,
                   DYNAMIC: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
                   CODES: tl.constexpr, BITS: tl.constexpr, BASES: tl.constexpr,
                   SCALES: tl.constexpr, POOL: tl.constexpr, CAP: tl.constexpr,
                   PAGE_BYTES: tl.constexpr, BD: tl.constexpr):
    """Return [tokens, BD] BF16 tile, with no global BF16 cache materialization."""
    d = tl.arange(0, BD)
    page = CACHE + (slots // BLOCK).to(tl.int64) * PAGE_BYTES
    token = slots % BLOCK
    mask = valid[:, None] & (d[None, :] < D)
    packed = tl.load(page[:, None] + CODES + token[:, None] * (D // 2)
                     + d[None, :] // 2, mask, 0).to(tl.int32)
    code = (packed >> ((d[None, :] % 2) * 4)) & 15
    bitmap = (page + BITS + token * (D // 8)).to(tl.pointer_type(tl.uint16))
    word = d // 32
    low = tl.load(bitmap[:, None] + word[None, :] * 2, mask, 0).to(tl.uint32)
    high = tl.load(bitmap[:, None] + word[None, :] * 2 + 1,
                   valid[:, None] & (word[None, :] * 32 + 16 < D), 0).to(tl.uint32)
    bit = low | (high << 16)
    shift = d[None, :] % 32
    exception = ((bit >> shift) & 1) != 0
    ordinal = libdevice.popc((bit & ((1 << shift).to(tl.uint32) - 1)).to(tl.int32))
    BASE_BITS: tl.constexpr = _base_bits(CAP)
    descriptor = tl.load((page + BASES + token * 4).to(tl.pointer_type(tl.uint32)), valid, 0)
    base = (descriptor & ((1 << BASE_BITS) - 1)).to(tl.int32)
    for preceding in tl.static_range(tl.cdiv(D, 32) - 1):
        count = ((descriptor >> (BASE_BITS + preceding * 6)) & 63).to(tl.int32)
        ordinal += tl.where(d[None, :] >= (preceding + 1) * 32, count[:, None], 0)
    exact = tl.load((page[:, None] + POOL).to(tl.pointer_type(tl.bfloat16))
                    + base[:, None] + ordinal,
                    mask & exception & (base[:, None] + ordinal < CAP), 0).to(tl.float32)
    if DYNAMIC:
        ptr = (page + SCALES + token * 8).to(tl.pointer_type(tl.float32))
        scale = tl.load(ptr, valid, 0)[:, None]
        offset = tl.load(ptr + 1, valid, 0)[:, None]
    else:
        lo = tl.load(LOWER + d, d < D, 0)[None, :]
        hi = tl.load(UPPER + d, d < D, 0)[None, :]
        scale, offset = (hi - lo) * 0.5, (hi + lo) * 0.5
    answer = tl.load(LUT + code) * scale + offset
    answer = tl.where(exception, exact, answer)
    return tl.where(mask, answer, 0.0).to(tl.bfloat16)


@triton.jit
def _gather(CACHE, LOWER, UPPER, LUT, SLOTS, OUT, N: tl.constexpr,
            DYNAMIC: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
            CODES: tl.constexpr, BITS: tl.constexpr, BASES: tl.constexpr,
            SCALES: tl.constexpr, POOL: tl.constexpr, CAP: tl.constexpr,
            PAGE_BYTES: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    slots = tl.load(SLOTS + row, row < N, -1)
    value = load_nuq4_tile(CACHE, LOWER, UPPER, LUT, slots,
        (row < N) & (slots >= 0), DYNAMIC, D, BLOCK, CODES, BITS, BASES,
        SCALES, POOL, CAP, PAGE_BYTES, BD)
    d = tl.arange(0, BD)
    tl.store(OUT + row[:, None] * D + d[None, :], value,
             (row[:, None] < N) & (d[None, :] < D))


class NUQ4PagedCache:
    def __init__(self, pages, layout, lower, upper, lut, *, dynamic=False):
        assert lower.is_cuda and lower.dtype == upper.dtype == lut.dtype == torch.float32
        assert lower.shape == upper.shape == (layout.width,) and lut.shape == (16,)
        assert lower.is_contiguous() and upper.is_contiguous() and lut.is_contiguous()
        assert lower.device == upper.device == lut.device
        assert torch.isfinite(lower).all() and torch.isfinite(upper).all()
        assert torch.isfinite(lut).all() and (upper >= lower).all()
        self.layout, self.dynamic = layout, dynamic
        self.lower, self.upper, self.lut = lower, upper, lut
        self.storage = torch.zeros((pages, layout.page_bytes), device=lower.device, dtype=torch.uint8)
        self.failed = torch.zeros((), device=lower.device, dtype=torch.int32)

    def bind(self, storage):
        assert storage.dtype == torch.uint8 and storage.device == self.lower.device
        assert storage.ndim == 2 and storage.shape[1] == self.layout.page_bytes
        assert storage.stride(1) == 1 and storage.stride(0) >= self.layout.page_bytes
        self.storage = storage

    def kernel_constants(self):
        return self.layout.constants() | dict(PAGE_BYTES=self.storage.stride(0))

    def append(self, values, slots, stats=None):
        assert values.is_cuda and values.dtype == torch.bfloat16
        assert values.shape == (slots.numel(), self.layout.width)
        assert slots.is_contiguous() and slots.dtype in (torch.int32, torch.int64)
        assert values.device == slots.device == self.storage.device
        if self.dynamic:
            assert stats.shape == (slots.numel(), 4) and stats.is_contiguous()
            assert stats.device == values.device and stats.dtype == torch.float32
        _reset_pages[(slots.numel(),)](self.storage, slots, self.layout.block, self.storage.stride(0))
        _append[(slots.numel(),)](values, slots, self.storage, self.lower, self.upper,
            self.lut, stats, self.failed, *values.stride(), self.dynamic,
            **self.kernel_constants(), enable_fp_fusion=False)

    def check(self):
        assert int(self.failed) == 0, "NUQ4 exception arena exhausted; result is invalid"

    def gather_for_validation(self, slots):
        self.check()
        out = torch.empty((slots.numel(), self.layout.width), device=slots.device, dtype=torch.bfloat16)
        _gather[(triton.cdiv(slots.numel(), 16),)](self.storage, self.lower, self.upper,
            self.lut, slots, out, slots.numel(), self.dynamic, **self.kernel_constants(),
            enable_fp_fusion=False)
        return out
