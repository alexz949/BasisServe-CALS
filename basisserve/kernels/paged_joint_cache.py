"""Page32 V96/R16 cache, FP32 bounds, and mapped host exact K.

One GPU page is [Hkv, 32, 128] BF16 cells: V96, R16, and 16 cells
per token reserved for the page's 128 FP32 minima and 128 FP32 maxima.
The equally sized host page stores exact post-RoPE K128. Logical addressing
comes exclusively from vLLM slot mappings and block tables.
"""

import torch
import triton as tr
import triton.language as tl
from cuda.bindings import runtime


@tr.jit
def _request_epoch(POS, SLOTS, EPOCH, N: tl.constexpr):
    t = tl.program_id(0) * 128 + tl.arange(0, 128)
    pos = tl.load(POS + t, t < N, other=-1)
    slot = tl.load(SLOTS + t, t < N, other=-1)
    tl.atomic_add(EPOCH + slot // 32, 1, (pos == 0) & (slot >= 0) & (t < N))


@tr.jit
def _append(K, V, POS, SLOTS, RIGHT, BIAS, ENCODER, ROPE, CACHE, HOST,
            N: tl.constexpr, H: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr,
            V0: tl.constexpr, V1: tl.constexpr, C0: tl.constexpr, C1: tl.constexpr,
            C2: tl.constexpr, RP0: tl.constexpr):
    t = tl.program_id(0) * 32 + tl.arange(0, 32)
    h = tl.program_id(1)
    d = tl.arange(0, 128)
    r = tl.arange(0, 16)
    slot = tl.load(SLOTS + t, t < N, other=-1)
    valid = (t < N) & (slot >= 0)
    pos = tl.load(POS + t, valid, other=0)
    base = tl.load(V + t[:, None] * V0 + h * V1 + r[None, :], valid[:, None], other=0)
    right = tl.load(RIGHT + h * 16 * 128 + r[:, None] * 128 + d[None, :])
    prediction = tl.dot(base, right, allow_tf32=False).to(tl.bfloat16).to(tl.float32)
    prediction += tl.load(BIAS + h * 128 + d)[None, :]
    prediction = prediction.to(tl.bfloat16).to(tl.float32)
    swapped = tl.gather(prediction, tl.broadcast_to(((d + 64) % 128)[None, :], (32, 128)), axis=1)
    cs = tl.load(ROPE + pos[:, None] * RP0 + (d % 64)[None, :], valid[:, None], other=0).to(tl.float32)
    sn = tl.load(ROPE + pos[:, None] * RP0 + (64 + d % 64)[None, :], valid[:, None], other=0).to(tl.float32)
    first = (prediction * cs).to(tl.bfloat16).to(tl.float32)
    second = (tl.where(d[None, :] < 64, -swapped, swapped) * sn).to(tl.bfloat16).to(tl.float32)
    rotated = (first + second).to(tl.bfloat16)
    key = tl.load(K + t[:, None] * K0 + h * K1 + d[None, :], valid[:, None], other=0)
    error = (key.to(tl.float32) - rotated.to(tl.float32)).to(tl.bfloat16)
    encoder = tl.load(ENCODER + h * 128 * 16 + d[:, None] * 16 + r[None, :])
    residual = tl.dot(error, encoder, allow_tf32=False).to(tl.bfloat16)
    address = (slot // 32)[:, None] * C0 + h * C1 + (slot % 32)[:, None] * C2
    value = tl.load(V + t[:, None] * V0 + h * V1 + d[None, :], valid[:, None] & (d[None, :] < 96), other=0)
    tl.store(CACHE + address + d[None, :], value, valid[:, None] & (d[None, :] < 96))
    tl.store(CACHE + address + 96 + r[None, :], residual, valid[:, None])
    host = HOST.to(tl.pointer_type(tl.bfloat16))
    host_address = ((slot // 32)[:, None] * H * 32 * 128 + h * 32 * 128
                    + (slot % 32)[:, None] * 128 + d[None, :])
    tl.store(host + host_address, key, valid[:, None])


@tr.jit
def _bounds(CACHE, HOST, TABLE, LENS, STARTS, H: tl.constexpr,
            T0: tl.constexpr, C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr):
    page = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    end = tl.load(LENS + b) - 64
    qlen = tl.load(STARTS + b + 1) - tl.load(STARTS + b)
    begin = tl.maximum(end - qlen, 0)
    changed = (page * 32 < end) & ((page + 1) * 32 > begin)
    if changed:
        block = tl.load(TABLE + b * T0 + page)
        t = tl.arange(0, 32)
        d = tl.arange(0, 128)
        host = HOST.to(tl.pointer_type(tl.bfloat16))
        key = tl.load(host + ((block * H + h) * 32 + t[:, None]) * 128 + d[None, :],
                      page * 32 + t[:, None] < end, other=0, cache_modifier=".cv").to(tl.float32)
        low = tl.min(tl.where(page * 32 + t[:, None] < end, key, float("inf")), 0)
        high = tl.max(tl.where(page * 32 + t[:, None] < end, key, -float("inf")), 0)
        base = CACHE + block * C0 + h * C1
        low_ptr = (base + (d // 8) * C2 + 112 + (d % 8) * 2).to(tl.pointer_type(tl.float32))
        high_ptr = (base + (16 + d // 8) * C2 + 112 + (d % 8) * 2).to(tl.pointer_type(tl.float32))
        tl.store(low_ptr, low)
        tl.store(high_ptr, high)


@tr.jit
def _gather(CACHE, HOST, TABLE, LENS, OUT, P: tl.constexpr, H: tl.constexpr,
            T0: tl.constexpr, C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr):
    page = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    length = tl.load(LENS + b)
    block = tl.load(TABLE + b * T0 + page, page * 32 < length, other=0)
    t = tl.arange(0, 32)
    d = tl.arange(0, 128)
    valid = page * 32 + t[:, None] < length
    host = HOST.to(tl.pointer_type(tl.bfloat16))
    key = tl.load(host + ((block * H + h) * 32 + t[:, None]) * 128 + d[None, :],
                  valid, other=0, cache_modifier=".cv")
    value = tl.load(CACHE + block * C0 + h * C1 + t[:, None] * C2 + d[None, :],
                    valid & (d[None, :] < 96), other=0)
    target = OUT + (((b * P + page) * H + h) * 32 + t[:, None]) * 224
    tl.store(target + d[None, :], key)
    tl.store(target + 128 + d[None, :], value, d[None, :] < 96)


class PagedJointCache:
    def __init__(self, gpu_cache):
        assert gpu_cache.dtype == torch.bfloat16 and gpu_cache.is_cuda
        assert gpu_cache.ndim == 4 and gpu_cache.shape[2:] == (32, 128)
        assert gpu_cache.stride(-1) == 1
        self.gpu = gpu_cache
        self.host = torch.empty(gpu_cache.shape, dtype=torch.bfloat16, pin_memory=True)
        status, pointer = runtime.cudaHostGetDevicePointer(self.host.data_ptr(), 0)
        assert status == runtime.cudaError_t.cudaSuccess
        self.pointer = int(pointer)
        self.epoch = torch.zeros(gpu_cache.shape[0], dtype=torch.int64, device=gpu_cache.device)

    def append(self, key, value, positions, slots, right, bias, encoder, rope):
        n, heads, _ = key.shape
        assert value.shape == (n, heads, 96)
        assert self.gpu.shape[1] == heads
        assert positions.numel() == slots.numel() == n
        _append[(tr.cdiv(n, 32), heads)](
            key, value, positions, slots, right, bias, encoder, rope, self.gpu, self.pointer,
            n, heads, *key.stride()[:2], *value.stride()[:2], *self.gpu.stride()[:3],
            rope.stride(0), num_warps=4,
        )
        _request_epoch[(tr.cdiv(n, 128),)](positions, slots, self.epoch, n)

    def update_bounds(self, table, lengths, query_starts):
        _bounds[(table.shape[1], self.gpu.shape[1], lengths.numel())](
            self.gpu, self.pointer, table, lengths, query_starts, self.gpu.shape[1],
            table.stride(0), *self.gpu.stride()[:3], num_warps=4,
        )

    def prefill_cache(self, table, lengths, max_length):
        pages = tr.cdiv(max_length, 32)
        batch, heads = lengths.numel(), self.gpu.shape[1]
        output = torch.empty((batch * pages, heads, 32, 224), dtype=self.gpu.dtype, device=self.gpu.device)
        _gather[(pages, heads, batch)](
            self.gpu, self.pointer, table, lengths, output, pages, heads,
            table.stride(0), *self.gpu.stride()[:3], num_warps=4,
        )
        blocks = torch.arange(batch * pages, device=table.device, dtype=table.dtype).view(batch, pages)
        return output, blocks

    def bounds_view(self):
        """Inspection only; converts the packed FP32 sidecar to [blocks,H,2,128]."""
        return self.gpu[..., 112:].contiguous().view(torch.float32).reshape(*self.gpu.shape[:2], 2, 128)
