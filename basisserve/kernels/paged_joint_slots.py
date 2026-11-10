"""Request-scoped persistent exact-K slots and paged V96 decode attention."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _reset(TABLE, EPOCH, OWNERS, LOOKUP, RESIDENT, LENS, H: tl.constexpr,
           CAP: tl.constexpr, T0: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0)
    b = row // H
    n = tl.load(LENS + b)
    block = tl.load(TABLE + b * T0, n > 0, other=0)
    epoch = tl.load(EPOCH + block)
    old_block = tl.load(OWNERS + row * 2)
    old_epoch = tl.load(OWNERS + row * 2 + 1)
    if (block != old_block) | (epoch != old_epoch):
        for offset in range(tl.cdiv(CAP, BC)):
            i = offset * BC + tl.arange(0, BC)
            tl.store(LOOKUP + row * CAP + i, -1, i < CAP)
        i = tl.arange(0, 2048)
        tl.store(RESIDENT + row * 2048 + i, -1)
    tl.store(OWNERS + row * 2, block)
    tl.store(OWNERS + row * 2 + 1, epoch)


@tr.jit
def _lookup(IDS, LOOKUP, SELECTED, KEEP, CAP: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, 2048)
    ids = tl.load(IDS + row * 2048 + i)
    slot = tl.load(LOOKUP + row * CAP + ids, ids >= 0, other=-1)
    tl.store(SELECTED + row * 2048 + i, slot)
    tl.store(KEEP + row * 2048 + slot, 1, slot >= 0)


@tr.jit
def _plan(IDS, LOOKUP, SELECTED, KEEP, RESIDENT, MISSING, COUNTS, CAP: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, 2048)
    offset = row * 2048 + i
    ids = tl.load(IDS + offset)
    slots = tl.load(SELECTED + offset)
    keep = tl.load(KEEP + offset)
    resident = tl.load(RESIDENT + offset)
    free = keep == 0
    free_ids = tl.sort(tl.where(free, i, 2048), descending=False)
    missing = (ids >= 0) & (slots < 0)
    missing_index = tl.cumsum(missing.to(tl.int32), 0) - 1
    assigned = tl.gather(free_ids, tl.maximum(missing_index, 0), 0)
    # Evicted IDs cannot be in the support: those slots were marked KEEP.
    tl.store(LOOKUP + row * CAP + resident, -1, free & (resident >= 0))
    tl.store(LOOKUP + row * CAP + ids, assigned, missing)
    tl.store(RESIDENT + offset, -1, free)
    tl.debug_barrier()
    tl.store(RESIDENT + row * 2048 + assigned, ids, missing)
    tl.store(SELECTED + offset, tl.where(missing, assigned, slots))
    tl.store(MISSING + offset, missing.to(tl.int32))
    tl.store(COUNTS + row * 2, tl.sum(((ids >= 0) & (slots >= 0)).to(tl.int32), 0))
    tl.store(COUNTS + row * 2 + 1, tl.sum(missing.to(tl.int32), 0))


@tr.jit
def _fetch(HOST, TABLE, IDS, SELECTED, MISSING, KEYS, H: tl.constexpr, T0: tl.constexpr):
    row = tl.program_id(0)
    i = tl.program_id(1) * 32 + tl.arange(0, 32)
    d = tl.arange(0, 128)
    ids = tl.load(IDS + row * 2048 + i)
    selected = tl.load(SELECTED + row * 2048 + i)
    missing = tl.load(MISSING + row * 2048 + i) != 0
    block = tl.load(TABLE + row // H * T0 + ids // 32, missing, other=0)
    host = HOST.to(tl.pointer_type(tl.bfloat16))
    k = tl.load(host + ((block[:, None] * H + row % H) * 32 + (ids % 32)[:, None]) * 128 + d[None, :],
                missing[:, None], other=0, cache_modifier=".cv")
    tl.store(KEYS + (row * 2048 + selected[:, None]) * 128 + d[None, :], k, missing[:, None])


@tr.jit
def _attention(Q, KEYS, CACHE, TABLE, IDS, SELECTED, PARTIAL, LSE,
               H: tl.constexpr, Q0: tl.constexpr, Q1: tl.constexpr, T0: tl.constexpr,
               C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr):
    head = tl.program_id(0)
    split = tl.program_id(1)
    row = head // 4
    t = split * 128 + tl.arange(0, 128)
    d = tl.arange(0, 128)
    ids = tl.load(IDS + row * 2048 + t)
    slots = tl.load(SELECTED + row * 2048 + t)
    valid = (ids >= 0) & (slots >= 0)
    q = tl.load(Q + head // (H * 4) * Q0 + head % (H * 4) * Q1 + d).to(tl.float32)
    k = tl.load(KEYS + (row * 2048 + slots[:, None]) * 128 + d[None, :], valid[:, None], other=0).to(tl.float32)
    scores = tl.where(valid, tl.sum(k * q[None, :], 1) * 0.08838834764831845, -float("inf"))
    maximum = tl.max(scores, 0)
    probability = tl.exp(scores - tl.where(maximum == -float("inf"), 0., maximum))
    total = tl.sum(probability, 0)
    block = tl.load(TABLE + row // H * T0 + ids // 32, valid, other=0)
    v = tl.load(CACHE + block[:, None] * C0 + row % H * C1 + (ids % 32)[:, None] * C2 + d[None, :],
                valid[:, None] & (d[None, :] < 96), other=0).to(tl.float32)
    out = tl.sum(probability[:, None] * v, 0) / tl.maximum(total, 1.e-30)
    tl.store(PARTIAL + (head * 16 + split) * 96 + d, out, d < 96)
    tl.store(LSE + head * 16 + split, maximum + tl.log(total))


@tr.jit
def _merge(PARTIAL, LSE, OUT, O0: tl.constexpr, O1: tl.constexpr, HEADS: tl.constexpr):
    head = tl.program_id(0)
    s = tl.arange(0, 16)
    d = tl.arange(0, 128)
    logs = tl.load(LSE + head * 16 + s)
    maximum = tl.max(logs, 0)
    weights = tl.exp(logs - tl.where(maximum == -float("inf"), 0., maximum))
    weights /= tl.maximum(tl.sum(weights, 0), 1.e-30)
    values = tl.load(PARTIAL + (head * 16 + s[:, None]) * 96 + d[None, :], d[None, :] < 96, other=0)
    tl.store(OUT + head // HEADS * O0 + head % HEADS * O1 + d, tl.sum(weights[:, None] * values, 0), d < 96)


class PagedJointSlots:
    def __init__(self, batch, heads, capacity, device):
        self.heads, self.capacity = heads, capacity
        self.keys = torch.empty(batch, heads, 2048, 128, device=device, dtype=torch.bfloat16)
        self.owners = torch.full((batch, heads, 2), -1, device=device, dtype=torch.int64)
        self.lookup = torch.empty(batch, heads, capacity, device=device, dtype=torch.int32)
        self.resident = torch.empty(batch, heads, 2048, device=device, dtype=torch.int32)
        self.selected = torch.empty_like(self.resident)
        self.keep = torch.empty_like(self.resident)
        self.missing = torch.empty_like(self.resident)
        self.counts = torch.empty(batch, heads, 2, device=device, dtype=torch.int32)
        self.partial = torch.empty(batch, heads * 4, 16, 96, device=device)
        self.lse = torch.empty(batch, heads * 4, 16, device=device)

    def attend(self, query, cache, table, lengths, support, output):
        batch, heads, capacity = lengths.numel(), self.heads, self.capacity
        _reset[(batch * heads,)](table, cache.epoch, self.owners, self.lookup, self.resident,
                                lengths, heads, capacity, table.stride(0), 1024)
        self.keep[:batch].zero_()
        _lookup[(batch * heads,)](support, self.lookup, self.selected, self.keep, capacity)
        _plan[(batch * heads,)](support, self.lookup, self.selected, self.keep,
                               self.resident, self.missing, self.counts, capacity, num_warps=8)
        _fetch[(batch * heads, 64)](cache.pointer, table, support, self.selected, self.missing,
                                    self.keys, heads, table.stride(0), num_warps=4)
        _attention[(batch * heads * 4, 16)](
            query, self.keys, cache.gpu, table, support, self.selected, self.partial, self.lse,
            heads, *query.stride()[:2], table.stride(0), *cache.gpu.stride()[:3], num_warps=8)
        _merge[(batch * heads * 4,)](self.partial, self.lse, output, *output.stride()[:2], heads * 4)
        return output
