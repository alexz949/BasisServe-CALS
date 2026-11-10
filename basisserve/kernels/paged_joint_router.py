"""Graph-compatible Page32/B16R16 routing over the native paged V96 cache."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _coarse(Q, CACHE, TABLE, LENS, OUT, H: tl.constexpr, P: tl.constexpr,
            Q0: tl.constexpr, Q1: tl.constexpr, T0: tl.constexpr,
            C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr):
    head = tl.program_id(0)
    b, hq = head // (H * 4), head % (H * 4)
    p = tl.program_id(1) * 16 + tl.arange(0, 16)
    d = tl.arange(0, 128)
    historical = tl.maximum(tl.load(LENS + b) - 64, 0)
    valid = (p < P) & (p * 32 < historical)
    block = tl.load(TABLE + b * T0 + p, valid, other=0)
    base = CACHE + block[:, None] * C0 + (hq // 4) * C1
    lowptr = (base + (d[None, :] // 8) * C2 + 112 + d[None, :] % 8 * 2).to(tl.pointer_type(tl.float32))
    highptr = (base + (16 + d[None, :] // 8) * C2 + 112 + d[None, :] % 8 * 2).to(tl.pointer_type(tl.float32))
    lo = tl.load(lowptr, valid[:, None], other=0)
    hi = tl.load(highptr, valid[:, None], other=0)
    q = tl.load(Q + b * Q0 + hq * Q1 + d).to(tl.float32)
    scores = tl.sum(tl.maximum(q[None, :] * lo, q[None, :] * hi), 1) * 0.08838834764831845
    scores += tl.log(tl.maximum(1, tl.minimum(32, historical - p * 32)).to(tl.float32))
    tl.store(OUT + head * P + p, tl.where(valid, scores, -float("inf")), p < P)


@tr.jit
def _select_coarse(SCORES, LENS, IDS, H: tl.constexpr, P: tl.constexpr, BP: tl.constexpr):
    row = tl.program_id(0)
    p = tl.arange(0, BP)
    q = tl.arange(0, 4)
    n = tl.load(LENS + row // H)
    actual_pages = tl.cdiv(n, 32)
    historical = tl.maximum(n - 64, 0)
    eligible = (p > 0) & (p < historical // 32) & (p < P)
    x = tl.load(SCORES + (row * 4 + q[:, None]) * P + p[None, :], p[None, :] < P, other=-float("inf"))
    x = tl.where(eligible[None, :], x, -float("inf"))
    m = tl.max(x, 1)
    z = tl.log(tl.sum(tl.exp(x - tl.where(m == -float("inf"), 0., m)[:, None]), 1)) + m
    score = tl.max(x - tl.where(z == -float("inf"), 0., z)[:, None], 0)
    score = tl.where((p == 0) | (p >= historical // 32), 3.4028234663852886e38, score)
    valid = (p < actual_pages) & (p < P)
    bits = score.to(tl.uint32, bitcast=True)
    key = tl.where(score >= 0., bits ^ 0x80000000, ~bits)
    threshold = tl.full((), 0, tl.uint32)
    target = tl.minimum(actual_pages, 512)
    for bit in range(31, -1, -1):
        trial = threshold | (tl.full((), 1, tl.uint32) << bit)
        count = tl.sum((valid & (key >= trial)).to(tl.int32), 0)
        threshold = tl.where(count >= target, trial, threshold)
    above = valid & (key > threshold)
    equal = valid & (key == threshold)
    chosen = above | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= target - tl.sum(above.to(tl.int32), 0)))
    index = tl.cumsum(chosen.to(tl.int32), 0) - 1
    tl.store(IDS + row * 512 + index, p, chosen)


@tr.jit
def _query_code(Q, RQ, OUT, H: tl.constexpr, Q0: tl.constexpr, Q1: tl.constexpr):
    head = tl.program_id(0)
    hq = head % (H * 4)
    d = tl.arange(0, 128)
    r = tl.arange(0, 16)
    q = tl.load(Q + head // (H * 4) * Q0 + hq * Q1 + d).to(tl.float32)
    rq = tl.load(RQ + (hq * 128 + d[:, None]) * 16 + r[None, :]).to(tl.float32)
    tl.store(OUT + head * 16 + r, tl.sum(q[:, None] * rq, 0))


@tr.jit
def _fine(Q, QC, CACHE, TABLE, LENS, IDS, RIGHT, BIAS, ROPE, OUT,
          H: tl.constexpr, Q0: tl.constexpr, Q1: tl.constexpr, T0: tl.constexpr,
          C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, RP0: tl.constexpr):
    candidate = tl.program_id(0)
    row = tl.program_id(1)
    b, h = row // H, row % H
    page = tl.load(IDS + row * 512 + candidate)
    historical = tl.maximum(tl.load(LENS + b) - 64, 0)
    r = tl.arange(0, 16)
    lse = tl.full((16,), -float("inf"), tl.float32)
    if (page >= 0) & (page * 32 < historical):
        t = page * 32 + tl.arange(0, 32)
        d = tl.arange(0, 128)
        valid = t < historical
        block = tl.load(TABLE + b * T0 + page)
        addr = CACHE + block * C0 + h * C1 + (t % 32)[:, None] * C2
        base = tl.load(addr + r[None, :], valid[:, None], other=0)
        right = tl.load(RIGHT + (h * 16 + r[:, None]) * 128 + d[None, :])
        pred = tl.dot(base, right, allow_tf32=False).to(tl.bfloat16).to(tl.float32)
        pred = (pred + tl.load(BIAS + h * 128 + d)[None, :]).to(tl.bfloat16).to(tl.float32)
        swapped = tl.gather(pred, tl.broadcast_to(((d + 64) % 128)[None, :], (32, 128)), 1)
        cs = tl.load(ROPE + t[:, None] * RP0 + (d % 64)[None, :], valid[:, None], other=0).to(tl.float32)
        sn = tl.load(ROPE + t[:, None] * RP0 + (64 + d % 64)[None, :], valid[:, None], other=0).to(tl.float32)
        lo = (pred * cs).to(tl.bfloat16).to(tl.float32)
        hi = (tl.where(d[None, :] < 64, -swapped, swapped) * sn).to(tl.bfloat16).to(tl.float32)
        rotated = (lo + hi).to(tl.bfloat16)
        q = tl.load(Q + b * Q0 + (h * 4 + r[None, :]) * Q1 + d[:, None], r[None, :] < 4, other=0)
        base_score = tl.dot(rotated, q, allow_tf32=False).to(tl.bfloat16).to(tl.float32)
        res = tl.load(addr + 96 + r[None, :], valid[:, None], other=0)
        qc = tl.load(QC + (row * 4 + r[None, :]) * 16 + r[:, None], r[None, :] < 4, other=0)
        res_score = tl.dot(res, qc, allow_tf32=False).to(tl.bfloat16).to(tl.float32)
        score = ((base_score + res_score).to(tl.bfloat16).to(tl.float32) * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)
        score = tl.where(valid[:, None], score, -float("inf"))
        maximum = tl.max(score, 0)
        total = tl.sum(tl.exp(score - tl.where(maximum == -float("inf"), 0., maximum)[None, :]), 0)
        lse = maximum + tl.log(total)
    tl.store(OUT + (row * 4 + r) * 512 + candidate, lse, r < 4)


@tr.jit
def _select_fine(SCORES, IDS, PAGES):
    row = tl.program_id(0)
    p = tl.arange(0, 512)
    q = tl.arange(0, 4)
    ids = tl.load(IDS + row * 512 + p)
    valid = (p > 0) & (ids >= 0)
    x = tl.load(SCORES + (row * 4 + q[:, None]) * 512 + p[None, :])
    x = tl.where(valid[None, :], x, -float("inf"))
    maximum = tl.max(x, 1)
    probability = tl.exp(x - tl.where(maximum == -float("inf"), 0., maximum)[:, None])
    score = tl.max(probability / tl.maximum(tl.sum(probability, 1), 1.e-30)[:, None], 0)
    score = tl.where(valid, score, -float("inf"))
    # Positive float bits preserve order. Break cutoff ties by candidate index.
    bits = score.to(tl.uint32, bitcast=True)
    key = tl.where(score >= 0., bits ^ 0x80000000, ~bits)
    target = tl.minimum(tl.sum(valid.to(tl.int32), 0), 61)
    threshold = tl.full((), 0, tl.uint32)
    for bit in range(31, -1, -1):
        trial = threshold | (tl.full((), 1, tl.uint32) << bit)
        count = tl.sum((valid & (key >= trial)).to(tl.int32), 0)
        threshold = tl.where(count >= target, trial, threshold)
    above = valid & (key > threshold)
    equal = valid & (key == threshold)
    chosen = above | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= target - tl.sum(above.to(tl.int32), 0)))
    chosen = chosen | ((p == 0) & (ids == 0))
    index = tl.cumsum(chosen.to(tl.int32), 0) - 1
    tl.store(PAGES + row * 62 + index, ids, chosen)


@tr.jit
def _pack(PAGES, LENS, IDS, H: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, 2048)
    n = tl.load(LENS + row // H)
    historical = tl.maximum(n - 64, 0)
    page = tl.load(PAGES + row * 62 + i // 32, i < 1984, other=-1)
    t = tl.where(i < 1984, page * 32 + i % 32, historical + i - 1984)
    valid = tl.where(i < 1984, (page >= 0) & (t < historical), t < n)
    tl.store(IDS + row * 2048 + i, tl.where(valid, t, -1))


class PagedJointRouter:
    def __init__(self, batch, heads, capacity, device):
        self.heads, self.pages = heads, tr.cdiv(capacity, 32)
        self.coarse = torch.empty(batch, heads, 4, self.pages, device=device)
        self.candidates = torch.empty(batch, heads, 512, device=device, dtype=torch.int32)
        self.query_code = torch.empty(batch, heads, 4, 16, device=device, dtype=torch.bfloat16)
        # Unlaunched candidate columns stay -inf for the lifetime of the router.
        self.fine = torch.full((batch, heads, 4, 512), -float("inf"), device=device)
        self.selected_pages = torch.empty(batch, heads, 62, device=device, dtype=torch.int32)
        self.support = torch.empty(batch, heads, 2048, device=device, dtype=torch.int32)

    def select(self, query, cache, table, lengths, right, bias, residual_query, rope):
        batch, heads, pages = lengths.numel(), self.heads, self.pages
        assert table.shape[1] >= pages
        self.candidates[:batch].fill_(-1)
        self.selected_pages[:batch].fill_(-1)
        _coarse[(batch * heads * 4, tr.cdiv(pages, 16))](
            query, cache.gpu, table, lengths, self.coarse, heads, pages,
            *query.stride()[:2], table.stride(0), *cache.gpu.stride()[:3])
        _select_coarse[(batch * heads,)](self.coarse, lengths, self.candidates, heads, pages, tr.next_power_of_2(pages), num_warps=8)
        _query_code[(batch * heads * 4,)](query, residual_query, self.query_code, heads, *query.stride()[:2])
        _fine[(min(512, pages), batch * heads)](
            query, self.query_code, cache.gpu, table, lengths, self.candidates,
            right, bias, rope, self.fine, heads, *query.stride()[:2], table.stride(0),
            *cache.gpu.stride()[:3], rope.stride(0), num_warps=4)
        _select_fine[(batch * heads,)](self.fine, self.candidates, self.selected_pages, num_warps=8)
        _pack[(batch * heads,)](self.selected_pages, lengths, self.support, heads)
        return self.support[:batch]
