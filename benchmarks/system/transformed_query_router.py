"""Diagnostic transformed-query router; omits legacy intermediate BF16 rounding."""
import torch
import triton
import triton.language as tl


@triton.jit
def _coefficients(Q, RIGHT, BIAS, C, S, QS0: tl.constexpr, QS1: tl.constexpr):
    head = tl.program_id(0)
    batch = head // 32
    hq = head % 32
    kv = hq // 4
    r = tl.arange(0, 32)
    d = tl.arange(0, 64)
    q0 = tl.load(Q + batch * QS0 + hq * QS1 + d).to(tl.float32)
    q1 = tl.load(Q + batch * QS0 + hq * QS1 + d + 64).to(tl.float32)
    a = tl.load(RIGHT + kv * 16 * 128 + r[:, None] * 128 + d[None, :], r[:, None] < 16, 0).to(tl.float32)
    b = tl.load(RIGHT + kv * 16 * 128 + r[:, None] * 128 + d[None, :] + 64, r[:, None] < 16, 0).to(tl.float32)
    bias0 = tl.load(BIAS + kv * 128 + d).to(tl.float32)
    bias1 = tl.load(BIAS + kv * 128 + d + 64).to(tl.float32)
    a = tl.where(r[:, None] == 16, bias0[None, :], a)
    b = tl.where(r[:, None] == 16, bias1[None, :], b)
    offsets = head * 17 * 64 + r[:, None] * 64 + d[None, :]
    tl.store(C + offsets, a * q0[None, :] + b * q1[None, :], r[:, None] < 17)
    tl.store(S + offsets, a * q1[None, :] - b * q0[None, :], r[:, None] < 17)


@triton.jit
def _pages(BASE, RES, QC, COS, SIN, C, S, OUT,
           T: tl.constexpr, P: tl.constexpr,
           BS0: tl.constexpr, BS1: tl.constexpr, BS2: tl.constexpr,
           RS0: tl.constexpr, RS1: tl.constexpr, RS2: tl.constexpr,
           SCALE: tl.constexpr):
    page = tl.program_id(0)
    head = tl.program_id(1)
    batch = head // 32
    kv = (head % 32) // 4
    t = page * 32 + tl.arange(0, 32)
    d = tl.arange(0, 64)
    r = tl.arange(0, 16)
    co = tl.load(COS + t[:, None] * 64 + d[None, :], t[:, None] < T, 0).to(tl.float32)
    si = tl.load(SIN + t[:, None] * 64 + d[None, :], t[:, None] < T, 0).to(tl.float32)
    cc = tl.load(C + head * 17 * 64 + r[None, :] * 64 + d[:, None])
    ss = tl.load(S + head * 17 * 64 + r[None, :] * 64 + d[:, None])
    # Exact real-arithmetic reassociation, evaluated with TF32x3 products.
    transformed = tl.dot(co, cc, input_precision='tf32x3') + tl.dot(si, ss, input_precision='tf32x3')
    z = tl.load(BASE + batch * BS0 + kv * BS1 + t[:, None] * BS2 + r[None, :], t[:, None] < T, 0).to(tl.float32)
    bc = tl.load(C + head * 17 * 64 + 16 * 64 + d)
    bs = tl.load(S + head * 17 * 64 + 16 * 64 + d)
    bias = tl.sum(co * bc[None, :] + si * bs[None, :], 1)
    base_score = tl.sum(z * transformed, 1) + bias
    residual = tl.load(RES + batch * RS0 + kv * RS1 + t[:, None] * RS2 + r[None, :], t[:, None] < T, 0).to(tl.float32)
    qc = tl.load(QC + head * 16 + r).to(tl.float32)
    residual_score = tl.sum(residual * qc[None, :], 1)
    score = ((base_score.to(tl.bfloat16).to(tl.float32) + residual_score.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32) * SCALE).to(tl.bfloat16).to(tl.float32)
    score = tl.where(t < T, score, -float('inf'))
    maximum = tl.max(score, 0)
    lse = maximum + tl.log(tl.sum(tl.exp(score - maximum), 0))
    tl.store(OUT + head * P + page, lse)


def allocate(query, base):
    coefficients = [torch.empty(query.shape[0] * 32, 17, 64, device=query.device, dtype=torch.float32) for _ in range(2)]
    output = torch.empty(query.shape[0], 8, 4, triton.cdiv(base.shape[2], 32), device=query.device)
    return coefficients, output


def run(query, base, residual, right, bias, cosine, sine, query_code, coefficients, output):
    global last_coefficient_kernel, last_page_kernel
    assert base.shape[-1] == residual.shape[-1] == 16 and base.shape[1] == 8
    c, s = coefficients
    last_coefficient_kernel = _coefficients[(query.shape[0] * 32,)](query, right, bias, c, s, query.stride(0), query.stride(1), num_warps=4)
    last_page_kernel = _pages[(output.shape[-1], query.shape[0] * 32)](
        base, residual, query_code, cosine, sine, c, s, output, base.shape[2], output.shape[-1],
        *base.stride()[:3], *residual.stride()[:3], 128 ** -.5, num_warps=4)
    return output
