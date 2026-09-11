"""Explicit FP32-input, FP32-accumulation TF32x3 matrix multiplication."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        AM: tl.constexpr, AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
        TM: tl.constexpr = 32, TN: tl.constexpr = 64, TK: tl.constexpr = 32):
    rows = tl.program_id(0) * TM + tl.arange(0, TM)
    cols = tl.program_id(1) * TN + tl.arange(0, TN)
    reduction = tl.arange(0, TK)
    acc = tl.full((TM, TN), 0, tl.float32)
    for start in range(tl.cdiv(K, TK)):
        ks = start * TK + reduction
        left = tl.load(A + rows[:, None] * AM + ks[None, :] * AK,
                       (rows[:, None] < M) & (ks[None, :] < K), other=0)
        right = tl.load(B + ks[:, None] * BK + cols[None, :] * BN,
                        (ks[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(left, right, acc, input_precision='tf32x3')
    tl.store(C + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


def fp32_tf32x3_mm(left, right):
    assert left.ndim == right.ndim == 2 and left.shape[1] == right.shape[0]
    assert left.dtype == right.dtype == torch.float32 and left.is_cuda and right.device == left.device
    output = torch.empty(left.shape[0], right.shape[1], device=left.device, dtype=torch.float32)
    _mm[(triton.cdiv(left.shape[0], 32), triton.cdiv(right.shape[1], 64))](
        left, right, output, left.shape[0], right.shape[1], left.shape[1], *left.stride(), *right.stride())
    return output
