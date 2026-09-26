"""Bounded, layer-local BF16 staging from quantized pages for prefill only."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from vllm.v1.attention.backends.utils import split_prefill_chunks

from basisserve.kernels.diffkv_prefill import kernel_diffkv_prefill_sm89
from basisserve.kernels.nuq4_cache import load_nuq4_tile


@dataclass
class PrefillChunk:
    records: torch.Tensor
    cu_query: torch.Tensor
    seq_lens: torch.Tensor
    table: torch.Tensor
    query_tokens: int
    max_kv_len: int
    max_query_len: int


class PrefillWorkspace:
    def __init__(self, max_tokens, max_context, device):
        self.capacity = triton.cdiv(max(max_tokens, max_context), 16) * 16
        self.query = torch.empty((max_tokens, 4, 128), device=device, dtype=torch.bfloat16)
        self.output = torch.empty_like(self.query)
        self.keys = torch.empty((self.capacity // 16, 16, 1, 128), device=device, dtype=torch.bfloat16)
        self.values = torch.empty_like(self.keys)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.query, self.output, self.keys, self.values))


def plan_prefill(cu_query_cpu, seq_upper_cpu, seq_lens, capacity):
    """Only use already-available CPU scheduling metadata; never read back GPU lengths."""
    assert cu_query_cpu.device.type == seq_upper_cpu.device.type == "cpu"
    assert cu_query_cpu.numel() == seq_upper_cpu.numel() + 1
    lengths = cu_query_cpu[1:] - cu_query_cpu[:-1]
    ids = torch.nonzero(lengths > 1).flatten()
    if not ids.numel():
        return []
    upper = seq_upper_cpu[ids]
    padded = (upper + 15) // 16 * 16
    chunks = []

    def upload(tensor):
        return tensor.to(torch.int32).pin_memory().to(seq_lens.device, non_blocking=True)

    for start, end in split_prefill_chunks(padded, capacity):
        selected = ids[start:end]
        offsets = torch.cat((torch.zeros(1, dtype=torch.int64), padded[start:end].cumsum(0)[:-1]))
        cu = torch.cat((torch.zeros(1, dtype=torch.int64), lengths[selected].cumsum(0)))
        records = upload(torch.stack((selected, offsets, cu[:-1]), dim=1))
        max_len = int(upper[start:end].max())
        table = offsets[:, None] // 16 + torch.arange(triton.cdiv(max_len, 16))[None, :]
        chunks.append(PrefillChunk(records, upload(cu), seq_lens.index_select(0, records[:, 0]),
            upload(table), int(cu[-1]), max_len, int(lengths[selected].max())))
    return chunks


@triton.jit
def _stage(Q, KC, VC, KLO, KHI, KLUT, VLO, VHI, VLUT, ROPE, CU, LENS, TABLE,
           RECORDS, QTMP, KTMP, VTMP, QS0: tl.constexpr, QS1: tl.constexpr,
           TS: tl.constexpr, KL: tl.constexpr, VL: tl.constexpr, BV: tl.constexpr,
           BN: tl.constexpr):
    item, tile = tl.program_id(0), tl.program_id(1)
    seq = tl.load(RECORDS + item * 3)
    dest_k = tl.load(RECORDS + item * 3 + 1)
    dest_q = tl.load(RECORDS + item * 3 + 2)
    start = tl.load(CU + seq)
    qlen = tl.load(CU + seq + 1) - start
    length = tl.load(LENS + seq)
    pos = tile * BN + tl.arange(0, BN)
    valid = pos < length
    d, dv = tl.arange(0, 128), tl.arange(0, BV)
    physical = tl.load(TABLE + seq * TS + pos // KL[1], valid, 0).to(tl.int64)
    slots = physical * KL[1] + pos % KL[1]
    k = load_nuq4_tile(KC, KLO, KHI, KLUT, slots, valid, False,
        KL[0], KL[1], KL[2], KL[3], KL[4], KL[5], KL[6], KL[7], KL[8], KL[9]).to(tl.float32)
    mate = tl.gather(k, tl.broadcast_to(((d + 64) % 128)[None, :], (BN, 128)), 1)
    cos = tl.load(ROPE + pos[:, None] * 128 + (d % 64)[None, :], valid[:, None], 0).to(tl.float32)
    sin = tl.load(ROPE + pos[:, None] * 128 + (d % 64)[None, :] + 64, valid[:, None], 0).to(tl.float32)
    k = (k * cos + tl.where(d[None, :] < 64, -mate, mate) * sin).to(tl.bfloat16)
    tl.store(KTMP + (dest_k + pos[:, None]) * 128 + d[None, :], k, valid[:, None])
    v = load_nuq4_tile(VC, VLO, VHI, VLUT, slots, valid, True,
        VL[0], VL[1], VL[2], VL[3], VL[4], VL[5], VL[6], VL[7], VL[8], VL[9])
    tl.store(VTMP + (dest_k + pos[:, None]) * 128 + dv[None, :], v,
             valid[:, None] & (dv[None, :] < VL[0]))
    for head in tl.static_range(4):
        q = tl.load(Q + (start + pos[:, None]) * QS0 + head * QS1 + d[None, :],
                     pos[:, None] < qlen, 0)
        tl.store(QTMP + ((dest_q + pos[:, None]) * 4 + head) * 128 + d[None, :],
                 q, pos[:, None] < qlen)


@triton.jit
def _scatter(SRC, DST, CU, RECORDS, OS0: tl.constexpr, OS1: tl.constexpr,
             D: tl.constexpr, BD: tl.constexpr, BM: tl.constexpr):
    item, block = tl.program_id(0), tl.program_id(1)
    seq = tl.load(RECORDS + item * 3)
    source = tl.load(RECORDS + item * 3 + 2)
    start = tl.load(CU + seq)
    length = tl.load(CU + seq + 1) - start
    row = block * BM + tl.arange(0, BM)
    d = tl.arange(0, BD)
    token, head = row // 4, row % 4
    mask = (token[:, None] < length) & (d[None, :] < D)
    value = tl.load(SRC + ((source + token[:, None]) * 4 + head[:, None]) * 128 + d[None, :], mask, 0)
    tl.store(DST + (start + token[:, None]) * OS0 + head[:, None] * OS1 + d[None, :], value, mask)


def staged_prefill(query, keys, values, rope, cu_query, seq_lens, block_table,
                   chunks, workspace, *, out):
    width = values.layout.width
    assert query.dtype == out.dtype == torch.bfloat16 and query.shape[1:] == (4, 128)
    assert out.shape == (query.shape[0], 4, width)
    assert query.stride(-1) == out.stride(-1) == 1
    for chunk in chunks:
        assert chunk.query_tokens <= workspace.query.shape[0]
        count = chunk.records.shape[0]
        _stage[(count, triton.cdiv(chunk.max_kv_len, 32))](
            query, keys.storage, values.storage, keys.lower, keys.upper, keys.lut,
            values.lower, values.upper, values.lut, rope, cu_query, seq_lens, block_table,
            chunk.records, workspace.query, workspace.keys, workspace.values,
            *query.stride()[:2], block_table.stride(0), tuple(keys.kernel_constants().values()),
            tuple(values.kernel_constants().values()), triton.next_power_of_2(width), 32,
            num_warps=4, num_stages=1, enable_fp_fusion=False)
        bm, bn = 128, 128
        kernel_diffkv_prefill_sm89[(chunk.query_tokens // (bm // 4) + count,)](
            workspace.query, workspace.keys, workspace.values, workspace.output,
            chunk.cu_query, chunk.seq_lens, chunk.table, 128 ** -0.5,
            *workspace.query.stride()[:2], *workspace.output.stride()[:2],
            *workspace.keys.stride()[:2], *workspace.values.stride()[:2], chunk.table.stride(0), count,
            GROUP=4, PAGE=16, BLOCK_M=bm, BLOCK_N=bn, VALUE_DIM=width,
            BLOCK_V=triton.next_power_of_2(width), num_warps=8, num_stages=2)
        _scatter[(count, triton.cdiv(chunk.max_query_len * 4, 32))](
            workspace.output, out, cu_query, chunk.records, *out.stride()[:2],
            width, triton.next_power_of_2(width), 32, num_warps=4)
    return out
