import pytest
import torch

from basisserve.kernels.nuq4_attention import nuq4_attention
from basisserve.kernels.nuq4_cache import NUQ4Layout, NUQ4PagedCache, nuq4_value_stats
from basisserve.kernels.nuq4_decode import decode_workspace, nuq4_decode
from basisserve.kernels.nuq4_prefill import PrefillWorkspace, plan_prefill, staged_prefill


@pytest.mark.parametrize("width", [32, 48, 64, 80, 96, 112, 128])
def test_bounded_staging_with_prefixes_and_interleaved_decode(width):
    torch.manual_seed(width)
    lengths, queries = [67, 35, 49, 17], [1, 17, 0, 5]
    pages = [(n + 15) // 16 for n in lengths]
    lo = torch.full((128,), -4.0, device="cuda")
    lut = torch.linspace(-1, 1, 16, device="cuda")
    kc = NUQ4PagedCache(sum(pages), NUQ4Layout(128), lo, -lo, lut)
    vc = NUQ4PagedCache(sum(pages), NUQ4Layout(width), lo[:width], -lo[:width], lut, dynamic=True)
    physical = torch.randperm(sum(pages), device="cuda", dtype=torch.int32)
    table = torch.zeros((len(pages), max(pages)), device="cuda", dtype=torch.int32)
    slots, offset = [], 0
    for seq, (length, count) in enumerate(zip(lengths, pages)):
        table[seq, :count] = physical[offset:offset+count]
        pos = torch.arange(length, device="cuda")
        slots.append(table[seq, pos // 16].long() * 16 + pos % 16)
        offset += count
    slots = torch.cat(slots)
    full_v = torch.randn(sum(lengths), width*8, device="cuda", dtype=torch.bfloat16)
    kc.append(torch.randn(sum(lengths), 128, device="cuda", dtype=torch.bfloat16), slots)
    vc.append(full_v[:, :width], slots, nuq4_value_stats(full_v))
    kc.check()
    vc.check()
    angles = torch.arange(max(lengths), device="cuda")[:, None] * (
        10000 ** (-torch.arange(64, device="cuda").float() / 64))[None]
    rope = torch.cat((angles.cos(), angles.sin()), 1).to(torch.bfloat16)
    query = torch.randn(sum(queries), 4, 128, device="cuda", dtype=torch.bfloat16)
    cu_cpu = torch.tensor([0, *torch.tensor(queries).cumsum(0).tolist()], dtype=torch.int32)
    cu = cu_cpu.cuda()
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    expected = nuq4_attention(query, kc, vc, rope, cu, lens, table)
    output = torch.full_like(expected, float("nan"))
    workspace = PrefillWorkspace(sum(queries), 64, query.device)
    chunks = plan_prefill(cu_cpu, torch.tensor(lengths) + 3, lens, workspace.capacity)
    assert len(chunks) == 2 and workspace.capacity == 64
    assert all(chunk.query_tokens <= 17 for chunk in chunks)
    scratch = decode_workspace(len(lengths), width, query.device)

    def step():
        staged_prefill(query, kc, vc, rope, cu, lens, table, chunks, workspace, out=output)
        nuq4_decode(query, kc, vc, rope, cu, lens, table, out=output, workspace=scratch)

    step()
    torch.testing.assert_close(output, expected, atol=0.008, rtol=0.015)
    if width in (64, 96):
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0.008, rtol=0.015)
    assert plan_prefill(torch.arange(5), torch.tensor(lengths), lens, workspace.capacity) == []
