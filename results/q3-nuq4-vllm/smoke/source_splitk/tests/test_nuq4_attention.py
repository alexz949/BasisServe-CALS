import pytest
import torch
from torch.nn import functional as F

from basisserve.kernels.nuq4_cache import NUQ4Layout, NUQ4PagedCache, nuq4_value_stats
from basisserve.kernels.nuq4_attention import nuq4_attention
from basisserve.kernels.nuq4_decode import nuq4_decode, decode_workspace


@pytest.mark.parametrize("width", [32, 64, 80, 96, 128])
@pytest.mark.parametrize("decode", [False, True])
def test_full_attention_packed_tiles(width, decode):
    torch.manual_seed(width)
    lengths = [67, 35]
    queries = [1, 1] if decode else [33, 17]
    lower, upper = torch.full((128,), -4.0, device="cuda"), torch.full((128,), 4.0, device="cuda")
    lut = torch.linspace(-1, 1, 16, device="cuda")
    kcache = NUQ4PagedCache(8, NUQ4Layout(128), lower, upper, lut)
    vcache = NUQ4PagedCache(8, NUQ4Layout(width), lower[:width], upper[:width], lut, dynamic=True)
    table = torch.tensor([[4, 2, 0, 7, 6], [1, 5, 3, 0, 0]], device="cuda", dtype=torch.int32)
    slots, key_rows, value_rows = [], [], []
    for seq, n in enumerate(lengths):
        pos = torch.arange(n, device="cuda")
        slots.append(table[seq, pos // 16].long() * 16 + pos % 16)
        key_rows.append(torch.randn(n, 128, device="cuda", dtype=torch.bfloat16))
        value_rows.append(torch.randn(n, width * 8, device="cuda", dtype=torch.bfloat16))
    slots = torch.cat(slots)
    keys, full_v = torch.cat(key_rows), torch.cat(value_rows)
    kcache.append(keys, slots)
    vcache.append(full_v[:, :width], slots, nuq4_value_stats(full_v))
    kcache.check()
    vcache.check()
    pos = torch.arange(max(lengths), device="cuda").float()
    freq = 10000 ** (-torch.arange(64, device="cuda").float() / 64)
    angles = pos[:, None] * freq[None, :]
    rope = torch.cat((angles.cos(), angles.sin()), 1).to(torch.bfloat16)
    query = torch.randn(sum(queries), 4, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, queries[0], sum(queries)], device="cuda", dtype=torch.int32)
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    actual = nuq4_attention(query, kcache, vcache, rope, cu, lens, table)
    kr = kcache.gather_for_validation(slots).float()
    vr = vcache.gather_for_validation(slots)
    expected, ki, qi = [], 0, 0
    for n, nq in zip(lengths, queries):
        k = kr[ki:ki+n]
        cos, sin = rope[:n, :64].float().repeat(1, 2), rope[:n, 64:].float().repeat(1, 2)
        mate = torch.cat((-k[:, 64:], k[:, :64]), 1)
        rotated = (k * cos + mate * sin).to(torch.bfloat16)
        mask = torch.arange(n, device="cuda")[None, :] <= n-nq+torch.arange(nq, device="cuda")[:, None]
        result = F.scaled_dot_product_attention(query[qi:qi+nq].transpose(0, 1)[None],
            rotated[None, None], vr[ki:ki+n][None, None], attn_mask=mask,
            enable_gqa=True).squeeze(0).transpose(0, 1)
        expected.append(result)
        ki, qi = ki+n, qi+nq
    torch.testing.assert_close(actual, torch.cat(expected), atol=0.008, rtol=0.015)
    assert torch.isfinite(actual).all()
    if decode:
        for splits in (1, 32):
            bv = 1 << (width-1).bit_length()
            workspace = (torch.empty(sum(queries), 4, splits, bv, device="cuda"),
                torch.empty(sum(queries), 4, splits, device="cuda"),
                torch.empty(sum(queries), 4, splits, device="cuda"))
            split_out = torch.empty_like(actual)
            nuq4_decode(query, kcache, vcache, rope, cu, lens, table, out=split_out, workspace=workspace)
            torch.testing.assert_close(split_out, torch.cat(expected), atol=0.008, rtol=0.015)


@pytest.mark.parametrize("width", [64, 96])
def test_4k_prefill_decode_and_graph(width):
    torch.manual_seed(42)
    n = 4096
    lower, upper = torch.full((128,), -4.0, device="cuda"), torch.full((128,), 4.0, device="cuda")
    lut = torch.linspace(-1, 1, 16, device="cuda")
    kc = NUQ4PagedCache(257, NUQ4Layout(128), lower, upper, lut)
    vc = NUQ4PagedCache(257, NUQ4Layout(width), lower[:width], upper[:width], lut, dynamic=True)
    keys = torch.randn(n+1, 128, device="cuda", dtype=torch.bfloat16)
    global_v = torch.randn(n+1, width*8, device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(n+1, device="cuda")
    stats = nuq4_value_stats(global_v)
    kc.append(keys[:n], slots[:n])
    vc.append(global_v[:n, :width], slots[:n], stats[:n])
    table = torch.arange(257, device="cuda", dtype=torch.int32)[None]
    freq = 10000 ** (-torch.arange(64, device="cuda").float() / 64)
    angles = torch.arange(n+1, device="cuda")[:, None] * freq[None]
    rope = torch.cat((angles.cos(), angles.sin()), 1).to(torch.bfloat16)
    query = torch.randn(n, 4, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, n], device="cuda", dtype=torch.int32)
    lens = torch.tensor([n], device="cuda", dtype=torch.int32)
    prefill = nuq4_attention(query, kc, vc, rope, cu, lens, table)
    k = kc.gather_for_validation(slots[:n]).float()
    rotated = (k * rope[:n, :64].float().repeat(1, 2) +
        torch.cat((-k[:, 64:], k[:, :64]), 1) * rope[:n, 64:].float().repeat(1, 2)).to(torch.bfloat16)
    v = vc.gather_for_validation(slots[:n])
    expected = F.scaled_dot_product_attention(query.transpose(0, 1)[None], rotated[None, None],
        v[None, None], is_causal=True, enable_gqa=True).squeeze(0).transpose(0, 1)
    torch.testing.assert_close(prefill, expected, atol=0.008, rtol=0.015)
    decode_q = query[-1:].contiguous()
    cu[1], lens[0] = 1, n+1
    out = torch.empty((1, 4, width), device="cuda", dtype=torch.bfloat16)
    workspace = decode_workspace(1, width, "cuda")

    def step():
        kc.append(keys[n:], slots[n:])
        vc.append(global_v[n:, :width], slots[n:], stats[n:])
        nuq4_decode(decode_q, kc, vc, rope, cu, lens, table, out=out, workspace=workspace)

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    graph.replay()
    kc.check()
    vc.check()
    k = kc.gather_for_validation(slots).float()
    rotated = (k * rope[:, :64].float().repeat(1, 2) +
        torch.cat((-k[:, 64:], k[:, :64]), 1) * rope[:, 64:].float().repeat(1, 2)).to(torch.bfloat16)
    v = vc.gather_for_validation(slots)
    expected = F.scaled_dot_product_attention(decode_q.transpose(0, 1)[None], rotated[None, None],
        v[None, None], enable_gqa=True).squeeze(0).transpose(0, 1)
    torch.testing.assert_close(out, expected, atol=0.008, rtol=0.015)
