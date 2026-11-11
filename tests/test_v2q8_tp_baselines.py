"""The V2Q8 tensor-parallel baselines keep the paper Table 2 per-request arithmetic (CPU, world size 1).

LRQK: the rank-averaged prefill/decode factor updates equal c1_lrqk's for one request, and a batched decode
gives every row exactly what that row gets alone (per-request convergence). ShadowKV: the state built from
externally supplied global SVD factors equals C1ShadowKVState, including a decode step."""

import torch

from basisserve.core import c1_lrqk
from basisserve.core.c1_shadowkv import C1ShadowKVState
from benchmarks.system import v2q8_tp_baselines as tp


def lrqk_inputs(seed, rows=1, heads=8, length=48, dim=16, rank=4):
    generator = torch.Generator().manual_seed(seed)
    draw = lambda *shape: torch.randn(*shape, generator=generator)
    return dict(bq=draw(rows, heads, rank, dim), ak=draw(rows, heads, 24, rank), bk=draw(rows, heads, rank, dim),
                active_k=draw(rows, heads, 24, dim), q=draw(rows, heads, 1, dim), k=draw(rows, heads, 1, dim))


def test_lrqk_prefill_factors_match_reference():
    torch.manual_seed(0)
    q, k = torch.randn(1, 8, 64, 16), torch.randn(1, 8, 64, 16)
    aq, ak = torch.randn(1, 8, 64, 4), torch.randn(1, 8, 64, 4)
    for tolerance in (0.0, 10.0):
        expected = c1_lrqk.prefill_factors(q, k, aq, ak, 2, tolerance)
        actual = tp.lrqk_prefill_factors(q, k, aq, ak, 2, tolerance)
        for left, right in zip(expected, actual, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_lrqk_decode_factors_match_reference_per_request():
    for tolerance in (0.0, 1e9, 0.5):
        rows = [lrqk_inputs(seed) for seed in range(3)]
        expected = [c1_lrqk.decode_factors(**row, iterations=2, tolerance=tolerance) for row in rows]
        batched = {name: torch.cat([row[name] for row in rows]) for name in rows[0]}
        actual = tp.lrqk_decode_factors(**batched, iterations=2, tolerance=tolerance)
        for index, reference in enumerate(expected):
            for left, right in zip(reference, actual, strict=True):
                torch.testing.assert_close(left[0], right[index], rtol=1e-5, atol=1e-5)


def test_shadowkv_state_with_supplied_svd_matches_reference():
    torch.manual_seed(2)
    heads, length, dim = 8, 4096, 128
    pre, post = torch.randn(1, heads, length, dim), torch.randn(1, heads, length, dim)
    cos, sin = torch.randn(1, length, 64), torch.randn(1, length, 64)
    options = dict(rank=16, budget=256, chunk=8, outliers=8)
    reference = C1ShadowKVState(pre, post, cos, sin, **options)
    u, s, v = torch.svd(pre.transpose(1, 2).reshape(1, length, heads * dim).float())
    sv = (torch.diag_embed(s[:, :16]) @ v.transpose(1, 2)[:, :16]).view(1, 16, heads, dim).transpose(1, 2)
    state = tp.ShadowKVState(pre, post, cos, sin, u[:, :, :16].contiguous(), sv.contiguous(), **options)
    for name in ("u", "sv", "landmarks", "landmark_ids", "fixed_ids", "fixed_key"):
        torch.testing.assert_close(getattr(reference, name), getattr(state, name), rtol=0, atol=0)
    query, key = torch.randn(1, heads * 4, 1, dim), torch.randn(1, heads, 1, dim)
    values = torch.randn(1, heads, length + 1, 96)
    torch.testing.assert_close(reference.decode(query, key, values, 0.1), state.decode(query, key, values, 0.1),
                               rtol=0, atol=0)
    merged = tp.ShadowKVState.merge([state, state])
    assert merged.u.shape[0] == 2 and merged.landmarks.shape[0] == 2
