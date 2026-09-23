from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from basisserve.core.llama31_8b_tp8_combined import load_decode_postprocess_extension
from basisserve.kernels.mapped_host_paged_attention import (
    mapped_host_bf16_empty,
    mapped_host_device_pointer,
)
from benchmarks.system.register_router import compile_register


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.fixture(scope="module")
def postprocess():
    return load_decode_postprocess_extension()


@pytest.fixture(scope="module")
def full_router():
    return compile_register(Path("/tmp/basisserve_tp8_full_router"), 16, 8)


def selected_reference(scores):
    probabilities = scores[..., 1:].softmax(-1).nan_to_num(0)
    order = probabilities.amax(2).argsort(dim=-1, descending=True, stable=True)
    sink = torch.zeros_like(order[..., :1])
    return torch.cat((sink, order[..., :61] + 1), dim=-1).sort(-1).values


@pytest.mark.parametrize("batch,heads,pages,pattern", [
    (1, 1, 62, "random"), (1, 1, 127, "random"),
    (1, 1, 256, "random"), (1, 1, 257, "random"),
    (1, 1, 512, "random"), (8, 1, 513, "late"),
    (1, 1, 1024, "random"), (2, 2, 1025, "random"),
    (1, 1, 2048, "random"), (1, 1, 2049, "late"),
    (1, 1, 4094, "random"), (1, 1, 4096, "late"),
    (1, 1, 513, "tied"), (1, 1, 4094, "tied"),
    (1, 1, 513, "empty_head"),
])
@torch.inference_mode()
def test_full_selection_and_persistent_slots(postprocess, batch, heads, pages, pattern):
    torch.manual_seed(71)
    historical = pages * 32 - 3
    end = historical + 64
    capacity = end + 17
    host = mapped_host_bf16_empty(batch=batch, kv_heads=heads, capacity=capacity)
    host.copy_(torch.arange(capacity).remainder(251).to(torch.bfloat16)[None, None, :, None])
    pointer = mapped_host_device_pointer(host)
    storage = torch.randn(batch, heads, 4, pages + 11, device="cuda")
    scores = storage[..., :pages]
    if pattern == "late":
        scores.zero_()
        scores[..., -61:] = 20
    elif pattern == "tied":
        scores.zero_()
    elif pattern == "empty_head":
        scores[:, :, 0] = -torch.inf
    selected = torch.empty(batch, heads, 62, device="cuda", dtype=torch.long)
    ids = torch.empty(batch, heads, 2048, device="cuda", dtype=torch.long)
    cache = torch.full((batch, heads, 2048, 128), -1, device="cuda", dtype=torch.bfloat16)
    resident = torch.full_like(ids, -1)
    lookup = torch.full((batch, heads, capacity), -1, device="cuda", dtype=torch.int32)
    slots = torch.empty_like(ids)
    missing = torch.empty_like(ids, dtype=torch.int32)
    counts = torch.empty(batch, heads, 2, device="cuda", dtype=torch.int32)
    original = scores.clone()
    for step in range(4):
        if step == 2:
            scores.copy_(original.roll(73, -1))
        elif step == 3:
            scores.copy_(original)
        expected_pages = selected_reference(scores)
        expected_ids = (expected_pages[..., None] * 32 + torch.arange(32, device="cuda")).flatten(-2)
        expected_ids.masked_fill_(expected_ids >= historical, -1)
        expected_ids = torch.cat((expected_ids, torch.arange(historical, end, device="cuda").expand(batch, heads, -1)), -1)
        # A partial page can leave cached tokens from earlier selections resident.
        previous = [set(row) - {-1} for row in resident.flatten(0, 1).cpu().tolist()]
        postprocess.select_pack_refresh(
            scores, selected, ids, pointer, cache, resident, lookup,
            slots, missing, counts, historical, end,
        )
        torch.testing.assert_close(selected, expected_pages, atol=0, rtol=0)
        torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
        if step == 0 and pattern == "late":
            assert selected.max().item() == pages - 1
        valid = ids >= 0
        assert bool((slots[~valid] == -1).all())
        assert bool(((slots[valid] >= 0) & (slots[valid] < 2048)).all())
        actual_keys = cache.gather(2, slots.clamp_min(0)[..., None].expand(-1, -1, -1, 128))
        expected_keys = ids.clamp_min(0).remainder(251).to(cache.dtype)[..., None].expand_as(actual_keys)
        torch.testing.assert_close(actual_keys[valid], expected_keys[valid], atol=0, rtol=0)
        torch.testing.assert_close(resident.gather(-1, slots.clamp_min(0))[valid], ids[valid], atol=0, rtol=0)
        current = [set(row) - {-1} for row in ids.flatten(0, 1).cpu().tolist()]
        expected_counts = torch.tensor([[len(old & new), len(new)] for old, new in zip(previous, current)], dtype=counts.dtype)
        torch.testing.assert_close(counts.flatten(0, 1).cpu(), expected_counts, atol=0, rtol=0)
        if step == 1:
            assert bool((missing == -1).all())


@pytest.mark.parametrize("batch,historical", [(1, 16385), (2, 65505), (1, 130001)])
@torch.inference_mode()
def test_full_router_scores_all_pages_against_math(full_router, batch, historical):
    torch.manual_seed(93)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16
    query = torch.randn(batch, 4, 1, 128, device="cuda", dtype=dtype)
    values = torch.randn(batch, 1, historical + 64, 96, device="cuda", dtype=dtype) * 0.2
    base = values[:, :, :historical, :16]
    residual = torch.randn(batch, 1, historical, 16, device="cuda", dtype=dtype) * 0.2
    right = torch.randn(1, 16, 128, device="cuda", dtype=dtype) * 0.2
    bias = torch.randn(1, 128, device="cuda", dtype=dtype) * 0.1
    rq = torch.randn(4, 128, 16, device="cuda", dtype=dtype) * 0.2
    angle = torch.randn(historical, 64, device="cuda")
    cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
    code = torch.empty(batch, 1, 4, 16, device="cuda", dtype=dtype)
    pages = (historical + 31) // 32
    storage = torch.full((batch, 1, 4, pages + 7), float("nan"), device="cuda")
    actual = storage[..., :pages]
    full_router.conditional_router_page_lse(
        query, base, residual, right, bias, rq, cos, sin, code, actual, 128**-0.5, False
    )
    key = (base.float() @ right.float()).bfloat16()
    key = (key.float() + bias[None, :, None].float()).bfloat16().float()
    x = ((key[..., :64] * cos.float()).bfloat16().float() - (key[..., 64:] * sin.float()).bfloat16().float()).bfloat16().float()
    y = ((key[..., 64:] * cos.float()).bfloat16().float() + (key[..., :64] * sin.float()).bfloat16().float()).bfloat16().float()
    score = torch.einsum("bhqd,bhtd->bhqt", query.reshape(batch, 1, 4, 128).float(), torch.cat((x, y), -1)).bfloat16().float()
    correction = torch.einsum("bhqr,bhtr->bhqt", code.float(), residual.float()).bfloat16().float()
    score = ((score + correction).bfloat16().float() * 128**-0.5).bfloat16().float()
    expected = F.pad(score, (0, (-historical) % 32), value=-torch.inf).reshape(batch, 1, 4, pages, 32).logsumexp(-1)
    assert bool(torch.isfinite(actual).all())
    assert bool(torch.isnan(storage[..., pages:]).all())
    torch.testing.assert_close(actual, expected, rtol=0.003, atol=0.003)
