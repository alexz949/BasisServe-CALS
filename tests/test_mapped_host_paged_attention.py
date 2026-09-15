from __future__ import annotations

import math
import os

import pytest
import torch

from basisserve.kernels.mapped_host_paged_attention import (
    append_mapped_host_key,
    conditional_router_append_decode,
    conditional_router_page_lse,
    gpu_paged_attention,
    mapped_host_bf16_empty,
    mapped_host_device_pointer,
    mapped_host_paged_attention,
    select_fixed_group_max_pages_cuda,
)
from basisserve.core.qwen3_8b_tp4_k_offload import (
    _feature_major_attention_view,
    conditional_router_page_log_mass,
    select_fixed_group_max_pages,
)


_CUDA_BUILD_AVAILABLE = torch.cuda.is_available() and bool(
    os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
)


def test_feature_major_attention_view_aliases_allgather_slot() -> None:
    batch = 3
    query_heads = 8
    value_dim = 80
    feature_major = torch.zeros(query_heads * value_dim, batch)
    attention = _feature_major_attention_view(
        feature_major,
        query_heads=query_heads,
        value_dim=value_dim,
    )
    expected = torch.arange(
        batch * query_heads * value_dim,
        dtype=feature_major.dtype,
    ).reshape(batch, query_heads, 1, value_dim)
    attention.copy_(expected)

    assert attention.data_ptr() == feature_major.data_ptr()
    torch.testing.assert_close(
        feature_major.transpose(0, 1),
        expected.reshape(batch, query_heads * value_dim),
    )


def _apply_rope_reference(
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin),
        dim=-1,
    )


@pytest.mark.skipif(
    not _CUDA_BUILD_AVAILABLE
    or (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8),
    reason="the BF16 Tensor-Core router requires a CUDA build node with SM80+",
)
@pytest.mark.parametrize("base_rank,residual_rank,group,page", [(16,8,4,32),(4,16,1,16),(32,20,8,64),(0,20,2,32)])
def test_conditional_router_page_lse_matches_pytorch_reference(base_rank, residual_rank, group, page) -> None:
    torch.manual_seed(20260902)
    device = torch.device("cuda")
    batch = 1
    kv_heads = 2
    query_heads = 2 * group
    tokens = 70
    query = torch.randn(
        batch,
        1,
        query_heads,
        128,
        dtype=torch.bfloat16,
        device=device,
    ).transpose(1, 2)
    value = torch.randn(
        batch,
        kv_heads,
        tokens,
        80,
        dtype=torch.bfloat16,
        device=device,
    )
    residual_code = torch.randn(
        batch,
        kv_heads,
        tokens,
        residual_rank,
        dtype=torch.bfloat16,
        device=device,
    )
    base_left = torch.randn(
        kv_heads, 80, base_rank, dtype=torch.bfloat16, device=device
    ) / 8
    base_right = torch.randn(
        kv_heads, base_rank, 128, dtype=torch.bfloat16, device=device
    ) / 8
    base_bias = torch.randn(
        kv_heads, 128, dtype=torch.bfloat16, device=device
    ) / 8
    residual_query = torch.randn(
        query_heads, 128, residual_rank, dtype=torch.bfloat16, device=device
    ) / 8
    angles = torch.randn(tokens, 64, dtype=torch.float32, device=device)
    rope_cos = angles.cos().to(torch.bfloat16)
    rope_sin = angles.sin().to(torch.bfloat16)
    base_code = torch.einsum("bgtv,gvr->bgtr", value, base_left)
    scale = 128**-0.5

    observed = conditional_router_page_lse(
        query,
        base_code,
        residual_code,
        base_right=base_right,
        base_bias=base_bias,
        residual_query=residual_query,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        scale=scale,
        page_size=page,
    )
    expected = conditional_router_page_log_mass(
        query,
        value,
        residual_code,
        base_left=base_left,
        base_right=base_right,
        base_bias=base_bias,
        residual_query=residual_query,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        page_size=page,
        page_chunk=2,
        scale=scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(observed, expected, rtol=1.0e-2, atol=2.0e-2)


@pytest.mark.parametrize(
    ("pages", "budget"),
    ((7, 16), (70, 16), (2048, 128)),
)
@pytest.mark.skipif(
    not _CUDA_BUILD_AVAILABLE
    or (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8),
    reason="the fused physical-page selector requires a CUDA build node",
)
def test_fused_group_max_page_selection_matches_pytorch_reference(
    pages: int,
    budget: int,
) -> None:
    torch.manual_seed(2718 + pages)
    scores = torch.randn(
        2,
        2,
        4,
        pages + 3,
        dtype=torch.float32,
        device="cuda",
    )[..., :pages]
    expected = select_fixed_group_max_pages(
        scores,
        pages_per_kv_head=budget,
        pinned_prefix_pages=1,
        force_current_page=True,
    )
    observed = select_fixed_group_max_pages_cuda(
        scores,
        pages_per_kv_head=budget,
        pinned_prefix_pages=1,
        force_current_page=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not _CUDA_BUILD_AVAILABLE
    or (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8),
    reason="the BF16 fused append requires a CUDA build node with SM80+",
)
@pytest.mark.parametrize("value_dim,base_rank,residual_rank", [(80,16,8),(96,4,16),(128,32,20),(37,0,20)])
def test_conditional_router_append_decode_matches_pytorch_reference(value_dim, base_rank, residual_rank) -> None:
    torch.manual_seed(31415)
    device = torch.device("cuda")
    batch = 1
    kv_heads = 2
    capacity = 7
    start = 5
    key = torch.randn(
        batch,
        1,
        kv_heads,
        128,
        dtype=torch.bfloat16,
        device=device,
    ).transpose(1, 2)
    value = torch.randn(
        batch,
        1,
        kv_heads,
        value_dim,
        dtype=torch.bfloat16,
        device=device,
    ).transpose(1, 2)
    base_left = torch.randn(
        kv_heads, value_dim, base_rank, dtype=torch.bfloat16, device=device
    ) / 8
    base_right = torch.randn(
        kv_heads, base_rank, 128, dtype=torch.bfloat16, device=device
    ) / 8
    base_bias = torch.randn(
        kv_heads, 128, dtype=torch.bfloat16, device=device
    ) / 8
    residual_encoder = torch.randn(
        kv_heads, 128, residual_rank, dtype=torch.bfloat16, device=device
    ) / 8
    angles = torch.randn(1, 64, dtype=torch.float32, device=device)
    rope_cos = angles.cos().to(torch.bfloat16)
    rope_sin = angles.sin().to(torch.bfloat16)
    value_cache = torch.zeros(
        batch, kv_heads, capacity, value_dim, dtype=torch.bfloat16, device=device
    )
    base_cache = torch.zeros(
        batch, kv_heads, capacity, base_rank, dtype=torch.bfloat16, device=device
    )
    residual_cache = torch.zeros(
        batch, kv_heads, capacity, residual_rank, dtype=torch.bfloat16, device=device
    )
    rope_cos_cache = torch.zeros(
        capacity, 64, dtype=torch.bfloat16, device=device
    )
    rope_sin_cache = torch.zeros_like(rope_cos_cache)

    expected_base = torch.einsum("bgtv,gvr->bgtr", value, base_left)
    expected_pre = torch.einsum(
        "bgtr,grd->bgtd", expected_base, base_right
    )
    expected_pre.add_(base_bias[None, :, None])
    expected_post = _apply_rope_reference(
        expected_pre,
        rope_cos[None, None],
        rope_sin[None, None],
    )
    expected_residual = torch.einsum(
        "bgtd,gdr->bgtr",
        key - expected_post,
        residual_encoder,
    )
    conditional_router_append_decode(
        key,
        value,
        base_left=base_left,
        base_right=base_right,
        base_bias=base_bias,
        residual_encoder=residual_encoder,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        value_cache=value_cache,
        base_cache=base_cache,
        residual_cache=residual_cache,
        rope_cos_cache=rope_cos_cache,
        rope_sin_cache=rope_sin_cache,
        start=start,
        write_rope=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(value_cache[:, :, start : start + 1], value)
    torch.testing.assert_close(
        base_cache[:, :, start : start + 1],
        expected_base,
        rtol=2.0e-2,
        atol=2.0e-2,
    )
    torch.testing.assert_close(
        residual_cache[:, :, start : start + 1],
        expected_residual,
        rtol=2.0e-2,
        atol=2.0e-2,
    )
    torch.testing.assert_close(rope_cos_cache[start], rope_cos[0])
    torch.testing.assert_close(rope_sin_cache[start], rope_sin[0])


@pytest.mark.skipif(
    not _CUDA_BUILD_AVAILABLE,
    reason="mapped-host attention requires a CUDA build node",
)
def test_mapped_host_page32_v80_matches_exact_selected_page_attention() -> None:
    torch.manual_seed(20260902)
    device = torch.device("cuda")
    batch = 1
    kv_heads = 2
    query_heads = 8
    sequence = 70
    capacity = 96
    page_size = 32
    exact_key = torch.randn(
        batch,
        sequence,
        kv_heads,
        128,
        dtype=torch.bfloat16,
        device=device,
    ).transpose(1, 2)
    assert not exact_key.is_contiguous()
    value = torch.randn(
        batch,
        kv_heads,
        capacity,
        80,
        dtype=torch.bfloat16,
        device=device,
    )
    query = torch.randn(
        batch,
        query_heads,
        1,
        128,
        dtype=torch.bfloat16,
        device=device,
    )
    page_ids = torch.tensor(
        [[[0, 2], [0, 2]]],
        dtype=torch.int64,
        device=device,
    )

    host_key = mapped_host_bf16_empty(
        batch=batch,
        kv_heads=kv_heads,
        capacity=capacity,
    )
    append_mapped_host_key(host_key, exact_key[:, :, :65], start=0)
    append_mapped_host_key(host_key, exact_key[:, :, 65:], start=65)
    padded_key = torch.nn.functional.pad(
        exact_key,
        (0, 0, 0, capacity - sequence),
    )
    observed = mapped_host_paged_attention(
        host_key,
        query,
        value,
        page_ids,
        sequence_length=sequence,
        splits=2,
        host_key_device_pointer=mapped_host_device_pointer(host_key),
    )
    gpu_observed = gpu_paged_attention(
        padded_key,
        query,
        value,
        page_ids,
        sequence_length=sequence,
        splits=2,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(host_key[:, :, :sequence], exact_key.cpu())
    paged_key = padded_key.reshape(batch, kv_heads, -1, page_size, 128)
    paged_value = value.reshape(batch, kv_heads, -1, page_size, 80)
    gather_key = page_ids[..., None, None].expand(
        batch, kv_heads, 2, page_size, 128
    )
    gather_value = page_ids[..., None, None].expand(
        batch, kv_heads, 2, page_size, 80
    )
    selected_key = paged_key.gather(2, gather_key).reshape(
        batch, kv_heads, -1, 128
    )
    selected_value = paged_value.gather(2, gather_value).reshape(
        batch, kv_heads, -1, 80
    )
    positions = (
        page_ids[..., None] * page_size
        + torch.arange(page_size, device=device)
    ).reshape(batch, kv_heads, -1)
    valid = positions < sequence
    expanded_key = selected_key.repeat_interleave(4, dim=1).float()
    expanded_value = selected_value.repeat_interleave(4, dim=1).float()
    scores = torch.matmul(query.float(), expanded_key.transpose(-1, -2))
    scores.mul_(1.0 / math.sqrt(128))
    scores.masked_fill_(
        ~valid.repeat_interleave(4, dim=1)[:, :, None],
        -torch.inf,
    )
    expected = torch.matmul(scores.softmax(dim=-1), expanded_value).to(
        torch.bfloat16
    )

    torch.testing.assert_close(observed, expected, rtol=2.0e-2, atol=2.0e-2)

    gpu_observed = gpu_paged_attention(
        padded_key,
        query,
        value,
        page_ids,
        sequence_length=sequence,
        splits=2,
    )
    torch.testing.assert_close(gpu_observed, observed, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not _CUDA_BUILD_AVAILABLE, reason='CUDA extension required')
@pytest.mark.parametrize('value_dim,group,page', [(80,4,32),(96,1,16),(128,8,64),(37,2,32),(128,4,1)])
def test_general_paged_attention(value_dim, group, page):
    torch.manual_seed(123)
    batch, heads, length, capacity = 2, 2, 2*page+3, 3*page+11
    key = torch.randn(batch, heads, capacity, 128, device='cuda', dtype=torch.bfloat16)
    value = torch.randn(batch, heads, capacity, value_dim, device='cuda', dtype=torch.bfloat16)
    query = torch.randn(batch, heads*group, 1, 128, device='cuda', dtype=torch.bfloat16)
    ids = torch.tensor([0,2,-1],device='cuda').expand(batch,heads,-1).contiguous()
    host = mapped_host_bf16_empty(batch=batch,kv_heads=heads,capacity=capacity)
    append_mapped_host_key(host,key[:,:,:length],start=0)
    positions = torch.cat((torch.arange(page,device='cuda'),torch.arange(2*page,min(3*page,length),device='cuda')))
    q = query.float().reshape(batch,heads,group,128)
    scores = (q @ key[:,:,positions].float().transpose(-1,-2)) / math.sqrt(128)
    expected = (scores.softmax(-1) @ value[:,:,positions].float()).reshape(batch,heads*group,1,value_dim)
    for splits in (1,3):
        for function, source in [(gpu_paged_attention,key),(mapped_host_paged_attention,host)]:
            # Feature-major output exercises noncontiguous output strides.
            backing = torch.empty(heads*group*value_dim,batch,device='cuda',dtype=torch.bfloat16)
            output = backing.T.reshape(batch,heads*group,1,value_dim)
            actual = function(source,query,value,ids,sequence_length=length,page_size=page,splits=splits,output=output)
            torch.testing.assert_close(actual.float(),expected,atol=0.004,rtol=0.015)
            empty = function(source,query,value,torch.full_like(ids,-1),sequence_length=length,page_size=page,splits=splits)
            assert torch.count_nonzero(empty)==0
