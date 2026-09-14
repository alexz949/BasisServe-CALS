from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from basisserve.kernels.compressed_v_decode_attention import (
    CompressedVDecodeWorkspace,
    c1_dense_gqa_v96_decode_attention_cuda,
    c1_pack_exact_key_pages_cuda,
    c1_paged_sparse_decode_attention_cuda,
    compressed_v_decode_attention_cuda,
    compressed_v_decode_attention_triton,
    compressed_v_prefill_attention,
    reference_c1_paged_sparse_decode_attention,
    reference_compressed_v_decode_attention,
    reference_compressed_v_prefill_attention,
    select_compressed_v_decode_cuda_splits,
)


def _pack_key_pages(
    key: torch.Tensor,
    selected_page_ids: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    batch, kv_heads, _, head_dim = map(int, key.shape)
    page_slots = int(selected_page_ids.shape[-1])
    packed = torch.zeros(
        batch,
        kv_heads,
        page_slots,
        page_size,
        head_dim,
        dtype=key.dtype,
        device=key.device,
    )
    for batch_index in range(batch):
        for kv_head in range(kv_heads):
            for page_slot in range(page_slots):
                page_id = int(selected_page_ids[batch_index, kv_head, page_slot])
                if page_id < 0:
                    continue
                start = page_id * page_size
                stop = min(start + page_size, int(key.shape[2]))
                packed[
                    batch_index,
                    kv_head,
                    page_slot,
                    : stop - start,
                ] = key[batch_index, kv_head, start:stop]
    return packed


@pytest.mark.parametrize(
    ("capability", "batch", "sequence_length", "expected"),
    [
        ((8, 9), 1, 127, 1),
        ((8, 9), 1, 128, 16),
        ((8, 9), 8, 1024, 8),
        ((8, 9), 64, 4096, 8),
        ((8, 9), 128, 512, 1),
        ((8, 9), 128, 8192, 16),
    ],
)
def test_offline_cuda_split_selection(
    capability: tuple[int, int],
    batch: int,
    sequence_length: int,
    expected: int,
) -> None:
    assert (
        select_compressed_v_decode_cuda_splits(
            capability=capability,
            batch=batch,
            sequence_length=sequence_length,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("capability", "batch", "sequence_length", "exception"),
    [
        ((8, 9), 3, 512, ValueError),
        ((8, 0), 1, 128, RuntimeError),
        ((8, 9), 1, 8193, ValueError),
    ],
)
def test_offline_cuda_split_selection_rejects_untuned_shapes(
    capability: tuple[int, int],
    batch: int,
    sequence_length: int,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        select_compressed_v_decode_cuda_splits(
            capability=capability,
            batch=batch,
            sequence_length=sequence_length,
        )


def test_shared_decode_workspace_returns_exact_contiguous_views() -> None:
    arena = CompressedVDecodeWorkspace.allocate(
        batch=3,
        query_heads=8,
        max_value_dim=112,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    workspace, output = arena.views(
        batch=3,
        query_heads=8,
        splits=16,
        value_dim=64,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert workspace.shape == (24, 16, 66)
    assert output.shape == (3, 8, 1, 64)
    assert workspace.is_contiguous()
    assert output.is_contiguous()


@pytest.mark.parametrize("value_dim", [3, 4])
def test_reference_matches_math_sdpa_gqa(value_dim: int) -> None:
    generator = torch.Generator().manual_seed(20260824 + value_dim)
    query = torch.randn(2, 8, 1, 4, dtype=torch.float64, generator=generator)
    key = torch.randn(2, 2, 7, 4, dtype=torch.float64, generator=generator)
    value = torch.randn(
        2,
        2,
        7,
        value_dim,
        dtype=torch.float64,
        generator=generator,
    )

    expected = F.scaled_dot_product_attention(
        query,
        key,
        value,
        enable_gqa=True,
    )
    observed = reference_compressed_v_decode_attention(query, key, value)
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


def test_paged_sparse_reference_matches_dense_when_all_pages_are_selected() -> None:
    generator = torch.Generator().manual_seed(20260830)
    query = torch.randn(1, 4, 1, 4, dtype=torch.float64, generator=generator)
    key = torch.randn(1, 2, 7, 4, dtype=torch.float64, generator=generator)
    value = torch.randn(1, 2, 7, 3, dtype=torch.float64, generator=generator)
    selected_page_ids = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.int64)
    packed = _pack_key_pages(key, selected_page_ids, page_size=4)

    observed = reference_c1_paged_sparse_decode_attention(
        query,
        packed,
        value,
        selected_page_ids,
    )
    expected = reference_compressed_v_decode_attention(query, key, value)

    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


def test_triton_entrypoint_rejects_non_cuda_inputs() -> None:
    query = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 3, 8, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 3, 5, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires CUDA"):
        compressed_v_decode_attention_triton(query, key, value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_valid_length_pointer_ignores_static_cache_padding() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260830)
    valid_length = 193
    cache_length = 257
    query = torch.randn(
        1,
        32,
        1,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    key = torch.randn(
        1,
        8,
        cache_length,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    value = torch.randn(
        1,
        8,
        cache_length,
        96,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    device_valid_length = torch.tensor(
        valid_length,
        dtype=torch.int64,
        device="cuda",
    )

    observed = compressed_v_decode_attention_triton(
        query,
        key,
        value,
        valid_sequence_length=device_valid_length,
    )
    expected = compressed_v_decode_attention_triton(
        query,
        key[:, :, :valid_length],
        value[:, :, :valid_length],
    )

    assert torch.isfinite(observed).all()
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((8, 0), (8, 9), (9, 0)),
    reason="requires SM80, SM89, or SM90",
)
@pytest.mark.parametrize("splits", [1, 4, 16])
def test_paged_sparse_cuda_matches_packed_reference(splits: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260830 + splits)
    query = torch.randn(
        1, 32, 1, 128, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    key = torch.randn(
        1, 8, 257, 128, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    value = torch.randn(
        1, 8, 257, 96, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    selected_page_ids = (
        torch.tensor([[[0, 2, 4, -1]]], dtype=torch.int64, device="cuda")
        .expand(1, 8, 4)
        .contiguous()
    )
    packed = _pack_key_pages(key, selected_page_ids, page_size=64)

    observed = c1_paged_sparse_decode_attention_cuda(
        query,
        packed,
        value,
        selected_page_ids,
        splits=splits,
    )
    expected = reference_c1_paged_sparse_decode_attention(
        query,
        packed,
        value,
        selected_page_ids,
    )

    torch.testing.assert_close(observed, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((8, 0), (8, 9), (9, 0)),
    reason="requires SM80, SM89, or SM90",
)
def test_exact_key_page_pack_cuda_matches_reference_with_partial_page() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260901)
    key = torch.randn(
        2,
        3,
        130,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    selected_page_ids = torch.tensor(
        [
            [[0, 2, -1], [1, 2, -1], [2, 0, -1]],
            [[2, 1, -1], [0, 2, -1], [1, 0, -1]],
        ],
        dtype=torch.int64,
        device="cuda",
    ).contiguous()

    observed = c1_pack_exact_key_pages_cuda(key, selected_page_ids)
    expected = _pack_key_pages(key, selected_page_ids, page_size=64)

    assert observed.is_contiguous()
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_triton_launches_on_the_input_device() -> None:
    torch.cuda.set_device(0)
    target = torch.device("cuda", 1)
    generator = torch.Generator(device=target).manual_seed(20260830)
    query = torch.randn(
        1, 32, 1, 128, dtype=torch.bfloat16, device=target, generator=generator
    )
    key = torch.randn(
        1, 8, 193, 128, dtype=torch.bfloat16, device=target, generator=generator
    )
    value = torch.randn(
        1, 8, 193, 96, dtype=torch.bfloat16, device=target, generator=generator
    )

    observed = compressed_v_decode_attention_triton(query, key, value)
    expected = F.scaled_dot_product_attention(
        query,
        key,
        value,
        enable_gqa=True,
    )

    assert observed.device == target
    assert torch.cuda.current_device() == 0
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)


def test_cuda_entrypoint_rejects_non_cuda_inputs() -> None:
    query = torch.randn(1, 4, 1, 128, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 3, 128, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 3, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires CUDA"):
        compressed_v_decode_attention_cuda(query, key, value)


def test_dense_gqa_v96_cuda_entrypoint_rejects_non_cuda_inputs() -> None:
    query = torch.randn(1, 4, 1, 128, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 3, 128, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 3, 96, dtype=torch.bfloat16)
    valid_sequence_length = torch.tensor(3, dtype=torch.int64)
    with pytest.raises(ValueError, match="requires CUDA"):
        c1_dense_gqa_v96_decode_attention_cuda(
            query,
            key,
            value,
            valid_sequence_length,
            splits=1,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((8, 0), (8, 9), (9, 0)),
    reason="requires SM80, SM89, or SM90",
)
@pytest.mark.parametrize("splits", [1, 4, 16, 64])
def test_dense_gqa_v96_cuda_matches_valid_prefix_reference(splits: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260831 + splits)
    valid_length = 257
    cache_length = 269
    query = torch.randn(
        1, 32, 1, 128, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    key = torch.randn(
        1,
        8,
        cache_length,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    value = torch.randn(
        1,
        8,
        cache_length,
        96,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    valid_sequence_length = torch.tensor(
        valid_length,
        dtype=torch.int64,
        device="cuda",
    )

    observed = c1_dense_gqa_v96_decode_attention_cuda(
        query,
        key,
        value,
        valid_sequence_length,
        splits=splits,
    )
    expected = reference_compressed_v_decode_attention(
        query,
        key[:, :, :valid_length],
        value[:, :, :valid_length],
    )

    assert torch.isfinite(observed).all()
    torch.testing.assert_close(observed, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9),
    reason="requires SM89",
)
@pytest.mark.parametrize("value_dim", [32, 48, 64, 80, 96, 112])
@pytest.mark.parametrize("splits", [1, 4, 16])
def test_cuda_splitk_decode_matches_triton(
    value_dim: int,
    splits: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        20260825 + value_dim + splits
    )
    query = torch.randn(
        3,
        8,
        1,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    key_storage = torch.randn(
        3,
        2,
        269,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    value_storage = torch.randn(
        3,
        2,
        269,
        value_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    key = key_storage[:, :, :257]
    value = value_storage[:, :, :257]

    observed = compressed_v_decode_attention_cuda(
        query,
        key,
        value,
        splits=splits,
    )
    feature_major = torch.empty(
        8 * value_dim,
        3,
        dtype=query.dtype,
        device=query.device,
    )
    returned = compressed_v_decode_attention_cuda(
        query,
        key,
        value,
        splits=splits,
        feature_major_output=feature_major,
    )
    assert returned.data_ptr() == feature_major.data_ptr()
    feature_major_as_attention = (
        feature_major.view(8, value_dim, 3).permute(2, 0, 1).unsqueeze(2).contiguous()
    )
    expected = compressed_v_decode_attention_triton(query, key, value)
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        feature_major_as_attention,
        expected,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9),
    reason="requires SM89",
)
@pytest.mark.parametrize("sequence_length", [1, 2, 3])
def test_cuda_fused_single_split_handles_short_contexts(
    sequence_length: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260825 + sequence_length)
    query = torch.randn(
        2, 8, 1, 128, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    key = torch.randn(
        2,
        2,
        sequence_length,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    value = torch.randn(
        2,
        2,
        sequence_length,
        64,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    observed = compressed_v_decode_attention_cuda(query, key, value, splits=1)
    expected = compressed_v_decode_attention_triton(query, key, value)
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9),
    reason="requires SM89",
)
@pytest.mark.parametrize("splits", [1, 8])
def test_cuda_splitk_supports_qwen3_32b_gqa_ratio_eight(splits: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260826 + splits)
    query = torch.randn(
        2, 16, 1, 128, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    key = torch.randn(
        2, 2, 257, 128, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    value = torch.randn(
        2, 2, 257, 64, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    feature_major = torch.empty(16 * 64, 2, dtype=query.dtype, device=query.device)

    token_major = compressed_v_decode_attention_cuda(
        query,
        key,
        value,
        splits=splits,
    )
    returned = compressed_v_decode_attention_cuda(
        query,
        key,
        value,
        splits=splits,
        feature_major_output=feature_major,
    )
    expected = compressed_v_decode_attention_triton(query, key, value)
    observed = returned.view(16, 64, 2).permute(2, 0, 1).unsqueeze(2).contiguous()
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(token_major, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("value_dim", [3, 4])
def test_prefill_reference_matches_causal_math_sdpa(value_dim: int) -> None:
    generator = torch.Generator().manual_seed(20260825 + value_dim)
    query = torch.randn(2, 8, 7, 4, dtype=torch.float64, generator=generator)
    key = torch.randn(2, 2, 7, 4, dtype=torch.float64, generator=generator)
    value = torch.randn(
        2,
        2,
        7,
        value_dim,
        dtype=torch.float64,
        generator=generator,
    )

    expected = F.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        enable_gqa=True,
    )
    observed = reference_compressed_v_prefill_attention(query, key, value)
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


def test_prefill_entrypoint_rejects_non_cuda_inputs() -> None:
    query = torch.randn(1, 4, 3, 8, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 3, 8, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 3, 5, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires CUDA"):
        compressed_v_prefill_attention(query, key, value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("sequence_length", [5, 8, 129])
@pytest.mark.parametrize("value_dim", [48, 64, 80, 112])
def test_flex_prefill_matches_causal_reference(
    value_dim: int,
    sequence_length: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        20260825 + value_dim + sequence_length
    )
    query = torch.randn(
        2,
        8,
        sequence_length,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    key = torch.randn(
        2,
        2,
        sequence_length,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    value = torch.randn(
        2,
        2,
        sequence_length,
        value_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    observed = compressed_v_prefill_attention(query, key, value)
    expected = reference_compressed_v_prefill_attention(query, key, value)
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("value_dim", [32, 96, 128])
def test_prefill_uses_tensor_device_and_its_current_stream(value_dim):
    previous = torch.cuda.current_device()
    stream = torch.cuda.Stream(device=1)
    with torch.cuda.stream(stream):
        generator = torch.Generator(device='cuda:1').manual_seed(20260828)
        q = torch.randn(1, 8, 257, 128, device='cuda:1',
            dtype=torch.bfloat16, generator=generator)
        k = torch.randn(1, 2, 257, 128, device='cuda:1',
            dtype=torch.bfloat16, generator=generator)
        v = torch.randn(1, 2, 257, value_dim, device='cuda:1',
            dtype=torch.bfloat16, generator=generator)
        # The caller's device differs from its tensors. Both the launch and
        # the consumer must stay ordered on device 1's selected stream.
        with torch.cuda.device(0):
            actual = compressed_v_prefill_attention(q, k, v)
            assert torch.cuda.current_device() == 0
        consumed = actual.clone()
        expected = reference_compressed_v_prefill_attention(q, k, v)
    stream.synchronize()
    assert torch.cuda.current_device() == previous
    assert consumed.device == q.device
    torch.testing.assert_close(consumed, expected, rtol=2e-2, atol=2e-2)
