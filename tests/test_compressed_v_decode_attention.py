from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from basisserve.kernels.compressed_v_decode_attention import (
    CompressedVDecodeWorkspace,
    compressed_v_decode_attention_cuda,
    compressed_v_decode_attention_triton,
    compressed_v_prefill_attention,
    reference_compressed_v_decode_attention,
    reference_compressed_v_prefill_attention,
    select_compressed_v_decode_cuda_splits,
)


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


def test_triton_entrypoint_rejects_non_cuda_inputs() -> None:
    query = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 3, 8, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 3, 5, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires CUDA"):
        compressed_v_decode_attention_triton(query, key, value)


def test_cuda_entrypoint_rejects_non_cuda_inputs() -> None:
    query = torch.randn(1, 4, 1, 128, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 3, 128, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 3, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires CUDA"):
        compressed_v_decode_attention_cuda(query, key, value)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (8, 9),
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
        feature_major.view(8, value_dim, 3)
        .permute(2, 0, 1)
        .unsqueeze(2)
        .contiguous()
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
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (8, 9),
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
@pytest.mark.parametrize("value_dim", [48, 64, 80, 112])
def test_flex_prefill_matches_causal_reference(value_dim: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260825 + value_dim)
    query = torch.randn(
        2,
        8,
        129,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    key = torch.randn(
        2,
        2,
        129,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    value = torch.randn(
        2,
        2,
        129,
        value_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    observed = compressed_v_prefill_attention(query, key, value)
    expected = reference_compressed_v_prefill_attention(query, key, value)
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)
