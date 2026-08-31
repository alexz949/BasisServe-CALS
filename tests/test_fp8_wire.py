from __future__ import annotations

import pytest
import torch

from basisserve.kernels.fp8_wire import (
    FP8_E4M3_DTYPE,
    FP8_E4M3_MAX,
    quantize_e4m3_static,
    quantize_e4m3_tensorwise_col_major,
    scaled_mm_e4m3_static,
)
from evaluation.calibrate_qwen3_32b_fp8_wire_from_snapshots import (
    project_process_latents,
)


def test_static_quantization_uses_dequantization_scale() -> None:
    value = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0])
    scale = torch.tensor(3.0 / FP8_E4M3_MAX)
    codes = quantize_e4m3_static(value, scale)
    reconstructed = codes.float() * scale

    assert codes.dtype == FP8_E4M3_DTYPE
    assert torch.isfinite(reconstructed).all()
    torch.testing.assert_close(reconstructed, value, rtol=0.08, atol=0.02)


def test_decoder_quantization_returns_column_major_matrix() -> None:
    decoder = torch.linspace(-3.0, 3.0, 16 * 32).view(16, 32)
    quantized, scale = quantize_e4m3_tensorwise_col_major(decoder)

    assert quantized.dtype == FP8_E4M3_DTYPE
    assert quantized.shape == decoder.shape
    assert quantized.stride(0) == 1
    assert quantized.stride(1) == decoder.shape[0]
    assert scale.dtype == torch.float32
    reconstructed = quantized.float() * scale
    torch.testing.assert_close(reconstructed, decoder, rtol=0.08, atol=0.02)


def test_scaled_mm_rejects_cpu_operands() -> None:
    left = torch.ones(2, 16).to(FP8_E4M3_DTYPE)
    right, right_scale = quantize_e4m3_tensorwise_col_major(torch.ones(16, 32))
    with pytest.raises(ValueError, match="CUDA"):
        scaled_mm_e4m3_static(
            left,
            right,
            left_scale=torch.ones((), dtype=torch.float32),
            right_scale=right_scale,
        )


def test_snapshot_projection_packs_sources_by_tp_process() -> None:
    activation = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [-1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0, 4.0],
        ]
    )
    encoders = torch.tensor(
        [
            [[1.0], [0.0]],
            [[0.0], [1.0]],
            [[1.0], [1.0]],
            [[1.0], [-1.0]],
        ]
    )

    local = project_process_latents(activation, encoders)

    assert len(local) == 4
    torch.testing.assert_close(local[0], torch.tensor([[1.0], [-1.0]]))
    torch.testing.assert_close(local[1], torch.tensor([[4.0], [2.0]]))
    torch.testing.assert_close(local[2], torch.tensor([[11.0], [0.0]]))
    torch.testing.assert_close(local[3], torch.tensor([[-1.0], [-8.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_scaled_mm_matches_dequantized_reference_on_cuda() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    left_value = torch.randn(64, 256, device=device)
    right_value = torch.randn(256, 512, device=device) / 8
    left_scale = (left_value.abs().amax() / FP8_E4M3_MAX).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    left = quantize_e4m3_static(left_value, left_scale)
    right, right_scale = quantize_e4m3_tensorwise_col_major(right_value)

    actual = scaled_mm_e4m3_static(
        left,
        right,
        left_scale=left_scale,
        right_scale=right_scale,
    )
    expected = (left.float() * left_scale) @ (right.float() * right_scale)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected, rtol=0.04, atol=0.12)
