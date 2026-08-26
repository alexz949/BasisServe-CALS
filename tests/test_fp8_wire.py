from __future__ import annotations

import torch

from basisserve.kernels.fp8_wire import (
    FP8_E4M3_MAX,
    decode_e4m3_bytes,
    quantize_e4m3_static,
)


def test_static_e4m3_wire_is_one_byte_and_round_trips_bits() -> None:
    value = torch.tensor([[0.0, 0.5, -2.0, 9.0]], dtype=torch.float32)
    scale = torch.tensor(9.0 / FP8_E4M3_MAX, dtype=torch.float32)
    quantized = quantize_e4m3_static(value, scale)
    wire = quantized.view(torch.uint8)

    assert quantized.element_size() == 1
    assert wire.element_size() == 1
    decoded = decode_e4m3_bytes(wire, dtype=torch.bfloat16)
    torch.testing.assert_close(
        decoded.float() * scale,
        value,
        rtol=0.08,
        atol=0.02,
    )


def test_source_scales_can_be_absorbed_into_bf16_decoder_rows() -> None:
    torch.manual_seed(11)
    tokens = 5
    source_width = 16
    hidden = 32
    source_scales = torch.tensor([0.1, 0.7, 2.0, 4.0])
    sources = [
        torch.randn(tokens, source_width) * source_scales[index]
        for index in range(len(source_scales))
    ]
    quantized = [
        quantize_e4m3_static(source, source_scales[index]).float()
        for index, source in enumerate(sources)
    ]
    decoder = torch.randn(len(source_scales) * source_width, hidden)
    decoder_blocks = decoder.view(len(source_scales), source_width, hidden)
    dequantized_reference = sum(
        code * source_scales[index] @ decoder_blocks[index]
        for index, code in enumerate(quantized)
    )
    scaled_decoder = decoder * source_scales.repeat_interleave(source_width)[:, None]
    bf16_decoder_input = torch.cat(quantized, dim=1) @ scaled_decoder

    torch.testing.assert_close(bf16_decoder_input, dequantized_reference)


def test_static_wire_rejects_invalid_cpu_scale() -> None:
    value = torch.ones(2, 4)
    for scale in (0.0, -1.0, float("inf"), float("nan")):
        try:
            quantize_e4m3_static(value, scale)
        except ValueError:
            continue
        raise AssertionError(f"invalid scale {scale} was accepted")
