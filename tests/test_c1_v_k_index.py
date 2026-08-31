from __future__ import annotations

import torch

from basisserve.core.c1_v_k_index import (
    apply_rotary,
    fit_c1_v_to_pre_rope_k,
    invert_rotary,
    page_center_rotary_embeddings,
    project_c1_v_to_pre_rope_k,
)


def test_per_head_v_to_k_fit_recovers_exact_map() -> None:
    generator = torch.Generator().manual_seed(37)
    value = torch.randn(2, 3, 29, 5, generator=generator, dtype=torch.float64)
    weight = torch.randn(3, 5, 8, generator=generator, dtype=torch.float64)
    key = torch.einsum("bhsv,hvd->bhsd", value, weight)

    factors = fit_c1_v_to_pre_rope_k(value, key)
    predicted = project_c1_v_to_pre_rope_k(value, factors)

    torch.testing.assert_close(predicted.double(), key, rtol=1.0e-5, atol=1.0e-5)


def test_qwen3_half_split_rotary_round_trip() -> None:
    generator = torch.Generator().manual_seed(41)
    value = torch.randn(2, 3, 7, 8, generator=generator)
    angles = torch.randn(2, 7, 4, generator=generator)
    angles = torch.cat((angles, angles), dim=-1)
    cos = angles.cos()
    sin = angles.sin()

    rotated = apply_rotary(value, cos, sin)
    restored = invert_rotary(rotated, cos, sin)

    torch.testing.assert_close(restored, value, rtol=1.0e-5, atol=1.0e-5)


def test_page_center_rotary_uses_short_final_page_center() -> None:
    positions = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
    cos, sin = page_center_rotary_embeddings(
        positions,
        -positions,
        page_size=4,
    )

    assert cos.flatten().tolist() == [1.0] * 4 + [5.0] * 4 + [8.0] * 2
    assert sin.flatten().tolist() == [-1.0] * 4 + [-5.0] * 4 + [-8.0] * 2
