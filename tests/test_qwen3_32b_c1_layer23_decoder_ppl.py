from __future__ import annotations

import pytest
import torch

from evaluation.eval_qwen3_32b_c1_layer23_decoder_ppl import (
    _add_comparisons,
    decoder_to_padded_o_weight,
)


def test_decoder_to_padded_o_weight_preserves_head_blocks() -> None:
    decoder = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)
    observed = decoder_to_padded_o_weight(
        decoder,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected = torch.zeros(3, 8)
    expected[:, 0:2] = decoder[0].T
    expected[:, 4:6] = decoder[1].T
    torch.testing.assert_close(observed, expected)


@pytest.mark.parametrize(
    "shape,head_dim",
    [
        ((2, 0, 3), 4),
        ((2, 5, 3), 4),
        ((2, 2), 4),
    ],
)
def test_decoder_to_padded_o_weight_rejects_invalid_geometry(
    shape: tuple[int, ...],
    head_dim: int,
) -> None:
    with pytest.raises(ValueError):
        decoder_to_padded_o_weight(
            torch.empty(shape),
            head_dim=head_dim,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_add_comparisons_uses_dense_and_ridge_controls() -> None:
    arms = [
        {"name": "dense", "ppl": {"ppl": 8.0}},
        {"name": "c1_ridge_d1e5", "ppl": {"ppl": 10.0}},
        {"name": "c1_qr0_65536", "ppl": {"ppl": 9.0}},
    ]
    _add_comparisons(arms)
    qr = arms[-1]
    assert qr["ppl_delta_vs_dense"] == pytest.approx(1.0)
    assert qr["ppl_relative_change_vs_dense"] == pytest.approx(0.125)
    assert qr["ppl_delta_vs_ridge"] == pytest.approx(-1.0)
    assert qr["ppl_relative_change_vs_ridge"] == pytest.approx(-0.1)
