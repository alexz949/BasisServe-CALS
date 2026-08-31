from __future__ import annotations

import torch
from torch import nn

from basisserve.core.pairwise_qr import unpack_upper_triangular
from evaluation.run_qwen3_32b_c1_layer23_tsqr_scaling import (
    _LayerTSQRCapture,
)


def test_layer_capture_streams_selected_rows_into_normalized_checkpoints() -> None:
    torch.manual_seed(20260827)
    module = nn.Linear(5, 2, bias=False, dtype=torch.float64)
    capture = _LayerTSQRCapture(
        module,
        input_width=5,
        fit_milestones=(10, 20),
        heldout_rows=5,
        accumulation_dtype=torch.float64,
    )
    selected_fit: list[torch.Tensor] = []
    try:
        positions = torch.tensor([[0, 3], [1, 4]])
        for _ in range(5):
            activation = torch.randn(2, 5, 5, dtype=torch.float64)
            selected_fit.append(
                torch.stack(
                    (activation[0, positions[0]], activation[1, positions[1]])
                ).reshape(-1, 5)
            )
            capture.begin("fit", positions)
            module(activation)
            capture.finish()
        capture.finish_split("fit")

        fit_rows = torch.cat(selected_fit)
        assert set(capture.fit_checkpoints) == {10, 20}
        for count in (10, 20):
            r = unpack_upper_triangular(
                capture.fit_checkpoints[count],
                dimension=5,
            )
            torch.testing.assert_close(
                r.T @ r,
                fit_rows[:count].T @ fit_rows[:count] / count,
                rtol=1e-11,
                atol=1e-11,
            )

        heldout = torch.randn(1, 5, 5, dtype=torch.float64)
        heldout_positions = torch.arange(5).unsqueeze(0)
        capture.begin("heldout", heldout_positions)
        module(heldout)
        capture.finish()
        capture.finish_split("heldout")
        assert capture.heldout_r is not None
        heldout_r = unpack_upper_triangular(capture.heldout_r, dimension=5)
        torch.testing.assert_close(
            heldout_r.T @ heldout_r,
            heldout[0].T @ heldout[0] / 5,
            rtol=1e-11,
            atol=1e-11,
        )
    finally:
        capture.close()
