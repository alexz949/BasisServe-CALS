from __future__ import annotations

from pathlib import Path

from safetensors.torch import load_file
import torch
from torch import nn

from basisserve.core.pairwise_qr import unpack_upper_triangular
from evaluation.run_qwen3_32b_c1_all_layer_tsqr_checkpoint import (
    _FinalLayerTSQRCapture,
    _updated_fit_config,
)


def test_final_layer_capture_writes_normalized_fit_and_heldout_r(
    tmp_path: Path,
) -> None:
    torch.manual_seed(20260828)
    module = nn.Linear(5, 2, bias=False, dtype=torch.float64)
    capture = _FinalLayerTSQRCapture(
        module,
        layer=3,
        output_dir=tmp_path,
        input_width=5,
        fit_rows=10,
        heldout_rows=5,
        accumulation_dtype=torch.float64,
    )
    expected: dict[str, list[torch.Tensor]] = {"fit": [], "heldout": []}
    try:
        positions = torch.arange(5).unsqueeze(0)
        for split, batches in (("fit", 2), ("heldout", 1)):
            for _ in range(batches):
                activation = torch.randn(1, 5, 5, dtype=torch.float64)
                expected[split].append(activation[0])
                capture.begin(split, positions)
                module(activation)
                capture.finish()
            capture.finish_split(split)
    finally:
        capture.close()

    for split in ("fit", "heldout"):
        record = capture.records[split]
        payload = load_file(str(tmp_path / record["file"]), device="cpu")
        r = unpack_upper_triangular(
            payload["r_upper_packed"],
            dimension=5,
        )
        rows = torch.cat(expected[split])
        torch.testing.assert_close(
            r.T @ r,
            rows.T @ rows / len(rows),
            rtol=1e-11,
            atol=1e-11,
        )


def test_updated_fit_config_records_qr0_refit_without_retraining_encoder(
    tmp_path: Path,
) -> None:
    base = {
        "covariance_damping": 1e-5,
        "cache_rank_per_head": 96,
        "encoder_sweeps": 5,
    }
    observed = _updated_fit_config(
        base,
        capture_dir=tmp_path / "capture",
        capture_manifest_sha256="abc123",
        model_path=tmp_path / "model",
    )
    assert observed["covariance_damping"] == 0.0
    assert observed["encoder_source_covariance_damping"] == 1e-5
    assert observed["decoder_solver"] == (
        "pairwise_tsqr_square_root_qr_lambda_0"
    )
    assert observed["fit_rows"] == 65_536
    assert observed["fit_windows"] == 256
    assert observed["validation_rows"] == 8_192
    assert observed["encoder_sweeps"] == 5
