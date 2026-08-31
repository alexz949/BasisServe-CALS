from pathlib import Path

import pytest
import torch

from evaluation.eval_qwen35_wo_compression_wikitext import (
    EXPECTED_FULL,
    EXPECTED_GDN,
    _private_rank_schedule,
    _result,
    _validate_private_factors,
)


def _factors(
    model_path: Path,
    layers: tuple[int, ...],
    *,
    rank: int = 192,
) -> dict:
    return {
        "model_path": str(model_path),
        "tp_size": 8,
        "layers": [
            {
                "layer_index": layer,
                "private_encoders": torch.empty(8, 512, rank, device="meta"),
            }
            for layer in layers
        ],
    }


def test_private_factor_validation_and_schedule(tmp_path: Path) -> None:
    gdn = _factors(tmp_path, EXPECTED_GDN)
    full = _factors(tmp_path, EXPECTED_FULL)
    _validate_private_factors(
        gdn,
        model_path=tmp_path,
        tp_size=8,
        expected_layers=EXPECTED_GDN,
    )
    schedule = _private_rank_schedule(gdn, full)
    assert schedule == {layer: 192 for layer in range(32)}


def test_private_factor_validation_rejects_layer_gap(tmp_path: Path) -> None:
    factors = _factors(tmp_path, EXPECTED_GDN[:-1])
    with pytest.raises(ValueError, match="coverage"):
        _validate_private_factors(
            factors,
            model_path=tmp_path,
            tp_size=8,
            expected_layers=EXPECTED_GDN,
        )


def test_result_reports_dense_relative_metrics() -> None:
    dense = {"nll_sum": 20.0, "tokens": 10, "ppl": 7.0}
    candidate = {"nll_sum": 21.0, "tokens": 10, "ppl": 8.0}
    result = _result(candidate, variant="candidate", dense=dense, metadata={})
    assert result["delta_mean_nll"] == pytest.approx(0.1)
    assert result["perplexity_ratio"] == pytest.approx(8.0 / 7.0)
