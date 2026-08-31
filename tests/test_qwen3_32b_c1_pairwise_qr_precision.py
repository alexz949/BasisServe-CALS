from __future__ import annotations

import pytest

from evaluation.check_qwen3_32b_c1_layer23_qr0_fp64 import (
    _reference_layer,
    _relative_change,
    _validate_sources,
)
from evaluation.compare_qwen3_32b_c1_pairwise_qr import EXPERIMENT_FORMAT


def _reference_payload() -> dict[str, object]:
    return {
        "format": EXPERIMENT_FORMAT,
        "status": "complete",
        "environment": {"work_dtype": "float32"},
        "layers": [{"layer": 23, "sentinel": True}, {"layer": 0}],
    }


def test_fp32_reference_selects_only_layer_23() -> None:
    assert _reference_layer(_reference_payload())["sentinel"] is True


def test_fp32_reference_rejects_wrong_precision() -> None:
    payload = _reference_payload()
    payload["environment"] = {"work_dtype": "float64"}
    with pytest.raises(ValueError, match="float32"):
        _reference_layer(payload)


def test_precision_source_validation_and_relative_change() -> None:
    source = {
        "snapshot_sha256": "snapshot",
        "factor_sha256": "factor",
        "fit_rows": 16,
        "heldout_row_start": 16,
        "heldout_rows": 8,
        "rank_per_head": 3,
    }
    _validate_sources(source, dict(source))
    changed = dict(source)
    changed["rank_per_head"] = 2
    with pytest.raises(ValueError, match="rank_per_head"):
        _validate_sources(source, changed)
    assert _relative_change(101.0, 100.0) == pytest.approx(0.01)
