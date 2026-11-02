import json

import pytest

from basisserve.compression import value as fitter
from basisserve.compression import ValueGeometry


def test_hybrid_v_fitting_covers_attention_layers_and_rejects_mamba(tmp_path):
    layers = (17, 38, 49, 60, 86)
    geometry = ValueGeometry("nemotron_h", 98, 64, 8, 128, 8192, layers)
    assert fitter._parse_layers("all", geometry=geometry) == layers
    assert fitter._parse_layers("17,86", geometry=geometry) == (17, 86)
    with pytest.raises(AssertionError):
        fitter._parse_layers("0,17", geometry=geometry)
    manifest = dict(
        format=fitter.COVARIANCE_SNAPSHOT_FORMAT,
        layers=list(layers),
        model=dict(
            model_type="nemotron_h",
            attention_type="gqa",
            num_hidden_layers=98,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
            hidden_size=8192,
        ),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert ValueGeometry.from_snapshot(tmp_path) == geometry
    assert fitter._snapshot_manifest(tmp_path, geometry=geometry)["layers"] == list(
        layers
    )
    manifest["layers"].append(0)
    path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        fitter._snapshot_manifest(tmp_path, geometry=geometry)
    other = ValueGeometry("qwen3", 64, 64, 8, 128, 5120, tuple(range(64)))
    assert fitter._parse_layers("all", geometry=other) == tuple(range(64))
    assert fitter._parse_layers("all", geometry=geometry) == layers
