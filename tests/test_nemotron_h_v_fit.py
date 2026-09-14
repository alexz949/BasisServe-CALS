import json

import pytest

from evaluation import fit_llama2_mha_c1_joint as fitter


def test_hybrid_v_fitting_covers_attention_layers_and_rejects_mamba(tmp_path, monkeypatch):
    names = ('FORMAT', 'LAYER_FORMAT', 'MODEL_LABEL', 'MODEL_TYPE', 'ATTENTION_TYPE',
        'NUM_LAYERS', 'NUM_HEADS', 'NUM_KV_HEADS', 'HEAD_DIM', 'HIDDEN_SIZE',
        'ATTENTION_LAYERS', 'FORMAL_ENCODER_SWEEPS')
    for name in names:
        monkeypatch.setattr(fitter, name, getattr(fitter, name))
    layers = (17, 38, 49, 60, 86)
    fitter.activate_nemotron_h_profile(layers)
    assert fitter._parse_layers('all') == layers
    assert fitter._parse_layers('17,86') == (17, 86)
    assert fitter.FORMAL_ENCODER_SWEEPS == 12
    with pytest.raises(AssertionError):
        fitter._parse_layers('0,17')
    manifest = dict(format=fitter.COVARIANCE_SNAPSHOT_FORMAT, layers=list(layers),
        model=dict(model_type='nemotron_h', attention_type='gqa', num_hidden_layers=98,
            num_attention_heads=64, num_key_value_heads=8, head_dim=128, hidden_size=8192))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    assert fitter._snapshot_manifest(tmp_path)['layers'] == list(layers)
    manifest['layers'].append(0)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        fitter._snapshot_manifest(tmp_path)
    fitter.activate_model_profile('qwen3_32b')
    assert fitter._parse_layers('all') == tuple(range(64))
    assert fitter.FORMAL_ENCODER_SWEEPS == 6
