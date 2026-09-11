"""Shared immutable artifacts and model loading for the Qwen3.5 hybrid run."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_identity(path, identity):
    root = Path(path)
    assert sha256(root / 'config.json') == identity['config_sha256']
    expected = identity['model_files_sha256']
    assert {p.name for p in root.glob('*.safetensors')} == set(expected)
    for name, digest in expected.items():
        assert sha256(root / name) == digest


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f'Existing artifact: {path}'
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        if path.suffix == '.json':
            stream.write(json.dumps(value, indent=2, allow_nan=False).encode())
        else:
            torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def load_model(path, device):
    from transformers import Qwen3_5ForCausalLM
    model = Qwen3_5ForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    return model


def full_layers(model):
    return {i: layer.self_attn for i, layer in enumerate(model.model.layers) if hasattr(layer, 'self_attn')}


def load_windows(root, split):
    payload = torch.load(Path(root) / 'windows.pt', weights_only=True, map_location='cpu')
    return payload[split].long()


def load_bank(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert payload['format'] == 'basisserve.qwen35.gated_v_als.v1'
    from basisserve.core.qwen35_gated_v_runtime import factor_hash
    assert factor_hash(payload['layers']) == payload['factor_sha256']
    return payload
