import json
from unittest.mock import Mock

import torch
import pytest

from basisserve.core.qwen3_8b_vllm_c1 import fold_value_weight
from basisserve.core import qwen3_8b_vllm_c1 as factors


@pytest.mark.parametrize("hidden_size", [4096, 5120])
def test_folded_value_projection_matches_explicit_encoder(hidden_size):
    generator = torch.Generator().manual_seed(102)
    dense = torch.randn(128, hidden_size, generator=generator) / 64
    encoder = torch.randn(128, 64, generator=generator) / 128**0.5
    hidden = torch.randn(3, hidden_size, generator=generator)
    observed = hidden @ fold_value_weight(dense, encoder).T
    expected = (hidden @ dense.T) @ encoder
    torch.testing.assert_close(observed, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("local_heads", [4, 8])
def test_tp8_coordinate_order_matches_sum_of_physical_sources(local_heads):
    generator = torch.Generator().manual_seed(123)
    coordinates = torch.randn(3, 8, local_heads, 64, generator=generator)
    decoders = torch.randn(8 * local_heads, 64, 17, generator=generator)
    arena = torch.cat([coordinates[:, rank].reshape(3, local_heads * 64).T.contiguous() for rank in range(8)])
    observed = arena.T @ decoders.reshape(8 * local_heads * 64, 17)
    expected = torch.einsum("bshr,shrn->bn", coordinates, decoders.reshape(8, local_heads, 64, 17))
    torch.testing.assert_close(observed, expected, rtol=2e-4, atol=5e-5)


def test_structure_validation_never_hashes(tmp_path, monkeypatch):
    manifest = {
        "format": "basisserve.qwen3_32b.gqa_c1_v96_joint.v1",
        "status": "complete", "layers": list(range(64)),
        "fit_config": {"hidden_size": 5120, "num_query_heads": 64,
                       "num_hidden_layers": 64, "cache_rank_per_head": 64,
                       "head_dim": 128, "num_physical_kv_heads": 8},
        "artifacts": {"0": {"file": "layer0.safetensors"}},
    }
    (tmp_path / "results.json").write_text(json.dumps(manifest))
    hasher = Mock()
    monkeypatch.setattr(factors, "file_sha256", hasher)
    monkeypatch.setattr(factors, "load_file", lambda *args, **kwargs: {
        "value_coordinate_encoders": torch.zeros(8, 128, 64),
        "head_output_decoders": torch.zeros(64, 64, 5120),
    })
    loaded = factors.load_manifest(tmp_path, None, validation="structure")
    encoder, decoder = factors.load_layer(
        tmp_path, loaded, 0, 7, device="cpu", dtype=torch.bfloat16,
        validation="structure")
    assert encoder.shape == (128, 64) and decoder.shape == (4096, 5120)
    hasher.assert_not_called()
    manifest["fit_config"]["cache_rank_per_head"] = 96
    (tmp_path / "results.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        factors.load_manifest(tmp_path, None, validation="structure")
    hasher.assert_not_called()
