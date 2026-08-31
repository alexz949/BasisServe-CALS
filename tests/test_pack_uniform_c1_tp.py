from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from basisserve.core.c1_tp_decode import C1TPFactorLoader, file_sha256
from evaluation.pack_qwen3_32b_uniform_c1_tp import (
    LEGACY_UNIFORM_C1_LAYER_FORMAT,
    pack_uniform_c1_checkpoint,
)


def _write_uniform_checkpoint(root: Path) -> tuple[Path, Path]:
    model_dir = root / "model"
    source_dir = root / "uniform"
    model_dir.mkdir()
    source_dir.mkdir()
    config_path = model_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "hidden_size": 6,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 3,
                "num_hidden_layers": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fit_config = {
        "model_config_sha256": file_sha256(config_path),
        "num_hidden_layers": 2,
        "num_physical_kv_heads": 2,
        "num_query_heads": 4,
        "head_dim": 3,
        "hidden_size": 6,
        "cache_rank_per_head": 2,
        "total_v_cache_rank": 4,
        "factor_dtype": "float32",
    }
    for layer in range(2):
        encoders = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        encoders = encoders + 100 * layer
        decoders = torch.arange(48, dtype=torch.float32).reshape(4, 2, 6)
        decoders = decoders + 1000 * layer
        tensor_path = source_dir / f"layer_{layer:03d}.safetensors"
        tensors = {
            "value_coordinate_encoders": encoders,
            "head_output_decoders": decoders,
        }
        save_file(tensors, str(tensor_path))
        payload = {
            "format": LEGACY_UNIFORM_C1_LAYER_FORMAT,
            "layer": layer,
            "fit_config": fit_config,
            "artifact": {
                "file": tensor_path.name,
                "sha256": file_sha256(tensor_path),
                "tensors": {
                    name: {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                    }
                    for name, tensor in tensors.items()
                },
            },
            "heldout": {"relative_mse": 0.01 * (layer + 1)},
            "selection": {"sweep": 5},
        }
        (source_dir / f"layer_{layer:03d}.json").write_text(
            json.dumps(payload) + "\n",
            encoding="utf-8",
        )
    return model_dir, source_dir


def test_pack_uniform_checkpoint_loads_through_canonical_tp_loader(
    tmp_path: Path,
) -> None:
    model_dir, source_dir = _write_uniform_checkpoint(tmp_path)
    output_dir = tmp_path / "packed"

    packed_result = pack_uniform_c1_checkpoint(
        source_dir,
        model=model_dir,
        output_dir=output_dir,
        uniform_rank=2,
        tp_size=2,
        command=("unit-test",),
    )

    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["selection"]["selected_schedule"] == [[2, 2], [2, 2]]
    assert result["selection"]["source_rank_sum"] == 8
    assert result["aggregate"]["v_retained_ratio"] == pytest.approx(2 / 3)
    assert result["aggregate"]["mean_heldout_relative_mse"] == pytest.approx(0.015)
    assert packed_result["result_sha256"] == file_sha256(output_dir / "result.json")

    loader = C1TPFactorLoader(
        output_dir,
        model_config=model_dir,
        tp_size=2,
        expected_result_sha256=packed_result["result_sha256"],
    )
    packed = loader.load_virtual_tp_layer(1)
    assert tuple(layer.source_ranks for layer in packed) == ((2, 2), (2, 2))
    assert packed[0].plan.source_widths == (4, 4)
    assert tuple(packed[0].local_encoder.shape) == (3, 2)
    assert tuple(packed[0].local_decoder.shape) == (4, 6)
    assert tuple(packed[0].global_decoder.shape) == (8, 6)


def test_pack_uniform_checkpoint_rejects_source_hash_mismatch(
    tmp_path: Path,
) -> None:
    model_dir, source_dir = _write_uniform_checkpoint(tmp_path)
    source_tensor = source_dir / "layer_001.safetensors"
    with source_tensor.open("ab") as handle:
        handle.write(b"corruption")
    output_dir = tmp_path / "packed"

    with pytest.raises(ValueError, match="source artifact hash mismatch"):
        pack_uniform_c1_checkpoint(
            source_dir,
            model=model_dir,
            output_dir=output_dir,
            uniform_rank=2,
            tp_size=2,
            command=("unit-test",),
        )

    assert not output_dir.exists()
    assert not (tmp_path / ".packed.packing").exists()
