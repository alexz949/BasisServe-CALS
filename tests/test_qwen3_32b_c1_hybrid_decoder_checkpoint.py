from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file
import torch

from evaluation.build_qwen3_32b_c1_hybrid_decoder_checkpoint import (
    FACTOR_FORMAT,
    _sha256,
    compose_checkpoint,
)


def _fit_config() -> dict:
    return {
        "model_config_sha256": "model-config",
        "attention_type": "gqa",
        "num_hidden_layers": 2,
        "num_query_heads": 4,
        "num_physical_kv_heads": 2,
        "head_dim": 3,
        "hidden_size": 5,
        "cache_rank_per_head": 2,
        "factor_dtype": "bfloat16",
        "covariance_damping": 1e-5,
    }


def _write_checkpoint(
    path: Path,
    *,
    encoders: list[torch.Tensor],
    decoder_offset: float,
    primary: bool,
    fallback_results_sha256: str | None = None,
) -> dict:
    path.mkdir()
    artifacts = {}
    records = []
    for layer, encoder in enumerate(encoders):
        artifact_path = path / f"layer_{layer:03d}.safetensors"
        decoder = torch.full(
            (4, 2, 5),
            decoder_offset + layer,
            dtype=torch.bfloat16,
        )
        save_file(
            {
                "value_coordinate_encoders": encoder,
                "head_output_decoders": decoder,
            },
            artifact_path,
        )
        artifact = {
            "file": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "tensors": {
                "value_coordinate_encoders": {
                    "shape": [2, 3, 2],
                    "dtype": "torch.bfloat16",
                },
                "head_output_decoders": {
                    "shape": [4, 2, 5],
                    "dtype": "torch.bfloat16",
                },
            },
        }
        artifacts[str(layer)] = artifact
        heldout = 0.1 + 0.1 * layer if primary else 0.3 - 0.1 * layer
        record = {
            "layer": layer,
            "artifact": artifact,
            "fit_config": _fit_config(),
            "fit": {
                "relative_mse": heldout / 2,
                "factor_dtype_relative_mse": heldout / 2,
            },
            "heldout": {
                "relative_mse": heldout,
                "factor_dtype_relative_mse": heldout,
            },
        }
        if primary:
            record["decoder_refit"] = {
                "base_fit_factor_dtype_relative_mse": (0.3 - 0.1 * layer) / 2,
                "base_heldout_factor_dtype_relative_mse": 0.3 - 0.1 * layer,
            }
        records.append(record)
    result = {
        "format": FACTOR_FORMAT,
        "status": "complete",
        "layers": [0, 1],
        "fit_config": _fit_config(),
        "records": records,
        "artifacts": artifacts,
        "aggregate": {},
    }
    if primary:
        result["fit_config"]["decoder_solver"] = "pairwise_tsqr_qr0"
        result["decoder_refit_source"] = {
            "base_results_sha256": fallback_results_sha256
        }
    (path / "results.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def test_compose_checkpoint_selects_exact_decoder_sources(tmp_path: Path) -> None:
    encoders = [
        torch.arange(12, dtype=torch.float32).reshape(2, 3, 2).to(torch.bfloat16),
        torch.arange(12, 24, dtype=torch.float32)
        .reshape(2, 3, 2)
        .to(torch.bfloat16),
    ]
    fallback_dir = tmp_path / "fallback"
    _write_checkpoint(
        fallback_dir,
        encoders=encoders,
        decoder_offset=20.0,
        primary=False,
    )
    fallback_sha256 = _sha256(fallback_dir / "results.json")
    primary_dir = tmp_path / "primary"
    _write_checkpoint(
        primary_dir,
        encoders=encoders,
        decoder_offset=10.0,
        primary=True,
        fallback_results_sha256=fallback_sha256,
    )

    output_dir = tmp_path / "hybrid"
    markdown = tmp_path / "hybrid.md"
    result = compose_checkpoint(
        primary_factor_dir=primary_dir,
        fallback_factor_dir=fallback_dir,
        fallback_layers=[1],
        output_dir=output_dir,
        output_markdown=markdown,
        command="compose-test",
    )

    assert result["fit_config"]["decoder_solver"] == (
        "hybrid_tsqr_qr0_with_ridge_guardrail"
    )
    assert result["decoder_composition"]["fallback_layers"] == [1]
    assert [row["decoder_selection"]["source"] for row in result["records"]] == [
        "tsqr_qr0",
        "ridge_fallback",
    ]
    assert result["aggregate"]["mean_heldout_factor_dtype_relative_mse"] == (
        pytest.approx(0.15)
    )
    assert torch.equal(
        load_file(output_dir / "layer_000.safetensors")["head_output_decoders"],
        torch.full((4, 2, 5), 10.0, dtype=torch.bfloat16),
    )
    assert torch.equal(
        load_file(output_dir / "layer_001.safetensors")["head_output_decoders"],
        torch.full((4, 2, 5), 21.0, dtype=torch.bfloat16),
    )
    assert _sha256(output_dir / "layer_001.safetensors") == result["artifacts"][
        "1"
    ]["sha256"]
    assert markdown.is_file()


def test_compose_checkpoint_rejects_encoder_mismatch_before_writing(
    tmp_path: Path,
) -> None:
    primary_encoders = [
        torch.zeros((2, 3, 2), dtype=torch.bfloat16),
        torch.ones((2, 3, 2), dtype=torch.bfloat16),
    ]
    fallback_encoders = [tensor.clone() for tensor in primary_encoders]
    fallback_encoders[1][0, 0, 0] = 7
    fallback_dir = tmp_path / "fallback"
    _write_checkpoint(
        fallback_dir,
        encoders=fallback_encoders,
        decoder_offset=20.0,
        primary=False,
    )
    primary_dir = tmp_path / "primary"
    _write_checkpoint(
        primary_dir,
        encoders=primary_encoders,
        decoder_offset=10.0,
        primary=True,
        fallback_results_sha256=_sha256(fallback_dir / "results.json"),
    )
    output_dir = tmp_path / "hybrid"

    with pytest.raises(ValueError, match="layer 1 Value encoders differ"):
        compose_checkpoint(
            primary_factor_dir=primary_dir,
            fallback_factor_dir=fallback_dir,
            fallback_layers=[1],
            output_dir=output_dir,
            output_markdown=tmp_path / "hybrid.md",
            command="compose-test",
        )

    assert not output_dir.exists()
    assert not output_dir.with_name("hybrid.partial").exists()
