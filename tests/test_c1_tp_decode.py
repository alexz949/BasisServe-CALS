from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from basisserve.core.c1_tp_decode import (
    C1TPFactorLoader,
    RAGGED_C1_FACTOR_FORMAT,
    file_sha256,
    headwise_c1_encode,
)
from basisserve.core.decoder_closed_rank_candidates import tensor_sha256
from evaluation.benchmark_qwen3_32b_c1_tp_decode import (
    _dense_local_decoder,
    _padded_decoder,
)
from evaluation.simulate_qwen3_32b_c1_tp_decode import _communication_bytes


def _write_checkpoint(
    root: Path,
    *,
    schedule: list[list[int]] | None = None,
    artifact_ranks: list[int] | None = None,
    artifact_sha256: str | None = None,
) -> tuple[Path, Path, torch.Tensor, torch.Tensor]:
    model_dir = root / "model"
    model_dir.mkdir()
    config = {
        "model_type": "qwen3",
        "hidden_size": 5,
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "head_dim": 4,
        "num_hidden_layers": 1,
    }
    config_path = model_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    factor_dir = root / "factors"
    factor_dir.mkdir()
    selected_schedule = [[2, 3, 1]] if schedule is None else schedule
    stored_ranks = [2, 3, 1] if artifact_ranks is None else artifact_ranks
    maximum_rank = max(stored_ranks)
    generator = torch.Generator().manual_seed(20260822)
    encoders = torch.randn(3, 4, maximum_rank, generator=generator)
    decoders = torch.randn(6, maximum_rank, 5, generator=generator)
    ranks = torch.tensor(stored_ranks, dtype=torch.int32)
    artifact_path = factor_dir / "layer_000.safetensors"
    save_file(
        {
            "value_coordinate_encoders": encoders,
            "head_output_decoders": decoders,
            "source_ranks": ranks,
        },
        artifact_path,
    )
    artifact = {
        "file": artifact_path.name,
        "sha256": file_sha256(artifact_path)
        if artifact_sha256 is None
        else artifact_sha256,
        "encoder_sha256": tensor_sha256(encoders),
        "decoder_sha256": tensor_sha256(decoders),
        "tensors": {
            "value_coordinate_encoders": {
                "shape": list(encoders.shape),
                "dtype": str(encoders.dtype),
            },
            "head_output_decoders": {
                "shape": list(decoders.shape),
                "dtype": str(decoders.dtype),
            },
            "source_ranks": {
                "shape": list(ranks.shape),
                "dtype": str(ranks.dtype),
            },
        },
    }
    source_rank_sum = sum(map(sum, selected_schedule))
    result = {
        "format": RAGGED_C1_FACTOR_FORMAT,
        "status": "complete",
        "layers": [0],
        "fit_config": {
            "model_config_sha256": file_sha256(config_path),
            "rank_schedule": selected_schedule,
            "source_rank_sum": source_rank_sum,
        },
        "selection": {
            "selected_schedule": selected_schedule,
            "source_rank_sum": source_rank_sum,
        },
        "artifacts": {"0": artifact},
    }
    (factor_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return factor_dir, model_dir, encoders, decoders


def test_headwise_encoder_is_one_flat_gemm(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = torch.Generator().manual_seed(20260825)
    local_attention = torch.randn(5, 8, 4, generator=generator)
    encoder = torch.randn(4, 3, generator=generator)
    expected = torch.matmul(local_attention, encoder).reshape(5, 24)
    original_mm = torch.mm
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def record_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        calls.append((tuple(left.shape), tuple(right.shape)))
        return original_mm(left, right)

    monkeypatch.setattr(torch, "mm", record_mm)
    observed = headwise_c1_encode(local_attention, encoder)

    assert calls == [((5 * 8, 4), (4, 3))]
    assert observed.shape == (5, 8 * 3)
    torch.testing.assert_close(observed, expected)


def test_factor_packer_preserves_tp_source_and_decoder_order(tmp_path: Path) -> None:
    factor_dir, model_dir, encoders, decoders = _write_checkpoint(tmp_path)
    loader = C1TPFactorLoader(factor_dir, model_config=model_dir, tp_size=3)
    packed = loader.load_virtual_tp_layer(0)

    assert loader.schedule == ((2, 3, 1),)
    assert len(loader.schedule_sha256) == 64
    assert len(loader.distributed_identity_sha256) == 64
    assert packed[0].plan.source_widths == (4, 6, 2)
    assert packed[0].plan.offsets == (0, 4, 10)
    assert all(
        layer.global_decoder.data_ptr() == packed[0].global_decoder.data_ptr()
        for layer in packed
    )
    for process_rank, layer in enumerate(packed):
        source_rank = (2, 3, 1)[process_rank]
        ownership = layer.ownership
        assert ownership.kv_head == process_rank
        assert ownership.query_head_start == 2 * process_rank
        assert ownership.query_head_stop == 2 * process_rank + 2
        torch.testing.assert_close(
            layer.local_encoder,
            encoders[process_rank, :, :source_rank],
        )
        expected_local_decoder = decoders[
            ownership.query_head_slice,
            :source_rank,
            :,
        ].reshape(2 * source_rank, 5)
        torch.testing.assert_close(layer.local_decoder, expected_local_decoder)

    expected_global_decoder = torch.cat(
        tuple(
            decoders[2 * source : 2 * source + 2, :source_rank, :].reshape(
                2 * source_rank,
                5,
            )
            for source, source_rank in enumerate((2, 3, 1))
        ),
        dim=0,
    )
    torch.testing.assert_close(packed[2].global_decoder, expected_global_decoder)

    generator = torch.Generator().manual_seed(20260823)
    attention = tuple(
        torch.randn(7, 2, 4, generator=generator) for _ in range(3)
    )
    local_latents = tuple(
        layer.encode_local_attention(value)
        for layer, value in zip(packed, attention)
    )
    compact_latent = torch.cat(local_latents, dim=1)
    observed = compact_latent @ packed[0].global_decoder
    expected = sum(
        (
            layer.decode_local_attention(value)
            for layer, value in zip(packed, attention)
        ),
        torch.zeros(7, 5),
    )
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)


def test_dense_and_padded_benchmark_baselines_preserve_c1_function(
    tmp_path: Path,
) -> None:
    factor_dir, model_dir, _, _ = _write_checkpoint(tmp_path)
    loader = C1TPFactorLoader(factor_dir, model_config=model_dir, tp_size=3)
    packed = loader.load_virtual_tp_layer(0)
    generator = torch.Generator().manual_seed(20260824)
    attention = torch.randn(9, 2, 4, generator=generator)
    local = packed[1]
    dense_decoder = _dense_local_decoder(local)
    torch.testing.assert_close(
        attention.flatten(start_dim=1) @ dense_decoder,
        local.decode_local_attention(attention),
        rtol=1e-5,
        atol=1e-5,
    )

    compact_latent = torch.randn(
        9,
        packed[0].plan.total_width,
        generator=generator,
    )
    maximum_width = max(packed[0].plan.source_widths)
    padded_blocks = []
    for source, width in enumerate(packed[0].plan.source_widths):
        block = torch.zeros(9, maximum_width)
        start = packed[0].plan.offsets[source]
        block[:, :width].copy_(compact_latent[:, start : start + width])
        padded_blocks.append(block)
    padded_latent = torch.cat(padded_blocks, dim=1)
    padded_decoder = _padded_decoder(
        packed[0].global_decoder,
        packed[0].plan,
    )
    torch.testing.assert_close(
        padded_latent @ padded_decoder,
        compact_latent @ packed[0].global_decoder,
        rtol=1e-5,
        atol=1e-5,
    )

    communication = _communication_bytes(packed, batch=2)
    assert communication["compact_ragged_allgather"][
        "total_wire_bytes_all_ranks"
    ] == 192
    assert communication["padded_allgather"][
        "total_wire_bytes_all_ranks"
    ] == 288
    assert communication["local_c1_allreduce"][
        "total_wire_bytes_all_ranks"
    ] == 160


def test_factor_loader_pins_result_hash(tmp_path: Path) -> None:
    factor_dir, model_dir, _, _ = _write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="result hash mismatch"):
        C1TPFactorLoader(
            factor_dir,
            model_config=model_dir,
            tp_size=3,
            expected_result_sha256="0" * 64,
        )


def test_factor_loader_rejects_artifact_hash_mismatch(tmp_path: Path) -> None:
    factor_dir, model_dir, _, _ = _write_checkpoint(
        tmp_path,
        artifact_sha256="0" * 64,
    )
    loader = C1TPFactorLoader(factor_dir, model_config=model_dir, tp_size=3)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        loader.load_layer(0, process_rank=0)


def test_factor_loader_rejects_artifact_schedule_mismatch(tmp_path: Path) -> None:
    factor_dir, model_dir, _, _ = _write_checkpoint(
        tmp_path,
        artifact_ranks=[1, 3, 2],
    )
    loader = C1TPFactorLoader(factor_dir, model_config=model_dir, tp_size=3)
    with pytest.raises(ValueError, match="rank mismatch"):
        loader.load_layer(0, process_rank=0)


def test_factor_loader_rejects_non_ownership_tp_size(tmp_path: Path) -> None:
    factor_dir, model_dir, _, _ = _write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="one physical KV head per rank"):
        C1TPFactorLoader(factor_dir, model_config=model_dir, tp_size=2)
