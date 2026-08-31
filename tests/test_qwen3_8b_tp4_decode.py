from __future__ import annotations

from pathlib import Path

from safetensors.torch import save_file
import torch

from basisserve.core.qwen3_8b_tp4_decode import (
    HEAD_DIM,
    HIDDEN_SIZE,
    KV_HEADS_PER_PROCESS,
    NUM_KV_HEADS,
    NUM_QUERY_HEADS,
    fold_local_c1_value_projection,
    load_qwen3_tp4_c1_factor_layer,
)
from evaluation.benchmark_qwen3_8b_tp4_decode import _segments


def _write_factors(root: Path, *, rank: int, selected: bool) -> Path:
    destination = root / "selected_factors" if selected else root
    destination.mkdir(parents=True)
    path = destination / "layer_000.safetensors"
    save_file(
        {
            "value_coordinate_encoders": torch.randn(
                NUM_KV_HEADS,
                HEAD_DIM,
                rank,
                dtype=torch.bfloat16,
            ),
            "head_output_decoders": torch.randn(
                NUM_QUERY_HEADS,
                rank,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
            ),
            "source_ranks": torch.full((NUM_KV_HEADS,), rank, dtype=torch.int32),
        },
        str(path),
    )
    return path


def test_loads_uniform_and_selected_factor_layouts(tmp_path: Path) -> None:
    uniform = tmp_path / "uniform"
    selected = tmp_path / "selected"
    uniform_path = _write_factors(uniform, rank=3, selected=False)
    selected_path = _write_factors(selected, rank=5, selected=True)

    uniform_layer = load_qwen3_tp4_c1_factor_layer(uniform, 0)
    selected_layer = load_qwen3_tp4_c1_factor_layer(selected, 0)
    assert uniform_layer.path == uniform_path
    assert uniform_layer.source_rank == 3
    assert tuple(uniform_layer.encoders.shape) == (NUM_KV_HEADS, HEAD_DIM, 3)
    assert selected_layer.path == selected_path
    assert selected_layer.source_rank == 5
    assert tuple(selected_layer.decoders.shape) == (
        NUM_QUERY_HEADS,
        5,
        HIDDEN_SIZE,
    )


def test_rejects_within_layer_source_rank_raggedness(tmp_path: Path) -> None:
    root = tmp_path / "ragged"
    path = _write_factors(root, rank=4, selected=True)
    tensors = {
        "value_coordinate_encoders": torch.randn(
            NUM_KV_HEADS, HEAD_DIM, 4, dtype=torch.bfloat16
        ),
        "head_output_decoders": torch.randn(
            NUM_QUERY_HEADS, 4, HIDDEN_SIZE, dtype=torch.bfloat16
        ),
        "source_ranks": torch.tensor([4, 4, 4, 3, 4, 4, 4, 4], dtype=torch.int32),
    }
    save_file(tensors, str(path))
    try:
        load_qwen3_tp4_c1_factor_layer(root, 0)
    except ValueError as error:
        assert "one rank for all eight" in str(error)
    else:
        raise AssertionError("ragged within-layer source ranks were accepted")


def test_folds_both_local_value_heads_independently() -> None:
    generator = torch.Generator().manual_seed(23)
    rank = 7
    dense = torch.randn(
        KV_HEADS_PER_PROCESS * HEAD_DIM,
        HIDDEN_SIZE,
        generator=generator,
    )
    bias = torch.randn(KV_HEADS_PER_PROCESS * HEAD_DIM, generator=generator)
    encoders = torch.randn(
        KV_HEADS_PER_PROCESS,
        HEAD_DIM,
        rank,
        generator=generator,
    )
    folded, folded_bias = fold_local_c1_value_projection(dense, encoders, bias)
    expected = torch.stack(
        [
            encoders[source].T
            @ dense[source * HEAD_DIM : (source + 1) * HEAD_DIM]
            for source in range(KV_HEADS_PER_PROCESS)
        ]
    ).reshape(KV_HEADS_PER_PROCESS * rank, HIDDEN_SIZE)
    expected_bias = torch.stack(
        [
            encoders[source].T
            @ bias[source * HEAD_DIM : (source + 1) * HEAD_DIM]
            for source in range(KV_HEADS_PER_PROCESS)
        ]
    ).reshape(KV_HEADS_PER_PROCESS * rank)
    torch.testing.assert_close(folded, expected)
    assert folded_bias is not None
    torch.testing.assert_close(folded_bias, expected_bias)


def test_decode_segments_use_requested_token_boundaries() -> None:
    segments = _segments([float(index) for index in range(1, 2049)])
    assert [(row["decode_step_start"], row["decode_step_stop"]) for row in segments] == [
        (1, 128),
        (129, 256),
        (257, 1024),
        (1025, 2048),
    ]
    assert segments[-1]["context_length_stop"] == 2049
