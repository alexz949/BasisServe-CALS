from __future__ import annotations

import hashlib
import json
from pathlib import Path

from safetensors.torch import save_file
import torch

from basisserve.core.qwen3_32b_tp4_decode import (
    FACTOR_FORMAT,
    HEAD_DIM,
    HIDDEN_SIZE,
    KV_HEADS_PER_PROCESS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_QUERY_HEADS,
    fold_ragged_local_c1_value_projection,
    load_qwen3_32b_tp4_c1_factor_layer,
)
from evaluation.benchmark_qwen3_32b_tp4_prefill import _SequenceChunkedMLP


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ragged_checkpoint(root: Path) -> tuple[Path, tuple[int, ...]]:
    root.mkdir()
    ranks = (2, 3, 4, 5, 6, 7, 8, 9)
    maximum_rank = max(ranks)
    path = root / "layer_000.safetensors"
    save_file(
        {
            "value_coordinate_encoders": torch.randn(
                NUM_KV_HEADS, HEAD_DIM, maximum_rank, dtype=torch.bfloat16
            ),
            "head_output_decoders": torch.randn(
                NUM_QUERY_HEADS, maximum_rank, HIDDEN_SIZE, dtype=torch.bfloat16
            ),
            "source_ranks": torch.tensor(ranks, dtype=torch.int32),
        },
        str(path),
    )
    schedule = [list(ranks)] + [[2] * NUM_KV_HEADS for _ in range(NUM_LAYERS - 1)]
    (root / "result.json").write_text(
        json.dumps(
            {
                "format": FACTOR_FORMAT,
                "status": "complete",
                "selection": {"selected_schedule": schedule},
                "artifacts": {
                    "0": {
                        "file": path.name,
                        "sha256": _sha256(path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path, ranks


def test_loads_and_preserves_source_raggedness(tmp_path: Path) -> None:
    path, ranks = _write_ragged_checkpoint(tmp_path / "checkpoint")
    layer = load_qwen3_32b_tp4_c1_factor_layer(path.parent, 0)
    assert layer.path == path
    assert layer.source_ranks == ranks
    assert layer.process_wire_widths == (
        8 * (2 + 3),
        8 * (4 + 5),
        8 * (6 + 7),
        8 * (8 + 9),
    )


def test_rejects_factor_hash_mismatch(tmp_path: Path) -> None:
    path, _ = _write_ragged_checkpoint(tmp_path / "checkpoint")
    manifest_path = path.parent / "result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["0"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        load_qwen3_32b_tp4_c1_factor_layer(path.parent, 0)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("corrupted factor hash was accepted")


def test_folds_two_different_local_source_ranks() -> None:
    generator = torch.Generator().manual_seed(17)
    dense = torch.randn(
        KV_HEADS_PER_PROCESS * HEAD_DIM,
        HIDDEN_SIZE,
        generator=generator,
    )
    bias = torch.randn(KV_HEADS_PER_PROCESS * HEAD_DIM, generator=generator)
    encoders = (
        torch.randn(HEAD_DIM, 3, generator=generator),
        torch.randn(HEAD_DIM, 5, generator=generator),
    )
    folded, folded_bias = fold_ragged_local_c1_value_projection(
        dense, encoders, bias
    )
    expected = torch.cat(
        (
            encoders[0].T @ dense[:HEAD_DIM],
            encoders[1].T @ dense[HEAD_DIM:],
        ),
        dim=0,
    )
    expected_bias = torch.cat(
        (
            encoders[0].T @ bias[:HEAD_DIM],
            encoders[1].T @ bias[HEAD_DIM:],
        )
    )
    assert tuple(folded.shape) == (8, HIDDEN_SIZE)
    torch.testing.assert_close(folded, expected)
    assert folded_bias is not None
    torch.testing.assert_close(folded_bias, expected_bias)


def test_sequence_chunked_mlp_matches_unchunked_tokenwise_module() -> None:
    class TokenwiseMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = torch.nn.Linear(7, 11, bias=False)
            self.up = torch.nn.Linear(7, 11, bias=False)
            self.down = torch.nn.Linear(11, 7, bias=False)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.down(torch.nn.functional.silu(self.gate(hidden_states)) * self.up(hidden_states))

    torch.manual_seed(29)
    inner = TokenwiseMLP()
    chunked = _SequenceChunkedMLP(inner, chunk_length=4)
    hidden_states = torch.randn(3, 10, 7)
    expected = inner(hidden_states)
    observed = chunked(hidden_states)
    torch.testing.assert_close(observed, expected, rtol=1.0e-5, atol=1.0e-6)
