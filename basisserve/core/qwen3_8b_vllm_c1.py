"""Validated uniform R64 factors for Qwen3-8B and Qwen3-32B TP8 serving."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file
import torch


TP_SIZE = 8
NUM_LAYERS = 36
HIDDEN_SIZE = 4096
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
VALUE_RANK = 64
LOCAL_WIDTH = 256
GLOBAL_WIDTH = 2048
FACTOR_FORMAT = "basisserve.qwen3_8b.gqa_c1_joint.v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path, expected_sha256: str | None, *, validation="sha256") -> dict:
    path = root / "results.json"
    assert validation in ("sha256", "structure")
    if validation == "sha256":
        assert file_sha256(path) == expected_sha256
    manifest = json.loads(path.read_text())
    fit = manifest["fit_config"]
    geometry = (fit["hidden_size"], fit["num_query_heads"], fit["num_hidden_layers"])
    assert geometry in ((4096, 32, 36), (5120, 64, 64))
    expected_format = FACTOR_FORMAT if geometry[0] == 4096 else "basisserve.qwen3_32b.gqa_c1_v96_joint.v1"
    assert manifest["format"] == expected_format and manifest["status"] == "complete"
    assert manifest["layers"] == list(range(geometry[2]))
    assert (fit["cache_rank_per_head"], fit["head_dim"], fit["hidden_size"],
            fit["num_query_heads"], fit["num_physical_kv_heads"]) == (64, 128, geometry[0], geometry[1], 8)
    return manifest


def load_layer(root: Path, manifest: dict, layer: int, rank: int, *, device, dtype,
               validation="sha256"):
    assert 0 <= rank < TP_SIZE
    record = manifest["artifacts"][str(layer)]
    path = root / record["file"]
    assert validation in ("sha256", "structure")
    if validation == "sha256":
        assert file_sha256(path) == record["sha256"]
    tensors = load_file(str(path), device="cpu")
    assert set(tensors) == {"value_coordinate_encoders", "head_output_decoders"}
    encoders = tensors["value_coordinate_encoders"]
    decoders = tensors["head_output_decoders"]
    hidden = manifest["fit_config"]["hidden_size"]
    heads = manifest["fit_config"]["num_query_heads"]
    assert tuple(encoders.shape) == (8, 128, 64)
    assert tuple(decoders.shape) == (heads, 64, hidden)
    # TP rank r owns KV head r and its contiguous group of query heads.
    # AllGather concatenation matches the checkpoint's query-head order.
    return (encoders[rank].to(device=device, dtype=dtype).contiguous(),
            decoders.reshape(heads * VALUE_RANK, hidden).to(device=device, dtype=dtype).contiguous())


def fold_value_weight(dense_value: torch.Tensor, encoder: torch.Tensor) -> torch.Tensor:
    assert dense_value.ndim == 2 and dense_value.shape[0] == HEAD_DIM
    assert dense_value.shape[1] in (4096, 5120)
    assert tuple(encoder.shape) == (HEAD_DIM, VALUE_RANK)
    return (encoder.float().T @ dense_value.float()).to(dense_value.dtype)
