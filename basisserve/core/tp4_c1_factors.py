"""Published Qwen3-32B and Llama-3.1-8B factors partitioned by KV head."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from basisserve.core.qwen3_8b_vllm_c1 import file_sha256

HEAD_DIM = 128
NUM_KV_HEADS = 8
TP_SIZE = 4
GEOMETRIES = {
    "basisserve.qwen3_32b.gqa_c1_v96_joint.v1": (5120, 64, 8, 128, 64),
    "basisserve.llama31_8b_instruct.gqa_c1_joint.v1": (4096, 32, 8, 128, 32),
}


def load_manifest(root: Path, expected_sha256: str) -> dict:
    path = root / "results.json"
    assert file_sha256(path) == expected_sha256
    manifest = json.loads(path.read_text())
    assert manifest["format"] in GEOMETRIES
    assert manifest["status"] == "complete"
    geometry = GEOMETRIES[manifest["format"]]
    assert manifest["layers"] == list(range(geometry[-1]))
    fit = manifest["fit_config"]
    assert (fit["hidden_size"], fit["num_query_heads"], fit["num_physical_kv_heads"],
            fit["head_dim"], fit["num_hidden_layers"]) == geometry
    assert fit["cache_rank_per_head"] in (64, 96)
    return manifest


def load_layer(root, manifest, layer, rank, *, device, dtype):
    assert 0 <= rank < TP_SIZE
    record = manifest["artifacts"][str(layer)]
    path = root / record["file"]
    # A bare file name inside the factor directory; Hugging Face snapshots store it as a symlink to a blob.
    assert Path(record["file"]).name == record["file"]
    assert file_sha256(path) == record["sha256"]
    tensors = load_file(str(path), device="cpu")
    assert set(tensors) == {"value_coordinate_encoders", "head_output_decoders"}
    encoders, decoders = tensors["value_coordinate_encoders"], tensors["head_output_decoders"]
    width = manifest["fit_config"]["cache_rank_per_head"]
    assert encoders.shape == (8, 128, width)
    heads = manifest["fit_config"]["num_query_heads"]
    hidden = manifest["fit_config"]["hidden_size"]
    assert decoders.shape == (heads, width, hidden)
    # Rank r owns KV heads 2r, 2r+1 and their contiguous query-head groups.
    return (encoders[rank * 2:(rank + 1) * 2].to(device=device, dtype=dtype).contiguous(),
            decoders.reshape(heads * width, hidden).to(device=device, dtype=dtype).contiguous())


def fold_value_weight(dense_value, encoder):
    assert dense_value.shape[0] == 256 and dense_value.shape[1] in (4096, 5120)
    assert encoder.shape[:2] == (2, 128) and encoder.shape[2] in (64, 96)
    folded = torch.bmm(encoder.float().transpose(1, 2), dense_value.float().reshape(2, 128, -1))
    return folded.flatten(0, 1).to(dense_value.dtype)


def local_decoder(decoder, rank, local_width):
    """Rows correspond to the same source-rank order as prepared AllGather."""
    assert 0 <= rank < TP_SIZE and decoder.shape[0] == TP_SIZE * local_width
    return decoder.narrow(0, rank * local_width, local_width)
