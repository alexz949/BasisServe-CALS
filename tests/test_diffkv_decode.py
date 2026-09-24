"""GPU correctness for the fixed-QK SM89 decode path."""

import pytest
import torch

from basisserve.kernels.diffkv_decode import _launch_config, diffkv_decode
from evaluation.benchmark_diffkv_prefill import make_case
from evaluation.summarize_vllm_qwen3_8b_c1 import kernel_category


def test_decode_profile_classification():
    assert kernel_category("kernel_diffkv_decode_sm89") == "attention"
    assert kernel_category("kernel_diffkv_decode_reduce_sm89") == "attention"


def test_decode_launch_config():
    assert _launch_config(1, 512, 128, 4, 64) == (16, 64, 8, 2)
    assert _launch_config(1, 512, 128, 8, 64) == (16, 64, 4, 2)
    assert _launch_config(1, 4096, 128, 8, 96) == (16, 128, 4, 2)
    assert _launch_config(8, 4096, 128, 8, 64) == (16, 128, 4, 2)
    assert _launch_config(16, 4096, 128, 4, 96) == (8, 64, 4, 3)
    assert _launch_config(32, 4096, 128, 8, 96) == (16, 32, 4, 2)
    assert _launch_config(64, 4096, 128, 8, 64) == (8, 32, 4, 2)
    assert _launch_config(64, 4096, 128, 8, 96) == (1, 64, 4, 3)
    assert _launch_config(128, 4096, 128, 4, 96) == (1, 64, 4, 3)
    assert _launch_config(256, 4096, 128, 8, 64) == (1, 64, 4, 3)
    assert _launch_config(256, 4096, 128, 8, 96) == (1, 32, 4, 3)


def _buffers(tokens: int, heads: int, value_dim: int, segments: int = 16):
    padded = 1 << (value_dim - 1).bit_length()
    return {
        "softmax_segm_output": torch.empty(
            tokens, heads, segments, padded, device="cuda", dtype=torch.float32
        ),
        "softmax_segm_max": torch.empty(
            tokens, heads, segments, device="cuda", dtype=torch.float32
        ),
        "softmax_segm_expsum": torch.empty(
            tokens, heads, segments, device="cuda", dtype=torch.float32
        ),
    }


def _value_rank(data, value_dim):
    if value_dim != data["v"].shape[-1]:
        data["v"] = torch.randn(
            *data["v"].shape[:-1],
            value_dim,
            device="cuda",
            dtype=data["v"].dtype,
        )
        data["out"] = torch.empty(
            data["q"].shape[0], data["q"].shape[1], value_dim,
            device="cuda", dtype=data["q"].dtype
        )
    return data


def _reference(data, query_lengths):
    answers = []
    query_offset = 0
    page = data["k"].shape[1]
    for sequence, query_length in enumerate(query_lengths):
        if query_length == 0:
            continue
        length = int(data["seqused_k"][sequence])
        block_count = (length + page - 1) // page
        block_ids = data["block_table"][sequence, :block_count].long()
        keys = data["k"][block_ids].flatten(0, 1)[:length, 0].float()
        values = data["v"][block_ids].flatten(0, 1)[:length, 0].float()
        query = data["q"][query_offset].float()
        scores = (query @ keys.T) * data["softmax_scale"]
        answers.append(scores.softmax(-1) @ values)
        query_offset += 1
    return torch.stack(answers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("page", [16, 32])
@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("value_dim", [64, 96])
@pytest.mark.parametrize(
    "segments,tile", [(1, 32), (2, 32), (4, 32), (8, 32), (16, 16), (16, 32)]
)
def test_paged_decode_matches_reference(page, heads, value_dim, segments, tile):
    torch.manual_seed(20260921)
    query_lengths = [1, 0, 1, 1]
    data = _value_rank(
        make_case(query_lengths, [0, 16, 254, 999], heads=heads, block_size=page),
        value_dim,
    )
    storage = torch.full(
        (sum(query_lengths) + 2, heads, value_dim + 16),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    data["out"] = storage[: sum(query_lengths), :, :value_dim]
    expected = _reference(data, query_lengths)
    diffkv_decode(
        **data,
        **_buffers(sum(query_lengths), heads, value_dim),
        split_threshold=128,
        max_sequence_length=max(data["seqused_k"]).item(),
        segments=segments,
        tile=tile,
        num_warps=4,
        num_stages=2,
    )
    torch.testing.assert_close(data["out"].float(), expected, atol=0.008, rtol=0.02)
    assert torch.isnan(storage[sum(query_lengths) :]).all()
    assert torch.isnan(storage[: sum(query_lengths), :, value_dim:]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("segments", [1, 16])
@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("value_dim", [64, 96])
def test_decode_writes_feature_major_output(segments, heads, value_dim):
    torch.manual_seed(20260921)
    query_lengths = [1, 1, 1, 1]
    data = _value_rank(
        make_case(query_lengths, [31, 255, 1023, 4095], heads=heads, block_size=16),
        value_dim,
    )
    feature_major = torch.full(
        (heads * value_dim, len(query_lengths)),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    data["out"] = feature_major.T.view(len(query_lengths), heads, value_dim)
    expected = _reference(data, query_lengths)
    diffkv_decode(
        **data,
        **_buffers(len(query_lengths), heads, value_dim),
        split_threshold=128,
        max_sequence_length=4096,
        segments=segments,
        tile=32,
        num_warps=4,
        num_stages=2,
    )
    torch.testing.assert_close(data["out"].float(), expected, atol=0.008, rtol=0.02)
    torch.testing.assert_close(
        feature_major.float(), expected.flatten(1).T, atol=0.008, rtol=0.02
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads,value_dim", [(4, 64), (8, 96)])
def test_decode_cuda_graph_replay_with_changed_inputs(heads, value_dim):
    torch.manual_seed(20260921)
    query_lengths = [1, 1, 1, 1]
    data = _value_rank(
        make_case(query_lengths, [15, 255, 1023, 4095],
                  heads=heads, block_size=16),
        value_dim,
    )
    arguments = data | _buffers(sum(query_lengths), heads, value_dim)
    for _ in range(3):
        diffkv_decode(**arguments, split_threshold=128, max_sequence_length=4096)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        diffkv_decode(**arguments, split_threshold=128, max_sequence_length=4096)
    for _ in range(2):
        data["q"].normal_()
        data["v"].normal_()
        graph.replay()
        expected = _reference(data, query_lengths)
        torch.testing.assert_close(data["out"].float(), expected, atol=0.008, rtol=0.02)
