"""GPU correctness independent of FA2, including replay with changing data."""

import pytest
import torch

from basisserve.kernels.diffkv_prefill import diffkv_prefill
from evaluation.benchmark_diffkv_prefill import make_case
from evaluation.summarize_vllm_qwen3_8b_c1 import kernel_category


def test_prefill_profile_classification():
    assert kernel_category("kernel_diffkv_prefill_sm89") == "attention"


def torch_reference(data, qs):
    answers = []
    offset = 0
    for i, qlen in enumerate(qs):
        length = int(data["seqused_k"][i])
        if qlen == 0:
            continue
        ids = data["block_table"][i, : (length + data["k"].shape[1] - 1) // data["k"].shape[1]].long()
        k = data["k"][ids].flatten(0, 1)[:length, 0].float()
        v = data["v"][ids].flatten(0, 1)[:length, 0].float()
        q = data["q"][offset:offset + qlen].transpose(0, 1).float()
        scores = (q @ k.T) * data["softmax_scale"]
        mask = torch.arange(length, device="cuda")[None, :] > (
            length - qlen + torch.arange(qlen, device="cuda")[:, None])
        scores.masked_fill_(mask, float("-inf"))
        answers.append((scores.softmax(-1) @ v).transpose(0, 1))
        offset += qlen
    return torch.cat(answers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("block_size", [16, 32])
def test_ragged_paged_prefill_and_graph(heads, block_size):
    torch.manual_seed(123)
    qs = [0, 1, 17, 65, 3, 0]
    data = make_case(qs, [16, 255, 31, 0, 1000, 16], heads, block_size)
    # Output is also allowed to have a larger physical token stride.
    storage = torch.full((sum(qs) + 3, heads, 80), float("nan"),
                         device="cuda", dtype=torch.bfloat16)
    data["out"] = storage[:sum(qs), :, :64]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            diffkv_prefill(**data)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        diffkv_prefill(**data)
    for _ in range(2):
        data["q"].normal_()
        data["v"].normal_()
        graph.replay()
        expected = torch_reference(data, qs)
        torch.testing.assert_close(data["out"].float(), expected, atol=0.008, rtol=0.02)
        assert torch.isnan(storage[sum(qs):]).all()
        assert torch.isnan(storage[:sum(qs), :, 64:]).all()
