from __future__ import annotations

import torch

from basisserve.core import (
    FixedTopKAllGatherOutput,
    fixed_topk_packet_bytes,
    pack_fixed_topk,
    unpack_fixed_topk,
)


def _topk_zero_fill(source: torch.Tensor, kept: int) -> torch.Tensor:
    indices = torch.topk(source.float().abs(), kept, dim=-1, sorted=False).indices
    expected = torch.zeros_like(source)
    expected.scatter_(-1, indices, source.gather(-1, indices))
    return expected


def test_fixed_topk_packet_sizes_match_qwen35_tp8_accounting() -> None:
    assert fixed_topk_packet_bytes(512, 256, torch.bfloat16) == 576
    assert fixed_topk_packet_bytes(512, 384, torch.bfloat16) == 832


def test_pack_round_trip_preserves_topk_with_non_byte_aligned_width() -> None:
    torch.manual_seed(9001)
    source = torch.randn(2, 3, 13)
    packet, indices, values = pack_fixed_topk(source, kept=5)
    reconstructed = unpack_fixed_topk(
        packet,
        width=13,
        kept=5,
        value_dtype=source.dtype,
    )

    torch.testing.assert_close(reconstructed, _topk_zero_fill(source, 5))
    assert packet.dtype == torch.uint8
    assert packet.shape == (2, 3, 22)
    assert torch.all(indices[..., 1:] >= indices[..., :-1])
    torch.testing.assert_close(values, source.gather(-1, indices))


def test_world_size_one_module_matches_topk_then_output_projection() -> None:
    torch.manual_seed(9002)
    weight = torch.randn(7, 13)
    hidden = torch.randn(5, 13)
    module = FixedTopKAllGatherOutput(weight, keep_ratio=5 / 13)

    output, gathered = module(hidden, return_gathered_input=True)
    expected_input = _topk_zero_fill(hidden, kept=5)
    expected_output = expected_input @ weight.transpose(0, 1)

    torch.testing.assert_close(gathered, expected_input)
    torch.testing.assert_close(output, expected_output)
    assert module.kept == 5
    assert module.raw_packet_bytes == 22
    assert module.packet_bytes == 24


def test_runtime_breakdown_reports_real_byte_payload() -> None:
    weight = torch.randn(16, 512, dtype=torch.bfloat16)
    hidden = torch.randn(3, 512, dtype=torch.bfloat16)
    module = FixedTopKAllGatherOutput(weight, keep_ratio=0.5)
    module.enable_runtime_breakdown()

    module(hidden)
    summary = module.runtime_breakdown_summary()

    assert summary["calls"] == 1
    assert summary["tokens"] == 3
    assert summary["dense_payload_bytes"] == 3 * 1024
    assert summary["compressed_payload_bytes"] == 3 * 576
    assert summary["packet_bytes_per_source_token"] == 576
    assert summary["collective"] == "single_value_dtype_carrier_all_gather"
