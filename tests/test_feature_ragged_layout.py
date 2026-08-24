from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


# Load by path so this CPU-only test never imports/builds the CUDA extension.
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "basisserve"
    / "kernels"
    / "feature_ragged_allgather.py"
)
spec = importlib.util.spec_from_file_location("feature_ragged_allgather_test_module", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

StaticRaggedPlan = module.StaticRaggedPlan
decode_feature_major = module.decode_feature_major
pure_torch_feature_major_reference = module.pure_torch_feature_major_reference


def test_feature_major_arena_is_exact_concatenation_transpose() -> None:
    torch.manual_seed(0)
    tokens = 7
    widths = (2, 5, 3, 4)
    parts = [torch.randn(tokens, width) for width in widths]

    arena = pure_torch_feature_major_reference(parts)
    dense = torch.cat(parts, dim=1)

    assert arena.shape == (sum(widths), tokens)
    torch.testing.assert_close(arena.transpose(0, 1), dense)


def test_one_big_gemm_matches_sourcewise_sum() -> None:
    torch.manual_seed(1)
    tokens = 11
    hidden = 13
    widths = (3, 1, 6, 2)
    parts = [torch.randn(tokens, width) for width in widths]
    decoder_parts = [torch.randn(width, hidden) for width in widths]
    decoder = torch.cat(decoder_parts, dim=0)

    arena = pure_torch_feature_major_reference(parts)
    actual = decode_feature_major(arena, decoder)
    expected = sum(part @ block for part, block in zip(parts, decoder_parts))

    torch.testing.assert_close(actual, expected)


def test_bias_path_matches_dense_addmm() -> None:
    torch.manual_seed(2)
    tokens = 5
    hidden = 9
    widths = (4, 2, 7)
    parts = [torch.randn(tokens, width) for width in widths]
    decoder = torch.randn(sum(widths), hidden)
    bias = torch.randn(hidden)

    arena = pure_torch_feature_major_reference(parts)
    actual = decode_feature_major(arena, decoder, bias)
    expected = torch.cat(parts, dim=1) @ decoder + bias

    torch.testing.assert_close(actual, expected)


def test_plan_offsets() -> None:
    plan = StaticRaggedPlan.from_source_widths((32, 64, 48, 96))
    assert plan.offsets == (0, 32, 96, 144)
    assert plan.total_width == 240
    assert plan.source_widths[2] == 48


def test_qwen3_tp8_head_ranks_expand_to_wire_widths() -> None:
    source_ranks = (32, 32, 48, 64, 64, 80, 96, 128)
    plan = StaticRaggedPlan.from_head_ranks(source_ranks, heads_per_source=8)

    assert plan.source_widths == (256, 256, 384, 512, 512, 640, 768, 1024)
    assert plan.total_width == 4352
