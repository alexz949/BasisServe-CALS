from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


# Load by path so this CPU-only test never imports/builds the CUDA extension.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODULE_PATH = (
    REPO_ROOT
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
PreparedUniformAllGather = module.PreparedUniformAllGather
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


class _FakeUniformPlan:
    local_width = 6
    total_width = 24
    backend = "uniform_nccl"
    ipc_algorithm = "auto"
    ipc_channels = 0

    def __init__(self) -> None:
        self.local = torch.zeros(6, 3)
        self.arena = torch.zeros(24, 3)
        self.gather_calls = 0

    def local_view(self) -> torch.Tensor:
        return self.local

    def local_view_fast(self) -> torch.Tensor:
        return self.local

    def gather_inplace(self) -> torch.Tensor:
        self.gather_calls += 1
        self.arena[:6].copy_(self.local)
        return self.arena

    def gather_inplace_fast(self) -> torch.Tensor:
        return self.gather_inplace()

    def gather(self, local: torch.Tensor, local_is_feature_major: bool) -> torch.Tensor:
        assert local_is_feature_major
        self.local.copy_(local)
        return self.gather_inplace()


def test_prepared_uniform_python_wrapper_preserves_zero_copy_view() -> None:
    impl = _FakeUniformPlan()
    prepared = PreparedUniformAllGather(
        impl,
        tokens=3,
        dtype=torch.float32,
    )

    view = prepared.local_feature_major_view()
    assert view.data_ptr() == impl.local.data_ptr()
    assert prepared.local_feature_major_view_fast().data_ptr() == impl.local.data_ptr()
    view.fill_(7.0)
    arena = prepared.gather_inplace()

    assert prepared.local_width == 6
    assert prepared.total_width == 24
    assert prepared.backend == "uniform_nccl"
    assert prepared.ipc_algorithm == "auto"
    assert prepared.ipc_channels == 0
    assert impl.gather_calls == 1
    torch.testing.assert_close(arena[:6], torch.full((6, 3), 7.0))
    prepared.gather_inplace_fast()
    assert impl.gather_calls == 2


def test_prepared_uniform_python_wrapper_copy_path() -> None:
    impl = _FakeUniformPlan()
    prepared = PreparedUniformAllGather(
        impl,
        tokens=3,
        dtype=torch.float32,
    )
    local = torch.arange(18, dtype=torch.float32).view(6, 3)
    arena = prepared.gather(local, local_is_feature_major=True)
    torch.testing.assert_close(arena[:6], local)
