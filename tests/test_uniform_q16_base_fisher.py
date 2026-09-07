import torch

from basisserve.core.c1_v_conditional_k_router import AffineReducedRankMap
from evaluation.fit_qwen3_8b_uniform_q16_base_fisher import pack, parser
from scripts.capture_qwen3_8b_q16 import UNIFORM_Q16_POSITIONS


def test_uniform_q16_joint_fit_cli_is_explicit():
    args = parser().parse_args([
        "--model", "/model", "--c1-checkpoint", "/c1",
        "--query-capture", "/uniform-q16",
    ])
    assert args.query_capture.name == "uniform-q16"
    assert args.initial_bank.name == "q8_qbase_fisher_bank"
    assert args.output_dir.name == "uniform32k_q16_base_fisher_r8"
    assert len(UNIFORM_Q16_POSITIONS) == 16


def test_joint_bank_packs_new_base_and_only_r8():
    torch.manual_seed(811)
    maps = tuple(AffineReducedRankMap(torch.randn(80, 16), torch.randn(16, 128), torch.randn(128))
                 for _ in range(8))
    residuals = {(16, 8): (torch.randn(8, 128, 8), torch.randn(32, 128, 8))}
    source = {
        "base_left_b16": torch.randn(8, 80, 16),
        "base_right_b16": torch.randn(8, 16, 128),
        "base_bias_b16": torch.randn(8, 128),
    }
    tensors, same = pack(maps, residuals, source)
    assert set(tensors) == set(source) | {"residual_encoder_b16_r8", "residual_query_b16_r8"}
    assert same == {name: False for name in source}
    assert tensors["base_left_b16"].shape == (8, 80, 16)
    assert tensors["residual_query_b16_r8"].shape == (32, 128, 8)
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in tensors.values())


def test_joint_bank_reports_unchanged_base_exactly():
    torch.manual_seed(812)
    maps = tuple(AffineReducedRankMap(torch.randn(80, 16), torch.randn(16, 128), torch.randn(128))
                 for _ in range(8))
    source = {
        "base_left_b16": torch.stack([item.left for item in maps]),
        "base_right_b16": torch.stack([item.right for item in maps]),
        "base_bias_b16": torch.stack([item.bias for item in maps]),
    }
    residuals = {(16, 8): (torch.randn(8, 128, 8), torch.randn(32, 128, 8))}
    _, same = pack(maps, residuals, source)
    assert all(same.values())
