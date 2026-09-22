"""Algebra and budget checks for Qwen3 STAR-KV V-only adaptive rank."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.train_qwen3_starkv_v50_adaptive import (
    DecomposeLinear,
    FULL_V_RANK,
    NUM_LAYERS,
    SKIP_LAYERS,
    TARGET_MEAN_V_RANK,
    TARGET_V_RANK_SUM,
    budget_report,
)
from model import _fuse_joint


def test_threshold_and_fusion():
    torch.manual_seed(7)
    linear = torch.nn.Linear(16, 8, bias=False)
    dec = DecomposeLinear(linear)
    x = torch.randn(3, 16)
    torch.testing.assert_close(dec(x), linear(x), atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        s = dec.Sigma.diag
        dec.Sigma.soft_thres_layer.alpha.copy_(((s[3] + s[4]) / 2).reshape(1))
    dec(x).square().mean().backward()
    alpha = dec.Sigma.soft_thres_layer.alpha
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all()
    fused = _fuse_joint(dec)
    assert fused.VS.out_features == 4
    torch.testing.assert_close(dec(x), fused(x), atol=1e-5, rtol=1e-5)


def test_v_cache_budget():
    assert TARGET_V_RANK_SUM == NUM_LAYERS * TARGET_MEAN_V_RANK == 18432
    exact = budget_report([TARGET_MEAN_V_RANK] * NUM_LAYERS)
    assert exact["exact_budget_match"]
    assert exact["v_cache_compression"] == 0.5
    adaptive = [FULL_V_RANK if i in SKIP_LAYERS else 465 for i in range(NUM_LAYERS)]
    report = budget_report(adaptive)
    assert report["within_budget"]
    assert report["mean_rank"] < TARGET_MEAN_V_RANK
    assert not budget_report([rank + 1 for rank in adaptive])["within_budget"]


if __name__ == "__main__":
    test_threshold_and_fusion()
    test_v_cache_budget()
    print("STAR-KV V50 adaptive algebra and budget tests passed")
