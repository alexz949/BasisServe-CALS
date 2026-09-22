"""Budget checks for Qwen3 STAR-KV V-only adaptive-rank training."""

from types import SimpleNamespace
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.train_qwen3_starkv_v_adaptive import (
    DecomposeLinear,
    FULL_V_RANK,
    NUM_LAYERS,
    SKIP_LAYERS,
    budget_report,
    project_adaptive_rank_budget,
    rank_profile,
)


def test_budget_scope():
    for target in (64, 128, 256):
        ranks = [FULL_V_RANK if layer in SKIP_LAYERS else target
                 for layer in range(NUM_LAYERS)]
        report = budget_report(ranks, target)
        assert report["exact_budget_match"]
        assert report["compressed_layer_rank_sum"] == 33 * target
        assert report["compressed_layer_mean_rank"] == target


def test_exact_projection_with_bfloat16_ties():
    torch.manual_seed(7)
    layers = []
    for layer in range(NUM_LAYERS):
        if layer in SKIP_LAYERS:
            projection = torch.nn.Linear(16, 8, bias=False)
        else:
            projection = DecomposeLinear(torch.nn.Linear(16, 8, bias=False)).to(torch.bfloat16)
            with torch.no_grad():
                projection.Sigma.diag.copy_(
                    torch.tensor([4.0, 3.0, 2.5, 2.5, 2.5, 2.0, 1.0, 0.5], dtype=torch.bfloat16)
                )
                projection.Sigma.soft_thres_layer.alpha.fill_(0.75 + layer / 100.0)
        layers.append(SimpleNamespace(self_attn=SimpleNamespace(v_proj=projection)))
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    projection = project_adaptive_rank_budget(model, target_mean_rank=4, min_rank=4)
    assert projection["exact"]
    assert projection["compressed_rank_sum"] == 33 * 4
    ranks = rank_profile(model)
    assert sum(rank for layer, rank in enumerate(ranks) if layer not in SKIP_LAYERS) == 132
    assert projection["bf16_cutoff_tie_breaks"]


if __name__ == "__main__":
    test_budget_scope()
    test_exact_projection_with_bfloat16_ties()
    print("STAR-KV adaptive-rank budget tests passed")
