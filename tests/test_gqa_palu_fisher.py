from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from evaluation.collect_gqa_palu_fisher import (
    _allocate_exact_official,
    _chunked_official_palu_loss_and_backward,
)


def test_v_only_fisher_uniform_schedule_has_expected_group_geometry() -> None:
    fisher = {
        f"model.layers.{layer}.self_attn.v_proj": float(layer + 1)
        for layer in range(8)
    }
    rank_map, rank_sum, total_rank = _allocate_exact_official(fisher)

    assert total_rank == 8 * 8 * 128
    assert rank_sum == sum(sum(ranks) for ranks in rank_map.values())
    assert all(len(ranks) == 8 for ranks in rank_map.values())
    assert all(len(set(ranks)) == 1 for ranks in rank_map.values())
    assert all(rank in {32, 64, 96, 128} for ranks in rank_map.values() for rank in ranks)


def test_v_only_fisher_rank64_target_uses_half_retained_budget() -> None:
    fisher = {
        f"model.layers.{layer}.self_attn.v_proj": float(layer + 1)
        for layer in range(64)
    }
    rank_map, rank_sum, total_rank = _allocate_exact_official(
        fisher,
        retained_ratio=0.5,
    )

    assert total_rank == 64 * 8 * 128
    assert abs(rank_sum / total_rank - 0.5) <= 32 / 128
    assert all(len(set(ranks)) == 1 for ranks in rank_map.values())
    assert all(rank in {32, 64, 96, 128} for ranks in rank_map.values() for rank in ranks)


def test_chunked_official_palu_loss_matches_full_double_shift_loss() -> None:
    class TinyBase(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(17, 8)

        def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
            del use_cache
            return SimpleNamespace(last_hidden_state=self.embed(input_ids))

    class TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = TinyBase()
            self.lm_head = nn.Linear(8, 17, bias=False)

    torch.manual_seed(7)
    model = TinyLM()
    model.lm_head.weight.requires_grad_(False)
    batch = torch.randint(0, 17, (2, 11))
    chunked = _chunked_official_palu_loss_and_backward(model, batch, chunk_size=3)
    chunked_grads = [
        parameter.grad.clone()
        for parameter in model.model.parameters()
    ]
    model.zero_grad(set_to_none=True)

    hidden = model.model(input_ids=batch[:, :-1]).last_hidden_state
    logits = model.lm_head(hidden).float()
    labels = nn.functional.pad(batch[:, 1:], (0, 1), value=-100)
    shifted_labels = labels[..., 1:].contiguous()
    reference = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), shifted_labels.reshape(-1)
    )
    reference.backward()

    torch.testing.assert_close(chunked, reference)
    for chunked_gradient, parameter in zip(chunked_grads, model.model.parameters()):
        torch.testing.assert_close(chunked_gradient, parameter.grad)
