from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from basisserve.core.qwen_suffix_replay import (
    capture_qwen_anchor_replay,
    replay_qwen_suffix,
)
from basisserve.core.sampled_terminal_kl import (
    gather_prediction_hidden,
    nested_prediction_positions,
    paired_terminal_kl_delta,
    parse_position_counts,
    selected_token_nll,
    streaming_output_logsumexp,
    teacher_terminal_statistics,
    terminal_kl_from_statistics,
)


def test_nested_prediction_positions_are_deterministic_and_nested() -> None:
    counts = parse_position_counts("2,4,7")
    first = nested_prediction_positions(
        num_windows=3,
        sequence_length=11,
        position_counts=counts,
        seed=123,
    )
    second = nested_prediction_positions(
        num_windows=3,
        sequence_length=11,
        position_counts=counts,
        seed=123,
    )

    assert torch.equal(first, second)
    assert first.shape == (3, 7)
    assert all(len(set(row[:count].tolist())) == count for row in first for count in counts)
    assert int(first.min()) >= 0
    assert int(first.max()) <= 9


def test_gather_prediction_hidden_uses_independent_positions_per_window() -> None:
    hidden = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
    positions = torch.tensor([[0, 4], [3, 1]])

    selected = gather_prediction_hidden(hidden, positions)

    assert torch.equal(selected[0], hidden[0, [0, 4]])
    assert torch.equal(selected[1], hidden[1, [3, 1]])


def test_streaming_statistics_reproduce_dense_full_vocab_kl_and_delta() -> None:
    generator = torch.Generator().manual_seed(7)
    teacher_hidden = torch.randn(2, 5, 6, generator=generator)
    anchor_hidden = torch.randn(2, 5, 6, generator=generator)
    candidate_hidden = torch.randn(2, 5, 6, generator=generator)
    weight = torch.randn(13, 6, generator=generator)
    teacher_logits = F.linear(teacher_hidden, weight)
    anchor_logits = F.linear(anchor_hidden, weight)
    candidate_logits = F.linear(candidate_hidden, weight)
    teacher_probabilities = teacher_logits.softmax(dim=-1)
    dense_anchor_kl = (
        teacher_probabilities
        * (teacher_logits.log_softmax(dim=-1) - anchor_logits.log_softmax(dim=-1))
    ).sum(dim=-1)
    dense_candidate_kl = (
        teacher_probabilities
        * (
            teacher_logits.log_softmax(dim=-1)
            - candidate_logits.log_softmax(dim=-1)
        )
    ).sum(dim=-1)

    statistics = teacher_terminal_statistics(
        teacher_hidden,
        weight,
        vocab_chunk_size=4,
    )
    anchor_lse = streaming_output_logsumexp(
        anchor_hidden, weight, vocab_chunk_size=4
    )
    candidate_lse = streaming_output_logsumexp(
        candidate_hidden, weight, vocab_chunk_size=4
    )
    anchor_kl = terminal_kl_from_statistics(statistics, anchor_hidden, anchor_lse)
    delta = paired_terminal_kl_delta(
        statistics,
        anchor_hidden=anchor_hidden,
        candidate_hidden=candidate_hidden,
        anchor_logsumexp=anchor_lse,
        candidate_logsumexp=candidate_lse,
    )

    assert torch.allclose(statistics.logsumexp, teacher_logits.logsumexp(-1), atol=1e-6)
    assert torch.allclose(anchor_kl, dense_anchor_kl, atol=2e-6)
    assert torch.allclose(delta, dense_candidate_kl - dense_anchor_kl, atol=2e-6)


def test_selected_token_nll_reproduces_dense_cross_entropy() -> None:
    hidden = torch.randn(2, 4, 5)
    weight = torch.randn(11, 5)
    labels = torch.randint(0, 11, (2, 4))
    logits = F.linear(hidden, weight)
    lse = streaming_output_logsumexp(hidden, weight, vocab_chunk_size=3)

    actual = selected_token_nll(hidden, lse, weight, labels)
    expected = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)

    assert torch.allclose(actual, expected, atol=1e-6)


class _ToyLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)

    def forward(self, hidden_states, *, scale, **_kwargs):
        return hidden_states + scale * self.linear(hidden_states)


class _ToyBackbone(nn.Module):
    def __init__(self, depth: int, width: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(17, width)
        self.layers = nn.ModuleList([_ToyLayer(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)

    def forward(self, input_ids, use_cache=False):
        assert not use_cache
        hidden = self.embed_tokens(input_ids)
        scale = torch.tensor(0.25, device=hidden.device)
        for layer in self.layers:
            hidden = layer(hidden, scale=scale, use_cache=False)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class _ToyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ToyBackbone(depth=4, width=8)


@pytest.mark.parametrize("intervention_layer", range(4))
def test_suffix_replay_matches_full_forward_after_single_layer_change(
    intervention_layer: int,
) -> None:
    torch.manual_seed(11)
    model = _ToyLM().eval()
    input_ids = torch.randint(0, 17, (2, 6))
    anchor = capture_qwen_anchor_replay(model, input_ids=input_ids)
    layer = model.model.layers[intervention_layer]
    with torch.no_grad():
        layer.linear.weight.add_(0.01)

    replayed = replay_qwen_suffix(
        model,
        anchor,
        intervention_layer=intervention_layer,
    )
    expected = model.model(input_ids=input_ids, use_cache=False).last_hidden_state

    assert torch.equal(replayed, expected)
