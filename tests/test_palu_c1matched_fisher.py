from types import SimpleNamespace

import torch
from torch import nn

from evaluation.collect_gqa_palu_fisher import _chunked_official_palu_loss_and_backward
from evaluation.collect_palu_c1matched_fisher import fisher_scalar


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(19, 8)
        self.projection = nn.Linear(8, 8, bias=False)

    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.projection(self.embedding(input_ids)).tanh())


def test_chunked_loss_and_gradients_match_full_shifted_loss():
    torch.manual_seed(17)
    model = nn.Module()
    model.model = TinyBase()
    model.lm_head = nn.Linear(8, 19, bias=False)
    for p in model.parameters():
        p.requires_grad_(False)
    model.model.projection.weight.requires_grad_(True)
    batch = torch.randint(0, 19, (1, 23))
    hidden = model.model(input_ids=batch[:, :-1]).last_hidden_state
    expected = nn.functional.cross_entropy(model.lm_head(hidden[:, :-1]).reshape(-1, 19), batch[:, 2:].reshape(-1))
    expected.backward()
    expected_grad = model.model.projection.weight.grad.clone()
    for chunk in (1, 4, 32):
        model.zero_grad(set_to_none=True)
        actual = _chunked_official_palu_loss_and_backward(model, batch, chunk_size=chunk)
        torch.testing.assert_close(actual, expected.detach(), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(model.model.projection.weight.grad, expected_grad, rtol=2e-5, atol=1e-7)


def test_merge_squares_before_root_not_gradients_or_scalars():
    gradients = torch.tensor([[1., 9.], [-1., 1.], [3., -4.], [-3., 0.]])
    direct = gradients.square().mean(0).sqrt().mean()
    shards = [gradients[::2].square().sum(0), gradients[1::2].square().sum(0)]
    assert abs(fisher_scalar(sum(shards), 4) - float(direct)) < 1e-6
    assert not torch.isclose(direct, gradients.mean(0).abs().mean())
    assert abs(float(direct) - sum(fisher_scalar(s, 2) for s in shards) / 2) > 1e-3
