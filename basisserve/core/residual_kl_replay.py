"""Read-only prefix forks and exact layer-suffix replay for routing KL probes."""

from __future__ import annotations

from contextlib import ExitStack
from copy import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from transformers.cache_utils import DynamicLayer

from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache


def fork_routing_prefix(prefix: RoutingDynamicCache) -> RoutingDynamicCache:
    """Share immutable prefix tensors; append operations replace fork tensors.

    This is deliberately restricted to non-sliding DynamicLayer caches, whose
    update uses concatenation, not in-place writes into the prefix allocation.
    """

    assert all(type(layer) is DynamicLayer for layer in prefix.layers)
    fork = copy(prefix)
    fork.layers = [copy(layer) for layer in prefix.layers]
    fork._routing_sidecars = list(prefix._routing_sidecars)
    return fork


def prefix_signature(prefix: RoutingDynamicCache) -> tuple:
    """Detect accidental mutation without copying or hashing the long cache."""

    rows = []
    for layer in prefix.layers:
        rows.append(tuple((x.data_ptr(), tuple(x.shape)) for x in (layer.keys, layer.values)))
    rows.append(tuple(None if x is None else (x.data_ptr(), tuple(x.shape))
                      for x in prefix._routing_sidecars))
    return tuple(rows)


@dataclass(frozen=True)
class CachedSuffixReplay:
    final_hidden: Tensor
    layer_inputs: tuple[Tensor, ...]
    layer_kwargs: tuple[dict[str, Any], ...]


@torch.inference_mode()
def capture_cached_suffix(
    model: nn.Module, input_ids: Tensor, cache: RoutingDynamicCache,
) -> CachedSuffixReplay:
    """Capture only suffix-token activations, not the long prefix activations."""

    layers = model.model.layers
    inputs: list[Tensor | None] = [None] * len(layers)
    contexts: list[dict[str, Any] | None] = [None] * len(layers)

    def make_hook(index: int):
        def hook(_module, args, kwargs):
            inputs[index] = args[0].detach()
            context = dict(kwargs)
            context.pop("past_key_values", None)
            contexts[index] = context
        return hook

    with ExitStack() as stack:
        for index, layer in enumerate(layers):
            handle = layer.register_forward_pre_hook(make_hook(index), with_kwargs=True)
            stack.callback(handle.remove)
        output = model.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    assert all(x is not None for x in inputs)
    assert all(x is not None for x in contexts)
    return CachedSuffixReplay(output.last_hidden_state.detach(), tuple(inputs), tuple(contexts))


@torch.inference_mode()
def replay_cached_suffix(
    model: nn.Module, anchor: CachedSuffixReplay, cache: RoutingDynamicCache,
    *, intervention_layer: int,
) -> Tensor:
    """Recompute the intervention and downstream layers on a fresh prefix fork."""

    assert 0 <= intervention_layer < len(anchor.layer_inputs)
    hidden = anchor.layer_inputs[intervention_layer]
    for index in range(intervention_layer, len(anchor.layer_inputs)):
        kwargs = dict(anchor.layer_kwargs[index], past_key_values=cache)
        hidden = model.model.layers[index](hidden, **kwargs)
    return model.model.norm(hidden)


def terminal_metrics(logits: Tensor, teacher_log_probability: Tensor, tokens: Tensor) -> dict:
    """Full-vocabulary teacher KL at every suffix position; next-token NLL at T-1."""

    log_probability = logits.float().log_softmax(dim=-1)
    assert log_probability.shape == teacher_log_probability.shape
    kl = (teacher_log_probability.exp() * (teacher_log_probability - log_probability)).sum(
        dim=-1, dtype=torch.float64,
    )
    nll = -log_probability[:, :-1].gather(-1, tokens[:, 1:, None]).squeeze(-1)
    assert bool(torch.isfinite(kl).all()) and bool(torch.isfinite(nll).all())
    assert float(kl.min()) > -1e-4
    return {
        "kl_sum": float(kl.sum()), "kl_positions": kl.numel(),
        "kl_mean": float(kl.mean()), "kl_by_position": kl.flatten().tolist(),
        "nll_sum": float(nll.double().sum()), "nll_positions": nll.numel(),
        "nll_mean": float(nll.double().mean()),
    }
