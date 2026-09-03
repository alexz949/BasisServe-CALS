"""Exact batch-local suffix replay for Qwen-style decoder models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class QwenAnchorReplay:
    """Anchor layer inputs and forward context retained for one input batch."""

    final_hidden: Tensor
    layer_inputs: tuple[Tensor, ...]
    layer_kwargs: tuple[Mapping[str, Any], ...]


def _qwen_layers(model: nn.Module) -> Sequence[nn.Module]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    assert backbone is not None
    assert layers is not None
    assert getattr(backbone, "norm", None) is not None
    return layers


@torch.inference_mode()
def capture_qwen_anchor_replay(
    model: nn.Module,
    *,
    input_ids: Tensor,
) -> QwenAnchorReplay:
    """Run the anchor once and retain every decoder-layer input and context."""

    layers = _qwen_layers(model)
    captured_inputs: list[Tensor | None] = [None] * len(layers)
    captured_kwargs: list[Mapping[str, Any] | None] = [None] * len(layers)
    handles = []

    def make_hook(index: int):
        def hook(_module, args, kwargs):
            assert args and isinstance(args[0], Tensor)
            captured_inputs[index] = args[0].detach()
            captured_kwargs[index] = dict(kwargs)

        return hook

    try:
        for index, layer in enumerate(layers):
            handles.append(
                layer.register_forward_pre_hook(make_hook(index), with_kwargs=True)
            )
        outputs = model.model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert not any(value is None for value in captured_inputs)
    assert not any(value is None for value in captured_kwargs)
    return QwenAnchorReplay(
        final_hidden=outputs.last_hidden_state.detach(),
        layer_inputs=tuple(captured_inputs),  # type: ignore[arg-type]
        layer_kwargs=tuple(captured_kwargs),  # type: ignore[arg-type]
    )


@torch.inference_mode()
def replay_qwen_suffix(
    model: nn.Module,
    anchor: QwenAnchorReplay,
    *,
    intervention_layer: int,
) -> Tensor:
    """Replay a modified layer and the unchanged anchor suffix exactly."""

    layers = _qwen_layers(model)
    assert len(anchor.layer_inputs) == len(layers)
    assert len(anchor.layer_kwargs) == len(layers)
    assert 0 <= intervention_layer < len(layers)
    hidden_states = anchor.layer_inputs[intervention_layer]
    for index in range(intervention_layer, len(layers)):
        hidden_states = layers[index](
            hidden_states,
            **anchor.layer_kwargs[index],
        )
    return model.model.norm(hidden_states)
