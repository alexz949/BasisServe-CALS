"""Device-correct native Mamba launches for dispatched Nemotron models."""
from functools import wraps

import torch


def install_mamba_device_guards(model):
    layers = []
    for index, block in enumerate(model.model.layers):
        if block.block_type != 'linear_attention':
            continue
        mixer = block.mixer
        original = mixer.forward

        def guarded_forward(*args, _forward=original, _mixer=mixer, **kwargs):
            # Accelerate may move inputs inside the wrapped forward, so use
            # the module's actual weight device rather than its incoming input.
            device = _mixer.in_proj.weight.device
            assert device.type == 'cuda'
            with torch.cuda.device(device):
                return _forward(*args, **kwargs)

        mixer.forward = wraps(original)(guarded_forward)
        layers.append(index)
    assert layers
    return layers
