"""Explicit host-mediated transfers for models dispatched across PCIe GPUs."""
import torch
from accelerate.utils.operations import recursively_apply, send_to_device


def host_staged_send_to_device(tensor, device, non_blocking=False, skip_keys=None):
    target = torch.device(f'cuda:{device}' if isinstance(device, int) else device)
    def stage(value):
        if value.is_cuda and target.type == 'cuda' and value.device != target:
            return value.cpu()
        return value
    # Preserve skipped top-level fields, especially model cache objects.
    if isinstance(tensor, dict) and skip_keys:
        skipped = {skip_keys} if isinstance(skip_keys, str) else set(skip_keys)
        staged = type(tensor)({key: value if key in skipped else recursively_apply(stage, value)
                              for key, value in tensor.items()})
    else:
        staged = recursively_apply(stage, tensor)
    return send_to_device(staged, device, non_blocking=non_blocking, skip_keys=skip_keys)


def install_host_staged_dispatch():
    import accelerate.hooks
    accelerate.hooks.send_to_device = host_staged_send_to_device
