"""Reconstruct Base K from resident C1 V and retain only residual codes."""
import torch
from basisserve.core.c1_v_conditional_k_router import _rotate_half


@torch.inference_mode()
def restore_routing_sidecar(value_codes, residual_codes, *, base_left,
                            base_right, base_bias, cos, sin):
    dtype, device = value_codes.dtype, value_codes.device
    # Match the original token-major projection layout after cache append.
    value_codes = value_codes.transpose(1, 2).contiguous().transpose(1, 2)
    predicted = torch.einsum('bhtv,hvr,hrd->bhtd', value_codes,
        base_left.to(device=device, dtype=dtype),
        base_right.to(device=device, dtype=dtype))
    predicted.add_(base_bias.to(device=device, dtype=dtype)[None, :, None, :])
    post = (predicted * cos.to(device=device, dtype=dtype).unsqueeze(1)
            + _rotate_half(predicted) * sin.to(device=device, dtype=dtype).unsqueeze(1))
    assert post.shape[:-1] == residual_codes.shape[:-1]
    return torch.cat((post, residual_codes), dim=-1).contiguous()
