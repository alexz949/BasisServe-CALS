"""Project head-major attention output without a full transposed copy."""
import torch


@torch.inference_mode()
def project_attention_output(attention, projection):
    batch, heads, length, width = attention.shape
    output = torch.empty((batch, length, projection.weight.shape[0]),
        device=attention.device, dtype=attention.dtype)
    for start in range(0, length, 1024):
        stop = min(start + 1024, length)
        block = attention[:, :, start:stop].transpose(1, 2).reshape(batch, stop-start, heads*width)
        output[:, start:stop].copy_(projection(block))
    return output
