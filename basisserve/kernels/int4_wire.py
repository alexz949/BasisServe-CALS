"""Reference packed INT4 latent communication (not KVQuant NUQ).

Symmetric round-to-nearest [-7, 7], one FP32 scale per row/source.
The scale bytes and packed nibbles share one uint8 NCCL packet. No
outlier side channel, cache quantization, or weight quantization is used.
"""
import torch
import torch.distributed as dist


def pack_int4(x):
    """Encode contiguous [tokens, local_features], including odd widths."""
    scale = (x.float().abs().amax(-1, keepdim=True) / 7).clamp_min(1e-30)
    codes = (x.float() / scale).round().clamp(-7, 7).to(torch.int16) + 8
    if x.shape[-1] % 2:
        codes = torch.nn.functional.pad(codes, (0, 1), value=8)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return torch.cat((scale.contiguous().view(torch.uint8), packed), dim=-1)


def unpack_int4(packet, width, dtype=torch.bfloat16):
    scale = packet[:, :4].contiguous().view(torch.float32)
    packed = packet[:, 4:]
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
    return ((codes[:, :width].float() - 8) * scale).to(dtype)


def all_gather_int4(local, group=None):
    """Return [tokens, source-major features]; only packed bytes cross ranks.

    All ranks must supply equal token counts/widths and finite inputs.
    Reference inference path; no autograd or CUDA-graph guarantee.
    """
    packet = pack_int4(local)
    world = dist.get_world_size(group)
    received = torch.empty((world * packet.shape[0], packet.shape[1]),
                           device=packet.device, dtype=torch.uint8)
    dist.all_gather_into_tensor(received, packet, group=group)
    decoded = unpack_int4(received, local.shape[1], local.dtype)
    return decoded.reshape(world, local.shape[0], local.shape[1]).permute(1, 0, 2).flatten(1)


def communication_bytes(tokens, local_width, world, hidden=4096):
    """Ring-model sent bytes per rank, not measured physical NIC traffic."""
    packet = tokens * (4 + (local_width + 1) // 2)
    dense = 2 * (world - 1) / world * tokens * hidden * 2
    wire = (world - 1) * packet
    return dict(local_packet_bytes=packet, scale_bytes=tokens * 4,
                dense_ring_sent_bytes=dense, int4_ring_sent_bytes=wire,
                effective_reduction=dense / wire,
                ideal_reduction=8 * hidden / (world * local_width))
