import torch
from basisserve.kernels.int4_wire import pack_int4, unpack_int4, communication_bytes


def test_roundtrip():
    for width in (1, 7, 512, 768):
        torch.manual_seed(width)
        x = torch.randn(5, width)
        x[0].zero_()
        packet = pack_int4(x)
        decoded = unpack_int4(packet, width, torch.float32)
        bound = x.abs().amax(-1, keepdim=True) / 14 + 1e-6
        assert packet.dtype == torch.uint8
        assert packet.shape == (5, 4 + (width + 1) // 2)
        assert ((decoded - x).abs() <= bound).all()
        assert torch.equal(decoded[0], x[0])


def test_budget():
    for world in (4, 8):
        budget = communication_bytes(1, 32 * 64 // world, world)
        assert budget["ideal_reduction"] == 16
        assert 15 < budget["effective_reduction"] < 16
