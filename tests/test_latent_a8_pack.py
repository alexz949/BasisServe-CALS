import pytest
import torch

from basisserve.kernels.fp8_wire import quantize_e4m3_static
from basisserve.kernels.latent_a8_pack import pack_latent_a8, unpack_latent_a8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows,width", [(1, 96), (4, 256), (16, 512), (128, 384), (257, 80)])
def test_fused_pack_exact(rows, width):
    torch.manual_seed(73)
    value = torch.randn(rows, width * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    scale = torch.tensor(0.00317, device="cuda", dtype=torch.float32)
    arena = torch.empty(width, rows, device="cuda", dtype=torch.uint8)
    with torch.inference_mode():
        expected = quantize_e4m3_static(value, scale)
        pack_latent_a8(value, scale, arena)
        assert torch.equal(arena, expected.view(torch.uint8).T)
        restored = torch.empty_like(value, memory_format=torch.contiguous_format)
        unpack_latent_a8(arena, scale, restored)
        torch.testing.assert_close(restored, (expected.float() * scale).to(torch.bfloat16), rtol=0, atol=0)
        codes = torch.empty(rows, width, device="cuda", dtype=torch.float8_e4m3fn)
        unpack_latent_a8(arena, scale, codes)
        assert torch.equal(codes.view(torch.uint8), expected.view(torch.uint8))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pack_latent_a8(value, scale, arena)
            unpack_latent_a8(arena, scale, codes)
        value.mul_(0.7)
        graph.replay()
        assert torch.equal(codes.view(torch.uint8), quantize_e4m3_static(value, scale).view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_direct_fp32_fp8_rounding():
    value = torch.tensor([[-0.193359375, 0.193359375]], device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.011369978077709675, device="cuda", dtype=torch.float32)
    arena = torch.empty(2, 1, device="cuda", dtype=torch.uint8)
    pack_latent_a8(value, scale, arena)
    assert torch.equal(arena.T, quantize_e4m3_static(value, scale).view(torch.uint8))
