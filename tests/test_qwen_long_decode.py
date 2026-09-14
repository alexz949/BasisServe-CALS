"""Full-context GQA decode parity at Qwen3-32B evaluation geometry."""
import pytest
import torch
from basisserve.kernels.compressed_v_decode_attention import (
    compressed_v_decode_attention_triton,
)


@pytest.mark.parametrize('rank', [32, 96, 128])
def test_long_decode(rank):
    generator = torch.Generator(device='cuda').manual_seed(42)
    q = torch.randn(1, 64, 1, 128, device='cuda', dtype=torch.bfloat16,
                    generator=generator)
    k = torch.randn(1, 8, 65537, 128, device='cuda', dtype=torch.bfloat16,
                    generator=generator)
    v = torch.randn(1, 8, 65537, rank, device='cuda', dtype=torch.bfloat16,
                    generator=generator)
    expected = torch.empty(1, 64, 1, rank, device='cuda', dtype=torch.float32)
    for group in range(8):
        logits = q[:, group*8:(group+1)*8].float() @ k[:, group:group+1].float().transpose(-1, -2)
        expected[:, group*8:(group+1)*8] = (logits / 128**0.5).softmax(-1) @ v[:, group:group+1].float()
    observed = compressed_v_decode_attention_triton(q, k, v)
    relative = (observed.float() - expected).square().sum() / expected.square().sum()
    assert relative.sqrt() < 0.01
