import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

from evaluation.chunked_prefill_mlp import ChunkedTokenwise


@torch.inference_mode()
def test_chunked_qwen_mlp_matches_prefill_and_decode():
    torch.manual_seed(19)
    config = Qwen3Config(hidden_size=64, intermediate_size=128)
    inner = Qwen3MLP(config).to(dtype=torch.bfloat16).eval()
    chunked = ChunkedTokenwise(inner, chunk_size=16)
    for length in (1, 16, 51):
        inputs = torch.randn(1, length, 64, dtype=torch.bfloat16)
        expected = inner(inputs)
        actual = chunked(inputs)
        assert actual.dtype == expected.dtype and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)


@torch.inference_mode()
def test_chunked_norm_matches_hidden_and_query_shapes():
    norm = Qwen3RMSNorm(64).to(dtype=torch.bfloat16).eval()
    chunked = ChunkedTokenwise(norm, chunk_size=16)
    for shape in ((1, 1, 64), (1, 51, 64), (1, 51, 8, 64)):
        inputs = torch.randn(shape, dtype=torch.bfloat16)
        torch.testing.assert_close(chunked(inputs), norm(inputs), rtol=0, atol=0)


@torch.inference_mode()
def test_chunked_mamba_gated_norm_matches_native():
    from transformers.models.zamba2.modeling_zamba2 import Zamba2RMSNormGated
    norm = Zamba2RMSNormGated(128, 32).to(dtype=torch.bfloat16).eval()
    chunked = ChunkedTokenwise(norm, chunk_size=16)
    for length in (1, 51):
        inputs = torch.randn(1, length, 128, dtype=torch.bfloat16)
        gate = torch.randn_like(inputs)
        torch.testing.assert_close(chunked(inputs, gate), norm(inputs, gate), rtol=0, atol=0)
