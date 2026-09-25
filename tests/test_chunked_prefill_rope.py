import pytest
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from benchmarks.system.chunked_prefill_rope import apply_prefill_rope_


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch,tokens", [(1, 1), (2, 2048), (2, 2051)])
@torch.inference_mode()
def test_chunked_rope_is_bitwise_equal_and_preserves_projection_layout(device, dtype, batch, tokens):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(31)
    q = torch.randn(batch, tokens, 32, 128, dtype=dtype, device=device).transpose(1, 2)
    k = torch.randn(batch, tokens, 8, 128, dtype=dtype, device=device).transpose(1, 2)
    angle = torch.randn(batch, tokens, 128, dtype=dtype, device=device)
    cos, sin = angle.cos(), angle.sin()
    expected_q, expected_k = apply_rotary_pos_emb(q, k, cos, sin)
    q_ptr, k_ptr, q_stride, k_stride = q.data_ptr(), k.data_ptr(), q.stride(), k.stride()
    observed_q, observed_k = apply_prefill_rope_(q, k, cos, sin)
    assert (observed_q.data_ptr(), observed_k.data_ptr()) == (q_ptr, k_ptr)
    assert (observed_q.stride(), observed_k.stride()) == (q_stride, k_stride)
    torch.testing.assert_close(observed_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(observed_k, expected_k, rtol=0, atol=0)
    bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
    assert torch.equal(observed_q.view(bits), expected_q.view(bits))
    assert torch.equal(observed_k.view(bits), expected_k.view(bits))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@torch.inference_mode()
def test_128k_rope_peak_allocation(tmp_path):
    import json

    torch.manual_seed(32)
    q = torch.randn(1, 131072, 32, 128, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    k = torch.randn(1, 131072, 8, 128, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    cos = torch.ones(1, 131072, 128, dtype=torch.bfloat16, device="cuda")
    sin = torch.full_like(cos, .5)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    expected_q, expected_k = apply_rotary_pos_emb(q, k, cos, sin)
    torch.cuda.synchronize()
    original_peak = torch.cuda.max_memory_allocated() - before
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    apply_prefill_rope_(q, k, cos, sin)
    torch.cuda.synchronize()
    chunked_peak = torch.cuda.max_memory_allocated() - before
    torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(k, expected_k, rtol=0, atol=0)
    assert torch.equal(q.view(torch.int16), expected_q.view(torch.int16))
    assert torch.equal(k.view(torch.int16), expected_k.view(torch.int16))
    assert chunked_peak < original_peak / 8
    report = dict(original_peak_extra_bytes=original_peak, chunked_peak_extra_bytes=chunked_peak,
                  bitwise_equal=True, context=131072, dtype="bfloat16")
    (tmp_path / "rope_memory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
