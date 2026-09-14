import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from evaluation.chunked_rope import chunked_rotary_inplace, chunked_rotary_preserve_keys


@torch.inference_mode()
def test_chunked_rope_exact_bf16_and_storage_reuse():
    torch.manual_seed(8)
    for length in (1, 1031):
        q = torch.randn(1, length, 8, 64, dtype=torch.bfloat16).transpose(1, 2)
        k = torch.randn(1, length, 2, 64, dtype=torch.bfloat16).transpose(1, 2)
        phase = torch.randn(1, length, 64)
        cos, sin = phase.cos().bfloat16(), phase.sin().bfloat16()
        expected_q, expected_k = apply_rotary_pos_emb(q, k, cos, sin)
        original_k = k.clone()
        shadow_q, shadow_k = chunked_rotary_preserve_keys(q.clone(), k, cos, sin)
        torch.testing.assert_close(k, original_k, rtol=0, atol=0)
        torch.testing.assert_close(shadow_q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(shadow_k, expected_k, rtol=0, atol=0)
        qptr, kptr = q.data_ptr(), k.data_ptr()
        actual_q, actual_k = chunked_rotary_inplace(q, k, cos, sin)
        assert actual_q.data_ptr() == qptr and actual_k.data_ptr() == kptr
        torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
