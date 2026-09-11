import torch
from transformers import LlamaConfig, LlamaForCausalLM, DynamicCache
from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention
from evaluation.uniform96_common import llama_attention_interface


@torch.inference_mode()
def test_llama_identity_norm_interface_matches_native_causal_attention():
    torch.manual_seed(19)
    config = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=64)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).eval()
    tokens = torch.randint(0, 64, (1, 19))
    native_cache = DynamicCache(config=config)
    native = model(tokens, past_key_values=native_cache, use_cache=True).logits
    continuation = model(torch.tensor([[7]]), past_key_values=native_cache, use_cache=True).logits
    for layer in model.model.layers:
        original = layer.self_attn
        llama_attention_interface(original)
        layer.self_attn = GQATiedVOQwen3Attention(original,
            v_proj_compressed_weight=original.v_proj.weight,
            o_decoder_weight=original.o_proj.weight, attention_backend='sdpa')
    model.eval()
    cache = DynamicCache(config=config)
    actual = model(tokens, past_key_values=cache, use_cache=True).logits
    actual_continuation = model(torch.tensor([[7]]), past_key_values=cache, use_cache=True).logits
    torch.testing.assert_close(actual, native)
    torch.testing.assert_close(actual_continuation, continuation)
    for a, b in zip(cache.layers, native_cache.layers, strict=True):
        torch.testing.assert_close(a.keys, b.keys)
        torch.testing.assert_close(a.values, b.values)
