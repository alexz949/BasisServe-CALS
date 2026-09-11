from types import MethodType
import torch
from transformers import LlamaConfig,LlamaForCausalLM,DynamicCache
from basisserve.checkpoint.c1_shadowkv_qwen3 import C1ShadowKVCache
from evaluation import eval_llama31_shadow as experiment


@torch.inference_mode()
def test_full_adapter_matches_native_llama_prefill_and_decode(monkeypatch):
    torch.manual_seed(17)
    config=LlamaConfig(hidden_size=64,intermediate_size=128,num_hidden_layers=2,
                       num_attention_heads=4,num_key_value_heads=2,head_dim=16,vocab_size=64)
    config._attn_implementation='sdpa'
    model=LlamaForCausalLM(config).eval()
    tokens=torch.randint(0,64,(1,19))
    native_cache=DynamicCache(config=config)
    native=model(tokens,past_key_values=native_cache,use_cache=True).logits
    next_ids=torch.tensor([[7]])
    continuation=model(next_ids,past_key_values=native_cache,use_cache=True).logits
    def prefill(q,k,v,scale):
        groups=q.shape[1]//k.shape[1]
        return torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(groups,1),
            v.repeat_interleave(groups,1),is_causal=True,scale=scale)
    monkeypatch.setattr(experiment,'compressed_v_prefill_attention',prefill)
    for layer in model.model.layers:
        layer.self_attn.shadow_enabled=False
        layer.self_attn.forward=MethodType(experiment.forward,layer.self_attn)
    cache=C1ShadowKVCache(config)
    actual=model(tokens,past_key_values=cache,use_cache=True).logits
    next_actual=model(next_ids,past_key_values=cache,use_cache=True).logits
    torch.testing.assert_close(actual,native)
    torch.testing.assert_close(next_actual,continuation)
    for a,b in zip(cache.layers,native_cache.layers,strict=True):
        torch.testing.assert_close(a.keys,b.keys)
        torch.testing.assert_close(a.values,b.values)
