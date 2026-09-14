"""Check displaced C1 attention lifetime with real Accelerate dispatch hooks."""
import gc
import sys
import weakref
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from accelerate import dispatch_model
from transformers import Qwen3Config, Qwen3ForCausalLM
from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention


def replace(model):
    refs = []
    for layer in model.model.layers:
        old = layer.self_attn
        refs.append((weakref.ref(old), weakref.ref(old.v_proj.weight),
                     weakref.ref(old.o_proj.weight)))
        layer.self_attn = GQATiedVOQwen3Attention(old,
            v_proj_compressed_weight=torch.zeros(16, 64),
            o_decoder_weight=torch.zeros(64, 32), attention_backend='triton')
    return refs


config = Qwen3Config(hidden_size=64, intermediate_size=128,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, vocab_size=128)
model = dispatch_model(Qwen3ForCausalLM(config), device_map={'': 'cpu'},
                       force_hooks=True)
gc.collect()
gc.disable()
refs = replace(model)
before = [[ref() is not None for ref in row] for row in refs]
collected = gc.collect()
after = [[ref() is not None for ref in row] for row in refs]
gc.enable()
print('columns: old_attention, old_v_weight, old_o_weight', flush=True)
print('before_explicit_gc', before, flush=True)
print('after_explicit_gc', after, 'collected', collected, flush=True)
assert all(all(row) for row in before)
assert not any(any(row) for row in after)
