"""Call upstream LRQK cache unchanged from the common evaluation attention hook."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'external/LRQK'))
sys.path.insert(0, '/home/zhangal/.cache/torch_extensions/lrqk_gather')
import lrqk_attention as upstream
from flash_attn import flash_attn_func


class OfficialLRQKState:
    @torch.inference_mode()
    def __init__(self, q, k, config, layer=0):
        self.config = config
        self.length = k.shape[2]
        self.steps = 0
        self.cache = upstream.LightAttentionIndicesFactory(
            num_lite_tokens=64, attn_topk=2048,
            num_key_value_groups=q.shape[1]//k.shape[1], r=32,
            max_iter=(2,2), tol=(0.01,0.01), capacity=self.length+1024,
            init_aq_ak_method=upstream.InitAQAK.randn)
        self.prefill_q, self.prefill_k = q, k

    def bind_values(self, v):
        # The common hook constructs routing state before computing prefill.
        # Upstream needs original V at construction of its CPU/GPU cache.
        self.cache.prefill(self.prefill_q, self.prefill_k, v)
        del self.prefill_q, self.prefill_k

    @torch.inference_mode()
    def decode(self, q, k, v, scale):
        selected_k, selected_v = self.cache.decode(q, k[:,:,-1:], v[:,:,-1:])
        self.length += 1
        self.steps += 1
        return flash_attn_func(q.transpose(1,2), selected_k.transpose(1,2),
            selected_v.transpose(1,2), softmax_scale=scale, causal=False).transpose(1,2)

    def statistics(self, kv_heads):
        return dict(length=self.length, decode_steps=self.steps,
            selected_per_query_head=self.cache.Kgpu.shape[2],
            implementation='upstream LightAttentionIndicesFactory unchanged',
            cpu_offload=True, rank=32, topk=2048, lite=64, tolerance=0.01)
