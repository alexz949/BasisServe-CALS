import torch
import pytest
from types import SimpleNamespace
from evaluation.llama_single_gpu_prefill import prefill,InplaceChunkedMLP
from evaluation.eval_k_routing_ruler import routing_forward


def test_mlp_buffer_reuse_preserves_values():
    inner=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.SiLU(),torch.nn.Linear(32,16))
    x=torch.randn(1,2051,16)
    with torch.inference_mode():
        expected=torch.cat([inner(x[:,s:s+1024]) for s in range(0,2051,1024)],dim=1)
        actual=InplaceChunkedMLP(inner)(x.clone())
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires FlashAttention CUDA')
def test_query_blocked_prefill_matches_full_flashattention():
    class Cache:
        def get_seq_length(self,layer):return 0
        def update(self,k,v,layer):return k,v
    torch.manual_seed(42)
    device='cuda';dtype=torch.bfloat16;hidden=64
    m=SimpleNamespace(num_attention_heads=4,num_key_value_heads=2,head_dim=16,value_head_dim=16,
        layer_idx=0,scaling=.25,_routing_arm='full',q_norm=torch.nn.Identity(),k_norm=torch.nn.Identity())
    for name,width in [('q_proj',64),('k_proj',32),('v_proj',32),('o_proj',64)]:
        setattr(m,name,torch.nn.Linear(hidden,width,bias=False).to(device=device,dtype=dtype))
    x=torch.randn(1,2051,hidden,device=device,dtype=dtype)
    angles=torch.randn(1,2051,8,device=device,dtype=dtype).repeat(1,1,2)
    cos,sin=angles.cos(),angles.sin()
    with torch.inference_mode():
        expected=routing_forward(m,x,(cos,sin),past_key_values=Cache())[0]
        actual=prefill(m,x,(cos,sin),past_key_values=Cache())[0]
    torch.testing.assert_close(actual,expected,atol=.005,rtol=.02)


def test_residual_buffer_reuse_matches_dense_decoder():
    from evaluation.llama_single_gpu_prefill import decoder
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__();self.linear=torch.nn.Linear(16,16)
        def forward(self,hidden_states,**kwargs):return self.linear(hidden_states),None
    inner=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.SiLU(),torch.nn.Linear(32,16))
    m=SimpleNamespace(input_layernorm=torch.nn.LayerNorm(16),post_attention_layernorm=torch.nn.LayerNorm(16),
        self_attn=Attention(),mlp=InplaceChunkedMLP(inner))
    x=torch.randn(1,2051,16)
    with torch.inference_mode():
        y=x+m.self_attn(m.input_layernorm(x))[0]
        normalized=m.post_attention_layernorm(y)
        expected=y+torch.cat([inner(normalized[:,s:s+1024]) for s in range(0,2051,1024)],dim=1)
        actual=decoder(m,x.clone())
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
