import pytest
import torch

from basisserve.core.c1_strict_recent_decode import strict_recent_token_ids, c1_strict_recent_decode
from basisserve.kernels.compressed_v_decode_attention import compressed_v_prefill_attention


@pytest.mark.parametrize('length',[31,64,2048,*range(2049,2081),65536,65537])
def test_two_recent_physical_pages_at_every_page_offset(length):
    torch.manual_seed(42)
    masses = torch.randn(1,2,4,(length+31)//32)
    ids = strict_recent_token_ids(masses,length)
    recent_start = max(0,((length+31)//32-2)*32)
    expected_count = length if length <= 2048 else 32+61*32+length-recent_start
    assert ids.shape == (1,2,expected_count)
    assert expected_count <= 2048
    for group in range(2):
        selected = ids[0,group].tolist()
        assert len(set(selected)) == len(selected)
        assert min(selected) >= 0 and max(selected) < length
        assert set(range(min(32,length))) <= set(selected)
        assert set(range(recent_start,length)) <= set(selected)
        if length > 2048:
            history = ids[0,group,32:32+61*32].reshape(61,32)
            assert torch.equal(history%32,torch.arange(32).expand(61,32))
            assert history.max() < recent_start
            assert torch.equal(ids[0,group,32+61*32:],torch.arange(recent_start,length))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_v96_triton_prefill_matches_dense_causal_reference():
    torch.manual_seed(9)
    q = torch.randn(1,8,129,128,device='cuda',dtype=torch.bfloat16)
    k = torch.randn(1,2,129,128,device='cuda',dtype=torch.bfloat16)
    v = torch.randn(1,2,129,96,device='cuda',dtype=torch.bfloat16)
    out = compressed_v_prefill_attention(q,k,v,scale=128**-.5)
    scores = q.float() @ k.repeat_interleave(4,1).float().transpose(-1,-2)/128**.5
    scores.masked_fill_(torch.ones(129,129,device='cuda',dtype=torch.bool).triu(1),-torch.inf)
    expected = scores.softmax(-1) @ v.repeat_interleave(4,1).float()
    torch.testing.assert_close(out.float(),expected,atol=.025,rtol=.025)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@torch.inference_mode()
def test_llama_adapter_identity_matches_native_prefill_and_decode():
    import copy
    from transformers import LlamaConfig,LlamaForCausalLM
    from transformers.cache_utils import DynamicCache
    from basisserve.core.llama_c1_ruler import LlamaC1RulerAttention,LlamaC1RulerCache
    torch.manual_seed(21)
    config = LlamaConfig(vocab_size=256,hidden_size=512,intermediate_size=768,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=1,
        head_dim=128,max_position_embeddings=1024)
    config._attn_implementation = 'sdpa'
    dense = LlamaForCausalLM(config).cuda().bfloat16().eval()
    compact = copy.deepcopy(dense)
    for layer in compact.model.layers:
        attention = layer.self_attn
        encoder = torch.eye(128,device='cuda')[None]
        decoder = attention.o_proj.weight.float().reshape(512,4,128).permute(1,2,0)
        router = dict(base_left_b16=torch.zeros(1,128,16,device='cuda'),
            base_right_b16=torch.zeros(1,16,128,device='cuda'),
            base_bias_b16=torch.zeros(1,128,device='cuda'),
            residual_encoder_b16_r8=torch.zeros(1,128,8,device='cuda'),
            residual_query_b16_r8=torch.zeros(4,128,8,device='cuda'))
        layer.self_attn = LlamaC1RulerAttention(attention,encoder,decoder,router)
        layer.self_attn.sparse = False
    dense_cache,compact_cache = DynamicCache(),LlamaC1RulerCache(160)
    for tokens in (torch.randint(0,256,(1,129),device='cuda'),torch.tensor([[7]],device='cuda')):
        expected = dense(input_ids=tokens,past_key_values=dense_cache,use_cache=True,logits_to_keep=1)
        observed = compact(input_ids=tokens,past_key_values=compact_cache,use_cache=True,logits_to_keep=1)
        torch.testing.assert_close(observed.logits.float(),expected.logits.float(),atol=.025,rtol=.025)
        assert compact_cache.get_seq_length() == dense_cache.get_seq_length()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_v96_triton_sparse_decode_matches_selected_reference():
    torch.manual_seed(19)
    length = 2081
    q = torch.randn(1,8,1,128,device='cuda',dtype=torch.bfloat16)
    k = torch.randn(1,2,length,128,device='cuda',dtype=torch.bfloat16)
    v = torch.randn(1,2,length,96,device='cuda',dtype=torch.bfloat16)
    sidecar = torch.randn(1,2,length,136,device='cuda',dtype=torch.bfloat16)
    u = torch.randn(8,128,8,device='cuda',dtype=torch.bfloat16)*.1
    out,ids = c1_strict_recent_decode(q,k,v,sidecar,u,scale=128**-.5)
    heads = ids.repeat_interleave(4,1)
    keys = k.repeat_interleave(4,1).gather(2,heads[...,None].expand(-1,-1,-1,128))
    values = v.repeat_interleave(4,1).gather(2,heads[...,None].expand(-1,-1,-1,96))
    scores = q.float() @ keys.float().transpose(-1,-2)/128**.5
    expected = scores.softmax(-1) @ values.float()
    torch.testing.assert_close(out.float(),expected,atol=.025,rtol=.025)
