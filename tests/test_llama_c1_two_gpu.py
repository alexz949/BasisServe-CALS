"""Verify the exact two-device dispatch/cache path used by the 64K evaluator."""

import copy
import pytest
import torch


@pytest.mark.skipif(torch.cuda.device_count() < 2,reason='two CUDA devices required')
@torch.inference_mode()
def test_dispatched_c1_v96_matches_one_device_with_resident_sidecars():
    from accelerate import dispatch_model
    from transformers import LlamaConfig,LlamaForCausalLM
    from basisserve.core.llama_c1_ruler import LlamaC1RulerAttention,LlamaC1RulerCache
    torch.manual_seed(37)
    config = LlamaConfig(vocab_size=256,hidden_size=512,intermediate_size=768,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=1,
        head_dim=128,max_position_embeddings=1024)
    config._attn_implementation = 'sdpa'
    single = LlamaForCausalLM(config).bfloat16().eval()
    for layer in single.model.layers:
        old = layer.self_attn
        encoder = torch.eye(128)[None,:,:96]
        decoder = old.o_proj.weight.float().reshape(512,4,128).permute(1,2,0)[:,:96]
        router = dict(base_left_b16=torch.zeros(1,96,16),base_right_b16=torch.zeros(1,16,128),
            base_bias_b16=torch.zeros(1,128),residual_encoder_b16_r8=torch.zeros(1,128,8),
            residual_query_b16_r8=torch.zeros(4,128,8))
        layer.self_attn = LlamaC1RulerAttention(old,encoder,decoder,router)
    multi = copy.deepcopy(single)
    single.cuda(0)
    dispatch_model(multi,device_map={'model.embed_tokens':0,'model.rotary_emb':0,
        'model.layers.0':0,'model.layers.1':1,'model.norm':1,'lm_head':1},
        skip_keys=['past_key_values','past_key_value'])
    caches = [LlamaC1RulerCache(160),LlamaC1RulerCache(160)]
    for ids in (torch.randint(0,256,(1,129),device='cuda:0'),torch.tensor([[7]],device='cuda:0')):
        expected = single(input_ids=ids,past_key_values=caches[0],use_cache=True,logits_to_keep=1)
        observed = multi(input_ids=ids,past_key_values=caches[1],use_cache=True,logits_to_keep=1)
        torch.testing.assert_close(observed.logits.cpu(),expected.logits.cpu(),atol=.01,rtol=.01)
    assert caches[1].lengths == [130,130]
    assert [t.device.index for t in caches[1].key_cache] == [0,1]
    assert [t.device.index for t in caches[1].sidecars] == [0,1]
