import pytest
import copy
from types import SimpleNamespace
import torch
from transformers import AutoModelForCausalLM, DynamicCache, NemotronHConfig

from basisserve.checkpoint.c1_attention_layers import c1_attention_layers
from basisserve.checkpoint.c1_lrqk_qwen3 import install_c1_lrqk, C1LRQKCache
from basisserve.checkpoint.c1_shadowkv_qwen3 import install_c1_shadowkv, C1ShadowKVCache
from basisserve.checkpoint.gqa_vo_nemotron_h import nemotron_h_c1_attention


@pytest.mark.parametrize('install,cache_type', [
    (install_c1_lrqk, C1LRQKCache), (install_c1_shadowkv, C1ShadowKVCache)])
def test_routing_install_preserves_native_recurrent_blocks_and_cache(install, cache_type):
    config = NemotronHConfig(hidden_size=256, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=128,
        layers_block_type=['linear_attention', 'full_attention', 'mlp', 'full_attention'])
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config).eval()
        encoder = torch.eye(128)[None]
        for index, attention in c1_attention_layers(model):
            model.model.layers[index].mixer = nemotron_h_c1_attention(attention,
                v_proj_compressed_weight=attention.v_proj.weight,
                o_decoder_weight=attention.o_proj.weight,
                value_coordinate_encoder=encoder)
    assert [i for i, _ in c1_attention_layers(model)] == [1, 3]
    recurrent = model.model.layers[0].mixer
    mlp = model.model.layers[2].mixer
    old_recurrent_forward, old_mlp_forward = recurrent.forward, mlp.forward
    before = {i: module.forward for i, module in c1_attention_layers(model)}
    install(model)
    assert model.model.layers[0].mixer is recurrent and recurrent.forward == old_recurrent_forward
    assert model.model.layers[2].mixer is mlp and mlp.forward == old_mlp_forward
    for index, module in c1_attention_layers(model):
        assert module.layer_idx == index and module.forward != before[index]
        assert module._forward_pre_hooks  # Native identity-position hook survives.
    native_cache, routing_cache = DynamicCache(config=config), cache_type(config=config)
    assert [type(layer) for layer in routing_cache.layers] == [type(layer) for layer in native_cache.layers]


@torch.inference_mode()
def test_ruler_native_install_uses_layer_ids_and_matches_dense_folded_attention(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHAttention
    from evaluation import eval_k_routing_ruler as driver
    from basisserve.kernels.compressed_v_decode_attention import reference_compressed_v_prefill_attention

    torch.manual_seed(112)
    config = NemotronHConfig(hidden_size=32, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layers_block_type=['linear_attention', 'full_attention', 'mlp', 'full_attention'])
    config._attn_implementation = 'sdpa'
    blocks, references, records = [], {}, []
    for index, kind in enumerate(config.layers_block_type):
        if kind != 'full_attention':
            blocks.append(SimpleNamespace(block_type=kind, mixer=torch.nn.Identity()))
            continue
        attention = NemotronHAttention(config, index).eval()
        reference = copy.deepcopy(attention)
        encoder = torch.stack([torch.linalg.qr(torch.randn(8, 8)).Q[:, :4] for _ in range(2)])
        decoder = torch.bmm(encoder.repeat_interleave(2, 0).mT,
            attention.o_proj.weight.T.reshape(4, 8, 32))
        name = f'layer_{index}.safetensors'
        save_file(dict(value_coordinate_encoders=encoder.contiguous(),
            head_output_decoders=decoder.contiguous(), source_ranks=torch.tensor([4, 4])), str(tmp_path/name))
        records.append(dict(layer=index, file=name, ranks=[4, 4]))
        weight = reference.v_proj.weight.reshape(2, 8, 32)
        reference.v_proj.weight.copy_(torch.bmm(encoder, torch.bmm(encoder.mT, weight)).reshape(16, 32))
        references[index] = reference
        blocks.append(SimpleNamespace(block_type=kind, mixer=attention))
    model = SimpleNamespace(config=config, model=SimpleNamespace(layers=blocks), eval=lambda: None)
    recurrent, mlp = blocks[0].mixer, blocks[2].mixer
    driver.install(model, tmp_path, dict(layers=records[::-1]), 'full', {})
    assert blocks[0].mixer is recurrent and blocks[2].mixer is mlp
    monkeypatch.setattr(driver, 'compressed_v_prefill_attention', reference_compressed_v_prefill_attention)
    cache, reference_cache = driver.RoutingCache(config), DynamicCache(config=config)
    for length in (23, 1, 1):
        for index in (1, 3):
            hidden = torch.randn(1, length, 32)
            actual, _ = blocks[index].mixer(hidden, attention_mask=None, past_key_values=cache)
            expected, _ = references[index](hidden, attention_mask=None, past_key_values=reference_cache)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            assert cache.layers[index].values.shape[-1] == 4


@torch.inference_mode()
def test_ruler_generate_native_hybrid_matches_unmodified_greedy_model(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from evaluation import eval_k_routing_ruler as driver
    from basisserve.kernels.compressed_v_decode_attention import reference_compressed_v_prefill_attention

    torch.manual_seed(117)
    config = NemotronHConfig(hidden_size=32, intermediate_size=64, vocab_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        mamba_num_heads=4, mamba_head_dim=8, ssm_state_size=4, n_groups=1, chunk_size=8,
        layers_block_type=['linear_attention','full_attention','mlp','full_attention'],
        eos_token_id=63)
    config._attn_implementation='sdpa'
    model=AutoModelForCausalLM.from_config(config).eval()
    reference=copy.deepcopy(model)
    records=[]
    for index, module in c1_attention_layers(model):
        name=f'layer_{index}.safetensors'
        save_file(dict(value_coordinate_encoders=torch.eye(8).repeat(2,1,1),
            head_output_decoders=module.o_proj.weight.T.reshape(4,8,32).contiguous(),
            source_ranks=torch.tensor([8,8])),str(tmp_path/name))
        records.append(dict(layer=index,file=name,ranks=[8,8]))
    driver.install(model,tmp_path,dict(layers=records),'full',{})
    monkeypatch.setattr(driver,'compressed_v_prefill_attention',reference_compressed_v_prefill_attention)
    row=dict(input_ids=[2,3,4,5,6,7,8,9,10])
    tokenizer=SimpleNamespace(eos_token_id=63)
    ids,first,stats,stopped=driver.generate(model,tokenizer,row,'full',4)
    cache=DynamicCache(config=config)
    tokens=torch.tensor([row['input_ids']])
    expected=[]
    for step in range(4):
        output=reference(input_ids=tokens,past_key_values=cache,use_cache=True,logits_to_keep=1)
        if step==0:
            torch.testing.assert_close(first,output.logits[0,-1],atol=2e-6,rtol=2e-5)
        token=int(output.logits[0,-1].argmax())
        expected.append(token)
        if token==63:
            break
        tokens=torch.tensor([[token]])
    assert len(ids)>1 and ids==expected and len(stats)==2
    assert stopped==(ids[-1]==63)
