from types import MethodType

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM, DynamicCache

from evaluation import eval_k_routing_ruler as runtime
from evaluation.fit_k_routing import fit_base


@torch.inference_mode()
def test_native_llama_identity_payload_matches_prefill_and_decode(tmp_path, monkeypatch):
    torch.manual_seed(51)
    config = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=128)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).eval()
    tokens = torch.randint(0,128,(1,23))
    native_cache = DynamicCache(config=config)
    native = model(tokens,past_key_values=native_cache,use_cache=True).logits
    continuation = model(torch.tensor([[7]]),past_key_values=native_cache,use_cache=True).logits
    records = []
    for i,layer in enumerate(model.model.layers):
        path = tmp_path/f'layer_{i}.safetensors'
        save_file(dict(value_coordinate_encoders=torch.eye(16).repeat(2,1,1),
            head_output_decoders=layer.self_attn.o_proj.weight.T.reshape(4,16,64).contiguous(),
            source_ranks=torch.tensor([16,16])),str(path))
        records.append(dict(layer=i,file=path.name,ranks=[16,16]))
    def prefill(q,k,v,scale):
        return torch.nn.functional.scaled_dot_product_attention(q,k,v,
            enable_gqa=True,is_causal=True,scale=scale)
    monkeypatch.setattr(runtime,'compressed_v_prefill_attention',prefill)
    runtime.install(model,tmp_path,dict(layers=records),'full',{})
    cache = runtime.RoutingCache(config)
    actual = model(tokens,past_key_values=cache,use_cache=True).logits
    after = model(torch.tensor([[7]]),past_key_values=cache,use_cache=True).logits
    torch.testing.assert_close(native,actual)
    torch.testing.assert_close(continuation,after)
    for a,b in zip(native_cache.layers,cache.layers,strict=True):
        torch.testing.assert_close(a.keys,b.keys)
        torch.testing.assert_close(a.values,b.values)


def test_base_uses_frozen_latent_and_excludes_diagnostic_rows():
    torch.manual_seed(29)
    raw = torch.randn(3,256,2,48)
    encoder = torch.randn(2,48,32)
    z = torch.einsum('ntgd,gdr->ntgr',raw,encoder)
    left,right = torch.randn(2,32,8),torch.randn(2,8,48)
    keys = torch.einsum('ntgr,grs,gsd->ntgd',z,left,right)+torch.randn(2,48)
    rows = torch.cat((raw,torch.randn_like(raw)),dim=-1)
    a = fit_base(rows,keys,encoder,2,torch.device('cpu'))[16]
    rows[2].fill_(1e6)
    keys[2].fill_(-1e6)
    b = fit_base(rows,keys,encoder,2,torch.device('cpu'))[16]
    for group in range(2):
        torch.testing.assert_close(a[group].weight,b[group].weight,rtol=0,atol=0)
        predicted = a[group].apply(z[:2,:,group].double())
        torch.testing.assert_close(predicted,keys[:2,:,group].double(),atol=3e-4,rtol=3e-5)


@torch.inference_mode()
def test_exact_page_oracle_has_same_output_as_dense_when_budget_covers_context():
    torch.manual_seed(31)
    q=torch.randn(1,4,1,16)
    k=torch.randn(1,2,73,16)
    v=torch.randn(1,2,73,12)
    identity=torch.eye(16).expand(4,-1,-1)
    result=runtime.c1_conditional_page_topk_attention(q,k,v,k,identity,page_size=32,
        exact_token_budget=2048,pinned_prefix_pages=1,scale=0.25,query_block_size=1)
    dense=torch.nn.functional.scaled_dot_product_attention(q,k,v,enable_gqa=True,scale=0.25)
    torch.testing.assert_close(result.output,dense)
    assert result.statistics['selected_tokens']==2*73


def test_capture_shards_restore_fit_then_diagnostic_order(tmp_path):
    from evaluation.fit_k_routing_captures import read_capture
    from evaluation.v96kl_common import write_json, sha256
    from basisserve.core.query_position_sampling import candidate_positions

    protocol = dict(sequence_length=256, num_shards=2)
    identity = dict(hkv=2, hq=4, head_dim=8)
    grid = candidate_positions(256)
    # A diagnostic row precedes a training row on disk. The fitter must use
    # the requested global IDs, not the shard's physical ordering.
    for shard, ids in enumerate(([64, 0], [1])):
        path = tmp_path / 'layer_000' / f'shard_{shard}.safetensors'
        path.parent.mkdir(exist_ok=True)
        tensors = {}
        for name, shape in dict(rows=(256, 2, 16), pre_rope_keys=(256, 2, 8),
                candidate_queries=(len(grid), 4, 8)).items():
            tensors[name] = torch.stack([torch.full(shape, window, dtype=torch.bfloat16) for window in ids])
        save_file(tensors, str(path))
        write_json(path.with_suffix('.json'), dict(status='complete', protocol=protocol,
            layer=0, candidate_positions=grid, window_ids=ids, sha256=sha256(path)))
    tensors, manifests = read_capture(tmp_path, 0, [0, 1, 64], protocol, identity)
    for tensor in tensors.values():
        tensor = torch.stack(list(tensor))
        assert tensor.flatten(1)[:, 0].tolist() == [0, 1, 64]
        assert all(torch.equal(row, torch.full_like(row, window))
                   for row, window in zip(tensor, [0, 1, 64], strict=True))
    assert [m['window_ids'] for m in manifests] == [[64, 0], [1]]
