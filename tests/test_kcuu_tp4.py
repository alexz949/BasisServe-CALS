"""CPU checks for source ownership, prompt semantics and graph dispatch policy."""
from types import SimpleNamespace
import pytest
import torch
from basisserve.core.tp4_c1_factors import local_decoder
from basisserve.core.tp4_step import classify_step


@pytest.mark.parametrize('heads', [32, 64])
def test_local_sum_matches_source_ordered_gather(heads):
    torch.manual_seed(19)
    # Keep this CPU algebra check small; production shapes are audited separately.
    z = torch.randn(5, heads, 64, dtype=torch.float64)
    decoder = torch.randn(heads * 64, 23, dtype=torch.float64)
    width = heads // 4 * 64
    pieces = [z.flatten(1)[:, rank * width:(rank + 1) * width] @
              local_decoder(decoder, rank, width) for rank in range(4)]
    expected = torch.einsum('thr,hro->to', z, decoder.view(heads, 64, 23))
    torch.testing.assert_close(sum(pieces), expected)
    torch.testing.assert_close(z.flatten(1) @ decoder, expected)


@pytest.mark.parametrize('lengths,flags,kind,route,tokens', [
    ([17], [True], 'prefill', 'ar', (17, 0)),
    ([1], [True], 'prefill', 'ar', (1, 0)),
    ([1, 1], [True, False], 'mixed', 'ar', (1, 1)),
    ([17, 1], [True, False], 'mixed', 'ar', (17, 1)),
    ([1, 1, 0], [False, False, True], 'decode', 'ag', (0, 2)),
    ([0], [False], 'empty', 'ag', (0, 0)),
])
def test_route_uses_prompt_semantics_not_query_length(lengths, flags, kind, route, tokens):
    step = classify_step(lengths, flags)
    assert step.kind == kind and step.collective('hybrid') == route
    assert (step.prefill_tokens, step.decode_tokens) == tokens
    assert step.collective('ag') == 'ag' and step.collective('ar') == 'ar'


@pytest.mark.parametrize('computed,prompt,known,expected', [(64, 65, 66, 'NONE'), (65, 65, 66, 'FULL'),
                                                         (0, 65, 66, 'NONE'), (66, 65, 100, 'NONE')])
def test_prompt_tail_and_recompute_cannot_replay_decode_graph(monkeypatch, computed, prompt, known, expected):
    import numpy as np
    from vllm.config import CUDAGraphMode
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from basisserve.vllm.kcuu_worker import KCuuDispatcher
    calls = []

    def dispatch(self, num_tokens, uniform_decode, has_lora, num_active_loras, valid_modes, invalid_modes):
        calls.append(valid_modes)
        mode = CUDAGraphMode.NONE if valid_modes == {CUDAGraphMode.NONE} else CUDAGraphMode.FULL
        return mode, SimpleNamespace(num_tokens=num_tokens)

    monkeypatch.setattr(CudagraphDispatcher, 'dispatch', dispatch)
    dispatcher = object.__new__(KCuuDispatcher)
    dispatcher.runner = SimpleNamespace(requests={'request': SimpleNamespace(num_tokens=known)}, input_batch=SimpleNamespace(req_ids=['request'],
        num_computed_tokens_cpu=np.array([computed]), num_prompt_tokens=np.array([prompt])))
    dispatcher.scheduled_tokens = {'request': 1}
    dispatcher.step_records = []
    mode, _ = dispatcher.dispatch(4, uniform_decode=True)
    assert mode.name == expected
    assert dispatcher.step_records[0]['prefill_tokens'] == int(expected == 'NONE')
    from basisserve.core.tp4_step import CURRENT_OUTPUT_STEP
    CURRENT_OUTPUT_STEP.set(None)


def test_llama_adapter_preserves_native_rope_without_qk_norm(monkeypatch):
    import basisserve.vllm.llama_tp4_c1 as module
    seen = {}

    class Projection(torch.nn.Module):
        def __init__(self, encoder, prefix, hidden_size, heads):
            super().__init__()
            assert (hidden_size, heads) == (4096, 32)
        def forward(self, hidden):
            return torch.ones(hidden.shape[0], 1024 + 256 + 128), None

    class Rotary(torch.nn.Module):
        def forward(self, positions, q, k):
            seen['shapes'] = (q.shape, k.shape)
            return q, k

    class Attention(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.layer_name = kwargs['prefix']

    monkeypatch.setattr(module, 'C1QKVParallelLinear', Projection)
    monkeypatch.setattr(module, 'Attention', Attention)
    monkeypatch.setattr(module, 'c1_diffkv_backend', lambda rank: None)
    original = SimpleNamespace(layer_name='model.layers.0.self_attn.attn')
    rotary = Rotary()
    base = SimpleNamespace(num_heads=8, num_kv_heads=2, scaling=128**-0.5,
                           qkv_proj=SimpleNamespace(bias=None), o_proj=SimpleNamespace(bias=None),
                           attn=original, rotary_emb=rotary)
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={original.layer_name: original}),
                          cache_config=None)
    attn = module.LlamaC1Attention(base, torch.empty(2, 128, 64), torch.empty(2048, 4096), None, cfg)
    assert attn.rotary_emb is rotary
    assert not hasattr(attn, 'q_norm') and not hasattr(attn, 'k_norm')
    # Intercept only the fused output op; run the actual projection/split/RoPE forward.
    monkeypatch.setattr(torch.ops.vllm, 'basisserve_c1_attention_output',
                        lambda q, k, v, d, name: v)
    output = attn(torch.arange(2), torch.empty(2, 4096))
    assert output.shape == (2, 128)
    assert seen['shapes'] == (torch.Size([2, 1024]), torch.Size([2, 256]))


@pytest.mark.parametrize('prefilling,expected', [([True, False, False], 'mixed'),
                                               ([False, False, True], 'decode')])
def test_metadata_preserves_single_token_prefill_and_ignores_padding(monkeypatch, prefilling, expected):
    from basisserve.vllm.diffkv_attention import c1_diffkv_backend
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
    monkeypatch.setattr(TritonAttentionMetadataBuilder, 'build',
                        lambda *args: SimpleNamespace(seq_lens=torch.ones(3, dtype=torch.int32)))
    builder = object.__new__(c1_diffkv_backend(64).get_builder_cls())
    common = SimpleNamespace(is_prefilling=torch.tensor(prefilling),
        query_start_loc_cpu=torch.tensor([0, 1, 2, 3]),
        seq_lens_cpu_upper_bound=torch.tensor([33, 33, 0]))
    metadata = builder.build(0, common)
    assert metadata.basisserve_step.kind == expected
    assert metadata.basisserve_step.prefill_tokens + metadata.basisserve_step.decode_tokens == 2
