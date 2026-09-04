from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM, StaticCache

import basisserve.checkpoint.gqa_vo_qwen3 as gqa_vo_qwen3
from basisserve.checkpoint.gqa_vo_qwen3 import (
    GQATiedVOQwen3Attention,
    _memory_bounded_gqa_sdpa,
    transition_qwen3_dense_prefill_cache_to_c1,
)
from basisserve.core.c1_k_routing_sidecar import (
    RoutingDynamicCache,
    build_routing_sidecar,
)
from basisserve.core.c1_k_reverse_shadow import ReverseShadowConfig
from evaluation.eval_qwen3_8b_c1_kq_routing_ruler import _chunked_exact_prefill


def _exact_c1_sdpa_model(
    model: Qwen3ForCausalLM,
    *,
    attention_backend: str = "sdpa",
) -> Qwen3ForCausalLM:
    candidate = copy.deepcopy(model)
    for layer in candidate.model.layers:
        attention = layer.self_attn
        identity_encoder = torch.eye(attention.head_dim).expand(
            attention.config.num_key_value_heads,
            -1,
            -1,
        )
        layer.self_attn = GQATiedVOQwen3Attention(
            attention,
            v_proj_compressed_weight=attention.v_proj.weight.detach().clone(),
            o_decoder_weight=attention.o_proj.weight.detach().clone(),
            v_proj_compressed_bias=(
                None
                if attention.v_proj.bias is None
                else attention.v_proj.bias.detach().clone()
            ),
            o_decoder_bias=(
                None
                if attention.o_proj.bias is None
                else attention.o_proj.bias.detach().clone()
            ),
            attention_backend=attention_backend,
            value_coordinate_encoder=identity_encoder,
        )
    return candidate


@torch.inference_mode()
def test_memory_bounded_gqa_sdpa_matches_full_head_expansion() -> None:
    torch.manual_seed(20260830)
    query = torch.randn(2, 8, 5, 4)
    key = torch.randn(2, 4, 7, 4)
    value = torch.randn(2, 4, 7, 3)
    mask = torch.rand(2, 1, 5, 7) > 0.2
    mask[..., 0] = True

    expected = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        scale=0.5,
        enable_gqa=True,
    )
    actual = _memory_bounded_gqa_sdpa(
        query,
        key,
        value,
        attention_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=0.5,
        kv_heads_per_chunk=2,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@torch.inference_mode()
def test_qwen3_attention_appends_post_rope_routing_sidecar_once() -> None:
    torch.manual_seed(31)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=0.0,
        _attn_implementation="sdpa",
    )
    model = _exact_c1_sdpa_model(Qwen3ForCausalLM(config).float()).eval()
    attention = model.model.layers[0].self_attn
    projector = torch.eye(4).expand(2, -1, -1).clone()
    attention.set_routing_projectors(projector, projector)
    cache = RoutingDynamicCache()
    input_ids = torch.arange(9).view(1, 9) % config.vocab_size

    for start, stop in ((0, 5), (5, 9)):
        model(
            input_ids=input_ids[:, start:stop],
            past_key_values=cache,
            use_cache=True,
        )

    exact_post_key = cache.layers[0].keys
    torch.testing.assert_close(
        cache.routing_sidecar(0),
        build_routing_sidecar(exact_post_key, projector),
    )

    cache.crop(6)
    assert cache.get_seq_length() == 6
    assert cache.routing_sidecar(0).shape[-2] == 6


@torch.inference_mode()
def test_full_rank_c1_sdpa_matches_qwen3_sdpa_full_and_cached() -> None:
    torch.manual_seed(20260829)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        _attn_implementation="sdpa",
    )
    dense = Qwen3ForCausalLM(config).float().eval()
    candidate = _exact_c1_sdpa_model(dense).float().eval()
    input_ids = torch.arange(12).view(1, 12) % config.vocab_size

    dense_logits = dense(input_ids=input_ids, use_cache=False).logits
    candidate_logits = candidate(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(
        candidate_logits,
        dense_logits,
        rtol=2e-6,
        atol=2e-6,
    )

    cached_logits = []
    for model in (dense, candidate):
        cache = DynamicCache()
        pieces = []
        for start, stop in ((0, 8), (8, 10), (10, 12)):
            pieces.append(
                model(
                    input_ids=input_ids[:, start:stop],
                    past_key_values=cache,
                    use_cache=True,
                ).logits
            )
        cached_logits.append(torch.cat(pieces, dim=1))
    torch.testing.assert_close(
        cached_logits[1],
        cached_logits[0],
        rtol=2e-6,
        atol=2e-6,
    )

    first_token, chunked_cache = _chunked_exact_prefill(
        candidate,
        input_ids,
        chunk_size=3,
    )
    assert first_token == int(dense_logits[0, -1].argmax().item())
    assert chunked_cache.get_seq_length() == input_ids.shape[1]


@torch.inference_mode()
def test_dense_prefill_transition_matches_dense_full_rank_decode() -> None:
    torch.manual_seed(20260831)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        _attn_implementation="sdpa",
    )
    dense = Qwen3ForCausalLM(config).float().eval()
    candidate = _exact_c1_sdpa_model(
        dense,
        attention_backend="dense_prefill",
    ).float().eval()
    prompt = torch.arange(12).view(1, 12) % config.vocab_size
    dense_cache = DynamicCache()
    candidate_cache = DynamicCache()

    for start, stop in ((0, 5), (5, 9), (9, 12)):
        expected = dense(
            input_ids=prompt[:, start:stop],
            past_key_values=dense_cache,
            use_cache=True,
        ).logits
        actual = candidate(
            input_ids=prompt[:, start:stop],
            past_key_values=candidate_cache,
            use_cache=True,
        ).logits
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

    key_pointers = [layer.keys.data_ptr() for layer in candidate_cache.layers]
    transition_qwen3_dense_prefill_cache_to_c1(
        candidate,
        candidate_cache,
        attention_backend="sdpa",
    )
    for layer_index, layer in enumerate(candidate.model.layers):
        attention = layer.self_attn
        assert attention.attention_backend == "sdpa"
        assert attention.dense_v_proj is None
        assert attention.dense_o_proj is None
        assert candidate_cache.layers[layer_index].keys.data_ptr() == key_pointers[
            layer_index
        ]
        torch.testing.assert_close(
            candidate_cache.layers[layer_index].values,
            dense_cache.layers[layer_index].values,
        )

    decode_token = torch.tensor([[17]])
    expected = dense(
        input_ids=decode_token,
        past_key_values=dense_cache,
        use_cache=True,
    ).logits
    actual = candidate(
        input_ids=decode_token,
        past_key_values=candidate_cache,
        use_cache=True,
    ).logits
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@torch.inference_mode()
def test_dense_prefill_cache_transition_projects_value_head_dimension() -> None:
    torch.manual_seed(20260901)
    dense_values = torch.randn(2, 3, 7, 8)
    exact_keys = torch.randn(2, 3, 7, 8)
    encoder = torch.randn(3, 8, 5)
    attention = SimpleNamespace(
        value_coordinate_encoder=encoder,
        dense_v_proj=object(),
        dense_o_proj=object(),
        attention_backend="dense_prefill",
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention)],
        )
    )
    cache = SimpleNamespace(
        layers=[SimpleNamespace(keys=exact_keys, values=dense_values.clone())],
    )

    expected = torch.einsum("bhsd,hdr->bhsr", dense_values, encoder)
    returned = transition_qwen3_dense_prefill_cache_to_c1(
        model,
        cache,
        attention_backend="cuda_dense",
    )

    assert returned is cache
    assert cache.layers[0].keys is exact_keys
    assert cache.layers[0].values.shape == (2, 3, 7, 5)
    torch.testing.assert_close(cache.layers[0].values, expected)
    assert attention.dense_v_proj is None
    assert attention.dense_o_proj is None
    assert attention.attention_backend == "cuda_dense"


@torch.inference_mode()
def test_c1_triton_backend_dispatches_after_sdpa_prefill(monkeypatch) -> None:
    torch.manual_seed(20260830)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        _attn_implementation="sdpa",
    )
    source = Qwen3ForCausalLM(config).float().eval()
    expected_model = _exact_c1_sdpa_model(source).float().eval()
    candidate = copy.deepcopy(expected_model).eval()
    expected_cache = DynamicCache()
    candidate_cache = StaticCache(candidate.config, max_cache_len=16)
    prompt = torch.arange(8).view(1, 8) % config.vocab_size
    expected_model(
        input_ids=prompt,
        past_key_values=expected_cache,
        use_cache=True,
    )
    candidate(
        input_ids=prompt,
        past_key_values=candidate_cache,
        use_cache=True,
    )

    calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    def fake_triton_decode(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        scale: float,
        valid_sequence_length: torch.Tensor | None,
    ) -> torch.Tensor:
        assert valid_sequence_length is not None
        valid = int(valid_sequence_length)
        calls.append((tuple(query.shape), tuple(key.shape), tuple(value.shape)))
        return F.scaled_dot_product_attention(
            query,
            key[:, :, :valid],
            value[:, :, :valid],
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )

    monkeypatch.setattr(
        gqa_vo_qwen3,
        "compressed_v_decode_attention_triton",
        fake_triton_decode,
    )
    for layer in candidate.model.layers:
        layer.self_attn.attention_backend = "triton"

    decode_token = torch.tensor([[11]])
    expected = expected_model(
        input_ids=decode_token,
        past_key_values=expected_cache,
        use_cache=True,
    ).logits
    actual = candidate(
        input_ids=decode_token,
        past_key_values=candidate_cache,
        use_cache=True,
    ).logits

    assert len(calls) == config.num_hidden_layers
    assert all(query_shape[-2] == 1 for query_shape, _, _ in calls)
    assert all(key_shape[-2] == 16 for _, key_shape, _ in calls)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@torch.inference_mode()
def test_c1_triton_backend_dispatches_complete_prefill(monkeypatch) -> None:
    torch.manual_seed(20260903)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        _attn_implementation="sdpa",
    )
    source = Qwen3ForCausalLM(config).float().eval()
    expected_model = _exact_c1_sdpa_model(source).float().eval()
    candidate = _exact_c1_sdpa_model(
        source,
        attention_backend="triton",
    ).float().eval()
    calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    def fake_triton_prefill(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        calls.append((tuple(query.shape), tuple(key.shape), tuple(value.shape)))
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
            enable_gqa=True,
        )

    monkeypatch.setattr(
        gqa_vo_qwen3,
        "compressed_v_prefill_attention",
        fake_triton_prefill,
    )
    input_ids = torch.arange(12).view(1, 12) % config.vocab_size
    expected = expected_model(input_ids=input_ids, use_cache=False).logits
    actual = candidate(input_ids=input_ids, use_cache=False).logits

    assert len(calls) == config.num_hidden_layers
    assert all(query_shape[-2] == 12 for query_shape, _, _ in calls)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@torch.inference_mode()
def test_c1_dense_cuda_backend_dispatches_after_sdpa_prefill(monkeypatch) -> None:
    torch.manual_seed(20260831)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        _attn_implementation="sdpa",
    )
    source = Qwen3ForCausalLM(config).float().eval()
    expected_model = _exact_c1_sdpa_model(source).float().eval()
    candidate = copy.deepcopy(expected_model).eval()
    expected_cache = DynamicCache()
    candidate_cache = StaticCache(candidate.config, max_cache_len=16)
    prompt = torch.arange(8).view(1, 8) % config.vocab_size
    expected_model(
        input_ids=prompt,
        past_key_values=expected_cache,
        use_cache=True,
    )
    candidate(
        input_ids=prompt,
        past_key_values=candidate_cache,
        use_cache=True,
    )

    calls: list[tuple[tuple[int, ...], int, tuple[int, ...]]] = []

    def fake_dense_cuda_decode(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid_sequence_length: torch.Tensor,
        *,
        scale: float,
        splits: int,
        workspace: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        valid = int(valid_sequence_length)
        calls.append((tuple(key.shape), splits, tuple(workspace.shape)))
        expected = F.scaled_dot_product_attention(
            query,
            key[:, :, :valid],
            value[:, :, :valid],
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )
        output.copy_(expected)
        return output

    monkeypatch.setattr(
        gqa_vo_qwen3,
        "c1_dense_gqa_v96_decode_attention_cuda",
        fake_dense_cuda_decode,
    )
    for layer in candidate.model.layers:
        layer.self_attn.attention_backend = "cuda_dense"

    decode_token = torch.tensor([[11]])
    expected = expected_model(
        input_ids=decode_token,
        past_key_values=expected_cache,
        use_cache=True,
    ).logits
    actual = candidate(
        input_ids=decode_token,
        past_key_values=candidate_cache,
        use_cache=True,
    ).logits

    assert len(calls) == config.num_hidden_layers
    assert all(key_shape[-2] == 16 for key_shape, _, _ in calls)
    assert all(split == 16 for _, split, _ in calls)
    assert all(shape == (4, 16, 10) for _, _, shape in calls)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@torch.inference_mode()
def test_c1_cuda_sparse_backend_routes_packs_and_dispatches(monkeypatch) -> None:
    torch.manual_seed(20260901)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        _attn_implementation="sdpa",
    )
    source = Qwen3ForCausalLM(config).float().eval()
    expected_model = _exact_c1_sdpa_model(source).float().eval()
    candidate = _exact_c1_sdpa_model(source).float().eval()
    expected_cache = DynamicCache()
    candidate_cache = RoutingDynamicCache()
    prompt = torch.arange(8).view(1, 8) % config.vocab_size
    projector = torch.eye(config.head_dim).expand(
        config.num_key_value_heads,
        -1,
        -1,
    ).clone()
    for layer in candidate.model.layers:
        layer.self_attn.set_routing_projectors(projector, projector)
    expected_model(
        input_ids=prompt,
        past_key_values=expected_cache,
        use_cache=True,
    )
    candidate(
        input_ids=prompt,
        past_key_values=candidate_cache,
        use_cache=True,
    )

    packed_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    sparse_calls: list[tuple[tuple[int, ...], int]] = []

    def fake_pack(
        exact_key: torch.Tensor,
        selected_page_ids: torch.Tensor,
        *,
        output: torch.Tensor,
    ) -> torch.Tensor:
        packed_calls.append((tuple(exact_key.shape), tuple(selected_page_ids.shape)))
        output.zero_()
        output[:, :, 0, : exact_key.shape[2]].copy_(exact_key)
        return output

    def fake_sparse(
        query: torch.Tensor,
        packed_key_pages: torch.Tensor,
        value: torch.Tensor,
        selected_page_ids: torch.Tensor,
        *,
        scale: float,
        splits: int,
        workspace: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        sparse_calls.append((tuple(packed_key_pages.shape), splits))
        valid = int(value.shape[2])
        expected = F.scaled_dot_product_attention(
            query,
            packed_key_pages[:, :, 0, :valid],
            value,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )
        output.copy_(expected)
        return output

    monkeypatch.setattr(gqa_vo_qwen3, "c1_pack_exact_key_pages_cuda", fake_pack)
    monkeypatch.setattr(
        gqa_vo_qwen3,
        "c1_paged_sparse_decode_attention_cuda",
        fake_sparse,
    )
    for layer in candidate.model.layers:
        attention = layer.self_attn
        attention.attention_backend = "cuda_sparse"
        attention.set_reverse_shadow_config(
            ReverseShadowConfig(
                page_size=64,
                exact_token_budget=64,
                selector="kq_svd",
            )
        )

    decode_token = torch.tensor([[11]])
    expected = expected_model(
        input_ids=decode_token,
        past_key_values=expected_cache,
        use_cache=True,
    ).logits
    actual = candidate(
        input_ids=decode_token,
        past_key_values=candidate_cache,
        use_cache=True,
    ).logits

    assert len(packed_calls) == config.num_hidden_layers
    assert len(sparse_calls) == config.num_hidden_layers
    assert all(page_id_shape == (1, 2, 1) for _, page_id_shape in packed_calls)
    assert all(packed_shape == (1, 2, 1, 64, 8) for packed_shape, _ in sparse_calls)
    assert all(splits == 1 for _, splits in sparse_calls)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
