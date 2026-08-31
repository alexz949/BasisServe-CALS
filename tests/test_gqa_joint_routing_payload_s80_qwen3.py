from __future__ import annotations

import copy

import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from basisserve.checkpoint.gqa_joint_routing_payload_s80_qwen3 import (
    S80LayerExport,
    S80Qwen3Attention,
    build_s80_joint_latent,
    compute_s80_routing_proxy_scores,
    install_qwen3_s80_factor_bank,
    load_s80_factor_bank,
    merge_s80_factor_banks,
    restore_qwen3_s80_layers,
    write_s80_factor_bank,
)
from basisserve.core.gqa_joint_routing_payload_s80 import (
    FoldedS80Factors,
    S80Layout,
)


def _config() -> Qwen3Config:
    return Qwen3Config(
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


def _exact_factors(attention) -> FoldedS80Factors:
    config = attention.config
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(attention.head_dim)
    routing_rank = head_dim // 2
    return FoldedS80Factors(
        v_joint_proj_weight=attention.v_proj.weight.detach().clone(),
        v_joint_proj_bias=(
            None
            if attention.v_proj.bias is None
            else attention.v_proj.bias.detach().clone()
        ),
        k_joint_encoder=torch.zeros(kv_heads, head_dim, head_dim),
        routing_query_factor=torch.randn(query_heads, head_dim, routing_rank),
        o_decoder_weight=attention.o_proj.weight.detach().clone(),
        o_decoder_bias=(
            None
            if attention.o_proj.bias is None
            else attention.o_proj.bias.detach().clone()
        ),
        head_to_kv_group=torch.arange(query_heads) // (query_heads // kv_heads),
    )


def test_joint_latent_and_routing_proxy_match_manual_factorization() -> None:
    torch.manual_seed(20260905)
    v_latent = torch.randn(2, 2, 5, 3, dtype=torch.float64)
    key = torch.randn(2, 2, 5, 4, dtype=torch.float64)
    key_encoder = torch.randn(2, 4, 3, dtype=torch.float64)
    joint = build_s80_joint_latent(v_latent, key, key_encoder)
    torch.testing.assert_close(
        joint,
        v_latent + torch.einsum("bgtd,gdr->bgtr", key, key_encoder),
    )

    query = torch.randn(2, 4, 3, 4, dtype=torch.float64)
    query_factor = torch.randn(4, 4, 2, dtype=torch.float64)
    scores = compute_s80_routing_proxy_scores(
        query,
        joint,
        query_factor,
        scaling=0.5,
    )
    selected_joint = joint[..., :2]
    expected = (
        torch.matmul(
            torch.einsum("bhnd,hdr->bhnr", query, query_factor),
            selected_joint.repeat_interleave(2, dim=1).transpose(-1, -2),
        )
        * 0.5
    )
    torch.testing.assert_close(scores, expected)


@torch.inference_mode()
def test_exact_v_s80_qwen3_matches_dense_full_and_cached() -> None:
    torch.manual_seed(20260906)
    dense = Qwen3ForCausalLM(_config()).float().eval()
    candidate = copy.deepcopy(dense)
    for layer in candidate.model.layers:
        attention = layer.self_attn
        layer.self_attn = S80Qwen3Attention(
            attention,
            factors=_exact_factors(attention),
            attention_backend="sdpa",
        )
    input_ids = torch.arange(13).view(1, 13) % dense.config.vocab_size
    dense_logits = dense(input_ids=input_ids, use_cache=False).logits
    candidate_logits = candidate(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(candidate_logits, dense_logits, rtol=2e-6, atol=2e-6)

    outputs = []
    for model in (dense, candidate):
        cache = DynamicCache()
        pieces = []
        for start, stop in ((0, 7), (7, 10), (10, 13)):
            pieces.append(
                model(
                    input_ids=input_ids[:, start:stop],
                    past_key_values=cache,
                    use_cache=True,
                ).logits
            )
        outputs.append(torch.cat(pieces, dim=1))
    torch.testing.assert_close(outputs[1], outputs[0], rtol=2e-6, atol=2e-6)


def test_checkpoint_round_trip_install_and_restore(tmp_path) -> None:
    torch.manual_seed(20260907)
    model = Qwen3ForCausalLM(_config()).float().eval()
    attention = model.model.layers[0].self_attn
    factors = _exact_factors(attention)
    layout = S80Layout(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        value_dim=8,
        key_dim=8,
        joint_rank=8,
        routing_rank=4,
    )
    root = tmp_path / "s80_bank"
    write_s80_factor_bank(
        root,
        layout=layout,
        layers=(
            S80LayerExport(
                layer_index=0,
                factors=factors,
                routing_weight=1.0,
                payload_normalizer=2.0,
                routing_normalizer=3.0,
                fit_diagnostics={"sweeps": 0},
            ),
        ),
        model_identifier="tiny-qwen3",
        model_config_sha256="test-config",
        initial_factor_sources={"payload": "identity", "routing": "random"},
        solver_configuration={"maximum_sweeps": 0},
        command="pytest",
        environment={"torch": torch.__version__},
    )
    manifest, loaded = load_s80_factor_bank(root)
    assert manifest["key_convention"] == "post_rope"
    assert manifest["routing_layout"] == {
        "stored_coordinates": 8,
        "routed_coordinates": 4,
        "routed_coordinate_start": 0,
        "selector_materialized": False,
    }
    assert "routing_latent_selector" not in manifest["artifacts"][0]["factor_shapes"]
    assert set(loaded) == {0}
    torch.testing.assert_close(
        loaded[0][0].v_joint_proj_weight,
        factors.v_joint_proj_weight,
    )

    original = model.model.layers[0].self_attn
    records = install_qwen3_s80_factor_bank(model, root, attention_backend="sdpa")
    assert len(records) == 1
    assert isinstance(model.model.layers[0].self_attn, S80Qwen3Attention)
    restore_qwen3_s80_layers(model, records)
    assert model.model.layers[0].self_attn is original


def test_merge_disjoint_checkpoint_shards(tmp_path) -> None:
    torch.manual_seed(20260908)
    model = Qwen3ForCausalLM(_config()).float().eval()
    layout = S80Layout(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        value_dim=8,
        key_dim=8,
        joint_rank=8,
        routing_rank=4,
    )
    inputs = []
    for layer_index in range(2):
        root = tmp_path / f"shard_{layer_index}"
        factors = _exact_factors(model.model.layers[layer_index].self_attn)
        write_s80_factor_bank(
            root,
            layout=layout,
            layers=(
                S80LayerExport(
                    layer_index=layer_index,
                    factors=factors,
                    routing_weight=1.0,
                    payload_normalizer=2.0,
                    routing_normalizer=3.0,
                    fit_diagnostics={"sweeps": 1},
                ),
            ),
            model_identifier="tiny-qwen3",
            model_config_sha256="test-config",
            initial_factor_sources={"statistics": f"shard-{layer_index}"},
            solver_configuration={
                "maximum_sweeps": 1,
                "elapsed_seconds_before_write": float(layer_index + 1),
            },
            command=f"build-shard-{layer_index}",
            environment={"torch": torch.__version__},
        )
        inputs.append(root)

    merged_root = tmp_path / "merged"
    merge_s80_factor_banks(
        merged_root,
        input_dirs=inputs,
        command="merge-test",
    )
    manifest, loaded = load_s80_factor_bank(merged_root)
    assert manifest["layer_coverage"] == [0, 1]
    assert len(manifest["source_banks"]) == 2
    assert set(loaded) == {0, 1}
