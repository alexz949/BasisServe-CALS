from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import save_file
import torch
from torch import nn

from basisserve.core.c1_shadow_kv import C1ShadowKeyValueCache, ShadowKeyConfig
from basisserve.core.c1_speculative_decode import (
    BlockCommitConfig,
    c1_shadow_greedy_decode,
    exact_c1_greedy_decode,
    transactional_exact_c1_greedy_decode,
)


class TinyDeterministicCacheLM(nn.Module):
    """A cache-sensitive causal LM with K and V head dimensions that differ."""

    def __init__(self, *, num_layers: int = 2, vocab_size: int = 29) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.register_buffer("device_anchor", torch.zeros(()))

    @staticmethod
    def _keys(input_ids: torch.Tensor, layer: int) -> torch.Tensor:
        values = input_ids.to(torch.float32)
        coordinates = torch.stack(
            (
                0.31 * values + 0.17 + layer,
                0.07 * values.square() + 0.23,
                0.19 * values + 0.41,
                0.11 * (values + 1).square() + 0.13,
            ),
            dim=-1,
        )
        return coordinates.unsqueeze(1)

    @staticmethod
    def _values(input_ids: torch.Tensor, layer: int) -> torch.Tensor:
        values = input_ids.to(torch.float32)
        return torch.stack(
            (0.5 * values + layer, 0.25 * values - layer),
            dim=-1,
        ).unsqueeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values,
        use_cache: bool,
        **kwargs,
    ) -> SimpleNamespace:
        del kwargs
        assert use_cache
        visible_keys = None
        for layer in range(self.num_layers):
            keys, values = past_key_values.update(
                self._keys(input_ids, layer),
                self._values(input_ids, layer),
                layer,
                {"cache_position": None},
            )
            assert values.shape[-1] == 2
            if layer == 0:
                visible_keys = keys
        assert visible_keys is not None
        query_tokens = int(input_ids.shape[1])
        base = int(visible_keys.shape[-2]) - query_tokens
        logits = torch.full(
            (1, query_tokens, self.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        for index in range(query_tokens):
            prefix = visible_keys[..., : base + index + 1, :]
            context_code = torch.floor(prefix.sum(dim=(-1, -2)) * 10).to(torch.long)
            predicted = (
                input_ids[:, index].to(torch.long) + context_code[:, 0] + 3
            ) % self.vocab_size
            logits[:, index].scatter_(1, predicted.unsqueeze(1), 100.0)
        return SimpleNamespace(logits=logits)


class IntentionallyBadShadowCache(C1ShadowKeyValueCache):
    def begin_draft(self) -> None:
        for layer in self.shadow_layers:
            if layer.shadow_key_committed is not None:
                layer.shadow_key_committed.values.zero_()
        super().begin_draft()


def test_transactional_exact_greedy_commits_one_token_target_state() -> None:
    model = TinyDeterministicCacheLM()
    prompt = torch.tensor([[2, 5, 1, 7]])
    expected = exact_c1_greedy_decode(model, prompt, max_new_tokens=9)
    cache = C1ShadowKeyValueCache(
        num_layers=model.num_layers,
        config=ShadowKeyConfig(bits=16, group_size=2, recent_exact_window=0),
    )

    observed = transactional_exact_c1_greedy_decode(
        model,
        prompt,
        cache=cache,
        max_new_tokens=9,
    )

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
    assert cache.committed_length == prompt.shape[1] + 9


def test_identity_shadow_accepts_every_block_and_catches_logit_alignment() -> None:
    model = TinyDeterministicCacheLM()
    prompt = torch.tensor([[2, 5, 1, 7]])
    expected = exact_c1_greedy_decode(model, prompt, max_new_tokens=11)
    cache = C1ShadowKeyValueCache(
        num_layers=model.num_layers,
        config=ShadowKeyConfig(bits=16, group_size=2, recent_exact_window=0),
    )

    observed = c1_shadow_greedy_decode(
        model,
        prompt,
        cache=cache,
        max_new_tokens=11,
        draft_length=4,
    )

    torch.testing.assert_close(observed.token_ids, expected, rtol=0.0, atol=0.0)
    assert len(observed.rounds) >= 3
    assert all(item.full_block_accepted for item in observed.rounds)
    assert all(item.first_rejection_index is None for item in observed.rounds)
    assert observed.metrics.target_correction_sync_calls == 0
    assert observed.metrics.target_commit_sync_calls == 11
    assert observed.metrics.target_accepted_replay_calls == 11
    assert observed.metrics.directly_committed_target_tokens == 0
    assert observed.metrics.verifier_top1_agreement == 1.0
    assert observed.metrics.mean_shadow_logit_kl == 0.0
    assert observed.metrics.block_sequential_top1_agreement == 1.0
    assert cache.committed_length == observed.token_ids.shape[1]


def test_bad_shadow_rejects_but_exact_correction_preserves_target_sequence() -> None:
    model = TinyDeterministicCacheLM()
    prompt = torch.tensor([[4, 3, 8]])
    expected = exact_c1_greedy_decode(model, prompt, max_new_tokens=13)
    cache = IntentionallyBadShadowCache(
        num_layers=model.num_layers,
        config=ShadowKeyConfig(bits=4, group_size=2, recent_exact_window=0),
    )

    observed = c1_shadow_greedy_decode(
        model,
        prompt,
        cache=cache,
        max_new_tokens=13,
        draft_length=5,
    )

    torch.testing.assert_close(observed.token_ids, expected, rtol=0.0, atol=0.0)
    rejected = [item for item in observed.rounds if not item.full_block_accepted]
    assert rejected
    assert all(item.accepted_length >= 1 for item in rejected)
    assert any(item.shadow_continuation_accepted == 0 for item in rejected)
    assert observed.metrics.rollback_count == len(rejected)
    assert observed.metrics.target_correction_sync_calls == len(rejected)
    assert observed.metrics.target_commit_sync_calls == 13
    assert (
        observed.metrics.target_accepted_replay_calls
        + observed.metrics.target_correction_sync_calls
        == 13
    )
    assert observed.metrics.verifier_top1_agreement < 1.0
    assert cache.committed_length == observed.token_ids.shape[1]
    for layer in cache.shadow_layers:
        assert layer.draft_key_provisional is None
        assert layer.draft_c1_value_provisional is None
        assert layer.target_key_pending is None
        assert layer.target_c1_value_pending is None


def test_direct_block_full_acceptance_commits_without_accepted_token_replay() -> None:
    model = TinyDeterministicCacheLM()
    prompt = torch.tensor([[2, 5, 1, 7]])
    expected = exact_c1_greedy_decode(model, prompt, max_new_tokens=8)
    cache = C1ShadowKeyValueCache(
        num_layers=model.num_layers,
        config=ShadowKeyConfig(bits=16, group_size=2, recent_exact_window=0),
    )

    observed = c1_shadow_greedy_decode(
        model,
        prompt,
        cache=cache,
        max_new_tokens=8,
        draft_length=4,
        block_commit=BlockCommitConfig(policy="direct_block"),
    )

    torch.testing.assert_close(observed.token_ids, expected, rtol=0.0, atol=0.0)
    assert observed.commit_policy == "direct_block"
    assert observed.metrics.commit_policy == "direct_block"
    assert observed.metrics.target_verification_calls == 2
    assert observed.metrics.target_commit_sync_calls == 0
    assert observed.metrics.target_accepted_replay_calls == 0
    assert observed.metrics.target_correction_sync_calls == 0
    assert observed.metrics.directly_committed_target_tokens == 8
    assert observed.metrics.sequentially_committed_target_tokens == 0
    assert all(item.directly_committed_length == 4 for item in observed.rounds)
    assert all(item.sequentially_committed_length == 0 for item in observed.rounds)
    assert cache.committed_length == prompt.shape[1] + 8


def test_direct_block_rejection_replays_only_the_correction_token() -> None:
    model = TinyDeterministicCacheLM()
    prompt = torch.tensor([[4, 3, 8]])
    expected = exact_c1_greedy_decode(model, prompt, max_new_tokens=13)
    cache = IntentionallyBadShadowCache(
        num_layers=model.num_layers,
        config=ShadowKeyConfig(bits=4, group_size=2, recent_exact_window=0),
    )

    observed = c1_shadow_greedy_decode(
        model,
        prompt,
        cache=cache,
        max_new_tokens=13,
        draft_length=5,
        block_commit=BlockCommitConfig(policy="direct_block"),
    )

    torch.testing.assert_close(observed.token_ids, expected, rtol=0.0, atol=0.0)
    rejected = [item for item in observed.rounds if not item.full_block_accepted]
    assert rejected
    assert observed.metrics.target_accepted_replay_calls == 0
    assert observed.metrics.target_commit_sync_calls == len(rejected)
    assert observed.metrics.target_correction_sync_calls == len(rejected)
    assert observed.metrics.sequentially_committed_target_tokens == len(rejected)
    assert (
        observed.metrics.directly_committed_target_tokens
        + observed.metrics.sequentially_committed_target_tokens
        == 13
    )
    assert all(item.sequentially_committed_length == 1 for item in rejected)
    assert cache.committed_length == prompt.shape[1] + 13


def test_direct_block_length_one_reduces_to_one_target_block_per_token() -> None:
    model = TinyDeterministicCacheLM()
    prompt = torch.tensor([[3, 1, 4]])
    expected = exact_c1_greedy_decode(model, prompt, max_new_tokens=6)
    cache = C1ShadowKeyValueCache(
        num_layers=model.num_layers,
        config=ShadowKeyConfig(bits=16, group_size=2, recent_exact_window=0),
    )

    observed = c1_shadow_greedy_decode(
        model,
        prompt,
        cache=cache,
        max_new_tokens=6,
        draft_length=1,
        block_commit=BlockCommitConfig(policy="direct_block"),
    )

    torch.testing.assert_close(observed.token_ids, expected, rtol=0.0, atol=0.0)
    assert observed.metrics.target_verification_calls == 6
    assert observed.metrics.target_commit_sync_calls == 0
    assert observed.metrics.directly_committed_target_tokens == 6


def test_max_new_tokens_zero_does_not_touch_model_or_cache() -> None:
    model = TinyDeterministicCacheLM()
    prompt = torch.tensor([[1, 2]])
    cache = C1ShadowKeyValueCache(
        num_layers=model.num_layers,
        config=ShadowKeyConfig(bits=16, group_size=2, recent_exact_window=0),
    )
    result = c1_shadow_greedy_decode(
        model,
        prompt,
        cache=cache,
        max_new_tokens=0,
        draft_length=4,
    )
    torch.testing.assert_close(result.token_ids, prompt)
    assert result.generated_token_ids == ()
    assert cache.committed_length == 0


def test_transformers_qwen3_c1_identity_shadow_matches_exact_greedy(
    tmp_path: Path,
) -> None:
    """Exercise the real Qwen3 mask/cache path with compressed C1 values."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from basisserve.checkpoint.gqa_vo_qwen3 import (
        GQATiedVOQwen3Attention,
        install_qwen3_gqa_vo_als_export,
    )
    from basisserve.core.c1_block_commit import (
        compare_exact_block_and_sequential_schedules,
    )
    from basisserve.core.c1_tp_decode import file_sha256

    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=37,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    value_head_dim = 2
    original_v_weights = [
        layer.self_attn.v_proj.weight.detach().clone() for layer in model.model.layers
    ]
    encoders_by_layer = []
    artifacts = {}
    for layer_index in range(config.num_hidden_layers):
        encoders = (
            torch.randn(
                config.num_key_value_heads,
                config.head_dim,
                value_head_dim,
            )
            / config.head_dim**0.5
        )
        decoders = (
            torch.randn(
                config.num_attention_heads,
                value_head_dim,
                config.hidden_size,
            )
            / value_head_dim**0.5
        )
        encoders_by_layer.append(encoders)
        artifact_path = tmp_path / f"layer_{layer_index:03d}.safetensors"
        save_file(
            {
                "value_coordinate_encoders": encoders,
                "head_output_decoders": decoders,
            },
            str(artifact_path),
        )
        artifacts[str(layer_index)] = {
            "file": artifact_path.name,
            "sha256": file_sha256(artifact_path),
        }
    manifest = {
        "format": "basisserve.qwen3_32b.gqa_c1_v96_joint.v1",
        "status": "complete",
        "layers": list(range(config.num_hidden_layers)),
        "fit_config": {
            "model_type": "qwen3",
            "hidden_size": config.hidden_size,
            "num_query_heads": config.num_attention_heads,
            "num_physical_kv_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "cache_rank_per_head": value_head_dim,
        },
        "artifacts": artifacts,
    }
    (tmp_path / "results.json").write_text(json.dumps(manifest), encoding="utf-8")

    records = install_qwen3_gqa_vo_als_export(model, tmp_path)
    assert len(records) == config.num_hidden_layers
    assert all(record.value_head_dim == value_head_dim for record in records)
    assert all(
        isinstance(layer.self_attn, GQATiedVOQwen3Attention)
        for layer in model.model.layers
    )
    expected_v = torch.bmm(
        encoders_by_layer[0].transpose(1, 2),
        original_v_weights[0].reshape(
            config.num_key_value_heads,
            config.head_dim,
            config.hidden_size,
        ),
    ).reshape(config.num_key_value_heads * value_head_dim, config.hidden_size)
    torch.testing.assert_close(
        model.model.layers[0].self_attn.v_proj.weight, expected_v
    )

    prompt = torch.tensor([[3, 9, 4, 12]], dtype=torch.long)
    dense_logits = model(input_ids=prompt, use_cache=False).logits
    from basisserve.core.c1_k_reverse_shadow import ReverseShadowConfig

    full_reverse = ReverseShadowConfig(
        page_size=2,
        exact_token_budget=4,
        selector="quest_minmax",
        landmark_dtype="float32",
    )
    for layer in model.model.layers:
        layer.self_attn.set_reverse_shadow_config(full_reverse)
    reverse_logits = model(input_ids=prompt, use_cache=False).logits
    torch.testing.assert_close(reverse_logits, dense_logits, rtol=1.0e-5, atol=1.0e-5)
    assert all(
        layer.self_attn.reverse_shadow_statistics()["queries"] == len(prompt[0])
        for layer in model.model.layers
    )
    for layer in model.model.layers:
        layer.self_attn.set_reverse_shadow_config(None)

    expected = exact_c1_greedy_decode(model, prompt, max_new_tokens=7)
    cache = C1ShadowKeyValueCache(
        num_layers=config.num_hidden_layers,
        config=ShadowKeyConfig(bits=16, group_size=2, recent_exact_window=0),
    )
    observed = c1_shadow_greedy_decode(
        model,
        prompt,
        cache=cache,
        max_new_tokens=7,
        draft_length=3,
    )

    torch.testing.assert_close(observed.token_ids, expected, rtol=0.0, atol=0.0)
    assert all(round_.full_block_accepted for round_ in observed.rounds)
    assert observed.metrics.target_correction_sync_calls == 0
    assert observed.metrics.target_commit_sync_calls == 7
    assert cache.committed_length == prompt.shape[1] + 7

    comparison = compare_exact_block_and_sequential_schedules(
        model,
        expected,
        sequential_cache=C1ShadowKeyValueCache(
            num_layers=config.num_hidden_layers,
            config=ShadowKeyConfig(bits=16, group_size=2, recent_exact_window=0),
        ),
        block_cache=C1ShadowKeyValueCache(
            num_layers=config.num_hidden_layers,
            config=ShadowKeyConfig(bits=16, group_size=2, recent_exact_window=0),
        ),
        prefill_length=prompt.shape[1],
        block_length=3,
    )
    assert comparison.evaluated_tokens == 7
    assert comparison.sequential_label_top1_matches == 7
    assert len(comparison.cache_drift) == 6

    direct_cache = C1ShadowKeyValueCache(
        num_layers=config.num_hidden_layers,
        config=ShadowKeyConfig(bits=16, group_size=2, recent_exact_window=0),
    )
    direct = c1_shadow_greedy_decode(
        model,
        prompt,
        cache=direct_cache,
        max_new_tokens=7,
        draft_length=3,
        block_commit=BlockCommitConfig(policy="direct_block"),
    )
    assert direct.metrics.target_accepted_replay_calls == 0
    assert (
        direct.metrics.directly_committed_target_tokens
        + direct.metrics.sequentially_committed_target_tokens
        == 7
    )
    assert direct_cache.committed_length == prompt.shape[1] + 7
