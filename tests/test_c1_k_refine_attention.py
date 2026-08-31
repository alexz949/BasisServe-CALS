from __future__ import annotations

from argparse import Namespace
import hashlib
import json
import math

import pytest
from safetensors.torch import save_file
import torch

from basisserve.core.c1_k_refine import (
    GQAKProxyFactors,
    GPUExactKeyPageStore,
    KProxyConfig,
    c1_k_refine_attention_reference,
    c1_k_refine_attention_streaming,
    k_refine_quality_statistics,
    project_proxy_key,
)
from evaluation.eval_qwen3_c1_k_refine_oracle import evaluate
from evaluation.fit_qwen3_c1_k_proxy import fit


def _case(
    *,
    sequence: int = 7,
    head_dim: int = 4,
    proxy_rank: int = 2,
    value_rank: int = 3,
    dtype: torch.dtype = torch.float64,
):
    generator = torch.Generator().manual_seed(20260826)
    batch, query_heads, kv_heads = 2, 4, 2
    query = torch.randn(
        batch, query_heads, 1, head_dim, generator=generator, dtype=dtype
    )
    exact_key = torch.randn(
        batch, kv_heads, sequence, head_dim, generator=generator, dtype=dtype
    )
    c1_value = torch.randn(
        batch, kv_heads, sequence, value_rank, generator=generator, dtype=dtype
    )
    key_encoder = torch.randn(
        kv_heads, head_dim, proxy_rank, generator=generator, dtype=dtype
    )
    query_encoder = torch.randn(
        query_heads, head_dim, proxy_rank, generator=generator, dtype=dtype
    )
    factors = GQAKProxyFactors(key_encoder, query_encoder)
    proxy_key = project_proxy_key(exact_key, factors)
    return query, exact_key, proxy_key, c1_value, factors


def _config(
    *,
    rank: int = 2,
    page_size: int = 3,
    budget: int = 0,
    recent: int = 0,
    policy: str = "proxy_exact_refine",
    mode: str = "max",
) -> KProxyConfig:
    return KProxyConfig(
        proxy_rank=rank,
        page_size=page_size,
        exact_token_budget=budget,
        recent_exact_window=recent,
        page_score_mode=mode,
        score_policy=policy,
        proxy_dtype="float32",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("proxy_rank", 0, "proxy rank"),
        ("page_size", 0, "page size"),
        ("exact_token_budget", -1, "budget"),
        ("recent_exact_window", -1, "recent"),
        ("proxy_dtype", "int8", "dtype"),
    ],
)
def test_config_validation(field: str, value: object, message: str) -> None:
    values = dict(
        proxy_rank=2,
        page_size=2,
        exact_token_budget=0,
        recent_exact_window=0,
        proxy_dtype="float32",
    )
    values[field] = value
    with pytest.raises(ValueError, match=message):
        KProxyConfig(**values).validate(4)


def test_page_budget_rounds_up_deterministically() -> None:
    assert _config(page_size=4, budget=0).page_budget == 0
    assert _config(page_size=4, budget=1).page_budget == 1
    assert _config(page_size=4, budget=5).page_budget == 2


def test_gpu_page_store_returns_padded_pages_and_deduplicates_reads() -> None:
    key = torch.arange(1 * 2 * 5 * 3, dtype=torch.float32).reshape(1, 2, 5, 3)
    store = GPUExactKeyPageStore(key, layer_idx=7)
    observed = store.get_pages(
        layer_idx=7,
        batch_indices=torch.tensor([0, 0, 0]),
        kv_head_indices=torch.tensor([1, 0, 1]),
        page_ids=torch.tensor([1, 0, 1]),
        page_size=3,
        device=torch.device("cpu"),
    )
    assert observed.shape == (3, 3, 3)
    torch.testing.assert_close(observed[0, :2], key[0, 1, 3:5])
    torch.testing.assert_close(observed[0], observed[2])
    assert torch.count_nonzero(observed[0, 2]) == 0
    assert store.last_request_count == 3
    assert store.last_unique_request_count == 2


def test_proxy_key_has_one_physical_copy_per_kv_head() -> None:
    _, exact_key, proxy_key, _, factors = _case()
    assert proxy_key.shape[:3] == exact_key.shape[:3]
    assert proxy_key.shape[1] == factors.key_encoder.shape[0] == 2
    assert factors.query_encoders.shape[0] == 4


def test_full_exact_matches_direct_gqa_attention_with_c1_values() -> None:
    query, exact_key, proxy_key, value, factors = _case()
    result = c1_k_refine_attention_reference(
        query,
        exact_key,
        proxy_key,
        value,
        factors,
        _config(policy="full_exact"),
    )
    head_to_kv = torch.tensor([0, 0, 1, 1])
    scores = torch.einsum(
        "bhd,bhsd->bhs",
        query[:, :, 0],
        exact_key.index_select(1, head_to_kv),
    ) / math.sqrt(query.shape[-1])
    expected = torch.einsum(
        "bhs,bhsv->bhv",
        torch.softmax(scores.float(), dim=-1),
        value.index_select(1, head_to_kv).float(),
    )[:, :, None].to(value.dtype)
    torch.testing.assert_close(result.output, expected, rtol=1e-6, atol=1e-7)
    assert result.output.shape == (2, 4, 1, 3)


def test_all_page_refinement_matches_full_exact() -> None:
    query, exact_key, proxy_key, value, factors = _case(sequence=7)
    refined = c1_k_refine_attention_reference(
        query,
        exact_key,
        proxy_key,
        value,
        factors,
        _config(page_size=3, budget=7),
    )
    exact = c1_k_refine_attention_reference(
        query,
        exact_key,
        proxy_key,
        value,
        factors,
        _config(page_size=3, policy="full_exact"),
    )
    torch.testing.assert_close(refined.output, exact.output, rtol=0, atol=0)
    torch.testing.assert_close(refined.mixed_scores, exact.mixed_scores, rtol=0, atol=0)


def test_zero_page_refinement_matches_proxy_only() -> None:
    case = _case()
    refined = c1_k_refine_attention_reference(*case, _config(budget=0))
    proxy = c1_k_refine_attention_reference(
        *case, _config(budget=6, policy="proxy_only")
    )
    torch.testing.assert_close(refined.output, proxy.output, rtol=0, atol=0)
    torch.testing.assert_close(refined.mixed_scores, proxy.mixed_scores, rtol=0, atol=0)
    assert refined.selected_token_count == 0


def test_selected_scores_are_replaced_and_unselected_scores_remain_proxy() -> None:
    query, exact_key, proxy_key, value, factors = _case(sequence=6)
    result = c1_k_refine_attention_reference(
        query,
        exact_key,
        proxy_key,
        value,
        factors,
        _config(page_size=2, budget=2),
    )
    exact = c1_k_refine_attention_reference(
        query,
        exact_key,
        proxy_key,
        value,
        factors,
        _config(page_size=2, policy="full_exact"),
    )
    for batch in range(2):
        for group in range(2):
            selected_page = int(result.selected_page_ids[batch, group, 0])
            heads = slice(group * 2, group * 2 + 2)
            tokens = slice(selected_page * 2, selected_page * 2 + 2)
            torch.testing.assert_close(
                result.mixed_scores[batch, heads, tokens],
                exact.mixed_scores[batch, heads, tokens],
            )
            unselected = torch.ones(6, dtype=torch.bool)
            unselected[tokens] = False
            torch.testing.assert_close(
                result.mixed_scores[batch, heads, unselected],
                result.proxy_scores[batch, heads, unselected],
            )


def test_quality_metrics_and_logical_counts_reach_exact_endpoint() -> None:
    query, exact_key, proxy_key, value, factors = _case(sequence=6)
    config = _config(page_size=2, budget=6)
    candidate = c1_k_refine_attention_reference(
        query, exact_key, proxy_key, value, factors, config
    )
    exact = c1_k_refine_attention_reference(
        query,
        exact_key,
        proxy_key,
        value,
        factors,
        _config(page_size=2, policy="full_exact"),
    )
    decoder = torch.randn(
        4, 3, 5, generator=torch.Generator().manual_seed(11), dtype=torch.float64
    )
    metrics = k_refine_quality_statistics(
        exact,
        candidate,
        config=config,
        head_dim=4,
        num_kv_heads=2,
        decoder=decoder,
    )
    assert metrics["mixed_raw_score_rmse"] == 0
    assert metrics["attention_kl_exact_to_mixed"] == 0
    assert metrics["c1_latent_relative_l2"] == 0
    assert metrics["c1_decoded_output_relative_l2"] == 0
    assert candidate.statistics["selected_token_fraction"] == 1
    assert candidate.statistics["estimated_key_bytes_avoided"] == 0


def test_sparse_exact_masks_only_unselected_valid_tokens() -> None:
    case = _case(sequence=6)
    result = c1_k_refine_attention_reference(
        *case, _config(page_size=2, budget=2, policy="sparse_exact")
    )
    assert torch.isfinite(result.mixed_scores).sum() == 2 * 4 * 2
    assert torch.isneginf(result.mixed_scores).sum() == 2 * 4 * 4
    exact = c1_k_refine_attention_reference(
        *case, _config(page_size=2, policy="full_exact")
    )
    metrics = k_refine_quality_statistics(
        exact,
        result,
        config=_config(page_size=2, budget=2, policy="sparse_exact"),
        head_dim=4,
        num_kv_heads=2,
    )
    assert metrics["mixed_raw_score_rmse"] == 0
    assert metrics["mixed_score_pearson"] == pytest.approx(1)
    assert metrics["mixed_score_evaluated_token_fraction"] == pytest.approx(1 / 3)


def test_sparse_exact_rejects_an_empty_support() -> None:
    with pytest.raises(ValueError, match="at least one selected"):
        c1_k_refine_attention_reference(
            *_case(), _config(budget=0, policy="sparse_exact")
        )


def test_recent_window_is_forced_and_ranked_pages_fill_remaining_budget() -> None:
    case = _case(sequence=9)
    result = c1_k_refine_attention_reference(
        *case, _config(page_size=3, budget=6, recent=2)
    )
    assert result.selected_page_ids.shape == (2, 2, 2)
    assert torch.all((result.selected_page_ids == 2).any(dim=-1))
    assert result.selected_token_count == 2 * 2 * 6


def test_padding_pages_are_not_selected_and_budget_is_page_granular() -> None:
    query, exact_key, proxy_key, value, factors = _case(sequence=7)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    result = c1_k_refine_attention_reference(
        query,
        exact_key,
        proxy_key,
        value,
        factors,
        _config(page_size=3, budget=4),
        mask,
    )
    assert not torch.any(result.selected_page_ids == 2)
    assert result.selected_token_count == 2 * 2 * 4


def test_gqa_page_selection_takes_max_across_consuming_query_heads() -> None:
    # One physical K head, two Q heads: each head makes a different page hot.
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    exact_key = torch.zeros(1, 1, 4, 2)
    proxy_key = torch.tensor([[[[10.0, 0.0], [9.0, 0.0], [0.0, 8.0], [0.0, 7.0]]]])
    value = torch.randn(1, 1, 4, 1, generator=torch.Generator().manual_seed(3))
    factors = GQAKProxyFactors(torch.eye(2)[None], torch.stack((torch.eye(2), torch.eye(2))))
    result = c1_k_refine_attention_reference(
        query,
        exact_key,
        proxy_key,
        value,
        factors,
        _config(rank=2, page_size=2, budget=2),
    )
    # Page 0 wins the physical max (10 versus 8); it is fetched once for both heads.
    assert result.selected_page_ids.tolist() == [[[0]]]


class _SpyPageStore:
    def __init__(self, exact_key: torch.Tensor) -> None:
        self.delegate = GPUExactKeyPageStore(exact_key)
        self.calls = 0

    def get_pages(self, **kwargs) -> torch.Tensor:
        self.calls += 1
        return self.delegate.get_pages(**kwargs)


def test_attention_reads_exact_keys_through_page_store_protocol() -> None:
    query, exact_key, proxy_key, value, factors = _case()
    store = _SpyPageStore(exact_key)
    result = c1_k_refine_attention_reference(
        query,
        None,
        proxy_key,
        value,
        factors,
        _config(budget=3),
        page_store=store,
    )
    assert store.calls == 1
    assert result.selected_token_count > 0


@pytest.mark.parametrize("mode", ["max", "logsumexp"])
@pytest.mark.parametrize("policy", ["proxy_only", "proxy_exact_refine", "sparse_exact", "full_exact"])
def test_streaming_matches_materialized(mode: str, policy: str) -> None:
    case = _case(sequence=7, dtype=torch.float32)
    config = _config(page_size=3, budget=3, mode=mode, policy=policy)
    materialized = c1_k_refine_attention_reference(*case, config)
    streaming = c1_k_refine_attention_streaming(*case, config)
    torch.testing.assert_close(streaming.output, materialized.output, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        streaming.selected_page_ids, materialized.selected_page_ids, rtol=0, atol=0
    )
    assert streaming.proxy_scores is None
    assert streaming.mixed_scores is None
    assert streaming.running_lse.shape == (2, 4)


def test_streaming_handles_mask_and_non_aligned_last_page() -> None:
    case = _case(sequence=7, dtype=torch.float32)
    mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool
    )
    config = _config(page_size=3, budget=3, recent=2)
    materialized = c1_k_refine_attention_reference(*case, config, mask)
    streaming = c1_k_refine_attention_streaming(*case, config, mask)
    torch.testing.assert_close(streaming.output, materialized.output, rtol=2e-6, atol=2e-6)
    assert torch.isfinite(streaming.output).all()


@pytest.mark.parametrize("magnitude", [1.0e4, -1.0e4])
def test_online_softmax_stays_finite_for_large_scores(magnitude: float) -> None:
    query, exact_key, proxy_key, value, factors = _case(dtype=torch.float32)
    query.fill_(magnitude)
    exact_key.mul_(100.0)
    proxy_key = project_proxy_key(exact_key, factors)
    result = c1_k_refine_attention_streaming(
        query,
        exact_key,
        proxy_key,
        value,
        factors,
        _config(budget=3),
    )
    assert torch.isfinite(result.output).all()
    assert torch.isfinite(result.running_lse).all()


def test_deterministic_inputs_produce_deterministic_selection() -> None:
    case = _case()
    first = c1_k_refine_attention_reference(*case, _config(budget=3))
    second = c1_k_refine_attention_reference(*case, _config(budget=3))
    torch.testing.assert_close(first.selected_page_ids, second.selected_page_ids)


def test_query_blocks_longer_than_one_are_rejected() -> None:
    query, exact_key, proxy_key, value, factors = _case()
    with pytest.raises(ValueError, match=r"\[batch, query heads, 1"):
        c1_k_refine_attention_reference(
            query.expand(-1, -1, 2, -1),
            exact_key,
            proxy_key,
            value,
            factors,
            _config(),
        )


def test_tiny_factor_export_and_layer_oracle_emit_json_and_markdown(tmp_path) -> None:
    generator = torch.Generator().manual_seed(14)
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    pair_heads = torch.arange(4).repeat_interleave(24)
    layer_path = capture_dir / "layer_000.safetensors"
    save_file(
        {
            "query": torch.randn(2, 4, 1, 3, generator=generator),
            "exact_key": torch.randn(2, 2, 5, 3, generator=generator),
            "c1_value": torch.randn(2, 2, 5, 2, generator=generator),
            "attention_mask": torch.ones(2, 5, dtype=torch.bool),
            "decoder": torch.randn(4, 2, 6, generator=generator),
            "fit_pair_query": torch.randn(96, 3, generator=generator),
            "fit_pair_key": torch.randn(96, 3, generator=generator),
            "fit_pair_query_head": pair_heads,
        },
        str(layer_path),
    )
    layer_sha = hashlib.sha256(layer_path.read_bytes()).hexdigest()
    (capture_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "basisserve.qwen3_8b.c1_k_refine_capture.v1",
                "artifacts": {
                    "0": {"file": layer_path.name, "sha256": layer_sha}
                },
                "calibration": {"pair_sampling": {"weight_mode": "uniform"}},
            }
        ),
        encoding="utf-8",
    )
    factor_dir = tmp_path / "factors"
    fit(
        Namespace(
            pair_bank=capture_dir,
            proxy_ranks="2",
            layers="all",
            num_query_heads=4,
            num_kv_heads=2,
            initialization="als",
            als_sweeps=1,
            ridge=1.0e-5,
            cg_iterations=24,
            cg_tolerance=1.0e-7,
            accumulation_dtype="float64",
            model_path=None,
            c1_export=None,
            output_dir=factor_dir,
        )
    )
    factor_manifest = json.loads(
        (factor_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert factor_manifest["query_head_to_kv_head"] == [0, 0, 1, 1]
    assert factor_manifest["artifacts"]["2"]["0"]["final_objective"] <= factor_manifest[
        "artifacts"
    ]["2"]["0"]["initial_objective"]

    output_json = tmp_path / "oracle.json"
    output_markdown = tmp_path / "oracle.md"
    evaluate(
        Namespace(
            capture=capture_dir,
            c1_export=None,
            k_proxy_export=factor_dir,
            layers="all",
            proxy_ranks="2",
            page_sizes="2",
            exact_token_budgets="0,2",
            recent_exact_windows="0",
            page_score_modes="max",
            policies="proxy_only,proxy_exact_refine",
            proxy_dtype="float32",
            attention_top_k=2,
            device="cpu",
            output_json=output_json,
            output_markdown=output_markdown,
        )
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["format"] == "basisserve.c1_k_refine_oracle.v1"
    assert len(payload["records"]) == 4
    assert "c1_latent_relative_l2" in payload["aggregate"]
    markdown = output_markdown.read_text(encoding="utf-8")
    assert "| logical K bytes |" in markdown
