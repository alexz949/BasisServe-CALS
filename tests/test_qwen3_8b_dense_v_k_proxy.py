import torch

from evaluation.eval_qwen3_8b_dense_v_k_proxy import (
    _aggregate,
    _accumulate_score_metrics,
    _finalize_score_sums,
    _new_score_sums,
    _record,
)


def test_exact_proxy_has_perfect_token_and_page_recall() -> None:
    exact_scores = torch.tensor(
        [[8.0, 7.0, 1.0, 0.0, 6.0, 5.0, 3.0, 2.0]]
    ).repeat(4, 1)
    sums = _new_score_sums([4])

    _accumulate_score_metrics(
        sums,
        exact_scores=exact_scores,
        proxy_scores=exact_scores.clone(),
        budgets=[4],
        page_size=2,
        page_token_budget=4,
        page_candidate_token_budget=6,
    )
    result = _finalize_score_sums(sums)

    assert result["budgets"]["4"]["mean_top_k_recall"] == 1.0
    assert result["page"]["mean_page_recall"] == 1.0
    assert result["page"]["mean_mass_page_recall"] == 1.0
    assert result["page"]["mean_top_k_covered_by_pages"] == 1.0
    assert result["page"]["mean_gqa_union_mass_page_recall"] == 1.0
    assert result["page"]["mean_gqa_union_amplification"] == 1.0
    assert result["page"]["mean_candidate_mass_page_recall"] == 1.0
    assert result["page"]["mean_reranked_mass_page_recall"] == 1.0
    assert result["page"]["mean_reranked_teacher_mass"] == result["page"][
        "mean_oracle_teacher_mass"
    ]
    assert torch.isclose(
        torch.tensor(result["page"]["mean_teacher_mass_selected"]),
        torch.tensor(result["page"]["mean_oracle_teacher_mass"]),
    )


def test_page_metrics_merge_across_layers() -> None:
    records = []
    for layer in (0, 17):
        scores = torch.tensor(
            [[8.0, 7.0, 1.0, 0.0, 6.0, 5.0, 3.0, 2.0]]
        ).repeat(4, 1)
        sums = _new_score_sums([4])
        _accumulate_score_metrics(
            sums,
            exact_scores=scores,
            proxy_scores=scores,
            budgets=[4],
            page_size=2,
            page_token_budget=4,
            page_candidate_token_budget=6,
        )
        metric_sums = {
            "squared_error": 0.0,
            "centered_energy": 1.0,
            "cosine_sum": 1.0,
            "vectors": 1,
        }
        records.append(
            _record(
                layer=layer,
                proxy="c1_v96_pre_rope",
                source="c1_v96",
                target="pre_rope_then_exact_rope",
                post_key_sums=metric_sums,
                pre_key_sums=metric_sums,
                score_sums=sums,
            )
        )

    aggregate = _aggregate(records, [4])

    assert aggregate[0]["score"]["budgets"]["4"]["mean_top_k_recall"] == 1.0
    assert aggregate[0]["score"]["page"]["mean_mass_page_recall"] == 1.0


def test_exact_rerank_recovers_oracle_pages_from_larger_candidate_set() -> None:
    exact_scores = torch.tensor(
        [[8.0, 7.0, 1.0, 0.0, 6.0, 5.0, 3.0, 2.0]]
    ).repeat(4, 1)
    proxy_scores = torch.tensor(
        [[8.0, 7.0, 7.0, 6.0, 5.0, 4.0, 0.0, -1.0]]
    ).repeat(4, 1)
    sums = _new_score_sums([4])

    _accumulate_score_metrics(
        sums,
        exact_scores=exact_scores,
        proxy_scores=proxy_scores,
        budgets=[4],
        page_size=2,
        page_token_budget=4,
        page_candidate_token_budget=6,
    )
    page = _finalize_score_sums(sums)["page"]

    assert page["mean_mass_page_recall"] == 0.5
    assert page["mean_candidate_mass_page_recall"] == 1.0
    assert page["mean_reranked_mass_page_recall"] == 1.0
    assert page["mean_reranked_teacher_mass"] == page[
        "mean_oracle_teacher_mass"
    ]
    assert page["mean_candidate_gqa_union_pages"] == 3.0
    assert page["mean_reranked_gqa_union_pages"] == 2.0
