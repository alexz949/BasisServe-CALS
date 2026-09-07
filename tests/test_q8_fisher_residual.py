import torch
from unittest.mock import patch

from basisserve.core.c1_v_conditional_k_router import AffineReducedRankMap
from basisserve.core.gqa_joint_routing_payload_s80_fisher import compact_softmax_fisher_loss
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _build_residual_statistics
from evaluation.fit_qwen3_8b_q8_fisher_residual import build_multi_query_statistics


def test_closed_form_base_refit_provenance_is_not_qaware():
    from evaluation.fit_qwen3_8b_q8_fisher_residual import frozen_base_specification, parser
    source = dict(format="basisserve.residual_kl_bank.v1", base_rank=16,
                  page_size=32, excluded_prefix_pages=1, fit_windows=64,
                  sequence_length=32768, fit_query="last token of each window")
    spec = frozen_base_specification(source, "closed_form_rrr")
    assert spec["format"] == "basisserve.closed_form_base_fisher_bank.v1"
    assert spec["base_query_positions"] == [] and "SVD" in spec["base_original_solver"]
    args = parser().parse_args(["--model", "model", "--c1-checkpoint", "c1",
                               "--base-kind", "closed_form_rrr"])
    assert args.base_kind == "closed_form_rrr"


def fixture():
    torch.manual_seed(319)
    rows = torch.randn(2, 12, 2, 8)
    queries = torch.randn(2, 2, 4, 4)
    angles = (torch.arange(12)[:, None] * torch.tensor([.2, .7])).repeat(1, 2)[None]
    maps = tuple(AffineReducedRankMap(torch.randn(3, 2), torch.randn(2, 4), torch.randn(4))
                 for _ in range(2))
    kwargs = dict(value_encoder=torch.randn(2, 4, 3), base_maps={16: maps},
                  cos=angles.cos(), sin=angles.sin(), page_size=2,
                  excluded_prefix_pages=1, device=torch.device("cpu"))
    return queries, rows, kwargs


def test_multi_query_pairs_and_loss_equal_sum_of_separate_causal_examples():
    queries, rows, kwargs = fixture()
    combined, _ = build_multi_query_statistics(queries, rows, query_positions=[5, 11], **kwargs)
    separate = []
    for index, position in enumerate([5, 11]):
        settings = dict(kwargs, cos=kwargs["cos"][:, :position + 1], sin=kwargs["sin"][:, :position + 1])
        one, _ = _build_residual_statistics(queries[:, index], rows[:, :position + 1], **settings)
        separate.append(one[16])
    stat = combined[16]
    assert stat.queries_by_head.shape == (4, 4, 4)
    for index, part in enumerate(separate):
        torch.testing.assert_close(stat.queries_by_head[:, index * 2:(index + 1) * 2], part.queries_by_head)
        torch.testing.assert_close(stat.fisher_grams_by_head[:, index * 2:(index + 1) * 2], part.fisher_grams_by_head)
    assert stat.teacher_fisher_energy == sum(p.teacher_fisher_energy for p in separate)
    factors = dict(routing_payload_encoders=torch.randn(2, 4, 2),
                   routing_query_factors=torch.randn(4, 4, 2))
    torch.testing.assert_close(compact_softmax_fisher_loss(stat, **factors),
                               sum(compact_softmax_fisher_loss(p, **factors) for p in separate),
                               rtol=1e-6, atol=1e-6)


def test_future_and_pinned_rows_do_not_enter_query_fisher():
    queries, rows, kwargs = fixture()
    original, _ = build_multi_query_statistics(queries[:, :1], rows, query_positions=[5], **kwargs)
    changed = rows.clone()
    changed[:, :2] += 100
    changed[:, 6:] -= 100
    modified, _ = build_multi_query_statistics(queries[:, :1], changed, query_positions=[5], **kwargs)
    torch.testing.assert_close(original[16].fisher_grams_by_head, modified[16].fisher_grams_by_head,
                               rtol=0, atol=0)
    assert original[16].teacher_fisher_energy == modified[16].teacher_fisher_energy


def test_single_terminal_query_reproduces_existing_builder():
    queries, rows, kwargs = fixture()
    old, old_reconstruction = _build_residual_statistics(queries[:, -1], rows, **kwargs)
    new, reconstruction = build_multi_query_statistics(queries[:, -1:], rows, query_positions=[11], **kwargs)
    torch.testing.assert_close(old[16].queries_by_head, new[16].queries_by_head, rtol=0, atol=0)
    torch.testing.assert_close(old[16].fisher_grams_by_head, new[16].fisher_grams_by_head, rtol=0, atol=0)
    assert old[16].teacher_fisher_energy == new[16].teacher_fisher_energy
    assert old_reconstruction == reconstruction["11"]


def test_query_order_does_not_change_fisher_objective():
    queries, rows, kwargs = fixture()
    first, _ = build_multi_query_statistics(queries, rows, query_positions=[5, 11], **kwargs)
    second, _ = build_multi_query_statistics(queries.flip(1), rows, query_positions=[11, 5], **kwargs)
    factors = dict(routing_payload_encoders=torch.randn(2, 4, 2),
                   routing_query_factors=torch.randn(4, 4, 2))
    torch.testing.assert_close(compact_softmax_fisher_loss(first[16], **factors),
                               compact_softmax_fisher_loss(second[16], **factors),
                               rtol=1e-6, atol=1e-6)


def test_window_major_features_are_computed_once_per_document_and_base():
    import evaluation.fit_qwen3_8b_q8_fisher_residual as builder
    queries, rows, kwargs = fixture()
    kwargs["base_maps"][8] = kwargs["base_maps"][16]
    rows_before, queries_before = rows.clone(), queries.clone()
    with patch.object(builder, "_value_codes", wraps=builder._value_codes) as codes, \
         patch.object(builder, "_apply_base", wraps=builder._apply_base) as base, \
         patch.object(builder, "_post_rope_rows", wraps=builder._post_rope_rows) as rope:
        result, _ = builder.build_multi_query_statistics(
            queries, rows, query_positions=[5, 11], **kwargs)
    assert codes.call_count == 2  # documents, not documents x queries
    assert base.call_count == rope.call_count == 4  # documents x Base alternatives
    assert set(result) == {8, 16}
    torch.testing.assert_close(rows, rows_before, rtol=0, atol=0)
    torch.testing.assert_close(queries, queries_before, rtol=0, atol=0)


def test_spread_unsorted_queries_and_partial_pages_match_single_query_oracle():
    _, rows, kwargs = fixture()
    queries = torch.randn(2, 3, 4, 4)
    positions = [10, 2, 6]  # full span; partial pages; not monotonically ordered
    combined, metrics = build_multi_query_statistics(queries, rows, query_positions=positions, **kwargs)
    energies = []
    for index, position in enumerate(positions):
        settings = dict(kwargs, cos=kwargs["cos"][:, :position + 1],
                        sin=kwargs["sin"][:, :position + 1])
        oracle, reconstruction = _build_residual_statistics(
            queries[:, index], rows[:, :position + 1], **settings)
        torch.testing.assert_close(combined[16].queries_by_head[:, index * 2:(index + 1) * 2],
                                   oracle[16].queries_by_head, rtol=0, atol=0)
        torch.testing.assert_close(combined[16].fisher_grams_by_head[:, index * 2:(index + 1) * 2],
                                   oracle[16].fisher_grams_by_head, rtol=2e-5, atol=2e-6)
        for key, value in reconstruction[16].items():
            torch.testing.assert_close(torch.tensor(metrics[str(position)][16][key]),
                                       torch.tensor(value), rtol=2e-5, atol=2e-6)
        energies.append(oracle[16].teacher_fisher_energy)
    torch.testing.assert_close(torch.tensor(combined[16].teacher_fisher_energy),
                               torch.tensor(sum(energies)), rtol=2e-5, atol=2e-6)
