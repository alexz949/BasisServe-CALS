import torch

from basisserve.core.page_selection_comparison import selection_details, compare_selection


def test_identical_page_sets_and_mass():
    torch.manual_seed(91)
    details = selection_details(torch.randn(4, 40), page_size=4, page_budget=3)
    result = compare_selection(details, details)
    assert result["page_recall"] == result["non_sink_page_recall"] == 1
    assert result["missed_count"] == result["extra_count"] == 0
    assert result["exact_mass"] == result["proxy_mass"]


def test_swapped_pages_and_mass_accounting():
    exact = torch.tensor([[8., 8., 4., 4., 2., 2., 0., 0.]])
    proxy = exact.clone()
    proxy[:, 2:4], proxy[:, 4:6] = exact[:, 4:6], exact[:, 2:4]
    a = selection_details(exact, page_size=2, page_budget=2)
    b = selection_details(proxy, page_size=2, page_budget=2)
    comparison = compare_selection(a, b)
    assert torch.where(comparison["missed"])[0].tolist() == [1]
    assert torch.where(comparison["extra"])[0].tolist() == [2]
    assert comparison["page_recall"] == .5 and comparison["non_sink_page_recall"] == 0
    assert abs(comparison["exact_mass"] - comparison["proxy_mass"] -
               comparison["missed_mass"] + comparison["extra_mass"]) < 1e-6


def test_equal_scores_have_rank_intervals_and_pinned_zero():
    details = selection_details(torch.zeros(4, 40), page_size=4, page_budget=3)
    assert details["rank_min"][0] == details["rank_max"][0] == 0
    assert torch.all(details["rank_min"][1:] == 1)
    assert torch.all(details["rank_max"][1:] == 9)
    assert int(details["selected"].sum()) == 3 and details["selected"][0]


def test_rank_csv_matches_saved_sets_and_boundary():
    from evaluation.export_page_rankings import page_row
    exact = torch.tensor([[8., 8., 4., 4., 2., 2., 0., 0.]]).repeat(4, 1)
    proxy = exact.clone()
    proxy[:, 2:4], proxy[:, 4:6] = exact[:, 4:6], exact[:, 2:4]
    details = {"exact": selection_details(exact, page_size=2, page_budget=2),
               "q32": selection_details(exact, page_size=2, page_budget=2),
               "qgram32": selection_details(proxy, page_size=2, page_budget=2)}
    tables = {"document": torch.tensor([72]), "query_position": torch.tensor([31231]),
              "page_count": torch.tensor([4]),
              "exact.teacher_mass": details["exact"]["full_mass"].mean(0)[None],
              "exact.teacher_non_sink_mass": details["exact"]["non_sink_mass"].mean(0)[None],
              "exact.teacher_mass_by_head": details["exact"]["full_mass"][None]}
    for arm, data in details.items():
        for key in ("group_score", "rank_min", "rank_max", "owner", "selected", "cutoff"):
            tables[f"{arm}.{key}"] = data[key][None]
    missed = page_row(tables, 0, 1, ("q32", "qgram32"), 15, 7)
    assert missed["q32.status"] == "intersection" and missed["qgram32.status"] == "missed"
    assert missed["exact.rank_min"] == 1 and missed["qgram32.rank_min"] == 2
    assert missed["qgram32.score_minus_cutoff"] < 0
    assert missed["exact.global_owner_head"] == 28
    extra = page_row(tables, 0, 2, ("q32", "qgram32"), 15, 7)
    assert extra["qgram32.status"] == "extra"
    pinned = page_row(tables, 0, 0, ("q32", "qgram32"), 15, 7)
    assert pinned["pinned"] and pinned["exact.rank_min"] == 0
