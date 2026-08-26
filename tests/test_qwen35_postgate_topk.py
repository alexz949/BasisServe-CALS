from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from basisserve.core.qwen35_postgate_topk import (
    Qwen35PostGateTopKRuntime,
    retained_count,
)
from basisserve.core.mlp_gram_srrqr import (
    StaticMLPMaskRuntime,
    contribution_gram,
    gram_subset_reweighting,
    gram_srrqr_coordinates,
)
from evaluation.eval_qwen35_postgate_topk_crossdomain import (
    _mcq_paired_metrics,
    _mixed_ratios,
    _runtime_specs,
    _variant_metadata,
    _variant_plan,
)
from evaluation.summarize_qwen35_postgate_topk_crossdomain import _profile_similarity


class _FullOwner(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.o_proj = nn.Linear(width, width, bias=False)
        self.o_proj.weight.data.copy_(torch.eye(width))


class _GDNOwner(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.out_proj = nn.Linear(width, width, bias=False)
        self.out_proj.weight.data.copy_(torch.eye(width))


class _MLPOwner(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(width, width, bias=False)
        self.down_proj.weight.data.copy_(torch.eye(width))


class _Layer(nn.Module):
    def __init__(self, kind: str, width: int) -> None:
        super().__init__()
        self.mlp = _MLPOwner(width)
        if kind == "full":
            self.self_attn = _FullOwner(width)
            self.linear_attn = None
        elif kind == "gdn":
            self.self_attn = None
            self.linear_attn = _GDNOwner(width)
        else:
            raise ValueError(kind)


class _LanguageModel(nn.Module):
    def __init__(self, layers: list[_Layer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _ModelContainer(nn.Module):
    def __init__(self, layers: list[_Layer]) -> None:
        super().__init__()
        self.language_model = _LanguageModel(layers)


class _MockModel(nn.Module):
    def __init__(self, kinds: list[str], width: int = 8) -> None:
        super().__init__()
        self.model = _ModelContainer([_Layer(kind, width) for kind in kinds])


def test_source_local_topk_respects_tp_blocks_and_restores() -> None:
    model = _MockModel(["full"])
    projection = model.model.language_model.layers[0].self_attn.o_proj
    source = torch.tensor([[1.0, 4.0, 3.0, 2.0, 8.0, 5.0, 7.0, 6.0]])
    with Qwen35PostGateTopKRuntime(
        model,
        intervention="full",
        keep_ratio=0.5,
        selection_scope="source_local",
        tp_size=2,
    ) as runtime:
        actual = projection(source)
        expected = torch.tensor([[0.0, 4.0, 3.0, 0.0, 8.0, 0.0, 7.0, 0.0]])
        torch.testing.assert_close(actual, expected)
        assert runtime.records[0].kept_per_source == 2
        assert runtime.records[0].kept_per_vector == 4
        summary, tensors = runtime.profile_snapshot()
        assert summary[0]["vectors"] == 1
        assert summary[0]["realized_ratio"] == pytest.approx(0.5)
        assert summary[0]["retained_input_energy"] == pytest.approx(
            float(expected.square().sum() / source.square().sum())
        )
        probability = next(iter(tensors.values()))
        torch.testing.assert_close(
            probability,
            torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        )
    torch.testing.assert_close(projection(source), source)
    assert not hasattr(projection, "_basisserve_postgate_topk")


def test_global_topk_can_concentrate_in_one_tp_source() -> None:
    model = _MockModel(["full"])
    projection = model.model.language_model.layers[0].self_attn.o_proj
    source = torch.tensor([[1.0, 4.0, 3.0, 2.0, 8.0, 5.0, 7.0, 6.0]])
    with Qwen35PostGateTopKRuntime(
        model,
        intervention="full",
        keep_ratio=0.5,
        selection_scope="global",
        tp_size=2,
    ):
        actual = projection(source)
    torch.testing.assert_close(
        actual,
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 8.0, 5.0, 7.0, 6.0]]),
    )


def test_intervention_selects_full_gdn_or_both() -> None:
    model = _MockModel(["full", "gdn"])
    full = model.model.language_model.layers[0].self_attn.o_proj
    gdn = model.model.language_model.layers[1].linear_attn.out_proj
    source = torch.arange(1.0, 9.0).unsqueeze(0)
    with Qwen35PostGateTopKRuntime(
        model,
        intervention="full",
        keep_ratio=0.5,
        selection_scope="global",
        tp_size=2,
    ) as runtime:
        assert len(runtime.records) == 1
        assert runtime.records[0].layer_kind == "full_attention"
        assert torch.count_nonzero(full(source)) == 4
        torch.testing.assert_close(gdn(source), source)
    with Qwen35PostGateTopKRuntime(
        model,
        intervention="both",
        keep_ratio=0.5,
        selection_scope="global",
        tp_size=2,
    ) as runtime:
        assert len(runtime.records) == 2
        assert torch.count_nonzero(full(source)) == 4
        assert torch.count_nonzero(gdn(source)) == 4


def test_mlp_intervention_targets_every_predown_wire() -> None:
    model = _MockModel(["full", "gdn"])
    source = torch.arange(1.0, 9.0).unsqueeze(0)
    projections = [layer.mlp.down_proj for layer in model.model.language_model.layers]
    with Qwen35PostGateTopKRuntime(
        model,
        intervention="mlp",
        keep_ratio=0.5,
        selection_scope="source_local",
        tp_size=2,
    ) as runtime:
        assert len(runtime.records) == 2
        assert {record.layer_kind for record in runtime.records} == {"mlp"}
        for projection in projections:
            assert torch.count_nonzero(projection(source)) == 4
    for projection in projections:
        torch.testing.assert_close(projection(source), source)


def test_mlp_output_weighted_score_uses_down_weight_column_norm() -> None:
    model = _MockModel(["full"], width=4)
    projection = model.model.language_model.layers[0].mlp.down_proj
    projection.weight.data.copy_(torch.diag(torch.tensor([100.0, 1.0, 1.0, 1.0])))
    source = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    with Qwen35PostGateTopKRuntime(
        model,
        intervention="mlp",
        keep_ratio=0.5,
        selection_scope="source_local",
        score="output_weighted",
        tp_size=2,
    ) as runtime:
        actual = projection(source)
        assert runtime.records[0].score == "output_weighted"
    torch.testing.assert_close(actual, torch.tensor([[100.0, 0.0, 3.0, 0.0]]))


def test_contribution_gram_matches_explicit_channel_contributions() -> None:
    activations = torch.tensor(
        [
            [1.0, 2.0, -1.0, 0.5],
            [3.0, -2.0, 4.0, 1.5],
            [-1.0, 0.25, 2.0, -3.0],
        ]
    )
    weight = torch.tensor(
        [
            [2.0, 0.0, 1.0, -1.0],
            [0.5, 3.0, -2.0, 4.0],
        ]
    )
    moment = activations.T @ activations / len(activations)
    actual = contribution_gram(moment, weight)
    explicit = torch.stack(
        [
            torch.outer(activations[:, channel], weight[:, channel]).reshape(-1)
            for channel in range(activations.shape[1])
        ],
        dim=1,
    )
    expected = explicit.T @ explicit / len(activations)
    torch.testing.assert_close(actual, expected)
    assert actual[0, 2] != 0.0


def test_gram_srrqr_returns_unique_real_coordinates() -> None:
    generator = torch.Generator().manual_seed(20260826)
    feature = torch.randn(5, 8, generator=generator)
    gram = feature.T @ feature
    result = gram_srrqr_coordinates(
        gram,
        3,
        pod_oversample=2,
        bound=4.0,
        max_swaps=16,
    )
    assert result.indices.shape == (3,)
    assert torch.unique(result.indices).numel() == 3
    assert int(result.indices.min()) >= 0
    assert int(result.indices.max()) < 8
    assert result.pod_dimension == 5
    assert result.pod_energy_fraction == pytest.approx(1.0, abs=1e-5)
    assert result.diagnostics.converged

    scaled = gram_srrqr_coordinates(
        gram * 1e-8,
        3,
        pod_oversample=2,
        bound=4.0,
        max_swaps=16,
    )
    torch.testing.assert_close(scaled.indices, result.indices)

    spiked = torch.diag(torch.logspace(0, -7, 8))
    spiked_result = gram_srrqr_coordinates(
        spiked,
        3,
        pod_oversample=2,
        bound=4.0,
        max_swaps=16,
    )
    assert spiked_result.indices.numel() == 3
    assert spiked_result.gram_positive_eigenvalue_count == 8


def test_gram_subset_reweighting_matches_explicit_least_squares() -> None:
    feature = torch.tensor(
        [
            [1.0, 0.5, -0.25, 2.0],
            [0.0, 1.5, 0.75, -1.0],
            [2.0, -0.5, 1.0, 0.25],
            [-1.0, 0.25, 2.0, 1.5],
            [0.5, 2.0, -1.5, 0.75],
        ],
        dtype=torch.float64,
    )
    gram = feature.T @ feature
    indices = torch.tensor([0, 2])
    result = gram_subset_reweighting(gram, indices)
    target = feature.sum(dim=1)
    expected = torch.linalg.lstsq(feature[:, indices], target).solution.float()
    torch.testing.assert_close(result.coefficients, expected)

    approximation = feature[:, indices] @ result.coefficients.double()
    expected_residual = float((target - approximation).square().sum() / target.square().sum())
    zero_fill = feature[:, indices].sum(dim=1)
    zero_fill_residual = float((target - zero_fill).square().sum() / target.square().sum())
    assert result.relative_residual == pytest.approx(expected_residual)
    assert result.relative_residual < zero_fill_residual


def test_static_mlp_mask_is_tp_balanced_and_restores() -> None:
    model = _MockModel(["full", "gdn"], width=8)
    masks = {
        0: torch.tensor([1, 0, 1, 0, 0, 1, 0, 1], dtype=torch.bool),
        1: torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.bool),
    }
    source = torch.arange(1.0, 9.0).unsqueeze(0)
    projections = [layer.mlp.down_proj for layer in model.model.language_model.layers]
    with StaticMLPMaskRuntime(model, masks, tp_size=2) as runtime:
        assert runtime.kept_per_source == 2
        torch.testing.assert_close(
            projections[0](source),
            source * masks[0],
        )
        torch.testing.assert_close(
            projections[1](source),
            source * masks[1],
        )
        assert all(row["retained_input_energy"] > 0.0 for row in runtime.profile_snapshot())
    for projection in projections:
        torch.testing.assert_close(projection(source), source)


def test_static_mlp_mask_applies_fixed_selected_channel_scales() -> None:
    model = _MockModel(["full"], width=8)
    mask = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1], dtype=torch.bool)
    scale = torch.tensor([2.0, 0.0, 0.5, 0.0, 0.0, -1.0, 0.0, 1.5])
    source = torch.arange(1.0, 9.0).unsqueeze(0)
    projection = model.model.language_model.layers[0].mlp.down_proj
    with StaticMLPMaskRuntime(
        model,
        {0: mask},
        scales={0: scale},
        tp_size=2,
    ) as runtime:
        torch.testing.assert_close(projection(source), source * scale)
        profile = runtime.profile_snapshot()[0]
        assert profile["transformed_input_energy"] == pytest.approx(
            float((source * scale).square().sum() / source.square().sum())
        )
    torch.testing.assert_close(projection(source), source)


def test_retained_count_and_variant_plan() -> None:
    assert retained_count(4096, 0.3) == 1229
    assert retained_count(512, 0.3) == 154
    plan = _variant_plan(("full", "gdn"), (0.5, 0.75), "source_local")
    assert [row["name"] for row in plan] == [
        "full_topk_50_source_local",
        "full_topk_75_source_local",
        "gdn_topk_50_source_local",
        "gdn_topk_75_source_local",
    ]

    mixed = _mixed_ratios(["0.75:0.5", "0.5:0.75", "0.75:0.5"])
    assert mixed == ((0.75, 0.5), (0.5, 0.75))
    mixed_plan = _variant_plan(("both",), (0.625,), "source_local", mixed)
    assert [row["name"] for row in mixed_plan] == [
        "both_topk_62p5_source_local",
        "full_topk_75_gdn_topk_50_source_local",
        "full_topk_50_gdn_topk_75_source_local",
    ]
    assert _runtime_specs(mixed_plan[1]) == (("full", 0.75), ("gdn", 0.5))


def test_asymmetric_family_budget_and_packet_metadata() -> None:
    model = _MockModel(["full", "gdn"], width=16)
    full = model.model.language_model.layers[0].self_attn.o_proj
    gdn = model.model.language_model.layers[1].linear_attn.out_proj
    source = torch.arange(1.0, 17.0).unsqueeze(0)
    spec = {
        "name": "full_topk_75_gdn_topk_50_source_local",
        "intervention": "mixed",
        "family_keep_ratios": {"full": 0.75, "gdn": 0.5},
        "selection_scope": "source_local",
    }
    with Qwen35PostGateTopKRuntime(
        model,
        intervention="full",
        keep_ratio=0.75,
        selection_scope="source_local",
        tp_size=2,
    ) as full_runtime, Qwen35PostGateTopKRuntime(
        model,
        intervention="gdn",
        keep_ratio=0.5,
        selection_scope="source_local",
        tp_size=2,
    ) as gdn_runtime:
        assert torch.count_nonzero(full(source)) == 12
        assert torch.count_nonzero(gdn(source)) == 8
        metadata = _variant_metadata(
            spec,
            [*full_runtime.records, *gdn_runtime.records],
        )
    assert metadata["per_family"]["full"]["kept_per_source"] == 6
    assert metadata["per_family"]["gdn"]["kept_per_source"] == 4
    assert metadata["average_bitmask_packet_bytes_per_source_per_layer"] == 11


def test_mcq_paired_metrics() -> None:
    dense = {
        "answered": 2,
        "accuracy": 0.5,
        "results": [
            {"idx": 0, "pred": 0, "correct": True, "scores": [2.0, 1.0]},
            {"idx": 1, "pred": 0, "correct": False, "scores": [1.5, 1.0]},
        ],
    }
    candidate = {
        "answered": 2,
        "accuracy": 0.5,
        "results": [
            {"idx": 0, "pred": 1, "correct": False, "scores": [1.0, 2.0]},
            {"idx": 1, "pred": 1, "correct": True, "scores": [1.0, 1.5]},
        ],
    }
    result = _mcq_paired_metrics(candidate, dense)
    assert result["accuracy_delta"] == pytest.approx(0.0)
    assert result["prediction_agreement"] == pytest.approx(0.0)
    assert result["dense_correct_to_candidate_wrong"] == 1
    assert result["candidate_rescues"] == 1
    assert result["choice_score_delta_rmse"] > 0.0


def test_profile_similarity_reports_overlap_and_shift() -> None:
    left = torch.arange(10.0, 0.0, -1.0)
    identical = _profile_similarity(left, left)
    assert identical["cosine"] == pytest.approx(1.0)
    assert identical["top10pct_jaccard"] == pytest.approx(1.0)
    assert identical["mean_absolute_probability_shift"] == pytest.approx(0.0)

    reversed_profile = _profile_similarity(left, left.flip(0))
    assert reversed_profile["cosine"] < 1.0
    assert reversed_profile["top10pct_jaccard"] == pytest.approx(0.0)
    assert reversed_profile["maximum_absolute_probability_shift"] == pytest.approx(9.0)
