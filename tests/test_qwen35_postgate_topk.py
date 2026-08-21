from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from basisserve.core.qwen35_postgate_topk import (
    Qwen35PostGateTopKRuntime,
    retained_count,
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


class _Layer(nn.Module):
    def __init__(self, kind: str, width: int) -> None:
        super().__init__()
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
