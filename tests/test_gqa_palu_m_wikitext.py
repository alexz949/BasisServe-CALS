from __future__ import annotations

import torch
from torch import nn

from evaluation.eval_gqa_palu_m_wikitext import (
    SUPPORTED_CHECKPOINT_FORMATS,
    install_palu_k_factors,
    install_palu_m_factors,
)
from evaluation.eval_gqa_palu_m_lm_eval import _layer_ranks, _task_names


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_proj = nn.Linear(7, 8, bias=False, dtype=torch.float64)
        self.v_proj = nn.Linear(7, 8, bias=False, dtype=torch.float64)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()


def test_all_generated_gqa_checkpoint_formats_are_supported() -> None:
    assert SUPPORTED_CHECKPOINT_FORMATS == {
        "basisserve.llama2_7b.iclr_v_factors.v1",
        "basisserve.llama31_8b.iclr_v_factors.v1",
        "basisserve.llama31_70b.iclr_v_factors.v1",
        "basisserve.qwen3_8b.iclr_v_factors.v1",
        "basisserve.qwen3_32b.iclr_v_factors.v1",
        "basisserve.llama31_8b.palu_m_v_only.v1",
        "basisserve.llama31_8b.palu_m_v_only_fisher.v1",
        "basisserve.llama31_70b.palu_m_v_only.v1",
        "basisserve.llama31_70b.palu_m_v_only_fisher.v1",
        "basisserve.qwen3_8b.palu_m_v_only.v1",
        "basisserve.qwen3_8b.palu_m_v_only_fisher.v1",
        "basisserve.qwen3_32b.palu_m_v_only.v1",
        "basisserve.qwen3_32b.palu_m_v_only_fisher.v1",
        "basisserve.llama31_8b.palu_m_k_only_fisher.v1",
        "basisserve.llama31_70b.palu_m_k_only_fisher.v1",
        "basisserve.qwen3_8b.palu_m_k_only_fisher.v1",
        "basisserve.qwen3_32b.palu_m_k_only_fisher.v1",
    }


def test_install_palu_m_factors_preserves_dense_k_and_factor_function() -> None:
    torch.manual_seed(43)
    model = _Model()
    dense_keys = [layer.self_attn.k_proj for layer in model.model.layers]
    payload = {}
    layer_ranks = [[2, 2], [1, 1]]
    for layer_index, ranks in enumerate(layer_ranks):
        payload[f"layers.{layer_index}.v_writer.weight"] = torch.randn(
            sum(ranks), 7, dtype=torch.float64
        )
        payload[f"layers.{layer_index}.v_decoder.weight"] = torch.randn(
            2, 4, ranks[0], dtype=torch.float64
        )

    records = install_palu_m_factors(
        model,
        payload,
        layer_ranks=layer_ranks,
        head_dim=4,
    )

    inputs = torch.randn(3, 5, 7, dtype=torch.float64)
    for layer_index, layer in enumerate(model.model.layers):
        writer = payload[f"layers.{layer_index}.v_writer.weight"]
        decoder = payload[f"layers.{layer_index}.v_decoder.weight"]
        latent = inputs @ writer.T
        rank = layer_ranks[layer_index][0]
        expected = torch.cat(
            [
                latent[..., :rank] @ decoder[0].T,
                latent[..., rank:] @ decoder[1].T,
            ],
            dim=-1,
        )
        torch.testing.assert_close(layer.self_attn.v_proj(inputs), expected)
        assert layer.self_attn.k_proj is dense_keys[layer_index]
    assert len(records) == 2


def test_install_grouped_palu_factors_use_full_group_output_width() -> None:
    torch.manual_seed(53)
    model = _Model()
    payload = {}
    layer_ranks = [[3], [2]]
    for layer_index, ranks in enumerate(layer_ranks):
        payload[f"layers.{layer_index}.v_writer.weight"] = torch.randn(
            sum(ranks), 7, dtype=torch.float64
        )
        payload[f"layers.{layer_index}.v_decoder.weight"] = torch.randn(
            1, 8, ranks[0], dtype=torch.float64
        )

    records = install_palu_m_factors(
        model,
        payload,
        layer_ranks=layer_ranks,
        head_dim=4,
    )

    inputs = torch.randn(3, 5, 7, dtype=torch.float64)
    for layer_index, layer in enumerate(model.model.layers):
        writer = payload[f"layers.{layer_index}.v_writer.weight"]
        decoder = payload[f"layers.{layer_index}.v_decoder.weight"]
        expected = (inputs @ writer.T) @ decoder[0].T
        torch.testing.assert_close(layer.self_attn.v_proj(inputs), expected)
    assert [record["heads_per_group"] for record in records] == [2, 2]


def test_install_palu_k_factors_preserves_dense_v_and_factor_function() -> None:
    torch.manual_seed(47)
    model = _Model()
    dense_values = [layer.self_attn.v_proj for layer in model.model.layers]
    payload = {}
    layer_ranks = [[2, 2], [1, 1]]
    for layer_index, ranks in enumerate(layer_ranks):
        payload[f"layers.{layer_index}.k_writer.weight"] = torch.randn(
            sum(ranks), 7, dtype=torch.float64
        )
        payload[f"layers.{layer_index}.k_decoder.weight"] = torch.randn(
            2, 4, ranks[0], dtype=torch.float64
        )

    records = install_palu_k_factors(
        model,
        payload,
        layer_ranks=layer_ranks,
        head_dim=4,
    )

    inputs = torch.randn(3, 5, 7, dtype=torch.float64)
    for layer_index, layer in enumerate(model.model.layers):
        writer = payload[f"layers.{layer_index}.k_writer.weight"]
        decoder = payload[f"layers.{layer_index}.k_decoder.weight"]
        latent = inputs @ writer.T
        rank = layer_ranks[layer_index][0]
        expected = torch.cat(
            [
                latent[..., :rank] @ decoder[0].T,
                latent[..., rank:] @ decoder[1].T,
            ],
            dim=-1,
        )
        torch.testing.assert_close(layer.self_attn.k_proj(inputs), expected)
        assert layer.self_attn.v_proj is dense_values[layer_index]
    assert len(records) == 2


def test_lm_eval_task_parser_and_layer_rank_compatibility() -> None:
    assert _task_names("openbookqa, hellaswag") == ["openbookqa", "hellaswag"]
    manifest = {"compression": {"ranks": [3, 3]}}
    assert _layer_ranks(manifest, num_layers=2) == [[3, 3], [3, 3]]

    manifest = {"compression": {"layer_ranks": [[1, 1], [2, 2]]}}
    assert _layer_ranks(manifest, num_layers=2) == [[1, 1], [2, 2]]
