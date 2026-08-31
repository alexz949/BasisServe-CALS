from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from safetensors.torch import load_file
import torch
from torch import nn

from basisserve.core.gqa_vo_svdllm import GQAVOLayout
from evaluation.capture_attention_o_proj_ppl_snapshots import _validate_attention
from evaluation.eval_qwen3_32b_c1_wikitext import (
    LAYER_ALLOCATION_FORMAT,
    RAGGED_FACTOR_FORMAT,
    _load_layer_allocation_results,
    _load_ragged_results,
    _sha256,
    fold_c1_to_padded_weights,
)
from evaluation.eval_qwen3_32b_c1_c4_ppl_shard import (
    _document_nll_from_logits,
)


def test_c4_document_nll_keeps_documents_paired() -> None:
    logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]],
            [[0.0, 3.0], [3.0, 0.0], [0.0, 3.0]],
        ]
    )
    input_ids = torch.tensor([[0, 0, 1], [1, 1, 0]])
    actual = _document_nll_from_logits(logits, input_ids)
    expected = torch.stack(
        [
            torch.nn.functional.cross_entropy(logits[0, :2], input_ids[0, 1:]),
            torch.nn.functional.cross_entropy(logits[1, :2], input_ids[1, 1:]),
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_qwen_layer_allocation_wikitext_loader_requires_exact_v_budget(
    tmp_path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")
    allocation_dir = tmp_path / "allocation"
    allocation_dir.mkdir()
    schedule = [[64] * 8 for _ in range(64)]
    source_rank_sum = 64 * 8 * 64
    result = {
        "format": LAYER_ALLOCATION_FORMAT,
        "status": "complete",
        "model_config_sha256": _sha256(model_path / "config.json"),
        "selection": {
            "candidate_ranks": [64],
            "selected_candidate": "uniform_anchor",
            "selected_schedule": schedule,
            "target_source_rank_sum": source_rank_sum,
            "selected_accounting": {"source_rank_sum": source_rank_sum},
        },
        "schedules": {
            name: {
                "schedule": schedule,
                "accounting": {"source_rank_sum": source_rank_sum},
            }
            for name in ("uniform_anchor", "mean_dp", "ucb_dp")
        },
        "selected_artifacts": {str(layer): {} for layer in range(64)},
    }
    (allocation_dir / "result.json").write_text(json.dumps(result))

    loaded = _load_layer_allocation_results(allocation_dir, model_path)
    assert loaded["selection"]["target_source_rank_sum"] == 32768

    result["selection"]["selected_schedule"][0] = [48] * 8
    (allocation_dir / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="V-cache budget"):
        _load_layer_allocation_results(allocation_dir, model_path)


def test_snapshot_geometry_uses_explicit_head_dim() -> None:
    config = SimpleNamespace(
        hidden_size=10,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=3,
        head_dim=3,
        model_type="synthetic",
    )
    geometry = _validate_attention(config, "gqa")
    assert geometry["head_dim"] == 3
    assert geometry["hidden_size"] == 10


def test_qwen_c1_padding_preserves_routed_function() -> None:
    torch.manual_seed(20260822)
    layout = GQAVOLayout(
        hidden_size=7,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=3,
        rank=2,
    )
    dense_v = torch.randn(layout.kv_width, layout.hidden_size)
    encoders = torch.randn(
        layout.num_key_value_heads,
        layout.head_dim,
        layout.rank,
    )
    decoders = torch.randn(
        layout.num_attention_heads,
        layout.rank,
        layout.hidden_size,
    )
    padded_v, padded_o, diagnostics = fold_c1_to_padded_weights(
        dense_v_weight=dense_v,
        encoders=encoders,
        decoders=decoders,
        layout=layout,
    )
    inputs = torch.randn(11, layout.hidden_size)
    physical = (inputs @ padded_v.T).reshape(
        len(inputs), layout.num_key_value_heads, layout.head_dim
    )
    routed = physical.repeat_interleave(
        layout.query_heads_per_kv_group, dim=1
    )
    actual = routed.reshape(len(inputs), layout.query_width) @ padded_o.T
    expected = torch.zeros_like(actual)
    for head in range(layout.num_attention_heads):
        group = head // layout.query_heads_per_kv_group
        dense_head = dense_v[
            group * layout.head_dim : (group + 1) * layout.head_dim
        ]
        latent = inputs @ dense_head.T @ encoders[group]
        expected.add_(latent @ decoders[head])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert diagnostics == {
        "maximum_dense_fold_error": 0.0,
        "maximum_qr_product_error": 0.0,
    }


def test_qwen_profile_has_eight_physical_encoder_groups() -> None:
    from evaluation import fit_llama2_mha_c1_joint as fitter

    fitter.activate_model_profile("qwen3_32b")
    mapping = fitter._head_to_kv_group()
    assert tuple(mapping.shape) == (64,)
    assert mapping.unique(sorted=True).tolist() == list(range(8))
    assert torch.bincount(mapping).tolist() == [8] * 8


def test_qwen_ragged_wikitext_loader_requires_exact_frozen_schedule(
    tmp_path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")
    factor_dir = tmp_path / "factors"
    factor_dir.mkdir()
    schedule = [[64] * 8 for _ in range(64)]
    result = {
        "format": RAGGED_FACTOR_FORMAT,
        "status": "complete",
        "layers": list(range(64)),
        "fit_config": {
            "model_config_sha256": _sha256(model_path / "config.json"),
            "rank_schedule": schedule,
        },
        "selection": {
            "selected_schedule": schedule,
            "source_rank_sum": 64 * 8 * 64,
        },
        "artifacts": {str(layer): {} for layer in range(64)},
    }
    (factor_dir / "result.json").write_text(json.dumps(result))

    loaded = _load_ragged_results(factor_dir, model_path)

    assert loaded["selection"]["source_rank_sum"] == 32768
    result["selection"]["selected_schedule"][0][0] = 80
    (factor_dir / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="rank budget"):
        _load_ragged_results(factor_dir, model_path)


def test_full_rank_bank_uses_exact_identity_endpoint(tmp_path, monkeypatch) -> None:
    from evaluation import fit_llama2_mha_c1_joint as fitter

    monkeypatch.setattr(fitter, "NUM_HEADS", 4)
    monkeypatch.setattr(fitter, "NUM_KV_HEADS", 2)
    monkeypatch.setattr(fitter, "HEAD_DIM", 3)
    monkeypatch.setattr(fitter, "HIDDEN_SIZE", 5)
    weight = torch.arange(60, dtype=torch.float32).reshape(5, 12)
    artifact_path = tmp_path / "layer_000.safetensors"
    record_path = tmp_path / "layer_000.json"
    args = SimpleNamespace(
        cache_rank=3,
        selection_boundaries="decoder-closed",
        encoder_initialization="activation-weighted-svd",
        encoder_initialization_seed=0,
        decoder_objective="full_layer",
    )
    fit_config = {
        "encoder_cg_mode": "fixed",
        "encoder_cg_relative_tolerance": 1e-8,
        "encoder_cg_max_iterations": 16,
    }
    record = fitter._write_exact_full_rank_layer(
        layer=0,
        weight=weight,
        fit_source={"file": "fit"},
        validation_source={"file": "heldout"},
        artifact_path=artifact_path,
        record_path=record_path,
        fit_config=fit_config,
        args=args,
        factor_dtype=torch.float32,
        started=0.0,
    )
    tensors = load_file(str(artifact_path), device="cpu")
    expected_A = torch.eye(3).repeat(2, 1, 1)
    expected_D = weight.T.reshape(4, 3, 5)
    torch.testing.assert_close(tensors["value_coordinate_encoders"], expected_A)
    torch.testing.assert_close(tensors["head_output_decoders"], expected_D)
    assert record["solver"]["method"] == "analytic_exact_full_rank_endpoint"
    assert record["fit"]["factor_dtype_relative_mse"] == 0.0
    assert record["heldout"]["factor_dtype_relative_mse"] == 0.0


def test_streaming_covariance_capture_uses_every_token_position() -> None:
    from evaluation.capture_attention_o_proj_covariances import (
        _StreamingCovarianceCapture,
    )

    modules = {
        0: nn.Linear(3, 2, bias=False),
        1: nn.Linear(3, 2, bias=False),
    }
    capture = _StreamingCovarianceCapture(modules, input_width=3)
    fit = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 10
    heldout = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3) / 7
    try:
        capture.begin("fit", rows=fit.shape[0] * fit.shape[1])
        for module in modules.values():
            module(fit)
        capture.finish()
        capture.begin("heldout", rows=heldout.shape[0] * heldout.shape[1])
        for module in modules.values():
            module(heldout)
        capture.finish()
        capture.normalize_and_offload("fit", expected_rows=8)
        capture.normalize_and_offload("heldout", expected_rows=4)
    finally:
        capture.close()

    expected_fit = fit.reshape(-1, 3).T @ fit.reshape(-1, 3) / 8
    expected_heldout = heldout.reshape(-1, 3).T @ heldout.reshape(-1, 3) / 4
    assert capture.rows == {"fit": 8, "heldout": 4}
    for layer in modules:
        torch.testing.assert_close(capture.sums["fit"][layer], expected_fit)
        torch.testing.assert_close(
            capture.sums["heldout"][layer], expected_heldout
        )


def test_covariance_matrix_to_head_blocks_matches_raw_activations(
    monkeypatch,
) -> None:
    from evaluation import fit_llama2_mha_c1_joint as fitter

    monkeypatch.setattr(fitter, "NUM_HEADS", 2)
    monkeypatch.setattr(fitter, "HEAD_DIM", 3)
    torch.manual_seed(20260823)
    activation = torch.randn(17, 6, dtype=torch.float64)
    matrix = activation.T @ activation / len(activation)
    from_matrix = fitter._covariance_matrix_to_blocks(
        matrix,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    from_raw = fitter._activation_covariance_blocks(
        activation,
        device=torch.device("cpu"),
        dtype=torch.float64,
        num_heads=2,
        head_dim=3,
    )
    torch.testing.assert_close(from_matrix, from_raw)
