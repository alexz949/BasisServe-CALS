from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch

from evaluation.allocate_qwen3_32b_c1_tp_source_global_kl import (
    NUM_KV_HEADS,
    NUM_LAYERS,
    _copy_batched_logits_to_cpu,
)
from evaluation.fit_qwen3_32b_c1_ragged_schedule_als import _load_allocation
from evaluation.run_qwen3_32b_c1_layer_global_kl_sharded import (
    FORMAT,
    PROFILE_FORMAT,
    _allocate_factorized_layer_ranks,
    _allocate_layer_ranks,
    _allocation_factorized_exponent,
    _domain_window_multiplier,
    _intervention_ranks,
    _layer_schedule_accounting,
    _load_secondary_windows,
    _merge_profile_shards,
    _profile_dataset_name,
    _records_at_position_count,
)
from evaluation.run_qwen3_32b_c1_tp_source_global_kl_sharded import _shard_layers


def test_batched_logit_copy_materializes_noncontiguous_view_on_cpu() -> None:
    full_logits = torch.arange(2 * 5 * 7, dtype=torch.bfloat16).reshape(2, 5, 7)
    logits = full_logits[:, :-1]

    copied = _copy_batched_logits_to_cpu(logits)

    assert not logits.is_contiguous()
    assert copied.device.type == "cpu"
    assert copied.is_contiguous()
    assert torch.equal(copied, logits)


def test_exact_dense_layer_endpoint_skips_decoder_closure(monkeypatch) -> None:
    import torch

    from evaluation import allocate_qwen3_32b_c1_tp_source_global_kl as common

    dense_o = torch.randn(common.HIDDEN_SIZE, common.NUM_QUERY_HEADS * common.HEAD_DIM)
    snapshot = type("Snapshot", (), {"dense_o_weight": dense_o})()

    def fail_if_called(**_kwargs):
        raise AssertionError("the exact dense endpoint must not run decoder closure")

    monkeypatch.setattr(common, "close_ragged_decoder_with_fixed_encoders", fail_if_called)
    factors, diagnostics = common._closed_factors(
        bank={},
        snapshot=snapshot,
        objective=None,
        source_ranks=[common.HEAD_DIM] * common.NUM_KV_HEADS,
        anchor_rank=64,
        decoder_relative_jitter=0.0,
        device=torch.device("cpu"),
    )

    assert factors.ranks == (common.HEAD_DIM,) * common.NUM_KV_HEADS
    assert diagnostics == {"closure": "exact_identity_dense_o_endpoint"}
    assert torch.equal(
        factors.A.float(),
        torch.eye(common.HEAD_DIM).expand(common.NUM_KV_HEADS, -1, -1),
    )


def _record(layer: int, rank: int, cost: float) -> dict:
    return {
        "layer": layer,
        "candidate_rank": rank,
        "terminal_kl_delta": {
            "mean": cost,
            "one_standard_error_ucb": cost + 0.01,
        },
    }


def test_position_count_projection_replaces_primary_metrics() -> None:
    record = _record(3, 32, 9.0)
    record["terminal_kl"] = {"mean": 8.0}
    record["nll"] = {"mean": 7.0}
    record["nll_delta"] = {"mean": 6.0}
    record["position_count_sweep"] = {
        "256": {
            "terminal_kl": {"mean": 1.0},
            "nll": {"mean": 2.0},
            "terminal_kl_delta": {"mean": 3.0},
            "nll_delta": {"mean": 4.0},
        }
    }

    projected = _records_at_position_count([record], 256)

    assert projected[0]["terminal_kl"]["mean"] == 1.0
    assert projected[0]["nll"]["mean"] == 2.0
    assert projected[0]["terminal_kl_delta"]["mean"] == 3.0
    assert projected[0]["nll_delta"]["mean"] == 4.0
    assert record["terminal_kl_delta"]["mean"] == 9.0


def test_layer_dp_preserves_v64_budget_and_uniform_tp_width() -> None:
    ranks = (32, 48, 64, 80, 96, 112, 128)
    records = []
    for layer in range(NUM_LAYERS):
        costs = {rank: 20.0 for rank in ranks if rank != 64}
        if layer == 0:
            costs[128] = -4.0
        if layer in (1, 2):
            costs[32] = 0.5
        records.extend(_record(layer, rank, cost) for rank, cost in costs.items())

    schedule, cost, contributions = _allocate_layer_ranks(
        records,
        candidate_ranks=ranks,
        anchor_rank=64,
        cost_key="mean",
    )

    assert schedule[0] == [128] * NUM_KV_HEADS
    assert schedule[1] == [32] * NUM_KV_HEADS
    assert schedule[2] == [32] * NUM_KV_HEADS
    assert all(layer == [64] * NUM_KV_HEADS for layer in schedule[3:])
    assert cost == pytest.approx(-3.0)
    assert sum(layer[0] for layer in schedule) == NUM_LAYERS * 64
    assert sum(map(sum, schedule)) == NUM_LAYERS * NUM_KV_HEADS * 64
    assert len(contributions) == NUM_LAYERS


def test_layer_schedule_accounting_has_zero_ragged_overhead() -> None:
    schedule = [[64] * NUM_KV_HEADS for _ in range(NUM_LAYERS)]
    schedule[0] = [96] * NUM_KV_HEADS
    schedule[1] = [32] * NUM_KV_HEADS

    accounting = _layer_schedule_accounting(schedule, anchor_rank=64)

    assert accounting["layer_rank_sum"] == NUM_LAYERS * 64
    assert accounting["source_rank_sum"] == NUM_LAYERS * NUM_KV_HEADS * 64
    assert accounting["changed_layers_from_anchor"] == 2
    assert accounting["ragged_padding_overhead"] == 0.0
    assert accounting["rectangular_collective_total_width"] == accounting[
        "uniform_anchor_total_width"
    ]

    schedule[2][0] = 48
    with pytest.raises(ValueError, match="one rank"):
        _layer_schedule_accounting(schedule, anchor_rank=64)


def test_layer_profile_merge_requires_one_record_per_layer_rank(tmp_path) -> None:
    configuration = {"allocation_unit": "decoder_layer"}
    ranks = (32, 48, 64, 80, 96, 112, 128)
    for shard_index in range(2):
        assigned = _shard_layers(shard_index, 2)
        records = [
            _record(layer, rank, float(layer + rank))
            for layer in assigned
            for rank in ranks
            if rank != 64
        ]
        payload = {
            "format": PROFILE_FORMAT,
            "status": "complete",
            "profile_shard_index": shard_index,
            "profile_shard_count": 2,
            "assigned_layers": list(assigned),
            "completed_layers": list(assigned),
            "configuration": configuration,
            "uniform_anchor": {"terminal_kl": {"mean": 0.0}},
            "records": records,
            "absolute_covariance_damping_by_layer": {
                str(layer): 1.0e-5 for layer in assigned
            },
            "elapsed_seconds": 1.0,
        }
        (tmp_path / f"shard_{shard_index:02d}.json").write_text(
            json.dumps(payload)
        )

    merged = _merge_profile_shards(
        tmp_path,
        shard_count=2,
        configuration=configuration,
        candidate_ranks=ranks,
    )

    assert len(merged["records"]) == NUM_LAYERS * 6
    assert len(merged["absolute_covariance_damping_by_layer"]) == NUM_LAYERS


def test_factorized_profile_uses_exactly_one_intervention_rank() -> None:
    args = type(
        "Args",
        (),
        {
            "anchor_rank": 64,
            "factorized_probe_rank": 96,
            "factorized_compression_probe_rank": None,
            "factorized_exponent": 1.0,
        },
    )()

    assert _intervention_ranks(args, (32, 48, 64, 80, 96, 112, 128)) == (96,)


def test_two_sided_factorized_profile_uses_both_endpoint_probes() -> None:
    args = SimpleNamespace(
        anchor_rank=64,
        factorized_probe_rank=96,
        factorized_compression_probe_rank=32,
        factorized_exponent=1.25,
    )

    assert _intervention_ranks(args, (32, 48, 64, 80, 96)) == (32, 96)


def test_finalize_can_override_allocation_exponent_without_reprofiling() -> None:
    args = SimpleNamespace(
        factorized_exponent=1.25,
        allocation_factorized_exponent=1.0,
    )

    assert _allocation_factorized_exponent(args) == 1.0
    args.allocation_factorized_exponent = None
    assert _allocation_factorized_exponent(args) == 1.25


def test_secondary_windows_are_sliced_for_equal_domain_mixture(tmp_path) -> None:
    path = tmp_path / "windows.safetensors"
    input_ids = torch.arange(32, dtype=torch.int32).reshape(4, 8)
    save_file({"input_ids": input_ids}, path)
    from evaluation import allocate_qwen3_32b_c1_tp_source_global_kl as common

    manifest = {
        "artifact": {"sha256": common._sha256(path)},
        "records": [{"sample_index": index} for index in range(4)],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    args = SimpleNamespace(
        secondary_windows=path,
        secondary_window_start=1,
        secondary_domain_name="wikitext2",
        profile_windows=1,
        confirmation_windows=1,
        sequence_length=4,
    )

    profile, confirmation, provenance = _load_secondary_windows(args)

    assert torch.equal(profile, input_ids[1:2, :4].long())
    assert torch.equal(confirmation, input_ids[2:3, :4].long())
    assert provenance["profile_indices"] == [1]
    assert provenance["confirmation_indices"] == [2]


@pytest.mark.parametrize(
    ("secondary_windows", "mode", "expected_name", "expected_multiplier"),
    (
        (None, "append", "c4_train_fresh_documents", 1),
        ("wiki", "append", "c4+wikitext2_train", 2),
        ("wiki", "replace", "wikitext2_train", 1),
    ),
)
def test_secondary_window_mode_controls_dataset_accounting(
    secondary_windows,
    mode,
    expected_name,
    expected_multiplier,
) -> None:
    args = SimpleNamespace(
        secondary_windows=secondary_windows,
        secondary_window_mode=mode,
        secondary_domain_name="wikitext2_train",
    )

    assert _profile_dataset_name(args) == expected_name
    assert _domain_window_multiplier(args) == expected_multiplier


def test_factorized_allocation_uses_heldout_local_error_curve() -> None:
    ranks = (32, 64, 128)
    factor_results = {
        rank: {
            "records": [
                {
                    "fit_config": {"cache_rank_per_head": rank},
                    "fit": {"factor_dtype_relative_mse": 10.0},
                    "heldout": {
                        "factor_dtype_relative_mse": (
                            0.4 if rank == 32 else 0.2
                        )
                    },
                }
                for _ in range(NUM_LAYERS)
            ]
        }
        for rank in (32, 64)
    }
    records = [
        {
            "layer": layer,
            "candidate_rank": 32,
            "terminal_kl_delta": {"mean": 0.2},
        }
        for layer in range(NUM_LAYERS)
    ]

    schedule, cost, contributions, method = _allocate_factorized_layer_ranks(
        records,
        factor_results,
        candidate_ranks=ranks,
        anchor_rank=64,
        probe_rank=32,
        local_error_split="heldout",
        exponent=1.0,
    )

    assert schedule == [[64] * NUM_KV_HEADS for _ in range(NUM_LAYERS)]
    assert cost == 0.0
    assert len(contributions) == NUM_LAYERS
    assert method["terminal_interventions_per_layer"] == 1
    assert method["local_error_split"] == "heldout"


def test_als_loader_accepts_only_uniform_per_layer_schedule(tmp_path) -> None:
    allocation_dir = tmp_path / "allocation"
    allocation_dir.mkdir()
    schedule = [[64] * NUM_KV_HEADS for _ in range(NUM_LAYERS)]
    result = {
        "format": FORMAT,
        "status": "complete",
        "model_config_sha256": "model-hash",
        "selection": {
            "selected_candidate": "mean_dp",
            "selected_schedule": schedule,
            "candidate_ranks": [32, 48, 64, 80, 96, 112, 128],
            "target_source_rank_sum": NUM_LAYERS * NUM_KV_HEADS * 64,
        },
    }
    (allocation_dir / "result.json").write_text(json.dumps(result))

    loaded, frozen = _load_allocation(
        allocation_dir,
        model_config_sha256="model-hash",
    )

    assert loaded["format"] == FORMAT
    assert frozen == tuple(tuple(layer) for layer in schedule)

    result["selection"]["selected_schedule"][0][0] = 48
    result["selection"]["selected_schedule"][0][1] = 80
    (allocation_dir / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="nonuniform layer rank"):
        _load_allocation(
            allocation_dir,
            model_config_sha256="model-hash",
        )
