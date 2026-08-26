from __future__ import annotations

import json

import pytest

from evaluation.allocate_qwen3_32b_c1_tp_source_global_kl import (
    NUM_KV_HEADS,
    NUM_LAYERS,
)
from evaluation.fit_qwen3_32b_c1_ragged_schedule_als import _load_allocation
from evaluation.run_qwen3_32b_c1_layer_global_kl_sharded import (
    FORMAT,
    PROFILE_FORMAT,
    _allocate_layer_ranks,
    _layer_schedule_accounting,
    _merge_profile_shards,
)
from evaluation.run_qwen3_32b_c1_tp_source_global_kl_sharded import _shard_layers


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
