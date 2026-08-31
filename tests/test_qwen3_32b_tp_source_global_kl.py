from __future__ import annotations

import json

import pytest
from safetensors.torch import save_file
import torch

from evaluation.allocate_qwen3_32b_c1_tp_source_global_kl import (
    FRESH_WINDOW_START,
    NUM_KV_HEADS,
    NUM_LAYERS,
    WINDOWS_FORMAT,
    _allocate,
    _fold_ragged_to_padded_weights,
    _schedule_accounting,
    _select_fresh_windows,
    _sha256,
)
from evaluation.prepare_qwen3_32b_global_kl_windows import _verify_extension
from evaluation.eval_qwen3_32b_c1_lm_eval import (
    _load_allocation as _load_lm_eval_allocation,
    _task_names as _lm_eval_task_names,
)
from evaluation.merge_lm_eval_task_shards import merge_shards
from evaluation.run_qwen3_32b_c1_tp_source_global_kl_sharded import (
    PROFILE_FORMAT,
    _merge_profile_shards,
    _shard_layers,
)
from evaluation.fit_qwen3_32b_c1_ragged_schedule_als import (
    ALLOCATION_FORMAT as RAGGED_INPUT_FORMAT,
    _load_allocation as _load_ragged_allocation,
)


def _record(layer: int, source: int, rank: int, cost: float) -> dict:
    return {
        "layer": layer,
        "source": source,
        "candidate_rank": rank,
        "terminal_kl_delta": {
            "mean": cost,
            "one_standard_error_ucb": cost + 0.01,
        },
    }


def test_c1_lm_eval_task_parser_rejects_duplicates() -> None:
    assert _lm_eval_task_names("boolq, piqa") == ["boolq", "piqa"]
    with pytest.raises(ValueError, match="duplicates"):
        _lm_eval_task_names("boolq,boolq")


def test_c1_lm_eval_allocation_requires_complete_selected_schedule(tmp_path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")
    allocation_dir = tmp_path / "allocation"
    allocation_dir.mkdir()
    result = {
        "format": "basisserve.qwen3_32b.gqa_c1.tp_source_global_kl_allocation.v2",
        "status": "complete",
        "model_config_sha256": _sha256(model_path / "config.json"),
        "selected_artifacts": {str(index): {} for index in range(NUM_LAYERS)},
        "selection": {"selected_schedule": [[96] * NUM_KV_HEADS]},
    }
    (allocation_dir / "result.json").write_text(json.dumps(result))

    with pytest.raises(ValueError, match="rank schedule"):
        _load_lm_eval_allocation(allocation_dir, model_path)


def test_lm_eval_task_shards_merge_disjoint_results(tmp_path) -> None:
    def shard(task: str, score: float) -> dict:
        return {
            "status": "complete",
            "arm": "c1_tp8_global_kl_mean_dp",
            "model": {"path": "model"},
            "checkpoint": {"result_sha256": "digest"},
            "compression": {"source_rank_sum": 49152},
            "protocol": {"tasks": [task], "num_fewshot": 0},
            "evaluation": {
                "results": {task: {"acc,none": score}},
                "n-samples": {task: {"effective": 1, "original": 1}},
            },
        }

    merged = merge_shards(
        [
            (tmp_path / "a.json", shard("hellaswag", 0.8)),
            (tmp_path / "b.json", shard("boolq", 0.7)),
        ]
    )

    assert list(merged["evaluation"]["results"]) == ["boolq", "hellaswag"]
    assert merged["protocol"]["tasks"] == ["boolq", "hellaswag"]
    assert merged["protocol"]["execution"] == "disjoint task shards"


def test_exact_qwen_tp_source_dp_preserves_rank96_budget() -> None:
    records = []
    for layer in range(NUM_LAYERS):
        for source in range(NUM_KV_HEADS):
            costs = {64: 20.0, 80: 20.0, 112: 20.0, 128: 20.0}
            if (layer, source) == (0, 0):
                costs[128] = -3.0
            if (layer, source) == (0, 1):
                costs[64] = 0.25
            records.extend(
                _record(layer, source, rank, cost)
                for rank, cost in costs.items()
            )

    schedule, cost, contributions = _allocate(
        records,
        candidate_ranks=(64, 80, 96, 112, 128),
        anchor_rank=96,
        cost_key="mean",
    )

    assert schedule[0][:2] == [128, 64]
    assert all(rank == 96 for rank in schedule[0][2:])
    assert all(rank == 96 for layer in schedule[1:] for rank in layer)
    assert cost == pytest.approx(-2.75)
    assert sum(map(sum, schedule)) == NUM_LAYERS * NUM_KV_HEADS * 96
    assert len(contributions) == NUM_LAYERS * NUM_KV_HEADS


def test_exact_qwen_tp_source_dp_supports_v64_seven_point_grid() -> None:
    ranks = (32, 48, 64, 80, 96, 112, 128)
    records = []
    for layer in range(NUM_LAYERS):
        for source in range(NUM_KV_HEADS):
            costs = {rank: 20.0 for rank in ranks if rank != 64}
            if (layer, source) == (0, 0):
                costs[128] = -4.0
            if (layer, source) == (0, 1):
                costs[32] = 0.5
            if (layer, source) == (0, 2):
                costs[32] = 0.5
            records.extend(
                _record(layer, source, rank, cost)
                for rank, cost in costs.items()
            )

    schedule, cost, contributions = _allocate(
        records,
        candidate_ranks=ranks,
        anchor_rank=64,
        cost_key="mean",
    )

    assert schedule[0][:3] == [128, 32, 32]
    assert all(rank == 64 for rank in schedule[0][3:])
    assert all(rank == 64 for layer in schedule[1:] for rank in layer)
    assert cost == pytest.approx(-3.0)
    assert sum(map(sum, schedule)) == NUM_LAYERS * NUM_KV_HEADS * 64
    assert len(contributions) == NUM_LAYERS * NUM_KV_HEADS


def test_ragged_als_loader_freezes_complete_v64_schedule(tmp_path) -> None:
    schedule = [[64] * NUM_KV_HEADS for _ in range(NUM_LAYERS)]
    allocation_dir = tmp_path / "allocation"
    allocation_dir.mkdir()
    result = {
        "format": RAGGED_INPUT_FORMAT,
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

    loaded, frozen = _load_ragged_allocation(
        allocation_dir,
        model_config_sha256="model-hash",
    )

    assert loaded["selection"]["selected_candidate"] == "mean_dp"
    assert frozen == tuple(tuple(layer) for layer in schedule)

    result["selection"]["selected_schedule"][0][0] = 80
    (allocation_dir / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="frozen rank budget"):
        _load_ragged_allocation(
            allocation_dir,
            model_config_sha256="model-hash",
        )


def test_qwen_schedule_accounting_separates_variable_and_padded_width() -> None:
    uniform = [[96] * NUM_KV_HEADS for _ in range(NUM_LAYERS)]
    adaptive = [list(layer) for layer in uniform]
    adaptive[0][0] = 128
    adaptive[0][1] = 64

    uniform_cost = _schedule_accounting(uniform, anchor_rank=96)
    adaptive_cost = _schedule_accounting(adaptive, anchor_rank=96)

    assert uniform_cost["source_rank_sum"] == NUM_LAYERS * NUM_KV_HEADS * 96
    assert adaptive_cost["ideal_variable_allgather_total_width"] == (
        uniform_cost["ideal_variable_allgather_total_width"]
    )
    assert adaptive_cost["padded_rectangular_allgather_total_width"] == (
        uniform_cost["padded_rectangular_allgather_total_width"] + 256
    )
    assert adaptive_cost["changed_sources_from_anchor"] == 2
    assert adaptive_cost["layers_with_padded_overhead"] == 1


def test_ragged_zero_padding_preserves_routed_gqa_function() -> None:
    generator = torch.Generator().manual_seed(20260822)
    num_sources, heads_per_source = 2, 2
    head_dim, hidden = 3, 5
    ranks = (1, 2)
    maximum_rank = max(ranks)
    num_heads = num_sources * heads_per_source
    dense_v = torch.randn(
        num_sources * head_dim, hidden, generator=generator
    )
    A = torch.zeros(num_sources, head_dim, maximum_rank)
    D = torch.zeros(num_heads, maximum_rank, hidden)
    for source, rank in enumerate(ranks):
        A[source, :, :rank] = torch.randn(
            head_dim, rank, generator=generator
        )
        heads = slice(
            source * heads_per_source, (source + 1) * heads_per_source
        )
        D[heads, :rank] = torch.randn(
            heads_per_source, rank, hidden, generator=generator
        )

    padded_v, padded_o = _fold_ragged_to_padded_weights(
        dense_v_weight=dense_v,
        A=A,
        D=D,
        source_ranks=ranks,
    )
    inputs = torch.randn(7, hidden, generator=generator)
    physical = (inputs @ padded_v.T).reshape(7, num_sources, head_dim)
    routed = physical.repeat_interleave(heads_per_source, dim=1)
    actual = routed.reshape(7, num_heads * head_dim) @ padded_o.T
    expected = torch.zeros_like(actual)
    for head in range(num_heads):
        source = head // heads_per_source
        rank = ranks[source]
        dense_group = dense_v[source * head_dim : (source + 1) * head_dim]
        latent = inputs @ dense_group.T @ A[source, :, :rank]
        expected.add_(latent @ D[head, :rank])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_global_kl_windows_use_fresh_tail_and_disjoint_documents(tmp_path) -> None:
    count = FRESH_WINDOW_START + 16
    windows = torch.arange(count * 4, dtype=torch.int32).reshape(count, 4)
    path = tmp_path / "windows.safetensors"
    save_file({"input_ids": windows}, str(path))
    manifest = {
        "format": WINDOWS_FORMAT,
        "artifact": {"sha256": _sha256(path)},
        "records": [
            {"document_id": f"document-{index}"} for index in range(count)
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    profile, confirmation, provenance = _select_fresh_windows(
        path,
        window_start=FRESH_WINDOW_START,
        profile_windows=8,
        confirmation_windows=8,
        sequence_length=4,
    )

    torch.testing.assert_close(profile, windows[320:328].long())
    torch.testing.assert_close(confirmation, windows[328:336].long())
    assert provenance["profile_indices"] == list(range(320, 328))
    assert provenance["confirmation_indices"] == list(range(328, 336))
    assert provenance["disjoint_from_als_fit_and_heldout_prefix"] is True


def test_window_extension_must_preserve_exact_prefix() -> None:
    prefix = torch.arange(3 * 2048).reshape(3, 2048)
    generated = torch.cat((prefix, torch.ones(2, 2048)), dim=0)
    _verify_extension(generated, prefix, extra=2)
    corrupted = generated.clone()
    corrupted[0, 0] += 1
    with pytest.raises(RuntimeError, match="audited 320-window prefix"):
        _verify_extension(corrupted, prefix, extra=2)


def test_two_profile_replicas_partition_all_layers() -> None:
    even = _shard_layers(0, 2)
    odd = _shard_layers(1, 2)

    assert even == tuple(range(0, NUM_LAYERS, 2))
    assert odd == tuple(range(1, NUM_LAYERS, 2))
    assert sorted(even + odd) == list(range(NUM_LAYERS))


def test_profile_merge_requires_complete_disjoint_shards(tmp_path) -> None:
    configuration = {"experiment": "unit-test"}
    ranks = (64, 80, 96, 112, 128)
    for shard_index in range(2):
        assigned = _shard_layers(shard_index, 2)
        records = [
            _record(layer, source, rank, float(layer + source + rank))
            for layer in assigned
            for source in range(NUM_KV_HEADS)
            for rank in ranks
            if rank != 96
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

    assert len(merged["records"]) == NUM_LAYERS * NUM_KV_HEADS * 4
    assert len(merged["absolute_covariance_damping_by_layer"]) == NUM_LAYERS
    assert [row["assigned_layers"][0] for row in merged["shards"]] == [0, 1]
