from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from basisserve.core.decoder_closed_rank_candidates import (
    close_fixed_ragged_decoder,
    enumerate_layer_rank_swaps,
    enumerate_schedule_rank_swaps,
    marginal_topk_proposal_ids,
    recover_ragged_anchor_from_folded,
    validate_rank_swap,
)
from basisserve.core.gqa_routed_ov_joint import (
    quadratic_from_target,
)
from basisserve.core.global_rank_sensitivity import (
    logits_logsumexp,
    teacher_kl_sum,
)
from scripts.build_qwen3_decoder_closed_rank_swap_candidates import (
    _resolve_decoder_closed_base,
)
from scripts.materialize_qwen3_decoder_closed_rank_swap import (
    _rank_schedule_sha256,
    _refinement_history,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exhaustive_directed_rank_swaps_preserve_every_budget() -> None:
    ranks = (32, 32, 128, 32, 48, 32, 128, 80)
    candidates = enumerate_layer_rank_swaps(layer_index=7, ranks=ranks)
    assert candidates
    assert len({item.candidate_id for item in candidates}) == len(candidates)
    assert len({item.ranks_after for item in candidates}) == len(candidates)
    for candidate in candidates:
        validate_rank_swap(candidate)
        assert sum(candidate.ranks_after) == sum(ranks)
        assert sum(a != b for a, b in zip(ranks, candidate.ranks_after)) == 2
    example = [
        item
        for item in candidates
        if item.receiver_group == 1 and item.donor_group == 4
    ]
    assert len(example) == 1
    assert example[0].ranks_after == (32, 48, 128, 32, 32, 32, 128, 80)


def test_schedule_enumeration_and_marginal_proposal_are_separate() -> None:
    schedule = (
        (32, 48, 32, 128),
        (32, 32, 32, 32),
        (128, 32, 64, 32),
    )
    candidates = enumerate_schedule_rank_swaps(schedule)
    costs = {}
    for layer, ranks in enumerate(schedule):
        for group, _ in enumerate(ranks):
            costs[(layer, group)] = {
                rank: float((128 - rank) * (group + 1) + layer)
                for rank in (32, 48, 64, 80, 96, 112, 128)
            }
    proposed = marginal_topk_proposal_ids(
        schedule=schedule,
        costs=costs,
        receiver_top_k=1,
        donor_top_k=1,
    )
    assert proposed
    assert proposed < {item.candidate_id for item in candidates}


def test_folded_anchor_recovery_and_fixed_decoder_closure() -> None:
    torch.manual_seed(20260722)
    groups = 2
    heads = 4
    head_dim = 4
    hidden = 6
    ranks = (2, 3)
    heads_per_group = heads // groups
    dense_v = torch.randn(groups * head_dim, hidden, dtype=torch.float64)
    dense_o = torch.randn(heads, head_dim, hidden, dtype=torch.float64)
    source_A = [
        torch.randn(head_dim, rank, dtype=torch.float64) for rank in ranks
    ]
    source_D = [
        torch.randn(heads_per_group, rank, hidden, dtype=torch.float64)
        for rank in ranks
    ]
    payloads = {}
    for selected_group, rank in enumerate(ranks):
        v = torch.randn(groups * rank, hidden, dtype=torch.float64)
        o = torch.randn(hidden, heads * rank, dtype=torch.float64)
        rows = slice(selected_group * head_dim, (selected_group + 1) * head_dim)
        v[selected_group * rank : (selected_group + 1) * rank] = (
            dense_v[rows].T @ source_A[selected_group]
        ).T
        first_head = selected_group * heads_per_group
        o[:, first_head * rank : (first_head + heads_per_group) * rank] = (
            source_D[selected_group].reshape(heads_per_group * rank, hidden).T
        )
        payloads[rank] = {
            "v_proj_compressed_weight": v,
            "v_proj_compressed_bias": None,
            "o_decoder_weight": o,
        }
    recovered = recover_ragged_anchor_from_folded(
        dense_v=dense_v,
        dense_o=dense_o,
        group_ranks=ranks,
        num_heads=heads,
        num_groups=groups,
        head_dim=head_dim,
        hidden_size=hidden,
        payload_for_rank=payloads.__getitem__,
        work_dtype=torch.float64,
        device="cpu",
    )
    assert recovered.maximum_value_encoder_error < 1e-12
    assert recovered.maximum_head_product_error < 1e-12

    rows = torch.randn(31, heads * head_dim, dtype=torch.float64)
    flat = rows.T @ rows / rows.shape[0] + 0.1 * torch.eye(
        heads * head_dim,
        dtype=torch.float64,
    )
    covariance = flat.reshape(heads, head_dim, heads, head_dim).permute(0, 2, 1, 3)
    objective = quadratic_from_target(
        covariance=covariance,
        target=dense_o,
        name="fixed_decoder",
        trace_normalize=False,
    )
    mapping = torch.tensor([0, 0, 1, 1])
    closure = close_fixed_ragged_decoder(
        objective=objective,
        A_unique=recovered.A_unique,
        source_D_heads=recovered.D_heads,
        head_to_kv_group=mapping,
        group_ranks=ranks,
    )
    assert closure.encoder_sha256_before_solve == closure.encoder_sha256_after_solve
    assert closure.decoder.relative_residuals[0] < 1e-10
    assert closure.decoder.matrix_dimensions == (10,)


def test_folded_anchor_recovery_can_use_explicit_full_rank_overrides() -> None:
    torch.manual_seed(20260728)
    groups, heads, head_dim, hidden = 2, 4, 4, 7
    ranks = (4, 2)
    heads_per_group = heads // groups
    dense_v = torch.randn(groups * head_dim, hidden, dtype=torch.float64)
    dense_o = torch.randn(heads, head_dim, hidden, dtype=torch.float64)
    source_A = (
        torch.randn(head_dim, head_dim, dtype=torch.float64),
        torch.randn(head_dim, 2, dtype=torch.float64),
    )
    source_D = (
        torch.randn(heads_per_group, head_dim, hidden, dtype=torch.float64),
        torch.randn(heads_per_group, 2, hidden, dtype=torch.float64),
    )
    payloads = {}
    for selected_group, rank in enumerate(ranks):
        v = torch.randn(groups * rank, hidden, dtype=torch.float64)
        o = torch.randn(hidden, heads * rank, dtype=torch.float64)
        rows = slice(selected_group * head_dim, (selected_group + 1) * head_dim)
        v[selected_group * rank : (selected_group + 1) * rank] = (
            dense_v[rows].T @ source_A[selected_group]
        ).T
        first_head = selected_group * heads_per_group
        o[:, first_head * rank : (first_head + heads_per_group) * rank] = (
            source_D[selected_group].reshape(heads_per_group * rank, hidden).T
        )
        payloads[rank] = {
            "v_proj_compressed_weight": v,
            "v_proj_compressed_bias": None,
            "o_decoder_weight": o,
        }

    recovered = recover_ragged_anchor_from_folded(
        dense_v=dense_v,
        dense_o=dense_o,
        group_ranks=ranks,
        num_heads=heads,
        num_groups=groups,
        head_dim=head_dim,
        hidden_size=hidden,
        payload_for_rank=payloads.__getitem__,
        use_full_rank_payloads=True,
        work_dtype=torch.float64,
        device="cpu",
    )
    torch.testing.assert_close(recovered.A_unique[0], source_A[0])
    torch.testing.assert_close(recovered.D_heads[:heads_per_group], source_D[0])
    assert recovered.maximum_value_encoder_error < 1e-12
    assert recovered.maximum_head_product_error < 1e-12


def test_conditional_kl_delta_matches_direct_cached_arithmetic() -> None:
    torch.manual_seed(20260723)
    teacher = torch.randn(2, 5, 11, dtype=torch.float64)
    base = teacher + 0.15 * torch.randn_like(teacher)
    candidate = teacher + 0.10 * torch.randn_like(teacher)
    teacher_lse = logits_logsumexp(teacher, vocab_chunk_size=4)
    base_sum, tokens = teacher_kl_sum(
        base,
        teacher,
        teacher_logsumexp=teacher_lse,
        vocab_chunk_size=4,
    )
    candidate_sum, candidate_tokens = teacher_kl_sum(
        candidate,
        teacher,
        teacher_logsumexp=teacher_lse,
        vocab_chunk_size=4,
    )
    assert candidate_tokens == tokens
    cached_delta = candidate_sum / tokens - base_sum / tokens
    teacher_probs = teacher.softmax(dim=-1)
    direct_base = F.kl_div(
        base.log_softmax(dim=-1), teacher_probs, reduction="batchmean"
    ) / teacher.shape[1]
    direct_candidate = F.kl_div(
        candidate.log_softmax(dim=-1), teacher_probs, reduction="batchmean"
    ) / teacher.shape[1]
    torch.testing.assert_close(
        torch.tensor(cached_delta, dtype=torch.float64),
        direct_candidate - direct_base,
        rtol=1e-6,
        atol=1e-7,
    )


def test_resolve_sequential_materialized_decoder_closed_base(tmp_path: Path) -> None:
    source = (tmp_path / "source").resolve()
    source.mkdir()
    schedule = (tmp_path / "materialized" / "schedule.json").resolve()
    selected_ranks = [[32, 48], [64, 32]]
    _write_json(schedule, {"selected_ranks": selected_ranks})
    _write_json(
        schedule.parent / "config.json",
        {
            "format": "basisserve.a3_gqa_vo.rank_bank.v1",
            "endpoint": "dc_gkl_one_swap",
            "full_rank_factor_overrides": True,
            "source_rank_bank": str(source),
            "rank_sum": sum(sum(layer) for layer in selected_ranks),
        },
    )

    resolved = _resolve_decoder_closed_base(
        schedule.parent,
        source_root=source,
        schedule_path=schedule,
        schedule_sha256=_sha256(schedule),
        schedule_rank_sum=sum(sum(layer) for layer in selected_ranks),
    )
    assert resolved == schedule.parent


def test_refinement_history_and_rank_schedule_hash_are_stable() -> None:
    first = {"iteration": 1, "candidate_id": "first"}
    second = {"iteration": 2, "candidate_id": "second"}
    assert _refinement_history({"dc_gkl_refinement": first}) == [first]
    assert _refinement_history({"dc_gkl_refinements": [first, second]}) == [
        first,
        second,
    ]
    assert _rank_schedule_sha256([[32, 48], [64, 32]]) == _rank_schedule_sha256(
        ((32, 48), (64, 32))
    )
