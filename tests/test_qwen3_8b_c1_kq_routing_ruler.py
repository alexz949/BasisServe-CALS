from __future__ import annotations

import pytest

from evaluation.eval_qwen3_8b_c1_kq_routing_ruler import (
    _arm_names,
    _assigned_work,
    _markdown,
    _sparse_policies,
)


def test_assignment_rank_offset_moves_smoke_sample_to_second_gpu() -> None:
    work = [(0, "sample0"), (1, "sample1"), (2, "sample2")]

    assert _assigned_work(work, rank=0, world_size=2, rank_offset=1) == [
        (1, "sample1")
    ]
    assert _assigned_work(work, rank=1, world_size=2, rank_offset=1) == [
        (0, "sample0"),
        (2, "sample2"),
    ]


def test_arm_names_encode_runtime_geometry() -> None:
    assert _arm_names(96, 32, 1024) == (
        "dense_k_c1_v96",
        "kq_r32_b1024_exact_k_c1_v96",
    )


@pytest.mark.parametrize("geometry", [(0, 32, 1024), (96, 0, 1024), (96, 32, 0)])
def test_arm_names_reject_nonpositive_geometry(geometry: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _arm_names(*geometry)


def test_markdown_uses_checkpoint_value_rank_and_runtime_budget() -> None:
    bf16_arm = "dense_k_dense_v"
    c1_arm, routing_arm = _arm_names(96, 32, 1024)
    task_row = {"accuracy": 1.0, "samples": 1}
    payload = {
        "metadata": {
            "arms": [bf16_arm, c1_arm, routing_arm],
            "value_rank": 96,
            "routing_rank": 32,
            "exact_token_budget": 1024,
            "sequence_length": 32768,
            "layers": 36,
            "tasks": ["niah_single_1"],
            "persistent_gpu_scalar_ratio": 0.5,
        },
        "summary": {
            "arms": [
                {
                    "arm": arm,
                    "task_balanced_accuracy": 1.0,
                    "tasks": {"niah_single_1": task_row},
                }
                for arm in (bf16_arm, c1_arm, routing_arm)
            ],
            "paired_routing_vs_c1": {
                "tasks": {
                    "niah_single_1": {
                        "sparse_regressions": 0,
                        "sparse_improvements": 0,
                    }
                },
                "all_samples": {
                    "sparse_regressions": 0,
                    "sparse_improvements": 0,
                },
            },
            "runtime_logical": {
                "queries": 36.0,
                "cpu_exact_key_bytes_fetched": float(1 << 20),
                "selected_token_fraction": 0.03125,
            },
        },
        "records": [{}],
    }

    markdown = _markdown(payload)

    assert "C1-V96" in markdown
    assert "R32/B1024" in markdown
    assert "C1-V64" not in markdown


def test_sparse_policies_include_adaptive_sweep_and_fixed_endpoints() -> None:
    policies = _sparse_policies(
        value_rank=96,
        routing_rank=32,
        base_token_budget=1024,
        adaptive_max_token_budget=2048,
        adaptive_tail_mass_ratios=(0.5, 0.25, 0.1),
    )

    assert [policy.token_budget for policy in policies] == [
        1024,
        1024,
        1024,
        1024,
        2048,
    ]
    assert [policy.adaptive_tail_mass_ratio for policy in policies] == [
        None,
        0.5,
        0.25,
        0.1,
        None,
    ]
    assert policies[1].arm == "kq_r32_b1024to2048_tail0p5_exact_k_c1_v96"
    assert policies[-1].arm == "kq_r32_b2048_exact_k_c1_v96"


def test_adaptive_markdown_reports_accuracy_traffic_and_refinements() -> None:
    bf16_arm = "dense_k_dense_v"
    c1_arm = "dense_k_c1_v96"
    policies = _sparse_policies(
        value_rank=96,
        routing_rank=32,
        base_token_budget=1024,
        adaptive_max_token_budget=2048,
        adaptive_tail_mass_ratios=(0.25,),
    )
    arms = [bf16_arm, c1_arm, *(policy.arm for policy in policies)]
    task_row = {"accuracy": 1.0, "samples": 1}
    runtime = {
        "queries": 36.0,
        "cpu_exact_key_bytes_fetched": float(1 << 20),
        "selected_token_fraction": 0.05,
        "adaptive_refinement_fraction": 0.5,
    }
    payload = {
        "metadata": {
            "arms": arms,
            "value_rank": 96,
            "routing_rank": 32,
            "layers": 36,
            "tasks": ["fwe"],
            "adaptive_budget_enabled": True,
            "sparse_policies": [
                {
                    "arm": policy.arm,
                    "token_budget": policy.token_budget,
                    "adaptive_max_token_budget": (
                        policy.adaptive_max_token_budget
                    ),
                    "adaptive_tail_mass_ratio": (
                        policy.adaptive_tail_mass_ratio
                    ),
                }
                for policy in policies
            ],
        },
        "summary": {
            "arms": [
                {
                    "arm": arm,
                    "task_balanced_accuracy": 1.0,
                    "tasks": {"fwe": task_row},
                }
                for arm in arms
            ],
            "runtime_logical_by_arm": {
                policy.arm: runtime for policy in policies
            },
        },
        "records": [{}],
    }

    markdown = _markdown(payload)

    assert "adaptive KQ-routing" in markdown
    assert "B1024->2048 tail>=0.25" in markdown
    assert "5.0000%" in markdown
    assert "50.00%" in markdown
