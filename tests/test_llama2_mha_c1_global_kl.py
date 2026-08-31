from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from evaluation.allocate_llama2_mha_c1_global_kl import (
    _parse_factor_dirs,
    _rank_invariant_fit_config,
    _schedule_stats,
    _select_windows,
    _summary,
)


def test_parse_factor_dirs_sorts_and_rejects_duplicates(tmp_path: Path) -> None:
    parsed = _parse_factor_dirs(
        [f"112={tmp_path / 'r112'}", f"64={tmp_path / 'r64'}"]
    )
    assert list(parsed) == [64, 112]
    with pytest.raises(ValueError, match="duplicate"):
        _parse_factor_dirs([f"64={tmp_path / 'a'}", f"64={tmp_path / 'b'}"])


def test_rank_invariant_config_removes_only_budget_fields() -> None:
    config = {
        "cache_rank_per_head": 64,
        "total_v_cache_rank": 2048,
        "v_retained_ratio": 0.5,
        "total_kv_retained_ratio_with_dense_k": 0.75,
        "fit_windows": 128,
        "decoder_objective": "full_layer",
    }
    assert _rank_invariant_fit_config(config) == {
        "fit_windows": 128,
        "decoder_objective": "full_layer",
    }


def test_select_windows_uses_disjoint_select_prefixes(tmp_path: Path) -> None:
    path = tmp_path / "windows.safetensors"
    input_ids = torch.arange(10 * 8, dtype=torch.int32).reshape(10, 8)
    split_codes = torch.tensor([0, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=torch.int8)
    save_file(
        {"input_ids": input_ids, "split_codes": split_codes},
        str(path),
    )
    profile, confirmation, provenance = _select_windows(
        path, profile_windows=2, confirmation_windows=3
    )
    torch.testing.assert_close(profile, input_ids[[1, 3]].long())
    torch.testing.assert_close(confirmation, input_ids[[4, 6, 7]].long())
    assert provenance["selected_window_indices"] == [1, 3, 4, 6, 7]
    with pytest.raises(ValueError, match="only 6"):
        _select_windows(path, profile_windows=4, confirmation_windows=3)


def test_schedule_stats_preserves_exact_budget() -> None:
    stats = _schedule_stats(
        {"layer_000": 64, "layer_001": 96, "layer_002": 128}
    )
    assert stats == {
        "rank_sum_per_head": 288,
        "minimum_rank": 64,
        "maximum_rank": 128,
        "mean_rank": 96,
        "histogram": {"64": 1, "96": 1, "128": 1},
    }


def test_summary_reports_dynamic_anchor_and_encoder_sweeps() -> None:
    schedule = {"layer_000": 64}
    metrics = {
        "rank_sum_per_head": 64,
        "confirmation": {"terminal_kl": {"mean": 0.1}},
        "test": {"ppl": 7.0},
    }
    text = _summary(
        {
            "command": "example",
            "geometry": {"num_hidden_layers": 1},
            "profile": {"anchor_rank": 64},
            "factorization": {"encoder_sweeps": 10},
            "selection": {
                "selected_candidate": "global_kl_mean",
                "selected_schedule": schedule,
            },
            "schedules": {
                "uniform_anchor": {**metrics, "schedule": schedule},
                "global_kl_mean": {**metrics, "schedule": schedule},
            },
        }
    )
    assert "10 encoder BCD sweeps" in text
    assert "uniform V64" in text
