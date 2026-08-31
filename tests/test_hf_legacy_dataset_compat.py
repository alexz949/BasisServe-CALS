from __future__ import annotations

from evaluation.hf_legacy_dataset_compat import (
    canonicalize_legacy_dataset_uri,
    compatibility_record,
)


def test_mathqa_legacy_hub_uri_is_canonicalized() -> None:
    revision = "c4f1cc784c04c4957b50c97858f23893b633eea6"
    original = f"hf://datasets/math_qa@{revision}/math_qa.py"
    assert canonicalize_legacy_dataset_uri(original) == (
        f"hf://datasets/allenai/math_qa@{revision}/math_qa.py"
    )


def test_unrelated_dataset_uri_is_unchanged() -> None:
    original = "hf://datasets/openai/gsm8k@main/README.md"
    assert canonicalize_legacy_dataset_uri(original) == original


def test_compatibility_record_declares_task_config_unchanged() -> None:
    record = compatibility_record()
    assert record["lm_eval_task_config_modified"] is False
    assert record["legacy_dataset_alias"] == "math_qa"
    assert record["canonical_hub_repository"] == "allenai/math_qa"
