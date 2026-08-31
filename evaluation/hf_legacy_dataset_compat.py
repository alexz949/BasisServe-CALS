"""Compatibility for legacy, unnamespaced Hugging Face dataset aliases.

lm-eval 0.4.11 intentionally names MathQA as ``math_qa``.  Modern versions of
``huggingface_hub`` still resolve that alias through the API, but their
filesystem URI parser rejects the resulting single-segment repository ID.
This module fixes only that URI parsing boundary; it does not modify the task
configuration passed to lm-eval.
"""

from __future__ import annotations

from typing import Any


LEGACY_MATHQA_DATASET = "math_qa"
CANONICAL_MATHQA_DATASET = "allenai/math_qa"


def canonicalize_legacy_dataset_uri(path: str) -> str:
    """Canonicalize the MathQA repository component in an HF filesystem URI."""

    marker = f"datasets/{LEGACY_MATHQA_DATASET}@"
    replacement = f"datasets/{CANONICAL_MATHQA_DATASET}@"
    if marker not in path:
        return path
    return path.replace(marker, replacement, 1)


def install_mathqa_alias_compatibility() -> dict[str, Any]:
    """Install an idempotent URI-only compatibility shim for MathQA."""

    from huggingface_hub import HfFileSystem

    if getattr(HfFileSystem.resolve_path, "_basisserve_mathqa_alias_compat", False):
        return compatibility_record()

    original_resolve_path = HfFileSystem.resolve_path

    def resolve_path(
        self: HfFileSystem,
        path: str,
        revision: str | None = None,
    ) -> Any:
        return original_resolve_path(
            self,
            canonicalize_legacy_dataset_uri(str(path)),
            revision,
        )

    resolve_path._basisserve_mathqa_alias_compat = True  # type: ignore[attr-defined]
    HfFileSystem.resolve_path = resolve_path
    return compatibility_record()


def compatibility_record() -> dict[str, Any]:
    """Return provenance suitable for inclusion in evaluation output."""

    return {
        "scope": "huggingface_hub filesystem URI resolution only",
        "lm_eval_task_config_modified": False,
        "legacy_dataset_alias": LEGACY_MATHQA_DATASET,
        "canonical_hub_repository": CANONICAL_MATHQA_DATASET,
    }


__all__ = [
    "CANONICAL_MATHQA_DATASET",
    "LEGACY_MATHQA_DATASET",
    "canonicalize_legacy_dataset_uri",
    "compatibility_record",
    "install_mathqa_alias_compatibility",
]
