#!/usr/bin/env python3
"""Analyze paired GSM8K strict/flexible sample logs from lm-eval 0.4.11."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRICT_FILTER = "strict-match"
FLEXIBLE_FILTER = "flexible-extract"
STRICT_PATTERN = re.compile(r"#### (\-?[0-9\.\,]+)")
FLEXIBLE_PATTERN = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-json", required=True)
    parser.add_argument("--c1-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--examples-per-category", type=int, default=3)
    return parser.parse_args()


def _unwrap_singleton(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete evaluation: {path}")
    protocol = payload.get("protocol", {})
    if protocol.get("tasks") != ["gsm8k"]:
        raise ValueError(f"expected only gsm8k in {path}")
    if not protocol.get("log_samples"):
        raise ValueError(f"samples were not logged in {path}")
    return payload


def _sample_index(payload: dict[str, Any]) -> dict[int, dict[str, dict[str, Any]]]:
    samples = payload["evaluation"]["samples"]["gsm8k"]
    indexed: dict[int, dict[str, dict[str, Any]]] = {}
    for sample in samples:
        doc_id = int(sample["doc_id"])
        filter_name = sample["filter"]
        if filter_name not in (STRICT_FILTER, FLEXIBLE_FILTER):
            raise ValueError(f"unexpected filter {filter_name!r}")
        by_filter = indexed.setdefault(doc_id, {})
        if filter_name in by_filter:
            raise ValueError(f"duplicate {filter_name} record for doc {doc_id}")
        by_filter[filter_name] = sample

    for doc_id, by_filter in indexed.items():
        if set(by_filter) != {STRICT_FILTER, FLEXIBLE_FILTER}:
            raise ValueError(f"missing filter record for doc {doc_id}")
        strict = by_filter[STRICT_FILTER]
        flexible = by_filter[FLEXIBLE_FILTER]
        if strict["doc_hash"] != flexible["doc_hash"]:
            raise ValueError(f"doc hash mismatch for doc {doc_id}")
        if strict["resps"] != flexible["resps"]:
            raise ValueError(f"filter responses differ for doc {doc_id}")
    return indexed


def _is_correct(sample: dict[str, Any]) -> bool:
    return bool(sample["exact_match"])


def _response(sample: dict[str, Any]) -> str:
    response = _unwrap_singleton(sample["resps"])
    if not isinstance(response, str):
        raise TypeError(f"unexpected response type: {type(response)}")
    return response


def _filtered_response(sample: dict[str, Any]) -> str:
    response = _unwrap_singleton(sample["filtered_resps"])
    if not isinstance(response, str):
        raise TypeError(f"unexpected filtered response type: {type(response)}")
    return response


def _contingency(left: bool, right: bool) -> str:
    if left and right:
        return "both_correct"
    if left:
        return "left_only"
    if right:
        return "right_only"
    return "both_wrong"


def _binomial_two_sided(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _format_features(response: str) -> dict[str, Any]:
    strict_matches = list(STRICT_PATTERN.finditer(response))
    flexible_matches = list(FLEXIBLE_PATTERN.finditer(response))
    first_strict = strict_matches[0] if strict_matches else None
    numeric_matches_after_first_strict = (
        list(FLEXIBLE_PATTERN.finditer(response[first_strict.end() :]))
        if first_strict is not None
        else []
    )
    return {
        "has_strict_marker": first_strict is not None,
        "strict_marker_count": len(strict_matches),
        "flexible_numeric_match_count": len(flexible_matches),
        "has_numeric_match_after_first_strict": bool(
            numeric_matches_after_first_strict
        ),
        "response_chars": len(response),
    }


def _example(
    doc_id: int,
    strict: dict[str, Any],
    flexible: dict[str, Any],
) -> dict[str, Any]:
    response = _response(strict)
    return {
        "doc_id": doc_id,
        "question": strict["doc"]["question"],
        "target": strict["target"],
        "strict_correct": _is_correct(strict),
        "flexible_correct": _is_correct(flexible),
        "strict_extracted": _filtered_response(strict),
        "flexible_extracted": _filtered_response(flexible),
        "format_features": _format_features(response),
        "response": response,
    }


def _within_arm(
    indexed: dict[int, dict[str, dict[str, Any]]],
    examples_per_category: int,
) -> dict[str, Any]:
    contingency: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {
        "strict_only": [],
        "flexible_only": [],
    }
    marker_count = 0
    trailing_number_count = 0
    extraction_equal_count = 0
    response_lengths: list[int] = []
    strict_only_trailing_number = 0
    flexible_only_missing_marker = 0

    for doc_id in sorted(indexed):
        strict = indexed[doc_id][STRICT_FILTER]
        flexible = indexed[doc_id][FLEXIBLE_FILTER]
        strict_correct = _is_correct(strict)
        flexible_correct = _is_correct(flexible)
        category = _contingency(strict_correct, flexible_correct)
        renamed = {
            "left_only": "strict_only",
            "right_only": "flexible_only",
        }.get(category, category)
        contingency[renamed] += 1

        response = _response(strict)
        features = _format_features(response)
        marker_count += int(features["has_strict_marker"])
        trailing_number_count += int(
            features["has_numeric_match_after_first_strict"]
        )
        response_lengths.append(features["response_chars"])
        extraction_equal_count += int(
            _filtered_response(strict) == _filtered_response(flexible)
        )
        if renamed == "strict_only":
            strict_only_trailing_number += int(
                features["has_numeric_match_after_first_strict"]
            )
        elif renamed == "flexible_only":
            flexible_only_missing_marker += int(not features["has_strict_marker"])

        if renamed in examples and len(examples[renamed]) < examples_per_category:
            examples[renamed].append(_example(doc_id, strict, flexible))

    total = len(indexed)
    strict_correct_count = contingency["both_correct"] + contingency["strict_only"]
    flexible_correct_count = (
        contingency["both_correct"] + contingency["flexible_only"]
    )
    return {
        "num_documents": total,
        "correct": {
            STRICT_FILTER: strict_correct_count,
            FLEXIBLE_FILTER: flexible_correct_count,
        },
        "strict_vs_flexible": dict(contingency),
        "mcnemar_exact_two_sided_p": _binomial_two_sided(
            contingency["strict_only"], contingency["flexible_only"]
        ),
        "format": {
            "has_strict_marker": marker_count,
            "has_numeric_match_after_first_strict": trailing_number_count,
            "strict_and_flexible_extractions_equal": extraction_equal_count,
            "strict_only_with_numeric_match_after_first_strict": (
                strict_only_trailing_number
            ),
            "flexible_only_with_missing_strict_marker": (
                flexible_only_missing_marker
            ),
            "response_chars_mean": statistics.fmean(response_lengths),
            "response_chars_median": statistics.median(response_lengths),
        },
        "examples": examples,
    }


def _cross_arm(
    dense: dict[int, dict[str, dict[str, Any]]],
    c1: dict[int, dict[str, dict[str, Any]]],
    examples_per_category: int,
) -> dict[str, Any]:
    if set(dense) != set(c1):
        raise ValueError("Dense and C1 document IDs differ")

    output: dict[str, Any] = {}
    for filter_name in (STRICT_FILTER, FLEXIBLE_FILTER):
        contingency: Counter[str] = Counter()
        examples: dict[str, list[dict[str, Any]]] = {
            "dense_only": [],
            "c1_only": [],
        }
        for doc_id in sorted(dense):
            dense_sample = dense[doc_id][filter_name]
            c1_sample = c1[doc_id][filter_name]
            if dense_sample["doc_hash"] != c1_sample["doc_hash"]:
                raise ValueError(f"Dense/C1 doc mismatch for doc {doc_id}")
            category = _contingency(
                _is_correct(dense_sample), _is_correct(c1_sample)
            )
            renamed = {
                "left_only": "dense_only",
                "right_only": "c1_only",
            }.get(category, category)
            contingency[renamed] += 1
            if renamed in examples and len(examples[renamed]) < examples_per_category:
                examples[renamed].append(
                    {
                        "doc_id": doc_id,
                        "question": dense_sample["doc"]["question"],
                        "target": dense_sample["target"],
                        "dense_extracted": _filtered_response(dense_sample),
                        "c1_extracted": _filtered_response(c1_sample),
                        "dense_response": _response(dense_sample),
                        "c1_response": _response(c1_sample),
                    }
                )
        output[filter_name] = {
            "contingency": dict(contingency),
            "net_c1_correct": contingency["c1_only"] - contingency["dense_only"],
            "mcnemar_exact_two_sided_p": _binomial_two_sided(
                contingency["dense_only"], contingency["c1_only"]
            ),
            "examples": examples,
        }
    return output


def main() -> None:
    args = parse_args()
    if args.examples_per_category < 0:
        raise ValueError("--examples-per-category must be non-negative")

    dense_path = Path(args.dense_json).expanduser().resolve()
    c1_path = Path(args.c1_json).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)

    dense_payload = _load(dense_path)
    c1_payload = _load(c1_path)
    dense_index = _sample_index(dense_payload)
    c1_index = _sample_index(c1_payload)

    payload = {
        "format": "basisserve.gsm8k_filter_diagnosis.v1",
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dense": str(dense_path),
            "c1": str(c1_path),
        },
        "protocol": {
            "lm_eval": dense_payload["environment"]["lm_eval"],
            "dense": dense_payload["protocol"],
            "c1": c1_payload["protocol"],
        },
        "within_arm": {
            "dense": _within_arm(dense_index, args.examples_per_category),
            "c1": _within_arm(c1_index, args.examples_per_category),
        },
        "dense_vs_c1": _cross_arm(
            dense_index, c1_index, args.examples_per_category
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
