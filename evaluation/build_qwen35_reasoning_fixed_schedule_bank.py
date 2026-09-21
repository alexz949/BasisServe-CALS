#!/usr/bin/env python3
"""Assemble a reasoning-calibrated Qwen3.5 V bank with the frozen C4 schedule."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.qwen35_hybrid_banks import LAYERS, save_bank  # noqa: E402


SCHEDULE = (80, 96, 96, 96, 192, 160, 192, 112)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--factors", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert len(SCHEDULE) == len(LAYERS)
    assert sum(SCHEDULE) == 128 * len(LAYERS)
    save_bank(
        args,
        SCHEDULE,
        128,
        "twosided",
        {
            "rank_allocation_source": "frozen_c4_two_sided_v128_schedule",
            "rank_allocation_recomputed_on_reasoning_calibration": False,
            "factor_calibration_domain": "reasoning_math_code",
            "factor_fit_composition": {"math": 128, "code": 128},
            "heldout_composition": {"c4": 64},
        },
    )
    print(
        f"[Bank] layers={dict(zip(LAYERS, SCHEDULE, strict=True))} "
        f"average_rank={sum(SCHEDULE) / len(SCHEDULE):.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
