#!/usr/bin/env python3
"""Build Qwen3-8B Section-3 PaLU M-LRD/G4-LRD factors from WT2 statistics."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import build_llama31_8b_palu_m_checkpoint as whitening  # noqa: E402
from evaluation import build_qwen3_8b_iclr_v_checkpoint as builder  # noqa: E402


if __name__ == "__main__":
    whitening.activate_model_profile("qwen3_8b")
    whitening.CALIBRATION_SAMPLES = 128
    whitening.SEQUENCE_LENGTH = 2048
    builder.activate_experiment_profile("qwen3_8b")
    if len(sys.argv) > 1 and sys.argv[1] == "prepare-whitening":
        whitening.main()
    else:
        sys.exit(builder.build(builder.parse_args()))
