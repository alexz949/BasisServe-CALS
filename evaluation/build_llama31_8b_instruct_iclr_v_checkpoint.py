#!/usr/bin/env python3
"""Build Llama-3.1-8B-Instruct V-side checkpoints for the ICLR experiment matrix.

``prepare-whitening`` computes the PaLU whitening factors on 2,048-token C4
windows; every other invocation builds one Dense/Weight-SVD/PaLU checkpoint.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import build_llama31_8b_palu_m_checkpoint as whitening  # noqa: E402
from evaluation import build_qwen3_8b_iclr_v_checkpoint as builder  # noqa: E402


if __name__ == "__main__":
    builder.activate_experiment_profile("llama31_8b_instruct")
    if len(sys.argv) > 1 and sys.argv[1] == "prepare-whitening":
        whitening.main()
    else:
        sys.exit(builder.build(builder.parse_args()))
