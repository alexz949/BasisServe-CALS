#!/usr/bin/env python3
"""Fit Qwen3-32B GQA C1 factors with true encoder/decoder ALS."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import fit_llama2_mha_c1_joint as fitter


if __name__ == "__main__":
    fitter.activate_model_profile("qwen3_32b")
    fitter.__doc__ = __doc__
    fitter.main()
