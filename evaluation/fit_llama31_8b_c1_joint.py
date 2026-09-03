#!/usr/bin/env python3
"""Fit Llama-3.1-8B GQA C1 factors at the fixed sweep-6 endpoint."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import fit_llama2_mha_c1_joint as fitter


if __name__ == "__main__":
    fitter.activate_model_profile("llama31_8b")
    fitter.__doc__ = __doc__
    fitter.main()
