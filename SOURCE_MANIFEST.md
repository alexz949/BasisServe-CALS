# Source manifest

Created from the CA-lowrank working tree on 2026-08-21.

- Repository: https://github.com/Zishan-Shao/BasisServe
- Baseline HEAD at extraction: c12e13c953b985f37e9a8128ea8e6d8642fa5fd2
- Source of truth: the current local worktree, including relevant changes newer
  than the baseline commit
- Policy: canonical implementation files were copied without refactoring and
  retain their original repository-relative paths

## Preserved groups

- basisserve/core/: C1, activation-aware V/Wo factorization, allocation,
  AllReduce, private AllGather, Top-K, and Qwen3.5 hybrid runtimes.
- basisserve/kernels/: Python wrapper and C++/CUDA/NCCL ragged-AllGather
  sources.
- basisserve/calibration/, analysis/, diagnostics/, checkpoint/, and
  sketching/: direct support dependencies of the entrypoints.
- evaluation/: calibration, C1 fitting, PPL, Global-KL, CKA allocation, PaLU
  reproduction, Qwen3.5 hybrid evaluation, and lm-eval entrypoints.
- scripts/: collective benchmarks and Qwen3.5 factor builder.
- palu/model/modules/: minimal PaLU low-rank modules imported by the
  comparison/reproduction code.
- tests/: selected CPU and distributed tests for the extracted surfaces.
- results/: Markdown summaries only; large tensors, checkpoints, caches, and
  raw experiment output are intentionally excluded.

## CALS-only files

The following files are new capsule-level aids rather than copies:

- README.md
- RESULTS.md
- SOURCE_MANIFEST.md
- requirements.txt
- evaluation/eval_llama2_c1_lm_eval.py
- tests/test_capsule_imports.py

basisserve/__init__.py and basisserve/analysis/__init__.py are minimal package
markers. basisserve/core/__init__.py is copied from the source tree.

## Refresh/audit

From the parent BasisServe repository, compare a preserved file with:

~~~bash
diff -u basisserve/core/gqa_routed_ov_joint.py \
  CALS/basisserve/core/gqa_routed_ov_joint.py
~~~

Do not refresh the capsule with a broad recursive copy: it is intentionally a
dependency-closed subset, not a mirror of the whole repository.
