# Query Q32 strong-RRQR check

Status: authorized real-query experiment and result checks completed successfully on 2026-09-10. Both f=2 and f=1.01 required zero swaps in all 12 layer/bin combinations. Detailed results: `results/evaluation/srrqr_q32/summary.md`. No new capture or residual fitting occurred.

Freeze the existing candidate capture `results/calibration/cpqr_candidates`: local C4 fit windows 0–63, 32768 tokens/window, layers 0/15/35, 512 positions/layer/window, 32 heads, BF16 Q. Recompute the identical FP64 per-head uncentered whitening (epsilon 1e-6), four 8K bins and eight selections/bin. Compare to `results/evaluation/cpqr_q32` using the same capture-manifest hash.

## Method

Reuse `basisserve/core/strong_rrqr.py`, which starts with SciPy CPQR and performs the Gu–Eisenstat fixed-rank volume-increasing swap test. For selected QR block R11 and remaining block R12, it uses

rho(i,j)^2 = (R11^-1 R12)[i,j]^2 + ||row_i(R11^-1)||_2^2 * ||residual_column_j||_2^2.

A swap is required when max rho exceeds f (with the existing FP64 stopping tolerance). This includes the residual-column term, not only interpolation coefficients. Test f=2 and the stricter f=1.01 independently from the same CPQR initialization, with at most 512 swaps. The existing solver rejects a run that reaches the cap without satisfying the bound. Report initial/final rho, swaps, convergence, selected positions, overlap, condition number, volume ratio, projection residual and the unconstrained rank-eight SVD reference error. No guarantee of optimal volume or routing improvement is asserted.

Reference: Gu and Eisenstat, *Efficient Algorithms for Computing a Strong Rank-Revealing QR Factorization*, 1996, fixed-rank Algorithm 4 and Lemma 3.1: https://math.berkeley.edu/~mgu/MA273/Strong_RRQR.pdf.

Each complete bin Gram is 128x128. Build its full FP64 PSD square-root feature A with A.T A = G, retaining all eigenvalues; only negative roundoff within the checked tolerance may be clipped. No POD truncation, query averaging, unit normalization or new weights. This preserves column geometry while avoiding the existing solver's enormous row-Gram diagnostic on the original 262144-dimensional feature matrix. Verify reconstructed Gram, identical CPQR initial pivot sequence, and baseline geometry against the prior explicit-feature CPQR result. Abort on a mismatch rather than silently changing the baseline.

Tests passed: 7 total, including the prior three CPQR geometry tests. New tests check full-Gram geometry/CPQR preservation, the equality between the predicted rho and the actual single-swap volume gain, a synthetic example requiring a swap at f=1.01, and a case requiring no swaps. Test environment `lowrank`, two CPU threads; log `results/logs/query_cpqr/srrqr_tests.log`.

## Completed execution

Working directory `/home/lz299/BasisServe-CALS`. Direct CPU execution, environment `lowrank`, no GPU, two OMP/MKL/OpenBLAS/PyTorch threads, 4 GiB host RAM budget estimate (no scheduler allocation). Fixed fit-only Q; no model forward. No original checkpoints, CPQR results, captures or manifests are overwritten. Existing output directory causes an assertion; shell noclobber protects the log.

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
set -o noclobber
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. \
python -u -m evaluation.audit_query_srrqr \
  --candidate-capture results/calibration/cpqr_candidates \
  --cpqr-reference results/evaluation/cpqr_q32 \
  --layers 0,15,35 --bounds 2,1.01 --max-swaps 512 \
  --output-dir results/evaluation/srrqr_q32 \
  > results/logs/query_cpqr/audit_srrqr_q32.log 2>&1
```

Per-layer JSONs retain commands, versions, source hashes, capture/reference hashes, full geometry and swap diagnostics. A no-swap result means CPQR already satisfies the tested threshold, not that sRRQR was skipped. Any changed subset still needs a separately confirmed residual-fit/common-query recall experiment before making routing-quality claims.
