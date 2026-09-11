# Q32 strong-RRQR versus CPQR

Completed 2026-09-10. Real-query audit and result checks exited 0. No warnings or errors appeared in the audit log.

## Fixed inputs and method

Reuse the same local dense-teacher BF16 candidate Q and completed CPQR reference: 64 fit windows, 32768 tokens/window, layers 0/15/35, 512 candidate positions, 32 heads of width 128. Keep FP64 uncentered per-head whitening, four 8K position bins, and eight selected positions/bin (Q32). No validation queries enter selection.

For each bin, a full-spectrum 128x128 Gram square-root feature preserves the geometry of the original window/head-concatenated feature. No eigenvalues were clipped in this run and no spectral truncation was performed. CPQR initialization and baseline geometry were checked against the earlier explicit-feature CPQR audit. Reuse the existing Gu–Eisenstat fixed-rank swap implementation at f=2 and f=1.01, independently initialized, with a 512-swap cap. Diagnostics use FP64 factors rather than the solver's returned FP32 basis.

## Results

| Layer | Maximum initial/final rho over bins | Swaps at f=2 | Swaps at f=1.01 | Unchanged bins, each bound | Maximum absolute interpolation coefficient |
|---|---:|---:|---:|---:|---:|
| 0 | 0.9997986173 | 0 | 0 | 4/4 | 0.1124477933 |
| 15 | 1.0018017149 | 0 | 0 | 4/4 | 0.1828943548 |
| 35 | 0.9999539288 | 0 | 0 | 4/4 | 0.2152119761 |

All 24 layer/bin/bound cases satisfied the stopping condition without swaps. All 96 selected pivot slots per bound retained the CPQR order. Selected positions, condition numbers, projection residuals and log volumes were unchanged; every volume ratio is exactly 1. No swap cap was reached. Full-spectrum root reconstruction had maximum Gram absolute error 5.5423e-13 across all bins.

The maximum rho occurs in layer 15, bin 1 (the second 8K interval). It represents approximately 0.18017% potential volume improvement from the best single swap at that initial set, below the 1% threshold for f=1.01. The other 11 bins have maximum rho below 1. Thus the observed zero-swap outcome means the CPQR selections already satisfy both tested strong-RRQR thresholds; it does not establish that every possible swap is non-improving or that global maximum-volume subsets were found. Tighter thresholds were not tested.

The unconstrained optimal rank-eight SVD projection still leaves 78.91%–86.87% of the full concatenated whitened feature energy, depending on layer/bin. This is evidence that these particular feature matrices are not well represented by eight arbitrary directions in Frobenius energy. It does not imply an equivalent routing error, explain downstream behavior, or prove that query sampling is ineffective.

No changed selector output was produced, so a duplicate residual fit for these same selections would not test a new sampling method. This run provides no new Fisher-loss, attention-mass recall or task-score measurement. Conclusions apply only to these three layers, Q32 and the two tested bounds.

## Execution and artifacts

Environment: `lowrank`, `/home/lz299/miniconda3/envs/lowrank/bin/python`. Direct CPU execution, two OMP/MKL/OpenBLAS/PyTorch threads, no GPU. The 4 GiB host-memory figure in the preparation protocol is a budget estimate, not a measured peak.

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

Per-layer results: `layer_000.json`, `layer_015.json`, `layer_035.json` in this directory. Records contain complete selections, bounds, stopping diagnostics, geometry, input/reference/source hashes, library versions and commands. Input candidate-manifest SHA256: `c7b10fb5cb4088d5ff45f641d508bc3ccfcc31fd7b766bbb5bd8fd87826e0160`.

Logs: `results/logs/query_cpqr/audit_srrqr_q32.log` and `results/logs/query_cpqr/srrqr_result_check.log`. The result check verifies 24 converged zero-swap cases, 192 unchanged pivot slots across the two bounds, equal initial/final rho, equal geometry dictionaries and unit volume ratios. Captures, CPQR outputs and routing checkpoints were preserved.
