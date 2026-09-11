# Q32 feature-space CPQR versus position-Gram pivoting

Completed 2026-09-10. All three layer audits exited successfully; the result consistency check also exited 0. No residual fitting or downstream evaluation was performed.

Inputs: newly recaptured dense-teacher BF16 Q from the local fixed V96-KL fit windows 0–63, 32K tokens/window, layers 0/15/35. Each layer has 512 candidate positions with 32 heads of width 128. Four 8K bins each select eight positions. Per-head uncentered whitening uses FP64 and epsilon 1e-6. A position feature concatenates all 64 windows and 32 heads, scaled by 1/sqrt(64*32); it has 262,144 coordinates. CPQR runs directly on the feature transpose using SciPy, not on a reconstructed Gram factor.

## Results

| Layer | Bins with identical pivot order | Maximum Gram absolute difference | Selected-block condition number range | Remaining projection energy fraction range | Minimum relative pivot gap |
|---|---:|---:|---:|---:|---:|
| 0 | 4/4 | 5.4854e-12 | 1.1105–1.2212 | 0.921405–0.923798 | 7.4440e-5 |
| 15 | 4/4 | 5.0306e-12 | 1.1691–1.3291 | 0.911547–0.914396 | 6.8817e-4 |
| 35 | 4/4 | 5.7980e-12 | 1.2625–1.5260 | 0.895644–0.903202 | 9.2140e-5 |

Production window-averaged Gram pivoting, pivoting on the explicitly concatenated feature Gram, and feature-space CPQR selected identical pivot sequences in every bin: 12/12 bins and 96/96 pivot slots. All selected feature blocks have numerical rank eight. CPQR and feature-Gram Cholesky geometry records are exactly equal, as expected for identical selections. The small Gram differences arise from different accumulation paths and pass the configured FP64 closeness check.

This supports CPQR/Cholesky equivalence for these captures, layers, Q32 budgets, and numerical environment. It does not establish all-layer or Q64 equivalence, or identity with the missing historical captures. No new selection manifest was installed into residual fitting.

Condition numbers show no evident selected-block ill-conditioning here. Remaining energy is ||Phi-Phi P_selected||_F^2 / ||Phi||_F^2 in the concatenated whitened feature space. Its approximately 90% value is not K error, Fisher NMSE, missed attention mass, or task error. These measurements alone do not establish low-rank query redundancy or predict sRRQR improvements. With unchanged selections there is no selector change to justify a duplicate residual fit for CPQR on these three layers; sRRQR remains a separate untested hypothesis.

## Execution and provenance

Environment `lowrank` (`/home/lz299/miniconda3/envs/lowrank/bin/python`), direct CPU execution, two OMP/MKL/OpenBLAS/PyTorch threads, no GPU. SciPy 1.15.3, PyTorch 2.8.0+cu128. No warning or error appeared in the audit log.

```bash
source /home/lz299/miniconda3/etc/profile.d/conda.sh
conda activate lowrank
set -o noclobber
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. \
python -u -m evaluation.audit_query_cpqr \
  --candidate-capture results/calibration/cpqr_candidates \
  --layers 0,15,35 --queries-per-bin 8 \
  --output-dir results/evaluation/cpqr_q32 \
  > results/logs/query_cpqr/audit_q32.log 2>&1
```

Output records: `layer_000.json`, `layer_015.json`, `layer_035.json` in this directory. Each contains full pivot sequences, per-bin metrics, source hashes, candidate-manifest hash, versions and execution command. Candidate manifest SHA256: `c7b10fb5cb4088d5ff45f641d508bc3ccfcc31fd7b766bbb5bd8fd87826e0160`.

Logs: `results/logs/query_cpqr/audit_q32.log` and `results/logs/query_cpqr/result_check.log`. The latter rechecks query shapes, sequence equality across all three implementations, identical geometry dictionaries, eight shared positions and selected rank eight in every bin. Original captures and routing checkpoints were not modified.
