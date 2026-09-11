# All-layer Q32: CPQR and strong-RRQR results

Completed 2026-09-10. Both GPU capture workers, CPU merge, CPQR audit, sRRQR audit and final result checks completed with exit code 0. No retries were needed.

## Fixed protocol

Local dense Qwen3-8B-Base; 64 fixed C4 fit windows of 32768 tokens, layers 0–35, 512 candidate positions/layer/window. BF16 captured queries retain all 32 heads of width 128. FP64 per-head uncentered whitening, epsilon 1e-6; four 8K bins each select eight positions (Q32). No validation queries, residual fitting, recall evaluation or downstream benchmark entered this experiment.

CPQR uses the explicit 262144-dimensional window/head-concatenated features. Strong RRQR uses their full-spectrum 128x128 bin-Gram square-root isometry and independently starts from matching CPQR at f=2 and f=1.01, with at most 512 swaps. Initial selections and baseline geometry are checked against the full-feature results.

## Aggregate result

| Comparison | Identical bins / 144 | Swaps |
|---|---:|---:|
| CPQR versus production Gram pivoting | 144 | n/a |
| Strong RRQR f=2 versus CPQR | 144 | 0 |
| Strong RRQR f=1.01 versus CPQR | 144 | 0 |

All 288 strong-RRQR cases satisfy their stopping bound. Maximum initial rho is 1.004078783580 at layer 1, zero-based bin 0; its best initial single-swap volume gain is 0.407878%.
This is below the f=1.01 threshold; a no-swap outcome does not prove global maximum volume or rule out smaller improving swaps. Both tested methods retain all 1152 selected position slots and their within-bin order when the identical-bin count is 144.

Maximum explicit-feature/production Gram absolute difference: 6.536993e-12. Maximum full-root Gram difference: 6.394885e-13. Root eigenvalues clipped: 0.
Selected-block condition numbers: 1.110502–1.525972. Remaining projection-energy fraction: 0.895644–0.923798. These are whitened concatenated-feature metrics, not routing loss or task scores.
New versus old capture agrees bitwise on 192/192 overlapping window/layer tensors (64 windows x layers 0/15/35). Input token hashes also agree.

## Per-layer results

| Layer | CPQR identical bins | Max initial rho | f=2 swaps | f=1.01 swaps |
|---|---:|---:|---:|---:|
| 0 | 4/4 | 0.9997986173 | 0 | 0 |
| 1 | 4/4 | 1.0040787836 | 0 | 0 |
| 2 | 4/4 | 0.9997879937 | 0 | 0 |
| 3 | 4/4 | 1.0000683175 | 0 | 0 |
| 4 | 4/4 | 0.9996495219 | 0 | 0 |
| 5 | 4/4 | 1.0011824779 | 0 | 0 |
| 6 | 4/4 | 0.9998446278 | 0 | 0 |
| 7 | 4/4 | 1.0018002198 | 0 | 0 |
| 8 | 4/4 | 0.9996136704 | 0 | 0 |
| 9 | 4/4 | 1.0033652658 | 0 | 0 |
| 10 | 4/4 | 1.0021794579 | 0 | 0 |
| 11 | 4/4 | 1.0009645011 | 0 | 0 |
| 12 | 4/4 | 1.0016604162 | 0 | 0 |
| 13 | 4/4 | 1.0024248396 | 0 | 0 |
| 14 | 4/4 | 1.0008650263 | 0 | 0 |
| 15 | 4/4 | 1.0018017149 | 0 | 0 |
| 16 | 4/4 | 0.9999764419 | 0 | 0 |
| 17 | 4/4 | 1.0026718330 | 0 | 0 |
| 18 | 4/4 | 1.0024252499 | 0 | 0 |
| 19 | 4/4 | 1.0007718972 | 0 | 0 |
| 20 | 4/4 | 1.0005183818 | 0 | 0 |
| 21 | 4/4 | 0.9996923193 | 0 | 0 |
| 22 | 4/4 | 0.9998807503 | 0 | 0 |
| 23 | 4/4 | 0.9996849133 | 0 | 0 |
| 24 | 4/4 | 1.0014541401 | 0 | 0 |
| 25 | 4/4 | 1.0019617730 | 0 | 0 |
| 26 | 4/4 | 1.0001366312 | 0 | 0 |
| 27 | 4/4 | 1.0004536924 | 0 | 0 |
| 28 | 4/4 | 0.9999174154 | 0 | 0 |
| 29 | 4/4 | 1.0012729923 | 0 | 0 |
| 30 | 4/4 | 1.0007537764 | 0 | 0 |
| 31 | 4/4 | 1.0000388226 | 0 | 0 |
| 32 | 4/4 | 1.0002394442 | 0 | 0 |
| 33 | 4/4 | 1.0005967701 | 0 | 0 |
| 34 | 4/4 | 1.0018311414 | 0 | 0 |
| 35 | 4/4 | 0.9999539288 | 0 | 0 |

## Execution and provenance

Environment: lowrank. Direct capture on A100 GPUs 5 and 6, two independent even/odd window workers, two CPU/OMP/MKL/OpenBLAS threads each. Both CPU audits use two threads, no GPU. Captured Q payload totals 9 GiB. Peak allocated GPU memory per capture worker was 20.157 GiB. Only capture emitted the torch_dtype deprecation warning; the audit logs have no warnings or failures.

Full capture/merge/audit commands and paths: [protocol](../../../docs/query_all_layers_protocol.md). Actual audit program commands (environment and thread limits as above):

```bash
/home/lz299/BasisServe-CALS/evaluation/audit_query_cpqr.py --candidate-capture results/calibration/cpqr_all_candidates --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35 --queries-per-bin 8 --output-dir results/evaluation/cpqr_all_q32
/home/lz299/BasisServe-CALS/evaluation/audit_query_srrqr.py --candidate-capture results/calibration/cpqr_all_candidates --cpqr-reference results/evaluation/cpqr_all_q32 --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35 --bounds 2,1.01 --max-swaps 512 --output-dir results/evaluation/srrqr_all_q32
```

Output: this directory contains 36 layer JSONs, audit.json and bins.csv (144 detailed bin rows). CPQR references are in results/evaluation/cpqr_all_q32. Capture is in results/calibration/cpqr_all_candidates. Original three-layer captures, results and router banks are preserved.

Candidate manifest SHA256: fbbc190c863dd5d8bd44079ee06e7c139fea360829828b47b509447bd382e05d. audit.json records all 72 result hashes. Input/reference/source hashes were rechecked.

Logs: results/logs/query_cpqr/capture_all_0.log, capture_all_1.log, capture_all_merge.log, audit_cpqr_all_q32.log, audit_srrqr_all_q32.log and all_result_check.log.

Within this all-layer Q32 experiment, the tested sRRQR bounds yield no new query subset, so rerunning residual fitting would not be an ablation of a changed selector. Tighter bounds, different Q budgets, different features or weights, and downstream behavior remain untested.
