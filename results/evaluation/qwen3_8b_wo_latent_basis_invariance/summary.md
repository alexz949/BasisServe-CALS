# Qwen3-8B Wo-C1 latent-basis invariance

Each source-private latent basis is independently rotated as `E'_p = E_p Q_p`, `D'_p = Q_p^T D_p`.

## Aggregate

- Layers: `36`; rotation trials: `108`
- Maximum FP32 product relative L2: `1.0470702e-06`
- Maximum FP32 heldout-output relative L2: `1.2788858e-06`
- Median BF16 deployment-probe relative L2 after factor requantization: `0.0040940396`
- Maximum BF16 deployment-probe relative L2 after factor requantization: `0.0045548934`

| Layer | FP32 product rel-L2 max | FP32 heldout rel-L2 max | BF16 probe rel-L2 median | BF16 probe rel-L2 max |
|---:|---:|---:|---:|---:|
| 0 | 1.0062878e-06 | 4.7356079e-07 | 0.0040511223 | 0.00405451 |
| 1 | 1.0192534e-06 | 7.9690943e-07 | 0.0039839926 | 0.0039863936 |
| 2 | 1.0193153e-06 | 8.551661e-07 | 0.0040309248 | 0.0040382212 |
| 3 | 1.0186002e-06 | 8.3902295e-07 | 0.0040166527 | 0.0040184325 |
| 4 | 1.0123096e-06 | 8.5620049e-07 | 0.0040330826 | 0.0040339753 |
| 5 | 1.0182702e-06 | 8.9628361e-07 | 0.0039790468 | 0.0039801509 |
| 6 | 1.019046e-06 | 9.2416463e-07 | 0.0040687844 | 0.0040698405 |
| 7 | 1.0155176e-06 | 9.6714879e-07 | 0.0040682638 | 0.0040784231 |
| 8 | 1.0152772e-06 | 1.0303961e-06 | 0.0040539503 | 0.0040577869 |
| 9 | 1.0223575e-06 | 1.0701057e-06 | 0.0040717642 | 0.004073773 |
| 10 | 1.0209168e-06 | 1.1067179e-06 | 0.0040950063 | 0.0040965527 |
| 11 | 1.0098432e-06 | 1.0925043e-06 | 0.0040738727 | 0.0040788092 |
| 12 | 1.0129758e-06 | 1.0836098e-06 | 0.0041399742 | 0.0041400637 |
| 13 | 1.0470702e-06 | 1.2788858e-06 | 0.0045545544 | 0.0045548934 |
| 14 | 1.0016344e-06 | 1.1575364e-06 | 0.004142846 | 0.004143842 |
| 15 | 1.0076333e-06 | 1.2200592e-06 | 0.0041589881 | 0.0041635414 |
| 16 | 1.0098378e-06 | 1.1625355e-06 | 0.004195618 | 0.0041976599 |
| 17 | 1.0037263e-06 | 1.1364802e-06 | 0.0041732136 | 0.0041827434 |
| 18 | 1.0116485e-06 | 1.21314e-06 | 0.0041232775 | 0.0041281008 |
| 19 | 1.015064e-06 | 1.0458121e-06 | 0.0041695987 | 0.0041714287 |
| 20 | 1.0063618e-06 | 1.2010569e-06 | 0.004186722 | 0.0041912147 |
| 21 | 1.0117944e-06 | 1.233084e-06 | 0.0041757929 | 0.0041762274 |
| 22 | 1.0222677e-06 | 1.1523915e-06 | 0.0042528696 | 0.0042548189 |
| 23 | 1.0138439e-06 | 1.0731834e-06 | 0.0040713996 | 0.004071455 |
| 24 | 1.0183826e-06 | 1.0163734e-06 | 0.0042152228 | 0.0042178044 |
| 25 | 1.0057765e-06 | 1.024119e-06 | 0.0040937876 | 0.0040942915 |
| 26 | 1.0053162e-06 | 1.0206448e-06 | 0.0041148453 | 0.0041184789 |
| 27 | 1.0059476e-06 | 1.0482241e-06 | 0.00412498 | 0.0041250545 |
| 28 | 1.0110415e-06 | 1.1348214e-06 | 0.0041043344 | 0.0041114045 |
| 29 | 1.0043133e-06 | 9.6959693e-07 | 0.0040649087 | 0.0040656156 |
| 30 | 1.0013573e-06 | 9.497134e-07 | 0.0040906072 | 0.0040906174 |
| 31 | 1.0048632e-06 | 1.0209539e-06 | 0.0041080811 | 0.0041093486 |
| 32 | 1.0055815e-06 | 9.6673324e-07 | 0.004048273 | 0.0040499335 |
| 33 | 9.9575789e-07 | 9.5428218e-07 | 0.0040531135 | 0.0040549883 |
| 34 | 1.0140982e-06 | 7.6317634e-07 | 0.0041018138 | 0.004103696 |
| 35 | 1.0202366e-06 | 7.1456764e-07 | 0.0040324568 | 0.0040559154 |

FP32 uses the complete heldout covariance and tests the same linear operator.
BF16 uses the deployed two-GEMM ordering with deterministic RMS-scaled probes. Its difference includes factor requantization and floating-point operation ordering, so it is a numerical gauge-sensitivity result, not representational error.

## Command

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_wo_latent_basis_invariance.py --phase1-dir results/checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s100 --covariance-dir results/calibration/qwen3_8b_c1_256f64h_full2048_cov --output-dir results/evaluation/qwen3_8b_wo_latent_basis_invariance --layers all --seeds 20260828,20260829,20260830 --bf16-probe-rows 256 --device cuda:0 --torch-num-threads 4
```
