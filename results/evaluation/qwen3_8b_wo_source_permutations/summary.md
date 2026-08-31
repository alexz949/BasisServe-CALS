# Qwen3-8B Wo-C1 source-permutation sanity test

The four stored BF16 `(encoder, decoder)` pairs are kept intact and exhaustively reassigned to the four logical TP input sources. No factor is refitted.

## Aggregate

- Layers: `36`
- Non-identity comparisons: `828`
- Comparisons strictly worse than the original mapping: `828`
- Layers where the original mapping is strictly best: `36`
- Mean original heldout relative MSE: `0.045983258`
- Mean permuted heldout relative MSE: `1.3098302`
- Median permutation/original MSE ratio: `29.7794x`

| Moved sources | Comparisons | Mean MSE | Mean gap | Median ratio |
|---:|---:|---:|---:|---:|
| 2 | 216 | 0.8321404 | 0.78615714 | 19.0039x |
| 3 | 288 | 1.2491299 | 1.2031466 | 28.6718x |
| 4 | 324 | 1.6822459 | 1.6362627 | 38.3597x |

## Per-layer

| Layer | Original MSE | Best wrong MSE | Mean wrong MSE | Worst wrong MSE | Best wrong ratio | Original best |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.0028626256 | 0.19520065 | 0.85200193 | 1.2794741 | 68.1894x | yes |
| 1 | 0.017374673 | 0.4596092 | 1.0731425 | 1.4626964 | 26.4528x | yes |
| 2 | 0.020727301 | 0.60301619 | 1.0955453 | 1.4473253 | 29.0928x | yes |
| 3 | 0.023332689 | 0.54818027 | 1.0925124 | 1.4528355 | 23.4941x | yes |
| 4 | 0.036498907 | 0.59377272 | 1.1289389 | 1.4826663 | 16.2682x | yes |
| 5 | 0.023886366 | 0.61657889 | 1.1698261 | 1.5282003 | 25.813x | yes |
| 6 | 0.03626938 | 0.64043327 | 1.1674445 | 1.5343118 | 17.6577x | yes |
| 7 | 0.042677772 | 0.71443176 | 1.2811389 | 1.66289 | 16.7401x | yes |
| 8 | 0.060725169 | 0.68564127 | 1.3385671 | 1.7310057 | 11.2909x | yes |
| 9 | 0.036515425 | 0.70500682 | 1.439119 | 1.960668 | 19.3071x | yes |
| 10 | 0.054282573 | 0.8167482 | 1.3789366 | 1.8476579 | 15.0462x | yes |
| 11 | 0.059287486 | 0.78981784 | 1.3911058 | 1.8274888 | 13.3218x | yes |
| 12 | 0.037992594 | 0.71451317 | 1.3551835 | 1.7831558 | 18.8066x | yes |
| 13 | 0.040307937 | 0.82073477 | 1.9186628 | 3.0318614 | 20.3616x | yes |
| 14 | 0.052164867 | 0.79486517 | 1.463808 | 1.8847008 | 15.2376x | yes |
| 15 | 0.048403509 | 0.99325836 | 1.5684472 | 2.0614278 | 20.5204x | yes |
| 16 | 0.057821913 | 0.62704287 | 1.4803529 | 1.9179309 | 10.8444x | yes |
| 17 | 0.04131724 | 0.68664053 | 1.378553 | 1.8213762 | 16.6187x | yes |
| 18 | 0.052146767 | 0.70665309 | 1.3831407 | 1.7939455 | 13.5512x | yes |
| 19 | 0.030603799 | 0.53531162 | 1.2791076 | 1.6979399 | 17.4917x | yes |
| 20 | 0.05663201 | 0.82803937 | 1.6537596 | 2.2998382 | 14.6214x | yes |
| 21 | 0.054737069 | 0.68290775 | 1.56251 | 2.0423285 | 12.4761x | yes |
| 22 | 0.029937178 | 0.62139607 | 1.3984651 | 1.8753183 | 20.7567x | yes |
| 23 | 0.048400847 | 0.70550439 | 1.3686189 | 1.8135793 | 14.5763x | yes |
| 24 | 0.030758753 | 0.55579419 | 1.1922957 | 1.5696748 | 18.0695x | yes |
| 25 | 0.054661233 | 0.76190823 | 1.3592122 | 1.7533133 | 13.9387x | yes |
| 26 | 0.075946314 | 0.74995953 | 1.3213326 | 1.7059914 | 9.87486x | yes |
| 27 | 0.060546889 | 0.75066252 | 1.3852646 | 1.8032112 | 12.398x | yes |
| 28 | 0.066278503 | 0.85924516 | 1.417441 | 1.8325986 | 12.9642x | yes |
| 29 | 0.070649435 | 0.65201795 | 1.2803641 | 1.6407413 | 9.22892x | yes |
| 30 | 0.051969803 | 0.47549862 | 1.2149928 | 1.577298 | 9.14952x | yes |
| 31 | 0.083187924 | 0.76045307 | 1.2952521 | 1.6916675 | 9.14139x | yes |
| 32 | 0.069568372 | 0.6421616 | 1.2655399 | 1.6460438 | 9.23065x | yes |
| 33 | 0.096674845 | 0.66503041 | 1.2481946 | 1.6197295 | 6.87904x | yes |
| 34 | 0.019490106 | 0.26164482 | 1.0516377 | 1.4648893 | 13.4245x | yes |
| 35 | 0.010759025 | 0.26708522 | 0.903472 | 1.2931244 | 24.8243x | yes |

These are heldout attention-layer output errors, not terminal-logit KL or PPL.
The checkpoint sweep was selected using this heldout split; the source mapping itself was fixed a priori and was not selected from the permutation family.

## Command

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_wo_source_permutations.py --phase1-dir results/checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s100 --covariance-dir results/calibration/qwen3_8b_c1_256f64h_full2048_cov --output-dir results/evaluation/qwen3_8b_wo_source_permutations --layers all --device cuda:0 --work-dtype float64 --torch-num-threads 4
```
