# Qwen3-32B C1 fixed-A pairwise QR experiment

This experiment changes only the decoder solver. Each layer uses the same BF16 encoder `A` loaded from the existing V96/damping-1e-5 checkpoint.

The damped square-root problem uses `delta = 1e-5 * tr(C) / 8192`:

```text
[R_X B; sqrt(delta) B] D ~= [R_X W; sqrt(delta) W]
```

It is therefore objective-equivalent to the current covariance damping, while avoiding decoder normal equations.

## Command

```bash
python evaluation/compare_qwen3_32b_c1_pairwise_qr.py --snapshot-dir results/calibration/qwen3_32b_c1_128f64h_p128 --factor-dir results/checkpoints/qwen3_32b_c1_v96_als_d1e5 --layers 23 0 --relative-damping 1e-5 --qr-block-rows 8192 --evaluation-row-chunk-size 256 --output-column-chunk-size 256 --work-dtype float32 --device cuda:0 --output-json results/evaluation/qwen3_32b_c1_fixed_a_pairwise_qr_l23_l0.json --output-markdown results/evaluation/qwen3_32b_c1_fixed_a_pairwise_qr_l23_l0.md
```

## Accuracy

| Layer | Solver | Fit relative MSE | Held-out relative MSE | Decoder norm |
|---:|---|---:|---:|---:|
| 23 | Cholesky + 1e-5 | 1.886032201e-02 | 1.416497377e-01 | 7.973691e+02 |
| 23 | Pairwise QR + lambda=0 | 1.849763568e-02 | 4.089572461e+00 | 8.540025e+03 |
| 23 | Pairwise QR + 1e-5 augmentation | 1.885573576e-02 | 1.416084985e-01 | 7.971211e+02 |
| 0 | Cholesky + 1e-5 | 1.647717930e-03 | 7.433591336e-03 | 2.175010e+02 |
| 0 | Pairwise QR + lambda=0 | 1.645772876e-03 | 7.443604904e-03 | 2.185022e+02 |
| 0 | Pairwise QR + 1e-5 augmentation | 1.645789546e-03 | 7.428196168e-03 | 2.170516e+02 |

## Solver agreement and timing

| Layer | Absolute delta | QR(damped) vs Cholesky decoder | QR(0) vs Cholesky decoder | Peak GPU GiB |
|---:|---:|---:|---:|---:|
| 23 | 1.086103507e-07 | 7.134248869e-03 | 1.047446874e+01 | 2.401 |
| 0 | 8.256858885e-09 | 3.440740833e-02 | 3.608884925e-02 | 2.409 |

| Layer | Cholesky baseline | Activation TSQR (s) | QR lambda=0 (s) | QR augmented (s) |
|---:|---|---:|---:|---:|
| 23 | existing checkpoint (no solve) | 1.249 | 0.588 | 0.869 |
| 0 | existing checkpoint (no solve) | 0.933 | 0.583 | 0.872 |

## Conclusion

- On layer 23, QR without damping lowers raw fit MSE by 1.923%, but held-out MSE becomes 28.871x the baseline and decoder norm becomes 10.710x larger.
- Matched damping restores layer-23 held-out MSE to within 0.029% of the checkpoint baseline.
- On control layer 0, removing damping changes held-out MSE by only 0.135%.
- Therefore square-root/TSQR removes normal equations as a numerical liability, but layer 23 still needs ridge as statistical regularization. The clean formulation is: QR provides stability; lambda controls generalization.

The baseline decoder is the stored BF16 checkpoint artifact. Decoder-space distances also include checkpoint factor quantization, so output MSE is the primary solver comparison. The held-out baseline metrics reproduce the checkpoint records to 1.9 ppm (layer 23) and 4.2 ppm (layer 0); the reconstructed absolute damping agrees to 1.8 ppm and 3.1 ppm respectively.
