# Qwen3-32B layer-23 QR(0) FP64 precision check

This check reruns only the unregularized layer-23 square-root solve in FP64. It uses the same raw BF16 activation snapshot, fixed BF16 encoder `A`, fit rows, and held-out rows as the saved FP32 experiment.

## Command

```bash
python evaluation/check_qwen3_32b_c1_layer23_qr0_fp64.py --snapshot-dir results/calibration/qwen3_32b_c1_128f64h_p128 --factor-dir results/checkpoints/qwen3_32b_c1_v96_als_d1e5 --fp32-reference-json results/evaluation/qwen3_32b_c1_fixed_a_pairwise_qr_l23_l0.json --qr-block-rows 8192 --evaluation-row-chunk-size 256 --output-column-chunk-size 256 --device cuda:0 --output-json results/evaluation/qwen3_32b_c1_layer23_qr0_fp64_check.json --output-markdown results/evaluation/qwen3_32b_c1_layer23_qr0_fp64_check.md
```

## FP32 versus FP64

| Metric | FP32 | FP64 | FP64 relative change |
|---|---:|---:|---:|
| Fit relative MSE | 1.849763567992e-02 | 1.849763566036e-02 | -0.000000% |
| Held-out relative MSE | 4.089572460683e+00 | 4.089556428191e+00 | -0.000392% |
| Decoder Frobenius norm | 8.540025006320e+03 | 8.540023697700e+03 | -0.000015% |
| Decoder maximum absolute value | 7.993603515625e+01 | 7.993066981532e+01 | -0.006712% |
| Minimum abs R diagonal | 6.655701145064e-05 | 6.655694234019e-05 | -0.000104% |
| R diagonal ratio | 4.663590476376e+03 | 4.663595069369e+03 | 0.000098% |

## Decision

- Maximum FP32/FP64 relative difference over fit MSE, held-out MSE, and decoder norm: 0.000392%.
- FP64 held-out MSE remains 28.871x the damped Cholesky baseline.
- FP64 decoder norm remains 10.710x the damped Cholesky baseline.
- FP32 numerical error excluded as the primary explanation: **True**.

FP64 reproduces the large-norm, poor-held-out QR(0) solution. The failure therefore comes from the unregularized statistical objective rather than FP32 roundoff.

The original FP32 decoder tensor was not persisted, so this check compares output metrics and decoder norms rather than an elementwise decoder distance. That is sufficient for testing whether FP32 roundoff caused the catastrophic held-out behavior.
