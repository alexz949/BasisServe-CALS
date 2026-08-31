# Qwen3-32B layer-23 QR(0) sample scaling with streaming TSQR

Raw layer activations are never written to disk. Fit TSQR checkpoints use 64, 128, and 256 independent C4 windows with 256 sampled positions per window. Every checkpoint is evaluated on the same 64 document-disjoint held-out windows with 128 sampled positions per window.
Decoder solves and metric evaluation use `float64`.

## Command

```bash
python evaluation/run_qwen3_32b_c1_layer23_tsqr_scaling.py --model-path Qwen/Qwen3-32B --windows results/calibration/qwen3_32b_c4_256f64h_s2048/windows.safetensors --factor-dir results/checkpoints/qwen3_32b_c1_v96_als_d1e5 --output-dir results/calibration/q32_l23_tsqr_16k32k64k --output-json results/evaluation/q32_l23_qr0_sample_scaling.json --output-markdown results/evaluation/q32_l23_qr0_sample_scaling.md --evaluate-only --device cuda:0 --solve-dtype float64 --output-column-chunk-size 256 --torch-num-threads 4
```

## Results

| Fit rows | Fit documents | Status | Fit relMSE | Held-out relMSE | Decoder norm | Min abs R diagonal | R diagonal ratio |
|---:|---:|:---|---:|---:|---:|---:|---:|
| 16384 | 64 | complete | 5.390420408e+159 | 6.392636036e+159 | 4.010863e+97 | 1.859439e-32 | 2833278415566879238731461558272.000 |
| 32768 | 128 | complete | 4.461522515e+153 | 5.149153326e+153 | 3.403907e+94 | 8.953223e-32 | 588639805126022820728112414720.000 |
| 65536 | 256 | complete | 9.930551386e+149 | 1.117478751e+150 | 5.258108e+92 | 2.658643e-31 | 203597766051507444068516888576.000 |

The stored damped-checkpoint decoder, evaluated on the same new held-out split, has relative MSE `1.066490729e+01`.

## Trend

- From 16384 to 65536 rows, QR(0) held-out MSE changes by -100.000%.
- Decoder norm changes by -99.999%.
- Minimum abs R diagonal changes by 1329.809%.
- Interpretation: strong sample-limited evidence: more independent calibration windows substantially reduce both decoder norm and held-out error
