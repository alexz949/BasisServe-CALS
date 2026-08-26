# Qwen3-32B layer-23 QR(0) sample scaling with streaming TSQR

Raw layer activations are never written to disk. Fit TSQR checkpoints use 64, 128, and 256 independent C4 windows with 256 sampled positions per window. Every checkpoint is evaluated on the same 64 document-disjoint held-out windows with 128 sampled positions per window.
Decoder solves and metric evaluation use `float64`.

## Command

```bash
python evaluation/run_qwen3_32b_c1_layer23_tsqr_scaling.py --model-path Qwen/Qwen3-32B --windows results/calibration/qwen3_32b_c4_256f64h_s2048/windows.safetensors --factor-dir results/checkpoints/qwen3_32b_c1_v96_als_d1e5 --output-dir results/calibration/q32_l23_tsqr_16k32k64k_l40s --output-json results/evaluation/q32_l23_qr0_sample_scaling_l40s.json --output-markdown results/evaluation/q32_l23_qr0_sample_scaling_l40s.md --fit-windows 256 --heldout-windows 64 --fit-positions-per-window 256 --heldout-positions-per-window 128 --milestones 16384 32768 65536 --position-seed 20260901 --batch-size 1 --model-dtype bfloat16 --attn-implementation sdpa --device cuda:0 --device-map balanced --max-memory-per-gpu-gib 44 --torch-num-threads 4 --output-column-chunk-size 256 --solve-dtype float64
```

## Results

| Fit rows | Fit documents | Status | Fit relMSE | Held-out relMSE | Decoder norm | Min abs R diagonal | R diagonal ratio |
|---:|---:|:---|---:|---:|---:|---:|---:|
| 16384 | 64 | complete | 1.782938237e-02 | 1.738650964e+01 | 2.788475e+04 | 7.263411e-06 | 43715.888 |
| 32768 | 128 | complete | 2.556115526e-02 | 2.127315835e-01 | 2.538538e+03 | 1.626272e-04 | 1921.941 |
| 65536 | 256 | complete | 3.093258222e-02 | 5.523626168e-02 | 7.546250e+02 | 4.255848e-04 | 740.623 |

The stored damped-checkpoint decoder, evaluated on the same new held-out split, has relative MSE `1.271486067e-01`.

## Trend

- From 16384 to 65536 rows, QR(0) held-out MSE changes by -99.682%.
- Decoder norm changes by -97.294%.
- Minimum abs R diagonal changes by 5759.297%.
- Interpretation: strong sample-limited evidence: more independent calibration windows substantially reduce both decoder norm and held-out error
