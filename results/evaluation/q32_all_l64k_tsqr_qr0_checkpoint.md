# Qwen3-32B all-layer 64k TSQR QR(0) checkpoint

All 64 Value encoders are fixed to the original uniform rank-96 C1 checkpoint. Every full-layer decoder is refit from 65,536 C4 rows with float32 streaming TSQR and a float64 square-root solve at `lambda=0`. Artifacts are stored in BF16.

- Checkpoint: `results/checkpoints/qwen3_32b_c1_v96_tsqr64k_qr0`
- Mean fit relMSE: `1.334718986e-01`
- Mean held-out relMSE: `2.416111942e+00`
- Base mean held-out relMSE: `1.049192073e-01`
- Layers improved on held-out: `61/64`

## WikiText-2 perplexity

The complete WikiText-2 test split was evaluated at sequence length 2,048 with
FP32 loss accumulation. The all-layer QR(0) checkpoint remained finite despite
the activation-space outliers in layers 9, 14, and 17.

| Checkpoint | PPL | Delta from ridge | Delta from dense |
|---|---:|---:|---:|
| Dense Qwen3-32B | `7.610497336` | - | - |
| Uniform C1 rank-96 ridge | `8.031913951` | - | `+0.421416615` |
| Uniform C1 rank-96 all-layer TSQR QR(0) | `8.051581866` | `+0.019667915` (`+0.245%`) | `+0.441084530` (`+5.796%`) |

- Evaluation tokens: `298862`
- Evaluation chunks: `146`
- Evaluation runtime: `128.507 s`
- Slurm job: `8277965` (`4x NVIDIA L40S`, completed with exit code `0:0`)
- Result JSON: `results/evaluation/qwen3_32b_c1_v96_tsqr64k_qr0_wikitext2_ppl.json`
- Log: `results/logs/c1_all_qr0_ppl_8277965.log`

The PPL degradation is much smaller than the worst per-layer held-out relative
MSE suggests. This is evidence that the downstream network suppresses much of
the error from the three ill-conditioned decoder fits. Pure QR(0) is still
worse than the ridge checkpoint, so the next useful checkpoint is a hybrid:
retain QR(0) for the 61 layers that improve held-out error and retain the ridge
decoder for layers 9, 14, and 17.

The exact PPL command was:

```bash
python -u evaluation/eval_qwen3_32b_c1_wikitext.py --model Qwen/Qwen3-32B --factor-dir results/checkpoints/qwen3_32b_c1_v96_tsqr64k_qr0 --output-json results/evaluation/qwen3_32b_c1_v96_tsqr64k_qr0_wikitext2_ppl.json --dataset wikitext2 --split test --seqlen 2048 --batch-size 1 --model-dtype bfloat16 --attn-implementation sdpa --device-map balanced --max-memory-per-gpu-gib 44 --torch-num-threads 4
```

## Per-layer results

| Layer | Fit relMSE | Held-out relMSE | Base held-out | Decoder norm | R diagonal ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.033012429e-03 | 5.380713781e-03 | 9.379329268e-03 | 1.725745e+02 | 79.820 |
| 1 | 2.428886871e-03 | 3.864010360e-03 | 7.305060854e-03 | 1.510359e+02 | 415.992 |
| 2 | 7.324201002e-03 | 1.155469903e-02 | 1.970768258e-02 | 1.463354e+02 | 69.419 |
| 3 | 9.826727514e-03 | 1.605956553e-02 | 2.541321393e-02 | 1.418435e+02 | 73.710 |
| 4 | 1.378259664e-02 | 2.316605709e-02 | 3.816828974e-02 | 1.521669e+02 | 83.525 |
| 5 | 9.888524910e-03 | 1.612659873e-02 | 2.790641373e-02 | 2.335975e+02 | 568.871 |
| 6 | 6.980466374e-03 | 1.365488000e-02 | 2.237960348e-02 | 1.524158e+02 | 129.595 |
| 7 | 2.919290376e-02 | 5.779431555e-02 | 9.693175503e-02 | 1.518725e+02 | 45.653 |
| 8 | 2.277162389e-02 | 3.796246160e-02 | 6.524641091e-02 | 1.554843e+02 | 73.892 |
| 9 | 1.247570623e-02 | 5.703378058e-01 | 3.818978126e-02 | 3.962477e+04 | 139738.414 |
| 10 | 1.880001501e-02 | 2.835884169e-02 | 4.662178769e-02 | 1.504532e+02 | 63.637 |
| 11 | 4.162618855e-02 | 6.086670872e-02 | 9.776427679e-02 | 1.575265e+02 | 75.605 |
| 12 | 2.048978866e-02 | 3.306775098e-02 | 6.206881917e-02 | 9.148405e+02 | 1194.661 |
| 13 | 2.844046783e-02 | 4.675320102e-02 | 7.197998545e-02 | 1.508904e+02 | 86.249 |
| 14 | 3.124351183e-02 | 5.504158743e+00 | 6.947892334e-02 | 4.520301e+05 | 1161346.586 |
| 15 | 1.918224194e-02 | 2.825693808e-02 | 4.321322535e-02 | 1.518061e+02 | 156.603 |
| 16 | 1.566771988e-02 | 2.447077216e-02 | 5.066336063e-02 | 5.118627e+02 | 874.119 |
| 17 | 6.221447414e+00 | 1.445391320e+02 | 2.558844200e-02 | 2.453118e+07 | 76039237.466 |
| 18 | 1.206323531e-02 | 2.040628586e-02 | 3.417637910e-02 | 3.872625e+02 | 915.075 |
| 19 | 1.923193279e-02 | 2.942733567e-02 | 4.894466332e-02 | 3.918894e+02 | 694.542 |
| 20 | 1.870439159e-02 | 3.486417207e-02 | 8.009031470e-02 | 8.112876e+02 | 2328.430 |
| 21 | 2.661912724e-02 | 4.216502148e-02 | 6.867789721e-02 | 1.949427e+02 | 211.741 |
| 22 | 3.095248092e-02 | 4.957146180e-02 | 7.280544430e-02 | 1.253646e+03 | 1412.006 |
| 23 | 3.093490322e-02 | 5.523899298e-02 | 1.271486067e-01 | 7.546250e+02 | 740.623 |
| 24 | 4.855005877e-02 | 8.196768455e-02 | 1.298697165e-01 | 1.534937e+02 | 56.531 |
| 25 | 3.705358091e-02 | 6.478189242e-02 | 1.203412833e-01 | 1.870538e+02 | 146.005 |
| 26 | 3.504981929e-02 | 6.389296482e-02 | 9.675503392e-02 | 3.207303e+02 | 284.651 |
| 27 | 4.176107755e-02 | 6.670295881e-02 | 1.071599838e-01 | 2.292543e+02 | 250.296 |
| 28 | 5.755437641e-02 | 1.020374168e-01 | 1.904603998e-01 | 2.209169e+02 | 239.874 |
| 29 | 4.644676654e-02 | 7.626579041e-02 | 1.265232174e-01 | 1.575967e+02 | 164.232 |
| 30 | 7.250971965e-02 | 1.271172204e-01 | 2.208745633e-01 | 2.011438e+02 | 187.691 |
| 31 | 5.834417380e-02 | 1.071534137e-01 | 1.596043610e-01 | 8.934511e+02 | 1042.490 |
| 32 | 3.330338336e-02 | 5.337611220e-02 | 8.320172795e-02 | 1.478991e+02 | 58.782 |
| 33 | 2.945559025e-02 | 4.589307503e-02 | 6.664901524e-02 | 1.842881e+03 | 2699.697 |
| 34 | 4.024821390e-02 | 5.812422132e-02 | 9.200295511e-02 | 1.458310e+02 | 27.650 |
| 35 | 6.213955455e-02 | 9.321001102e-02 | 1.439995794e-01 | 1.476420e+02 | 21.323 |
| 36 | 4.556392366e-02 | 6.887964097e-02 | 1.002167931e-01 | 1.462177e+02 | 74.606 |
| 37 | 5.045337454e-02 | 8.456260691e-02 | 1.229091412e-01 | 1.482940e+02 | 32.331 |
| 38 | 5.078849410e-02 | 9.087919210e-02 | 1.299053873e-01 | 6.694267e+02 | 711.909 |
| 39 | 4.547386058e-02 | 7.031436040e-02 | 1.025004371e-01 | 1.564718e+02 | 143.267 |
| 40 | 4.262143863e-02 | 6.695406112e-02 | 1.018969815e-01 | 2.169979e+02 | 496.722 |
| 41 | 2.997048698e-02 | 4.967007503e-02 | 7.094542719e-02 | 1.453283e+02 | 27.345 |
| 42 | 4.543661481e-02 | 7.250788558e-02 | 1.052005079e-01 | 1.460725e+02 | 70.828 |
| 43 | 5.277781940e-02 | 7.582078829e-02 | 1.103293506e-01 | 1.386889e+02 | 14.516 |
| 44 | 5.700287073e-02 | 8.581183893e-02 | 1.255111880e-01 | 1.436230e+02 | 36.752 |
| 45 | 4.944382393e-02 | 7.910944618e-02 | 1.146794389e-01 | 1.443200e+02 | 25.172 |
| 46 | 5.236022288e-02 | 8.062791595e-02 | 1.156975057e-01 | 4.608557e+02 | 384.919 |
| 47 | 5.964126144e-02 | 9.439863137e-02 | 1.410338317e-01 | 1.442333e+02 | 21.391 |
| 48 | 4.695938644e-02 | 7.821196970e-02 | 1.146551000e-01 | 1.475789e+02 | 35.125 |
| 49 | 4.961866909e-02 | 8.048057631e-02 | 1.179656930e-01 | 1.432093e+02 | 17.026 |
| 50 | 4.877236499e-02 | 8.388500402e-02 | 1.224046986e-01 | 1.454243e+02 | 24.059 |
| 51 | 4.724206531e-02 | 6.884599531e-02 | 1.014954393e-01 | 1.431431e+02 | 22.188 |
| 52 | 5.369369497e-02 | 9.147459865e-02 | 1.327435632e-01 | 1.559308e+02 | 83.191 |
| 53 | 4.390105619e-02 | 7.140234373e-02 | 1.063341668e-01 | 1.592553e+02 | 121.848 |
| 54 | 6.749965650e-02 | 1.103281117e-01 | 1.664834082e-01 | 1.496980e+02 | 38.913 |
| 55 | 5.145316215e-02 | 9.588134040e-02 | 1.410173335e-01 | 1.435886e+02 | 19.390 |
| 56 | 4.458862540e-02 | 1.042351831e-01 | 1.720447612e-01 | 1.506171e+02 | 61.106 |
| 57 | 6.442292331e-02 | 1.165970057e-01 | 1.757567957e-01 | 1.437224e+02 | 25.882 |
| 58 | 4.730674936e-02 | 1.145758145e-01 | 1.974788733e-01 | 1.512052e+02 | 45.331 |
| 59 | 4.027627319e-02 | 1.034915393e-01 | 1.613458536e-01 | 1.471522e+02 | 22.078 |
| 60 | 4.191529711e-02 | 1.355282498e-01 | 2.663476737e-01 | 1.675168e+02 | 136.902 |
| 61 | 3.476193405e-02 | 7.961211708e-02 | 1.308681642e-01 | 1.505350e+02 | 73.373 |
| 62 | 4.802231456e-02 | 1.539661670e-01 | 4.280882028e-01 | 1.834371e+02 | 310.446 |
| 63 | 1.470876070e-02 | 2.994976760e-02 | 5.168204732e-02 | 1.625430e+02 | 38.144 |

## Command

```bash
python evaluation/run_qwen3_32b_c1_all_layer_tsqr_checkpoint.py --model-path Qwen/Qwen3-32B --windows results/calibration/qwen3_32b_c4_256f64h_s2048/windows.safetensors --factor-dir results/checkpoints/qwen3_32b_c1_v96_als_d1e5 --capture-dir results/calibration/q32_all_l64k_tsqr_l40s --checkpoint-dir results/checkpoints/qwen3_32b_c1_v96_tsqr64k_qr0 --output-markdown results/evaluation/q32_all_l64k_tsqr_qr0_checkpoint.md --fit-windows 256 --heldout-windows 64 --fit-positions-per-window 256 --heldout-positions-per-window 128 --position-seed 20260901 --batch-size 1 --model-dtype bfloat16 --attn-implementation sdpa --device-map balanced --max-memory-per-gpu-gib 44 --torch-num-threads 4 --output-column-chunk-size 256
```
