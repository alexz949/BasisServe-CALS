# Qwen3-32B C1 layer-23 TSQR decoder PPL

The model is loaded once. Dense PPL is measured before installing the uniform rank-96 C1 checkpoint; QR(0) arms then replace only the layer-23 decoder. The padded Hugging Face V/O path measures function quality, not compressed-cache performance.

## Results

| Arm | PPL | Mean NLL | ΔPPL vs dense | ΔPPL vs ridge |
|:---|---:|---:|---:|---:|
| dense | 7.610497336 | 2.029528523 | +0.000000000 | -0.421416614 |
| c1_ridge_d1e5 | 8.031913951 | 2.083422850 | +0.421416614 | +0.000000000 |
| c1_qr0_32768 | 8.029302738 | 2.083097692 | +0.418805401 | -0.002611213 |
| c1_qr0_65536 | 8.026668362 | 2.082769543 | +0.416171025 | -0.005245589 |

## Command

```bash
evaluation/eval_qwen3_32b_c1_layer23_decoder_ppl.py --model Qwen/Qwen3-32B --factor-dir results/checkpoints/qwen3_32b_c1_v96_als_d1e5 --capture-dir results/calibration/q32_l23_tsqr_16k32k64k_l40s --output-json results/evaluation/q32_c1_l23_qr0_32k64k_wikitext2_ppl.json --output-markdown results/evaluation/q32_c1_l23_qr0_32k64k_wikitext2_ppl.md --milestones 32768 65536 --dataset wikitext2 --split test --seqlen 2048 --batch-size 1 --model-dtype bfloat16 --attn-implementation sdpa --device cuda:0 --device-map balanced --max-memory-per-gpu-gib 44 --torch-num-threads 4 --output-column-chunk-size 256
```
