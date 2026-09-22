# Qwen3-8B Wo-only C1 AllGather vs LR-AllReduce rank sweep

All headline pairs have equal ideal TP4 ring traffic. Quality is measured after folding each factorized map into an equivalent BF16 `o_proj`; V and the KV cache remain dense.

| AG rank | Retained | AR rank | Bytes/row/rank | AG PPL | AR PPL | AG MCQ | AR MCQ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 12.50% | 256 | 768 | 44.180626 | 107.069514 | 40.683 | 38.363 |
| 192 | 18.75% | 384 | 1152 | 13.688772 | 31.726789 | 52.374 | 42.136 |
| 256 | 25.00% | 512 | 1536 | 9.404409 | 14.430708 | 60.090 | 50.256 |
| 384 | 37.50% | 768 | 2304 | 7.678032 | 8.396637 | 70.974 | 58.042 |
| 512 | 50.00% | 1024 | 3072 | 7.281949 | 7.546591 | 71.177 | 70.791 |
| 640 | 62.50% | 1280 | 3840 | 7.120592 | 7.227802 | 70.728 | 70.939 |
| 768 | 75.00% | 1536 | 4608 | 7.048183 | 7.115327 | 70.398 | 70.728 |
| 896 | 87.50% | 1792 | 5376 | 7.009468 | 7.074950 | 70.590 | 70.577 |
| 1024 | 100.00% | 2048 | 6144 | 7.002026 | 7.039705 | 70.430 | 70.941 |

Dense reference: PPL `7.00202554`, MCQ `70.430`.

MCQ is the unweighted mean of ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande, BoolQ, and OpenBookQA (zero-shot).

Plot-ready long-form data: `figure_data.csv`.

## Command

```bash
evaluation/summarize_qwen3_8b_wo_c1_ag_ar_rank_sweep.py --evaluations results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r1024/c1_allgather/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r1024/lr_allreduce/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r128/c1_allgather/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r128/lr_allreduce/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r192/c1_allgather/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r192/lr_allreduce/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r256/c1_allgather/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r256/lr_allreduce/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r384/c1_allgather/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r384/lr_allreduce/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r512/c1_allgather/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r512/lr_allreduce/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r640/c1_allgather/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r640/lr_allreduce/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r768/c1_allgather/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r768/lr_allreduce/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r896/c1_allgather/results.json results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r896/lr_allreduce/results.json --existing-ppl results/evaluation/qwen3_8b_wo_c1_uniform_rank_sweep/results.json --dense-quality ICLR-results/qwen3-8b/quality/Q3-8B-Dense/result.json --expected-ranks 128,192,256,384,512,640,768,896,1024 --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/summary
```
