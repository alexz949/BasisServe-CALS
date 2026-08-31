# Llama-2-7B C1 per-TP-source Global-KL allocation

## Outcome

| Schedule | Confirmation KL | WikiText-2 PPL | Changed sources | Padded overhead |
|:---|---:|---:|---:|---:|
| uniform_anchor | 0.161442895 | 6.896116100 | 0 | 0.00% |
| mean_dp | 0.120138462 | 6.578383605 | 213 | 41.41% |
| ucb_dp | 0.124142352 | — | 160 | 35.16% |

Selected: **mean_dp**.

## Selected TP-source ranks

| Layer | Source ranks 0–7 |
|---:|:---|
| 0 | [32, 32, 32, 32, 32, 32, 32, 32] |
| 1 | [32, 32, 32, 80, 80, 64, 96, 96] |
| 2 | [96, 32, 96, 96, 96, 80, 48, 64] |
| 3 | [80, 96, 80, 64, 64, 64, 32, 48] |
| 4 | [48, 96, 96, 80, 96, 96, 80, 96] |
| 5 | [32, 48, 96, 64, 96, 96, 96, 96] |
| 6 | [96, 64, 96, 80, 96, 80, 96, 96] |
| 7 | [32, 32, 96, 32, 96, 64, 96, 48] |
| 8 | [96, 64, 32, 32, 96, 64, 64, 96] |
| 9 | [96, 96, 96, 64, 64, 96, 64, 80] |
| 10 | [80, 80, 96, 64, 96, 96, 96, 64] |
| 11 | [96, 32, 80, 48, 96, 64, 96, 96] |
| 12 | [96, 32, 64, 32, 32, 64, 96, 96] |
| 13 | [96, 48, 96, 96, 96, 96, 48, 96] |
| 14 | [80, 32, 80, 96, 80, 96, 64, 96] |
| 15 | [80, 48, 96, 32, 80, 64, 96, 96] |
| 16 | [96, 96, 96, 96, 96, 64, 64, 80] |
| 17 | [48, 64, 80, 48, 96, 32, 80, 96] |
| 18 | [64, 64, 96, 96, 32, 64, 64, 32] |
| 19 | [64, 80, 32, 80, 64, 32, 96, 64] |
| 20 | [64, 96, 96, 96, 48, 64, 48, 32] |
| 21 | [48, 32, 96, 64, 32, 32, 48, 64] |
| 22 | [32, 48, 80, 32, 32, 48, 96, 32] |
| 23 | [96, 48, 96, 48, 48, 80, 32, 48] |
| 24 | [80, 48, 48, 32, 32, 64, 64, 32] |
| 25 | [48, 32, 80, 48, 32, 48, 32, 32] |
| 26 | [80, 64, 32, 80, 64, 64, 64, 32] |
| 27 | [32, 48, 64, 32, 32, 32, 32, 32] |
| 28 | [80, 96, 32, 96, 32, 48, 48, 32] |
| 29 | [32, 32, 64, 32, 32, 32, 48, 32] |
| 30 | [48, 32, 32, 96, 48, 32, 32, 32] |
| 31 | [32, 32, 32, 32, 32, 32, 96, 32] |

The DP constrains ideal variable-size collective width. Padded rectangular AllGather cost is diagnostic only.

## Command

`evaluation/allocate_llama2_mha_c1_tp_source_global_kl.py --model /home/lz299/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 --windows results/cache/llama2_7b_c4_s2048_fit192_select64_seed20260901/windows.safetensors --snapshot-dir results/llama2_7b_mha_o_proj_ppl_snapshots/c4_fit128_p2048_alltok --output-dir results/llama2_mha_v25/c1_tp_source_global_kl_v64_r32_96_b4_c4p8c8_s512_wt146 --factor-dir 32=results/llama2_mha_v25/c1_joint_v32_c4_128x2048_fixedheld_s10_cg16 --factor-dir 48=results/llama2_mha_v25/c1_joint_v48_c4_128x2048_fixedheld_s10_cg16 --factor-dir 64=results/llama2_mha_v25/c1_joint_v64_c4_128x2048_fixedheld_s10_cg16 --factor-dir 80=results/llama2_mha_v25/c1_joint_v80_c4_128x2048_fixedheld_s10_cg16 --factor-dir 96=results/llama2_mha_v25/c1_joint_v96_c4_128x2048_fixedheld_s10_cg16 --anchor-rank 64 --candidate-ranks 32,48,64,80,96 --profile-windows 8 --confirmation-windows 8 --sequence-length 512 --batch-size 4 --fit-windows 128 --covariance-damping 1e-7 --covariance-row-chunk-size 8192 --decoder-relative-jitter 0 --vocab-chunk-size 8192 --eval-seqlen 2048 --eval-max-chunks 146 --torch-num-threads 2 --local-files-only`
