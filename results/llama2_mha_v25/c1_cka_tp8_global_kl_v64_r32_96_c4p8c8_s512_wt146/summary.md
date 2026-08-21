# Llama-2-7B C1 per-TP-source Global-KL allocation

## Outcome

| Schedule | Confirmation KL | WikiText-2 PPL | Changed sources | Padded overhead |
|:---|---:|---:|---:|---:|
| uniform_anchor | 0.161452338 | 6.895931596 | 0 | 0.00% |
| mean_dp | 0.106762856 | 6.600288685 | 232 | 43.75% |
| ucb_dp | 0.110147623 | — | 164 | 41.41% |

Selected: **mean_dp**.

## Selected TP-source ranks

| Layer | Source ranks 0–7 |
|---:|:---|
| 0 | [32, 32, 32, 32, 32, 32, 32, 32] |
| 1 | [32, 32, 48, 32, 80, 32, 96, 48] |
| 2 | [96, 48, 96, 80, 80, 48, 96, 32] |
| 3 | [64, 48, 96, 96, 96, 80, 32, 48] |
| 4 | [64, 80, 80, 80, 96, 32, 96, 96] |
| 5 | [64, 32, 48, 80, 96, 96, 80, 96] |
| 6 | [96, 96, 96, 96, 96, 32, 96, 80] |
| 7 | [32, 32, 64, 80, 32, 96, 32, 32] |
| 8 | [32, 96, 96, 96, 48, 64, 80, 32] |
| 9 | [80, 96, 96, 32, 96, 96, 48, 96] |
| 10 | [32, 96, 80, 48, 96, 80, 96, 96] |
| 11 | [96, 96, 32, 48, 64, 32, 80, 96] |
| 12 | [96, 32, 64, 32, 96, 32, 80, 96] |
| 13 | [96, 96, 96, 96, 80, 48, 64, 80] |
| 14 | [96, 32, 96, 80, 96, 80, 32, 96] |
| 15 | [80, 96, 80, 96, 96, 32, 96, 80] |
| 16 | [80, 64, 96, 96, 96, 80, 96, 96] |
| 17 | [96, 64, 80, 32, 64, 96, 32, 32] |
| 18 | [64, 64, 64, 96, 96, 32, 32, 96] |
| 19 | [96, 32, 80, 48, 80, 96, 48, 64] |
| 20 | [80, 32, 64, 96, 32, 96, 48, 48] |
| 21 | [96, 32, 32, 32, 64, 96, 96, 32] |
| 22 | [32, 32, 80, 32, 48, 80, 96, 64] |
| 23 | [96, 32, 96, 32, 48, 48, 80, 32] |
| 24 | [32, 80, 48, 48, 64, 48, 32, 64] |
| 25 | [96, 80, 96, 32, 32, 32, 32, 48] |
| 26 | [32, 80, 64, 64, 64, 32, 64, 96] |
| 27 | [32, 32, 48, 32, 48, 48, 48, 32] |
| 28 | [32, 48, 48, 80, 32, 32, 96, 32] |
| 29 | [32, 96, 32, 32, 32, 96, 80, 32] |
| 30 | [32, 48, 96, 32, 96, 32, 32, 32] |
| 31 | [32, 32, 32, 32, 96, 32, 96, 32] |

The DP constrains ideal variable-size collective width. Padded rectangular AllGather cost is diagnostic only.
Head allocation: **train_activation_cka_balanced_tp_groups**.

## Command

`evaluation/allocate_llama2_mha_c1_tp_source_global_kl.py --model /home/lz299/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 --windows results/cache/llama2_7b_c4_s2048_fit192_select64_seed20260901/windows.safetensors --snapshot-dir results/llama2_7b_mha_o_proj_ppl_snapshots/c4_fit128_p2048_alltok --head-groups results/llama2_mha_v25/c1_cka_tp8_v64_screen_c4_train64_held64/result.json --output-dir results/llama2_mha_v25/c1_cka_tp8_global_kl_v64_r32_96_c4p8c8_s512_wt146 --factor-dir 32=results/llama2_mha_v25/c1_joint_v32_c4_128x2048_fixedheld_s10_cg16 --factor-dir 48=results/llama2_mha_v25/c1_joint_v48_c4_128x2048_fixedheld_s10_cg16 --factor-dir 64=results/llama2_mha_v25/c1_joint_v64_c4_128x2048_fixedheld_s10_cg16 --factor-dir 80=results/llama2_mha_v25/c1_joint_v80_c4_128x2048_fixedheld_s10_cg16 --factor-dir 96=results/llama2_mha_v25/c1_joint_v96_c4_128x2048_fixedheld_s10_cg16 --anchor-rank 64 --candidate-ranks 32,48,64,80,96 --profile-windows 8 --confirmation-windows 8 --sequence-length 512 --batch-size 4 --fit-windows 128 --covariance-damping 1e-7 --covariance-row-chunk-size 8192 --decoder-relative-jitter 0 --vocab-chunk-size 8192 --eval-seqlen 2048 --eval-max-chunks 146 --torch-num-threads 2 --local-files-only`
