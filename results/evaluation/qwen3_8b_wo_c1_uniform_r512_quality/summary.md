# Qwen3-8B Wo-only uniform rank-512 quality

Every compressed arm uses the same C4 calibration covariances and uniform rank. V and the KV cache remain dense. Factors are folded into equivalent BF16 `o_proj` weights for quality isolation.

| Arm | WikiText-2 PPL | Change vs dense | Heldout output MSE |
|---|---:|---:|---:|
| `dense` | 7.00202554 | +0.000% | — |
| `joint_c1` | 7.28194919 | +3.998% | 0.045990029 |
| `independent_local_c1` | 7.25140044 | +3.561% | 0.048528799 |
| `wire_matched_lr_allreduce` | 7.54659120 | +7.777% | 0.086512558 |
| `capacity_matched_lr_allreduce` | 7.03970536 | +0.538% | 0.016642956 |

Protocol:

- WikiText-2 test, concatenated non-overlapping 2048-token chunks.
- BF16 model execution, SDPA attention, FP32 loss accumulation.
- Joint and independent-local C1 use the same private-AllGather wire.

## Command

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_wo_c1_uniform_rank_quality.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --phase1-dir results/checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s20 --independent-local-dir results/checkpoints/qwen3_8b_wo_c1_independent_local_r512_joint20 --output-dir results/evaluation/qwen3_8b_wo_c1_uniform_r512_quality --dataset wikitext2 --split test --seqlen 2048 --batch-size 1 --model-dtype bfloat16 --attn-implementation sdpa --device cuda:0 --torch-num-threads 4 --local-files-only
```
