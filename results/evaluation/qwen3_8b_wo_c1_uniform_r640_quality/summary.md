# Qwen3-8B Wo-only uniform rank-640 quality

Every compressed arm uses the same C4 calibration covariances and uniform rank. V and the KV cache remain dense. Factors are folded into equivalent BF16 `o_proj` weights for quality isolation.

| Arm | WikiText-2 PPL | Change vs dense | Heldout output MSE |
|---|---:|---:|---:|
| `dense` | 7.00202554 | +0.000% | — |
| `joint_c1` | 7.12059246 | +1.693% | 0.025244215 |
| `independent_local_c1` | 7.11920026 | +1.673% | 0.026363483 |
| `wire_matched_lr_allreduce` | 7.22780202 | +3.224% | 0.058503875 |
| `capacity_matched_lr_allreduce` | 7.01337058 | +0.162% | 0.0060047525 |

Protocol:

- WikiText-2 test, concatenated non-overlapping 2048-token chunks.
- BF16 model execution, SDPA attention, FP32 loss accumulation.
- Joint and independent-local C1 use the same private-AllGather wire.

## Command

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_wo_c1_uniform_rank_quality.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --phase1-dir results/checkpoints/qwen3_8b_wo_c1_uniform_r640_s20 --independent-local-dir results/checkpoints/qwen3_8b_wo_c1_independent_local_r640_s20 --output-dir results/evaluation/qwen3_8b_wo_c1_uniform_r640_quality --dataset wikitext2 --split test --seqlen 2048 --batch-size 1 --model-dtype bfloat16 --attn-implementation sdpa --device cuda:0 --torch-num-threads 4 --local-files-only
```
